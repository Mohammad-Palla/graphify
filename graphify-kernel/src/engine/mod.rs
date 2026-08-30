//! A config-driven port of `engine.py::_extract_generic`.
//!
//! # Why this exists
//!
//! `_extract_generic` is ONE 3,144-line function driving fifteen
//! `LanguageConfig`s. The `js/` and `py/` walkers each re-derive the slice of it
//! their language reaches, which was right for two languages and is not right
//! for ten: re-deriving the same skeleton ten times is ten chances for it to
//! drift from the Python in a way no single language's parity run would reveal.
//!
//! So this module mirrors the Python's own structure instead. The skeleton is
//! shared and its control flow matches `walk` / `walk_calls` branch for branch;
//! everything a language does differently is either DATA on [`EngineConfig`] --
//! the same fields `LanguageConfig` carries -- or one of the nine [`LangHooks`]
//! methods, which sit at exactly the nine points where the Python has an
//! `_is_<lang>` guard.
//!
//! `js/` and `py/` deliberately stay as they are. They are gated at DIVERGENT 0
//! over 15,000+ files, and rewriting a walker that already passes buys nothing
//! and risks a regression the corpora might not catch.
//!
//! # The hook points, and why exactly these
//!
//! Taken from an inventory of every `_is_<lang>` guard in `walk` and
//! `walk_calls`, not invented:
//!
//! ```text
//! walk
//!   import branch          -> config.import_handler
//!   class, pre-add_node    -> class_metadata      (C# alone: is_nested_type/is_partial)
//!   class, post-edges      -> on_class            (10 languages: inheritance, annotations)
//!   between class & fn     -> before_function     (7: field/property declarations)
//!   function, post-edges   -> on_function         (all: params, return type, annotations)
//!   trailing               -> extra_walk          (7: enum_constant, companion_object, ...)
//! walk_calls
//!   call branch            -> call_info           (8: callee/receiver extraction)
//!   the tgt_nid decision   -> defers              (3: python, csharp, java)
//!   raw_call construction  -> raw_call_extra      (4: lang tag, receiver_type)
//! ```
//!
//! Every method has a no-op default, so a language implements only what its
//! Python guards actually do -- and a language with no guards at all (Lua,
//! Groovy) needs no impl beyond the config.

use std::collections::{HashMap, HashSet};

use tree_sitter::{Language, Node};

use crate::ids::{make_id_ascii, normalize_id_ascii};
use crate::js::ast::text_checked;
use crate::js::emit::{EdgeRow, NodeRow, RawCall, Val};

pub mod calls;
pub mod walk;

pub type R<T> = Result<T, &'static str>;

/// See `js::MAX_DEPTH`. A Rust stack overflow is a SIGSEGV that takes the whole
/// pool worker down, where Python raises a catchable `RecursionError`.
pub const MAX_DEPTH: u32 = 1000;

/// Membership test for a config set.
///
/// A LINEAR scan over a `&'static [&'static str]`, never a binary search. These
/// sets have at most a handful of entries so the scan is free, and the
/// alternative has already cost this project real time: `BUILTIN_GLOBALS` is
/// written grouped by language rather than sorted, and `binary_search` over it
/// silently reported most of the set as absent -- 64 of 264 gson files
/// DIVERGENT, on names as ordinary as `set` and `next`.
#[inline]
pub fn has(set: &'static [&'static str], kind: &str) -> bool {
    set.contains(&kind)
}

/// The data half of `LanguageConfig`, field for field.
///
/// Every field here is the Python attribute of the same name. Fields the Python
/// carries but no ported language uses yet are present anyway, so that adding
/// such a language is a config change rather than a skeleton change.
pub struct EngineConfig {
    /// The kernel's language key, e.g. `"java"`. Matches `languages::supported`.
    pub language: &'static str,
    /// The grammar, as the linked crate provides it.
    pub grammar: fn() -> Language,

    pub class_types: &'static [&'static str],
    pub function_types: &'static [&'static str],
    pub import_types: &'static [&'static str],
    pub call_types: &'static [&'static str],
    pub function_boundary_types: &'static [&'static str],

    pub name_field: &'static str,
    pub name_fallback_child_types: &'static [&'static str],
    pub body_field: &'static str,
    pub body_fallback_child_types: &'static [&'static str],

    pub call_function_field: &'static str,
    pub call_accessor_node_types: &'static [&'static str],
    pub call_accessor_field: &'static str,
    /// Empty string means "unset", matching the Python default `""` rather than
    /// `None`; the Python tests it for truthiness.
    pub call_accessor_object_field: &'static str,

    /// `function_label_parens`: when false a function's label is bare.
    pub function_label_parens: bool,

    pub hooks: &'static (dyn LangHooks + Sync),
}

/// Whether a hook consumed the node. `Handled::Yes` means the Python returned.
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Handled {
    Yes,
    No,
}

/// What one call site resolved to, before the defer decision.
#[derive(Default)]
pub struct CallInfo {
    pub callee_name: Option<String>,
    pub is_member_call: bool,
    pub is_this_field_call: bool,
    pub member_receiver: Option<String>,
    /// Swift's separate receiver slot: the Python writes
    /// `swift_receiver or member_receiver` into the raw_call.
    pub swift_receiver: Option<String>,
    /// Kotlin / C# fully-qualified call prefixes.
    pub qualified_prefix: Option<String>,
}

/// The per-language blocks. Every method defaults to doing nothing, so a
/// language implements only the guards its Python actually has.
pub trait LangHooks {
    /// `config.import_handler`. Runs for a node in `import_types`.
    fn import_handler<'tree>(&self, _ctx: &mut Ctx<'_, 'tree>, _node: Node<'tree>) -> R<()> {
        Ok(())
    }

    /// C#'s `metadata` on a class node, computed BEFORE `add_node`.
    fn class_metadata<'tree>(
        &self,
        _ctx: &Ctx<'_, 'tree>,
        _node: Node<'tree>,
        _parent_class_nid: Option<&str>,
    ) -> R<Vec<(&'static str, Val)>> {
        Ok(Vec::new())
    }

    /// The per-language class block: inheritance, interfaces, annotations,
    /// record components. Runs after the class node and its containment edge,
    /// before the body is recursed into.
    fn on_class<'tree>(
        &self,
        _ctx: &mut Ctx<'_, 'tree>,
        _node: Node<'tree>,
        _class_nid: &str,
        _class_name: &str,
        _line: usize,
    ) -> R<()> {
        Ok(())
    }

    /// The declarations that sit BETWEEN the class and function branches --
    /// fields, properties, annotation elements. Returning `Handled::Yes` means
    /// the Python returned and the node is consumed.
    fn before_function<'tree>(
        &self,
        _ctx: &mut Ctx<'_, 'tree>,
        _node: Node<'tree>,
        _parent_class_nid: Option<&str>,
    ) -> R<Handled> {
        Ok(Handled::No)
    }

    /// The per-language function block: parameter types, return type,
    /// annotations. Runs after the function node and its edge.
    fn on_function<'tree>(
        &self,
        _ctx: &mut Ctx<'_, 'tree>,
        _node: Node<'tree>,
        _func_nid: &str,
        _func_name: &str,
        _line: usize,
        _parent_class_nid: Option<&str>,
    ) -> R<()> {
        Ok(())
    }

    /// The trailing `_<lang>_extra_walk` slot, before the default recurse.
    fn extra_walk<'tree>(
        &self,
        _ctx: &mut Ctx<'_, 'tree>,
        _node: Node<'tree>,
        _parent_class_nid: Option<&str>,
    ) -> R<Handled> {
        Ok(Handled::No)
    }

    /// Extract the callee and receiver from a call node. Returning `None` means
    /// "use the generic accessor path" (`call_function_field` +
    /// `call_accessor_node_types`), which is what a language with no call guard
    /// does.
    fn call_info<'tree>(&self, _ctx: &Ctx<'_, 'tree>, _node: Node<'tree>) -> R<Option<CallInfo>> {
        Ok(None)
    }

    /// Whether this call must defer to receiver-typed cross-file resolution
    /// instead of binding to a bare name here.
    fn defers(&self, _info: &CallInfo) -> bool {
        false
    }

    /// Extra keys appended to a `raw_call`, in Python's order, after `receiver`.
    fn raw_call_extra<'tree>(
        &self,
        _ctx: &Ctx<'_, 'tree>,
        _info: &CallInfo,
        _receiver_types: &HashMap<String, String>,
    ) -> Vec<(&'static str, Val)> {
        Vec::new()
    }
}

/// Everything `_extract_generic` keeps in its local scope for one file.
pub struct Ctx<'a, 'tree> {
    pub cfg: &'static EngineConfig,
    pub src: &'a [u8],
    pub str_path: &'a str,
    pub stem: String,
    pub file_nid: String,

    pub nodes: Vec<NodeRow>,
    pub seen_ids: HashSet<String>,
    pub edges: Vec<EdgeRow>,
    pub raw_calls: Vec<RawCall>,

    pub callable_def_nids: HashSet<String>,
    pub callable_class_nids: HashSet<String>,
    pub function_bodies: Vec<(String, Node<'tree>)>,

    /// `<lang>_field_types`: {class_nid: {field_name: declared_type}}.
    pub field_types: HashMap<String, HashMap<String, String>>,
    /// `<lang>_method_scopes`, keyed by the body's byte range rather than
    /// Python's `id(body)`: unique within one tree and stable across the clone
    /// the call pass makes, which a pointer would not be.
    pub method_scopes: HashMap<(usize, usize), (Node<'tree>, String)>,

    pub label_to_nid: HashMap<String, String>,
    pub nid_to_sf: HashMap<String, String>,
    pub seen_call_pairs: HashSet<(String, String)>,
}

impl<'a, 'tree> Ctx<'a, 'tree> {
    pub fn mkid(&self, parts: &[&str]) -> R<String> {
        make_id_ascii(parts).ok_or("non_ascii_id")
    }

    pub fn text(&self, node: Node) -> R<&'a str> {
        text_checked(node, self.src).ok_or("invalid_utf8_text")
    }

    /// `add_node`. `metadata`, `type` and `scope_chain` are only ever set by the
    /// C# namespace handler, so the extra fields arrive through `class_metadata`
    /// rather than being assumed absent.
    pub fn add_node_meta(
        &mut self,
        nid: &str,
        label: &str,
        line: usize,
        extra: Vec<(&'static str, Val)>,
    ) {
        if !self.seen_ids.insert(nid.to_string()) {
            return;
        }
        let mut fields = vec![
            ("label", Val::S(label.to_string())),
            ("file_type", Val::Static("code")),
            ("source_file", Val::S(self.str_path.to_string())),
            ("source_location", Val::S(format!("L{line}"))),
        ];
        fields.extend(extra);
        self.nodes.push(NodeRow {
            id: nid.to_string(),
            fields,
        });
    }

    pub fn add_node(&mut self, nid: &str, label: &str, line: usize) {
        self.add_node_meta(nid, label, line, Vec::new());
    }

    pub fn add_edge(&mut self, src: &str, tgt: &str, relation: &'static str, line: usize) {
        self.edges.push(EdgeRow {
            source: src.to_string(),
            target: tgt.to_string(),
            relation,
            fields: vec![
                ("confidence", Val::Static("EXTRACTED")),
                ("source_file", Val::S(self.str_path.to_string())),
                ("source_location", Val::S(format!("L{line}"))),
                ("weight", Val::F(1.0)),
            ],
        });
    }

    /// `add_edge(..., context=...)`: same shape, `context` LAST.
    pub fn add_edge_ctx(
        &mut self,
        src: &str,
        tgt: &str,
        relation: &'static str,
        line: usize,
        context: &'static str,
    ) {
        self.edges.push(EdgeRow {
            source: src.to_string(),
            target: tgt.to_string(),
            relation,
            fields: vec![
                ("confidence", Val::Static("EXTRACTED")),
                ("source_file", Val::S(self.str_path.to_string())),
                ("source_location", Val::S(format!("L{line}"))),
                ("weight", Val::F(1.0)),
                ("context", Val::Static(context)),
            ],
        });
    }

    /// The import edge appended directly by a handler: `context` THIRD, right
    /// after `relation`. A different key order from `add_edge_ctx`, and the
    /// order reaches the pickled result, so it is its own emitter.
    pub fn add_import_edge(&mut self, tgt: &str, line: usize) {
        let src = self.file_nid.clone();
        self.edges.push(EdgeRow {
            source: src,
            target: tgt.to_string(),
            relation: "imports",
            fields: vec![
                ("context", Val::Static("import")),
                ("confidence", Val::Static("EXTRACTED")),
                ("source_file", Val::S(self.str_path.to_string())),
                ("source_location", Val::S(format!("L{line}"))),
                ("weight", Val::F(1.0)),
            ],
        });
    }

    /// `ensure_named_node`: a SOURCELESS stub when the name is not defined in
    /// this file, so the corpus-level rewire can collapse it onto the real
    /// definition (#1402). `namespace_stack` is empty for every language ported
    /// here, so the scoped probe is `_make_id(stem, "", name)`.
    pub fn ensure_named_node(&mut self, name: &str, _line: usize) -> R<String> {
        let scoped = self.mkid(&[&self.stem.clone(), "", name])?;
        if self.seen_ids.contains(&scoped) {
            return Ok(scoped);
        }
        let bare = self.mkid(&[name])?;
        if !self.seen_ids.contains(&bare) {
            self.seen_ids.insert(bare.clone());
            self.nodes.push(NodeRow {
                id: bare.clone(),
                fields: vec![
                    ("label", Val::S(name.to_string())),
                    ("file_type", Val::Static("code")),
                    ("source_file", Val::Static("")),
                    ("source_location", Val::Static("")),
                    ("origin_file", Val::S(self.str_path.to_string())),
                ],
            });
        }
        Ok(bare)
    }

    /// The bare `_make_id(stem, base) else _make_id(base)` stub the Java/Groovy
    /// parent emitter uses. NOT `ensure_named_node`: no `origin_file` key, and
    /// the scoped probe omits the empty namespace part.
    pub fn ensure_parent_node(&mut self, name: &str) -> R<String> {
        let scoped = self.mkid(&[&self.stem.clone(), name])?;
        if self.seen_ids.contains(&scoped) {
            return Ok(scoped);
        }
        let bare = self.mkid(&[name])?;
        if !self.seen_ids.contains(&bare) {
            self.seen_ids.insert(bare.clone());
            self.nodes.push(NodeRow {
                id: bare.clone(),
                fields: vec![
                    ("label", Val::S(name.to_string())),
                    ("file_type", Val::Static("code")),
                    ("source_file", Val::Static("")),
                    ("source_location", Val::Static("")),
                ],
            });
        }
        Ok(bare)
    }

    /// The #1899 guard: a name that normalizes to nothing would collapse
    /// `_make_id(prefix, name)` onto its path-derived prefix.
    pub fn normalizes_to_something(&self, name: &str) -> R<bool> {
        Ok(!normalize_id_ascii(name).ok_or("non_ascii_id")?.is_empty())
    }
}

// ── The per-file driver ──────────────────────────────────────────────────────

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tree_sitter::Parser;

use crate::ids::file_stem;
use crate::js::ast::children;
use crate::js::emit;
use crate::Outcome;

fn tree_depth(root: Node) -> u32 {
    let mut stack = vec![(root, 1u32)];
    let mut max = 1u32;
    while let Some((n, d)) = stack.pop() {
        if d > max {
            max = d;
            if max > MAX_DEPTH {
                return max;
            }
        }
        for c in children(n) {
            stack.push((c, d + 1));
        }
    }
    max
}

/// The whole per-file pipeline, shared by every language on this engine.
///
/// `receiver_types_for` is the one thing that varies structurally: Java and C#
/// build a per-method receiver table from the class's field types, and other
/// languages build nothing. It is a plain function rather than a hook because it
/// runs BETWEEN the two walks, not inside either.
pub fn run<'py>(
    py: Python<'py>,
    cfg: &'static EngineConfig,
    path: &str,
    source: &[u8],
    receiver_types_for: fn(&Ctx, Node, &HashMap<String, String>) -> R<HashMap<String, String>>,
) -> PyResult<Outcome<'py>> {
    match extract(py, cfg, path, source, receiver_types_for) {
        Ok(dict) => Ok(Outcome::Native(dict)),
        Err(reason) => Ok(Outcome::Defer(reason)),
    }
}

fn extract<'py>(
    py: Python<'py>,
    cfg: &'static EngineConfig,
    path: &str,
    source: &[u8],
    receiver_types_for: fn(&Ctx, Node, &HashMap<String, String>) -> R<HashMap<String, String>>,
) -> Result<Bound<'py, PyDict>, &'static str> {
    // One validation of the whole buffer makes every later `text()` sound by
    // construction -- see `js::extract` for the U+FFFD divergence this prevents.
    if std::str::from_utf8(source).is_err() {
        return Err("source_not_utf8");
    }
    let stem = file_stem(path).ok_or("path_needs_pathlib")?;
    let file_nid = make_id_ascii(&[path]).ok_or("non_ascii_path")?;

    let mut parser = Parser::new();
    parser
        .set_language(&(cfg.grammar)())
        .map_err(|_| "grammar_load_failed")?;
    let tree = parser.parse(source, None).ok_or("parse_failed")?;
    let root = tree.root_node();
    // Python attaches a `parse_errors` block and keeps going; its recovery is
    // authoritative and reproducing it is a separate surface, so defer.
    if root.has_error() {
        return Err("parse_error");
    }
    if tree_depth(root) > MAX_DEPTH {
        return Err("tree_too_deep");
    }

    let mut ctx = Ctx {
        cfg,
        src: source,
        str_path: path,
        stem,
        file_nid: file_nid.clone(),
        nodes: Vec::new(),
        seen_ids: HashSet::new(),
        edges: Vec::new(),
        raw_calls: Vec::new(),
        callable_def_nids: HashSet::new(),
        callable_class_nids: HashSet::new(),
        function_bodies: Vec::new(),
        field_types: HashMap::new(),
        method_scopes: HashMap::new(),
        label_to_nid: HashMap::new(),
        nid_to_sf: HashMap::new(),
        seen_call_pairs: HashSet::new(),
    };

    let file_label = path.rsplit('/').next().unwrap_or(path).to_string();
    ctx.add_node(&file_nid, &file_label, 1);

    walk::walk(&mut ctx, root, None)?;

    // ── Call-graph pass ─────────────────────────────────────────────────────
    // `label_to_nid` is built from the COMPLETED node list, exactly as in
    // Python: a call may name something declared later in the file.
    for n in &ctx.nodes {
        let mut sf = String::new();
        let mut label = String::new();
        for (k, v) in &n.fields {
            match (*k, v) {
                ("source_file", Val::S(s)) => sf = s.clone(),
                ("source_file", Val::Static(s)) => sf = s.to_string(),
                ("label", Val::S(s)) => label = s.clone(),
                _ => {}
            }
        }
        ctx.nid_to_sf.insert(n.id.clone(), sf);
        let normalised = label.trim_matches(|c| c == '(' || c == ')').trim_start_matches('.');
        ctx.label_to_nid.insert(normalised.to_string(), n.id.clone());
    }

    // Every per-method receiver table is built BEFORE any body is walked, as in
    // Python, where the whole dict comprehension is evaluated up front.
    let scopes: Vec<((usize, usize), (Node, String))> = ctx
        .method_scopes
        .iter()
        .map(|(k, (n, c))| (*k, (*n, c.clone())))
        .collect();
    let empty: HashMap<String, String> = HashMap::new();
    let mut tables: HashMap<(usize, usize), HashMap<String, String>> = HashMap::new();
    for (body_key, (method_node, class_nid)) in scopes {
        let fields = ctx.field_types.get(&class_nid).unwrap_or(&empty).clone();
        tables.insert(body_key, receiver_types_for(&ctx, method_node, &fields)?);
    }

    let bodies: Vec<(String, Node)> = ctx.function_bodies.clone();
    for (caller_nid, body) in bodies {
        let key = (body.start_byte(), body.end_byte());
        let table = tables.get(&key).cloned().unwrap_or_default();
        calls::walk_calls(&mut ctx, body, &caller_nid, &table)?;
    }

    // ── Clean edges ─────────────────────────────────────────────────────────
    let mut clean: Vec<&EdgeRow> = Vec::with_capacity(ctx.edges.len());
    for e in &ctx.edges {
        let target_ok = ctx.seen_ids.contains(&e.target)
            || matches!(e.relation, "imports" | "imports_from" | "re_exports");
        if ctx.seen_ids.contains(&e.source) && target_ok {
            clean.push(e);
        }
    }

    let out = PyDict::new(py);
    let nodes = PyList::empty(py);
    for n in &ctx.nodes {
        let is_callable = ctx.callable_def_nids.contains(&n.id);
        let is_class = ctx.callable_class_nids.contains(&n.id);
        nodes
            .append(emit::node_to_py(py, n, is_callable, is_class).map_err(|_| "py_error")?)
            .map_err(|_| "py_error")?;
    }
    let edges = PyList::empty(py);
    for e in clean {
        edges
            .append(emit::edge_to_py(py, e).map_err(|_| "py_error")?)
            .map_err(|_| "py_error")?;
    }
    let raw_calls = PyList::empty(py);
    for c in &ctx.raw_calls {
        raw_calls
            .append(emit::raw_call_to_py(py, c).map_err(|_| "py_error")?)
            .map_err(|_| "py_error")?;
    }
    out.set_item("nodes", nodes).map_err(|_| "py_error")?;
    out.set_item("edges", edges).map_err(|_| "py_error")?;
    out.set_item("raw_calls", raw_calls).map_err(|_| "py_error")?;
    Ok(out)
}

/// The default for a language that builds no receiver table.
pub fn no_receiver_types(
    _ctx: &Ctx,
    _method_node: Node,
    _fields: &HashMap<String, String>,
) -> R<HashMap<String, String>> {
    Ok(HashMap::new())
}

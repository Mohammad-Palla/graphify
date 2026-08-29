//! The Python native walker.
//!
//! A transliteration of the Python-live slice of `engine.py::_extract_generic`
//! -- the branches reachable when `config` is `_PYTHON_CONFIG` -- plus the
//! `_python_*` helpers they call. Same two rules as the JS walker:
//!
//! 1. **Recurse by default, defer on a guard.** Python's `walk` ends in
//!    `for child in node.children: walk(child, None)`, so a declaration nested
//!    inside an `if`, a `try` or a `with` is found exactly as a top-level one is.
//! 2. **Defer with a reason, never guess.** Anything whose Python behaviour is
//!    not reproduced here returns `Err(reason)` and the whole file goes to
//!    Python.
//!
//! # What makes the Python slice much smaller than the JS one
//!
//! `_extract_generic` is ~3,100 lines covering fourteen languages, but almost all
//! of it sits behind `_is_csharp` / `_is_java` / `_is_swift` / ... guards that are
//! dead for Python. `_PYTHON_CONFIG` also leaves `static_prop_types`,
//! `helper_fn_names`, `container_bind_methods` and `event_listener_properties`
//! empty, which removes four more whole branches, and it sets no
//! `sanitize_symbol_name_fn`, `resolve_function_name_fn` or `extra_walk_fn`.
//! `namespace_stack` and `scope_stack` are only ever pushed by the C# namespace
//! handler, so for Python they are permanently empty -- which is why `add_node`
//! here never emits `metadata`, `type` or `scope_chain`.
//!
//! # Why import resolution is a Python callback
//!
//! `_import_python` resolves a RELATIVE import by walking `Path.parent`, joining
//! a dotted module name and probing the filesystem
//! (`_probe_python_module_candidate`: `is_dir()`, `__init__.py`, `is_file()`).
//! Reimplementing pathlib's normalization plus those probes would be a large
//! surface for no gain -- it is I/O, not walking -- so it stays in Python behind
//! [`imports::Resolver`], exactly as the JS walker does for
//! `_resolve_js_import_target`.

use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tree_sitter::{Node, Parser};

use crate::ids::{file_stem, make_id_ascii, normalize_id_ascii};
use crate::js::ast::{children, text_checked};
use crate::js::emit::{self, EdgeRow, NodeRow, RawCall, Val};
use crate::Outcome;

pub mod calls;
pub mod facts;
pub mod helpers;
pub mod imports;
pub mod rationale;
pub mod walk;
pub mod xfile;

/// See `js::MAX_DEPTH`. Same reasoning, same bound: a Rust stack overflow is a
/// SIGSEGV that takes the whole pool worker down, where Python raises a catchable
/// `RecursionError`. Staying an order of magnitude below Python's 10,000 limit
/// keeps us out of the band where the two sides could disagree about whether a
/// file is extractable at all.
const MAX_DEPTH: u32 = 1000;

pub type R<T> = Result<T, &'static str>;

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

/// Everything `_extract_generic` keeps in its local scope for one Python file.
pub struct Ctx<'a, 'tree> {
    pub src: &'a [u8],
    pub str_path: &'a str,
    pub stem: String,
    pub file_nid: String,
    pub res: &'a imports::Resolver<'a>,

    pub nodes: Vec<NodeRow>,
    pub seen_ids: HashSet<String>,
    pub edges: Vec<EdgeRow>,
    pub raw_calls: Vec<RawCall>,

    pub callable_def_nids: HashSet<String>,
    pub callable_class_nids: HashSet<String>,
    pub local_bound_names: HashMap<String, HashSet<String>>,
    pub function_bodies: Vec<(String, Node<'tree>)>,

    // Call-graph pass. Empty until the declaration walk has finished, exactly as
    // in Python, where `label_to_nid` is built from the completed `nodes` list.
    pub label_to_nid: HashMap<String, String>,
    pub nid_to_sf: HashMap<String, String>,
    pub seen_call_pairs: HashSet<(String, String)>,
    pub seen_indirect_pairs: HashSet<(String, String)>,
}

impl<'a, 'tree> Ctx<'a, 'tree> {
    pub fn mkid(&self, parts: &[&str]) -> R<String> {
        make_id_ascii(parts).ok_or("non_ascii_id")
    }

    pub fn text(&self, node: Node) -> R<&'a str> {
        text_checked(node, self.src).ok_or("invalid_utf8_text")
    }

    /// `add_node`. For Python `metadata`, `type` and `scope_chain` are never set
    /// (see the module docstring), so the dict is always these five keys.
    pub fn add_node(&mut self, nid: &str, label: &str, line: usize) {
        if !self.seen_ids.insert(nid.to_string()) {
            return;
        }
        self.nodes.push(NodeRow {
            id: nid.to_string(),
            fields: vec![
                ("label", Val::S(label.to_string())),
                ("file_type", Val::Static("code")),
                ("source_file", Val::S(self.str_path.to_string())),
                ("source_location", Val::S(format!("L{line}"))),
            ],
        });
    }

    /// `add_edge` with its default confidence/weight, in ITS key order --
    /// `context` AFTER `weight`. `_semantic_reference_edge` and the `calls` /
    /// `indirect_call` literals use different orders and are built at their own
    /// call sites; see `emit.rs` for why that matters.
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

    /// `add_edge(..., context=...)`: same shape, `context` last.
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

    /// `_semantic_reference_edge(...)`: a DIFFERENT key order from `add_edge` --
    /// `context` third, right after `relation`. Appended straight to `edges` in
    /// Python, bypassing `add_edge`, which is why it is its own emitter here.
    pub fn add_semantic_reference_edge(
        &mut self,
        src: &str,
        tgt: &str,
        context: &'static str,
        line: usize,
    ) {
        self.edges.push(EdgeRow {
            source: src.to_string(),
            target: tgt.to_string(),
            relation: "references",
            fields: vec![
                ("context", Val::Static(context)),
                ("confidence", Val::Static("EXTRACTED")),
                ("source_file", Val::S(self.str_path.to_string())),
                ("source_location", Val::S(format!("L{line}"))),
                ("weight", Val::F(1.0)),
            ],
        });
    }

    /// `ensure_named_node`. Emits a SOURCELESS stub when the name is not defined
    /// in this file, so the corpus-level rewire can collapse it onto the real
    /// definition (#1402: a sourced stub bakes the referencing file's path into
    /// the id and blocks the rewire).
    ///
    /// `namespace_stack` is always empty for Python, so the first id is
    /// `_make_id(stem, "", name)` -- `make_id_ascii` drops the empty part, which
    /// is what Python's `if p` filter does.
    /// `line` is unused, exactly as in Python: the stub is emitted SOURCELESS
    /// (`source_file` and `source_location` both empty), so there is nowhere for
    /// it to go. Kept in the signature to match the call sites one-for-one.
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

    /// `normalize_id(name)` truthiness -- the #1899 guard that keeps a name which
    /// normalizes to nothing from collapsing an id onto its (path-derived) prefix.
    pub fn normalizes_to_something(&self, name: &str) -> R<bool> {
        Ok(!normalize_id_ascii(name).ok_or("non_ascii_id")?.is_empty())
    }
}

pub fn walk_python<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
    res: &crate::Resolvers<'py>,
) -> PyResult<Outcome<'py>> {
    match extract(py, path, source, &res.py) {
        Ok(dict) => Ok(Outcome::Native(dict)),
        Err(reason) => Ok(Outcome::Defer(reason)),
    }
}

fn extract<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
    res: &imports::Resolver<'py>,
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
        .set_language(&tree_sitter_python::LANGUAGE.into())
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
        src: source,
        str_path: path,
        stem,
        file_nid: file_nid.clone(),
        res,
        nodes: Vec::new(),
        seen_ids: HashSet::new(),
        edges: Vec::new(),
        raw_calls: Vec::new(),
        callable_def_nids: HashSet::new(),
        callable_class_nids: HashSet::new(),
        local_bound_names: HashMap::new(),
        function_bodies: Vec::new(),
        label_to_nid: HashMap::new(),
        nid_to_sf: HashMap::new(),
        seen_call_pairs: HashSet::new(),
        seen_indirect_pairs: HashSet::new(),
    };

    // `add_node(file_nid, path.name, 1)`. `path.name` is the last component; the
    // walker only ever sees paths produced by a directory walk, and `file_stem`
    // above has already refused anything pathlib would renormalize.
    let file_label = path.rsplit('/').next().unwrap_or(path).to_string();
    ctx.add_node(&file_nid, &file_label, 1);

    walk::walk(&mut ctx, root, None)?;

    // ── Call-graph pass ─────────────────────────────────────────────────────
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
        // `type == "namespace"` nodes are skipped in Python; only the C#
        // namespace handler emits one, so there is nothing to skip here.
        let normalised = label.trim_matches(|c| c == '(' || c == ')').trim_start_matches('.');
        ctx.label_to_nid.insert(normalised.to_string(), n.id.clone());
    }

    let bodies: Vec<(String, Node)> = ctx.function_bodies.clone();
    for (caller_nid, body) in bodies {
        calls::walk_calls(&mut ctx, body, &caller_nid, &HashSet::new())?;
    }

    // ── Module-level dispatch tables (#1566) ────────────────────────────────
    calls::scan_module_dispatch(&mut ctx, root)?;

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

    // Docstring / rationale-comment material, from THIS parse. Without it
    // `_extract_python_rationale` parses the file a second time, because the
    // kernel returns before `_extract_generic` fills `parsed_out`.
    //
    // A SEPARATE deferral axis: a file this pass cannot handle simply omits the
    // key and Python's rationale pass runs for that file alone, while its nodes
    // and edges still come from here.
    match rationale::collect(&ctx, root) {
        Ok(items) => {
            let payload = rationale::to_py(py, &items).map_err(|_| "py_error")?;
            out.set_item("py_rationale", payload).map_err(|_| "py_error")?;
        }
        Err(_reason) => { /* rationale defers; Python collects it for this file */ }
    }

    // Cross-file import material, from THIS parse. Without it
    // `_resolve_cross_file_imports` parses and walks every Python file a THIRD
    // time, in the serial parent -- 7.1s of django's phase 3. A third separate
    // deferral axis, for the same reason as the other two.
    match xfile::collect(&ctx, root) {
        Ok(x) => {
            let payload = xfile::to_py(py, &x).map_err(|_| "py_error")?;
            out.set_item("py_xfile", payload).map_err(|_| "py_error")?;
        }
        Err(_reason) => { /* xfile defers; the parent parses this file itself */ }
    }

    // Symbol-resolution facts, from THIS parse. Without them
    // `_collect_python_facts_one_file` parses every Python file a SECOND time.
    // A fourth separate deferral axis.
    match facts::collect(&ctx, root) {
        Ok(f) => {
            let payload = facts::to_py(py, &f).map_err(|_| "py_error")?;
            out.set_item("py_symbol_facts", payload).map_err(|_| "py_error")?;
        }
        Err(_reason) => { /* facts defer; Python collects them for this file */ }
    }
    Ok(out)
}

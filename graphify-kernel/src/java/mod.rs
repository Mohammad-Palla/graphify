//! The Java native walker.
//!
//! A transliteration of the Java-live slice of `engine.py::_extract_generic` --
//! the branches reachable when `config` is `_JAVA_CONFIG` -- plus the `_java_*`
//! helpers they call. Same two rules as the JS and Python walkers:
//!
//! 1. **Recurse by default, defer on a guard.**
//! 2. **Defer with a reason, never guess.**
//!
//! # What `_JAVA_CONFIG` switches off
//!
//! `static_prop_types`, `helper_fn_names`, `container_bind_methods`,
//! `event_listener_properties`, `name_fallback_child_types`,
//! `body_fallback_child_types` and `call_accessor_node_types` are all EMPTY, and
//! `resolve_function_name_fn`, `sanitize_symbol_name_fn` and `extra_walk_fn` are
//! all `None`. Those branches are therefore absent below rather than ported as
//! dead code -- if a future config gives Java any of them, this walker must be
//! updated, which is why they are named here.
//!
//! `extra_walk_fn` being `None` does not mean Java has no extra walk:
//! `_java_extra_walk` is called from `walk` directly, behind an `_is_java`
//! guard, not through the config. It handles `enum_constant`.
//!
//! `namespace_stack` and `scope_stack` are only ever pushed by the C# namespace
//! handler, so for Java they are permanently empty -- which is why `add_node`
//! here never emits `metadata`, `type` or `scope_chain`.
//!
//! # No import resolver
//!
//! Unlike JS and Python, `_import_java` touches no filesystem: a Java import
//! names a package and the handler keeps only its last dotted segment. So there
//! is no callback into Python here and no I/O to defer on.

use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tree_sitter::{Node, Parser};

use crate::ids::{file_stem, make_id_ascii, normalize_id_ascii};
use crate::js::ast::{children, text_checked};
use crate::js::emit::{self, EdgeRow, NodeRow, RawCall, Val};
use crate::Outcome;

pub mod calls;
pub mod consts;
pub mod helpers;
pub mod imports;
pub mod walk;

/// See `js::MAX_DEPTH`. Same reasoning, same bound.
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

/// Everything `_extract_generic` keeps in its local scope for one Java file.
pub struct Ctx<'a, 'tree> {
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

    /// `java_field_types`: {class_nid: {field_name: declared_type}}. Built by the
    /// declaration walk, read by the call pass for receiver resolution.
    pub java_field_types: HashMap<String, HashMap<String, String>>,
    /// `java_method_scopes`, keyed by the body node's id in Python. Keyed here by
    /// the body's byte range, which is unique within one tree and, unlike a
    /// pointer, is stable across the clone in the call pass.
    pub java_method_scopes: HashMap<(usize, usize), (Node<'tree>, String)>,

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

    /// `add_node`. For Java `metadata`, `type` and `scope_chain` are never set
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

    /// `add_edge` with its default confidence/weight, `context` absent.
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

    /// The import edge `_import_java` appends directly. It carries `context`
    /// THIRD -- right after `relation` -- which is a different key order from
    /// `add_edge`, so it is built here rather than reusing one.
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

    /// `ensure_named_node`. Emits a SOURCELESS stub when the name is not defined
    /// in this file, so the corpus-level rewire can collapse it onto the real
    /// definition (#1402).
    ///
    /// `namespace_stack` is always empty for Java, so the first id is
    /// `_make_id(stem, "", name)`.
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

    /// The bare `_make_id(stem, base) else _make_id(base)` stub the Java parent
    /// emitter uses. NOT `ensure_named_node`: it emits no `origin_file` key.
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

    pub fn normalizes_to_something(&self, name: &str) -> R<bool> {
        Ok(!normalize_id_ascii(name).ok_or("non_ascii_id")?.is_empty())
    }
}

pub fn walk_java<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
    _res: &crate::Resolvers<'py>,
) -> PyResult<Outcome<'py>> {
    match extract(py, path, source) {
        Ok(dict) => Ok(Outcome::Native(dict)),
        Err(reason) => Ok(Outcome::Defer(reason)),
    }
}

fn extract<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
) -> Result<Bound<'py, PyDict>, &'static str> {
    if std::str::from_utf8(source).is_err() {
        return Err("source_not_utf8");
    }
    let stem = file_stem(path).ok_or("path_needs_pathlib")?;
    let file_nid = make_id_ascii(&[path]).ok_or("non_ascii_path")?;

    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_java::LANGUAGE.into())
        .map_err(|_| "grammar_load_failed")?;
    let tree = parser.parse(source, None).ok_or("parse_failed")?;
    let root = tree.root_node();
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
        nodes: Vec::new(),
        seen_ids: HashSet::new(),
        edges: Vec::new(),
        raw_calls: Vec::new(),
        callable_def_nids: HashSet::new(),
        callable_class_nids: HashSet::new(),
        function_bodies: Vec::new(),
        java_field_types: HashMap::new(),
        java_method_scopes: HashMap::new(),
        label_to_nid: HashMap::new(),
        nid_to_sf: HashMap::new(),
        seen_call_pairs: HashSet::new(),
        seen_indirect_pairs: HashSet::new(),
    };

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
        let normalised = label.trim_matches(|c| c == '(' || c == ')').trim_start_matches('.');
        ctx.label_to_nid.insert(normalised.to_string(), n.id.clone());
    }

    // `java_receiver_types`: one table per method body, built BEFORE any body is
    // walked (Python builds the whole dict comprehension first), because a
    // method's table is derived from its class's field types and every class has
    // been walked by now.
    let scopes: Vec<((usize, usize), (Node, String))> = ctx
        .java_method_scopes
        .iter()
        .map(|(k, (n, c))| (*k, (*n, c.clone())))
        .collect();
    let empty_fields: HashMap<String, String> = HashMap::new();
    let mut receiver_types_by_body: HashMap<(usize, usize), HashMap<String, String>> =
        HashMap::new();
    for (body_key, (method_node, class_nid)) in scopes {
        let fields = ctx.java_field_types.get(&class_nid).unwrap_or(&empty_fields).clone();
        let table = calls::method_receiver_types(&ctx, method_node, &fields)?;
        receiver_types_by_body.insert(body_key, table);
    }

    let bodies: Vec<(String, Node)> = ctx.function_bodies.clone();
    let empty_table: HashMap<String, String> = HashMap::new();
    for (caller_nid, body) in bodies {
        let key = (body.start_byte(), body.end_byte());
        let table = receiver_types_by_body.get(&key).unwrap_or(&empty_table).clone();
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

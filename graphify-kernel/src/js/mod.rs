//! The JS/TS native walker.
//!
//! This is a transliteration of the TypeScript-live slice of
//! `engine.py::_extract_generic` -- its `walk`, its `walk_calls`, and the
//! `_js_*` / `_ts_*` helpers they call -- not a reimplementation of what those
//! functions are *for*. Where the Python looks redundant (a singular
//! `parameter` field checked alongside the plural `parameters`, a `seen_ids`
//! probe before an enum member) the port keeps it, because in every one of those
//! places the Python carries an issue number for a wrong edge that the shape
//! prevents.
//!
//! # The two rules that make this safe
//!
//! 1. **Recurse by default, defer on a guard.** Python's `walk` ends in
//!    `for child in node.children: walk(child)`, so a declaration nested inside
//!    an `if`, a `try` or a bare block is found and emitted exactly as a
//!    top-level one is. An earlier version of this walker SKIPPED statement
//!    kinds outright on the strength of a probe showing `if (a) { helper(); }`
//!    emits nothing -- the probe tested a statement containing a CALL, not one
//!    containing a DECLARATION, and one file silently lost 45 raw_calls and
//!    every node but the file node. So the only kinds that stop the walk here
//!    are the ones Python has an explicit branch for.
//! 2. **Defer with a reason, never guess.** Every construct whose Python
//!    behaviour is not reproduced below returns `Err(reason)`, which becomes a
//!    whole-file deferral. A missing rule must cost a deferral, never a
//!    plausible-looking node.
//!
//! # Why import resolution is a Python callback
//!
//! `_import_js` resolves each specifier through `_resolve_js_import_target`,
//! which probes the filesystem: extension candidates, index files, `is_file()`
//! tests, tsconfig `paths` aliases, workspace package manifests. That work is
//! I/O-bound -- filesystem calls measured ~5% of phase 2 against ~88% for the
//! walk -- so porting it would reproduce a large, heavily special-cased surface
//! for almost no gain. It stays in Python, reached through the memoized
//! [`imports::Resolver`] callback. See that module for why a callback beats
//! pre-resolving a specifier list.

use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tree_sitter::{Node, Parser};

use crate::ids::{file_stem, make_id_ascii, normalize_id_ascii};
use crate::Outcome;

pub mod ast;
pub mod calls;
pub mod emit;
pub mod imports;
pub mod pat;
pub mod walk;

use ast::{children, text_checked};
use emit::{EdgeRow, NodeRow, RawCall, Val};
use imports::Resolver;
use walk::Dialect;

/// The deepest AST this walker will recurse into. Past it, the file defers.
///
/// Every traversal here is recursive, mirroring Python's, so a deep enough tree
/// overflows the Rust stack -- and a Rust stack overflow is a SIGSEGV, not a
/// catchable exception. That is uniquely bad in this design: `try_extract`
/// converts any Python-visible failure into a deferral, but a segfault takes the
/// whole pool worker down and `ProcessPoolExecutor` reports it as a
/// `BrokenProcessPool` for every file that worker held. Graphify's own limit is a
/// `RecursionError` (`sys.setrecursionlimit(10_000)`), which `_safe_extract`
/// catches and turns into an error result.
///
/// 1000 sits deliberately between the two. Measured over 11,266 JS/TS files in
/// the Bun corpus, the deepest real tree is **230** levels and only 21 files
/// exceed 50 -- so the bound costs nothing. The one file past it is a deliberate
/// 9 MB fixture of ~100k nested `for` loops (depth 320,004) on which Python
/// raises `RecursionError`; deferring hands it back to Python, which produces the
/// authoritative error result it always did.
///
/// Staying an order of magnitude under Python's 10,000 also keeps us out of the
/// band where the two implementations could disagree about whether a file is
/// extractable at all: anything the kernel accepts, Python would too.
const MAX_DEPTH: u32 = 1000;

/// The deepest path in `root`, measured iteratively.
///
/// Checked once, before any recursive traversal runs, because there are nine
/// mutually-recursive walkers here (`walk`, `walk_calls`, the pattern collectors,
/// the nested-declaration scan, ...) and threading a depth counter through all of
/// them would make stack safety depend on not having missed one. This makes it
/// depend on a single test instead.
fn tree_depth(root: Node) -> u32 {
    let mut stack = vec![(root, 1u32)];
    let mut max = 1u32;
    while let Some((n, d)) = stack.pop() {
        if d > max {
            max = d;
            if max > MAX_DEPTH {
                return max; // no need to find out HOW deep
            }
        }
        for c in children(n) {
            stack.push((c, d + 1));
        }
    }
    max
}

/// A deferral reason. `Err` everywhere in this module means "hand the file to
/// Python", with a `&'static str` naming the construct so the counters can rank
/// the gaps instead of reporting one opaque percentage.
pub type R<T> = Result<T, &'static str>;

/// An insertion-ordered string map, for the payloads that reach Python as a
/// `dict` whose key order is part of the output (`ts_type_table`).
#[derive(Default)]
pub struct OrderedMap {
    order: Vec<(String, String)>,
    keys: HashSet<String>,
}

impl OrderedMap {
    /// `table.setdefault`-with-a-guard: Python's callers all test
    /// `if name not in table` first, so a second binding of the same name never
    /// wins.
    pub fn insert_if_absent(&mut self, k: &str, v: &str) {
        if self.keys.insert(k.to_string()) {
            self.order.push((k.to_string(), v.to_string()));
        }
    }
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
    pub fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (k, v) in &self.order {
            d.set_item(k, v)?;
        }
        Ok(d)
    }
}

/// Everything `_extract_generic` keeps in its local scope for one file.
///
/// Held as one struct rather than threaded as arguments because the Python is a
/// closure over ~30 mutable locals, and passing them individually would let a
/// call site quietly read a stale copy of one -- the exact shape of the five
/// silent-wrong-output defects this project's earlier pool work produced.
pub struct Ctx<'a, 'tree> {
    pub src: &'a [u8],
    pub str_path: &'a str,
    pub stem: String,
    pub file_nid: String,
    pub res: &'a Resolver<'a>,
    pub dialect: Dialect,

    pub nodes: Vec<NodeRow>,
    pub seen_ids: HashSet<String>,
    pub edges: Vec<EdgeRow>,
    pub raw_calls: Vec<RawCall>,

    pub callable_def_nids: HashSet<String>,
    pub callable_class_nids: HashSet<String>,
    pub local_bound_names: HashMap<String, HashSet<String>>,
    pub closure_locals_by_body: HashMap<usize, HashSet<String>>,
    pub function_bodies: Vec<(String, Node<'tree>)>,
    pub type_table: OrderedMap,
    pub js_external_imports: HashSet<String>,

    // Call-graph pass. Empty until the declaration walk has finished, exactly as
    // in Python, where `label_to_nid` is built from the completed `nodes` list.
    pub label_to_nid: HashMap<String, String>,
    pub nid_to_sf: HashMap<String, String>,
    pub seen_call_pairs: HashSet<(String, String)>,
    pub seen_indirect_pairs: HashSet<(String, String)>,
    pub seen_dyn_import_pairs: HashSet<(String, String)>,
    pub tracked_body_ids: HashSet<usize>,
}

impl<'a, 'tree> Ctx<'a, 'tree> {
    /// `_make_id(*parts)`, deferring rather than guessing on non-ASCII.
    pub fn mkid(&self, parts: &[&str]) -> R<String> {
        make_id_ascii(parts).ok_or("non_ascii_id")
    }

    /// `add_node`. Returns early when the id is already known, like Python.
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

    /// `add_edge` with its default confidence/weight, in ITS key order:
    /// `context` last, after `weight`. The import and call literals in Python use
    /// a different order and are built by their own emitters.
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

    pub fn text(&self, node: Node) -> R<&'a str> {
        text_checked(node, self.src).ok_or("invalid_utf8_text")
    }

    /// `normalize_id(name)` truthiness -- the guard that keeps a minified `$`
    /// from collapsing an id onto its parent and leaking the scan path (#1899).
    pub fn normalizes_to_something(&self, name: &str) -> R<bool> {
        Ok(!normalize_id_ascii(name).ok_or("non_ascii_id")?.is_empty())
    }
}

/// Parse and run exactly one full-tree traversal. See `lib.rs::debug_traversal_cost`.
pub fn debug_traversal_cost(source: &[u8], language: &str) -> Option<u32> {
    let lang: tree_sitter::Language = match language {
        "typescript" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        "tsx" => tree_sitter_typescript::LANGUAGE_TSX.into(),
        "javascript" => tree_sitter_javascript::LANGUAGE.into(),
        _ => return None,
    };
    let mut parser = Parser::new();
    parser.set_language(&lang).ok()?;
    let tree = parser.parse(source, None)?;
    Some(tree_depth(tree.root_node()))
}

/// The entry points registered in `languages::walker_for`.
///
/// One walker, three grammars. `.tsx` needs its own grammar rather than the plain
/// TypeScript one -- `language_typescript` silently fails on JSX expressions and
/// drops any call nested inside them -- and `.js` needs both a different grammar
/// and the `_JS_CONFIG` dispatch sets. Nothing else differs, so the three are
/// thin wrappers over the same [`extract`] rather than three walkers that could
/// drift apart.
pub fn walk_typescript<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
    resolver: &Resolver<'py>,
) -> PyResult<Outcome<'py>> {
    let lang = tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into();
    finish(extract(py, path, source, resolver, &lang, Dialect::TypeScript))
}

pub fn walk_tsx<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
    resolver: &Resolver<'py>,
) -> PyResult<Outcome<'py>> {
    let lang = tree_sitter_typescript::LANGUAGE_TSX.into();
    finish(extract(py, path, source, resolver, &lang, Dialect::TypeScript))
}

pub fn walk_javascript<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
    resolver: &Resolver<'py>,
) -> PyResult<Outcome<'py>> {
    let lang = tree_sitter_javascript::LANGUAGE.into();
    finish(extract(py, path, source, resolver, &lang, Dialect::JavaScript))
}

fn finish<'py>(r: Result<Bound<'py, PyDict>, &'static str>) -> PyResult<Outcome<'py>> {
    match r {
        Ok(dict) => Ok(Outcome::Native(dict)),
        Err(reason) => Ok(Outcome::Defer(reason)),
    }
}

fn extract<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
    res: &Resolver<'py>,
    lang: &tree_sitter::Language,
    dialect: Dialect,
) -> Result<Bound<'py, PyDict>, &'static str> {
    // Every `text()` below reads a byte range as `&str`. Python's `_read_text`
    // decodes with `errors="replace"`, turning invalid bytes into U+FFFD, which
    // Rust cannot do without allocating per call -- so the helpers were quietly
    // yielding "" instead, and an identifier containing invalid UTF-8 would have
    // produced a different name, a different shadow set, and a different set of
    // indirect_call edges. Silent, and absent from all four corpora, so parity
    // would never have shown it.
    //
    // One validation of the whole buffer makes every later `text()` sound by
    // construction. It is a SIMD-accelerated scan (GB/s) against a walk that is
    // orders of magnitude slower, and it replaces a per-node check that had to be
    // remembered at ~20 call sites.
    if std::str::from_utf8(source).is_err() {
        return Err("source_not_utf8");
    }
    let stem = file_stem(path).ok_or("path_needs_pathlib")?;
    let file_nid = make_id_ascii(&[path]).ok_or("non_ascii_path")?;

    let mut parser = Parser::new();
    parser.set_language(lang).map_err(|_| "grammar_load_failed")?;
    let tree = parser.parse(source, None).ok_or("parse_failed")?;
    let root = tree.root_node();
    // Python attaches a `parse_errors` block and keeps going; its recovery is
    // authoritative and reproducing it is a separate surface, so defer. Measured
    // at 4 of 900 Bun `.ts` files, so the coverage cost is ~0.4%.
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
        dialect,
        nodes: Vec::new(),
        seen_ids: HashSet::new(),
        edges: Vec::new(),
        raw_calls: Vec::new(),
        callable_def_nids: HashSet::new(),
        callable_class_nids: HashSet::new(),
        local_bound_names: HashMap::new(),
        closure_locals_by_body: HashMap::new(),
        function_bodies: Vec::new(),
        type_table: OrderedMap::default(),
        js_external_imports: HashSet::new(),
        label_to_nid: HashMap::new(),
        nid_to_sf: HashMap::new(),
        seen_call_pairs: HashSet::new(),
        seen_indirect_pairs: HashSet::new(),
        seen_dyn_import_pairs: HashSet::new(),
        tracked_body_ids: HashSet::new(),
    };

    // Module-scoped, computed once per file before the walk, exactly as Python
    // does: a name bound by an import of a module outside the corpus shadows in
    // every scope.
    ctx.js_external_imports = imports::external_import_names(&ctx, root)?;

    let file_label = path.rsplit('/').next().unwrap_or(path).to_string();
    ctx.add_node(&file_nid, &file_label, 1);

    walk::walk(&mut ctx, root, None)?;

    // ── Call-graph pass ─────────────────────────────────────────────────────
    for n in &ctx.nodes {
        let mut sf = String::new();
        let mut label = String::new();
        for (k, v) in &n.fields {
            if *k == "source_file" {
                if let Val::S(s) = v {
                    sf = s.clone();
                }
            } else if *k == "label" {
                if let Val::S(s) = v {
                    label = s.clone();
                }
            }
        }
        ctx.nid_to_sf.insert(n.id.clone(), sf);
        // `type == "namespace"` nodes are skipped in Python; TS never emits one
        // (only the C# namespace handler does), so there is nothing to skip.
        let normalised = label.trim_matches(|c| c == '(' || c == ')').trim_start_matches('.');
        ctx.label_to_nid.insert(normalised.to_string(), n.id.clone());
    }

    let tracked: Vec<usize> = ctx.function_bodies.iter().map(|(_, b)| b.id()).collect();
    ctx.tracked_body_ids.extend(tracked);

    let bodies: Vec<(String, Node)> = ctx.function_bodies.clone();
    for (caller_nid, body) in bodies {
        let extra = ctx
            .closure_locals_by_body
            .get(&body.id())
            .cloned()
            .unwrap_or_default();
        calls::walk_calls(&mut ctx, body, &caller_nid, &extra)?;
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

    // The receiver table is a SEPARATE full-tree pass in Python, run after the
    // walk, and its constructor-injection entries (populated during the walk)
    // win on a name clash because they were inserted first.
    walk::receiver_type_table(&mut ctx, root)?;
    if !ctx.type_table.is_empty() {
        let tt = PyDict::new(py);
        tt.set_item("path", path).map_err(|_| "py_error")?;
        tt.set_item("table", ctx.type_table.to_py(py).map_err(|_| "py_error")?)
            .map_err(|_| "py_error")?;
        out.set_item("ts_type_table", tt).map_err(|_| "py_error")?;
    }
    Ok(out)
}


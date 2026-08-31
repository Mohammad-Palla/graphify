//! Zig: a BESPOKE walker, like `bash/`, `go/` and `rust/`.
//!
//! `extract_zig` is a hand-written walk with no `LanguageConfig`, so it reaches
//! the kernel through `BespokeGrammar("tree_sitter_zig")` rather than through
//! the `(ts_module, ts_language_fn)` routing table.
//!
//! # Three things here that look like bugs and are not
//!
//! Each is reproduced deliberately, because the kernel must match
//! `extract_zig` INCLUDING where it gives up:
//!
//! 1. **`variable_declaration` never recurses into unrecognised children.** The
//!    Python's final `return` is unconditional, so a `const x = someCall();`
//!    contributes nothing and its subtree is not walked.
//! 2. **An anonymous container drops its methods.** The struct/enum/union arms
//!    recurse only `if name_node`, and return either way.
//! 3. **Call resolution is a LINEAR SCAN over `nodes` in insertion order**, not
//!    a label map. `next((n for n in nodes if n["label"] in (f"{c}()",
//!    f".{c}()")), None)` picks the first NODE matching either spelling -- which
//!    is not the same answer a dict keyed on the label would give when both a
//!    free function `foo()` and a method `.foo()` exist in one file. Keeping the
//!    scan is the only way to stay byte-identical.

use std::collections::HashSet;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tree_sitter::{Node, Parser};

use crate::engine::R;
use crate::ids::{file_stem, make_id_ascii};
use crate::js::ast::{children, text_checked};
use crate::js::emit::{self, EdgeRow, NodeRow, RawCall, Val};
use crate::Outcome;

/// The container kinds a `variable_declaration` can bind, in the Python's
/// membership order.
const VALUE_KINDS: &[&str] = &[
    "struct_declaration",
    "enum_declaration",
    "union_declaration",
    "builtin_function",
    "field_expression",
];

struct Ctx<'a, 'tree> {
    src: &'a [u8],
    str_path: &'a str,
    stem: String,
    file_nid: String,
    nodes: Vec<NodeRow>,
    edges: Vec<EdgeRow>,
    raw_calls: Vec<RawCall>,
    seen_ids: HashSet<String>,
    function_bodies: Vec<(String, Node<'tree>)>,
    seen_call_pairs: HashSet<(String, String)>,
}

impl<'a, 'tree> Ctx<'a, 'tree> {
    fn text(&self, node: Node) -> R<&'a str> {
        text_checked(node, self.src).ok_or("invalid_utf8_text")
    }

    fn mkid(&self, parts: &[&str]) -> R<String> {
        make_id_ascii(parts).ok_or("non_ascii_id")
    }

    fn add_node(&mut self, nid: &str, label: &str, line: usize) {
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

    /// `add_edge`. Zig never passes `context`, so unlike Go's there is no
    /// optional slot -- adding one would be dead weight and an invitation to
    /// emit a key the Python does not.
    fn add_edge(&mut self, src: &str, tgt: &str, relation: &'static str, line: usize) {
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
}

/// `_extract_import`: `@import("std")` / `@cImport(...)` -> an `imports_from`
/// edge naming the module.
///
/// Called with the `variable_declaration` node, not its value -- the Python
/// passes `node`, so the scan starts one level above the builtin.
///
/// The control flow is fiddly and is reproduced literally:
/// * the inner `bi` / `args` scan keeps assigning, so the LAST
///   `builtin_identifier` and the LAST `arguments` child win;
/// * a `builtin_function` that is not `@import`/`@cImport` does NOT stop the
///   outer loop -- the next child still gets a turn;
/// * the `return` inside the argument loop fires on the first string argument
///   whether or not it yielded a module name, so `@import("")` emits nothing
///   AND stops;
/// * a `field_expression` child recurses and then returns unconditionally.
fn extract_import<'tree>(ctx: &mut Ctx<'_, 'tree>, node: Node<'tree>) -> R<()> {
    for child in children(node) {
        if child.kind() == "builtin_function" {
            let mut bi: Option<&str> = None;
            let mut args: Option<Node> = None;
            for c in children(child) {
                if c.kind() == "builtin_identifier" {
                    bi = Some(ctx.text(c)?);
                } else if c.kind() == "arguments" {
                    args = Some(c);
                }
            }
            let is_import = matches!(bi, Some("@import") | Some("@cImport"));
            if let (true, Some(args)) = (is_import, args) {
                for arg in children(args) {
                    if !matches!(arg.kind(), "string_literal" | "string") {
                        continue;
                    }
                    // `.strip('"')` removes EVERY leading and trailing quote,
                    // not just one, and Rust's `trim_matches` is the same rule.
                    let raw = ctx.text(arg)?.trim_matches('"');
                    let module_name = raw
                        .rsplit('/')
                        .next()
                        .unwrap_or("")
                        .split('.')
                        .next()
                        .unwrap_or("");
                    if !module_name.is_empty() {
                        let tgt = ctx.mkid(&[module_name])?;
                        let line = node.start_position().row + 1;
                        let file_nid = ctx.file_nid.clone();
                        ctx.add_edge(&file_nid, &tgt, "imports_from", line);
                    }
                    return Ok(());
                }
            }
        } else if child.kind() == "field_expression" {
            extract_import(ctx, child)?;
            return Ok(());
        }
    }
    Ok(())
}

fn walk<'tree>(
    ctx: &mut Ctx<'_, 'tree>,
    node: Node<'tree>,
    parent_struct_nid: Option<&str>,
) -> R<()> {
    let t = node.kind();

    if t == "function_declaration" {
        if let Some(name_node) = node.child_by_field_name("name") {
            let func_name = ctx.text(name_node)?.to_string();
            let line = node.start_position().row + 1;
            let func_nid = match parent_struct_nid {
                Some(p) => {
                    let nid = ctx.mkid(&[p, &func_name])?;
                    ctx.add_node(&nid, &format!(".{func_name}()"), line);
                    ctx.add_edge(p, &nid, "method", line);
                    nid
                }
                None => {
                    let nid = ctx.mkid(&[&ctx.stem.clone(), &func_name])?;
                    ctx.add_node(&nid, &format!("{func_name}()"), line);
                    let f = ctx.file_nid.clone();
                    ctx.add_edge(&f, &nid, "contains", line);
                    nid
                }
            };
            if let Some(body) = node.child_by_field_name("body") {
                ctx.function_bodies.push((func_nid, body));
            }
        }
        // Returns whether or not the declaration had a name.
        return Ok(());
    }

    if t == "variable_declaration" {
        // Both scans keep assigning: the LAST matching child wins in each case.
        let mut name_node: Option<Node> = None;
        let mut value_node: Option<Node> = None;
        for child in children(node) {
            if child.kind() == "identifier" {
                name_node = Some(child);
            } else if VALUE_KINDS.contains(&child.kind()) {
                value_node = Some(child);
            }
        }

        let vkind = value_node.map(|v| v.kind());
        if vkind == Some("struct_declaration") {
            if let (Some(nn), Some(vn)) = (name_node, value_node) {
                let struct_name = ctx.text(nn)?.to_string();
                let line = node.start_position().row + 1;
                let struct_nid = ctx.mkid(&[&ctx.stem.clone(), &struct_name])?;
                ctx.add_node(&struct_nid, &struct_name, line);
                let f = ctx.file_nid.clone();
                ctx.add_edge(&f, &struct_nid, "contains", line);
                for child in children(vn) {
                    walk(ctx, child, Some(&struct_nid))?;
                }
            }
            return Ok(());
        }

        if matches!(vkind, Some("enum_declaration") | Some("union_declaration")) {
            if let (Some(nn), Some(vn)) = (name_node, value_node) {
                let type_name = ctx.text(nn)?.to_string();
                let line = node.start_position().row + 1;
                let type_nid = ctx.mkid(&[&ctx.stem.clone(), &type_name])?;
                ctx.add_node(&type_nid, &type_name, line);
                let f = ctx.file_nid.clone();
                ctx.add_edge(&f, &type_nid, "contains", line);
                // Zig enums and tagged unions declare methods exactly as structs
                // do, so the same recursion applies -- without it the whole
                // method layer of an enum is dropped.
                for child in children(vn) {
                    walk(ctx, child, Some(&type_nid))?;
                }
            }
            return Ok(());
        }

        if matches!(vkind, Some("builtin_function") | Some("field_expression")) {
            extract_import(ctx, node)?;
        }
        // Unconditional: an unrecognised binding is NOT descended into.
        return Ok(());
    }

    for child in children(node) {
        walk(ctx, child, parent_struct_nid)?;
    }
    Ok(())
}

/// The first node whose label is `callee()` or `.callee()`, scanning `nodes` in
/// INSERTION order. See the module doc: a label map would answer differently.
fn find_target(ctx: &Ctx<'_, '_>, callee: &str) -> Option<String> {
    let plain = format!("{callee}()");
    let dotted = format!(".{callee}()");
    for n in &ctx.nodes {
        if let Some((_, Val::S(label))) = n.fields.first() {
            if *label == plain || *label == dotted {
                return Some(n.id.clone());
            }
        }
    }
    None
}

fn walk_calls<'tree>(
    ctx: &mut Ctx<'_, 'tree>,
    node: Node<'tree>,
    caller_nid: &str,
) -> R<()> {
    // A nested function is its own caller and has its own `function_bodies`
    // entry, so the walk stops at the boundary.
    if node.kind() == "function_declaration" {
        return Ok(());
    }
    if node.kind() == "call_expression" {
        if let Some(fn_node) = node.child_by_field_name("function") {
            let fn_text = ctx.text(fn_node)?;
            let callee = fn_text.rsplit('.').next().unwrap_or("").to_string();
            let is_member_call = fn_text.contains('.');
            let tgt = find_target(ctx, &callee);
            // The Python's `elif` hangs off `if tgt_nid and tgt_nid != caller_nid`,
            // NOT off `if tgt_nid`. So a call whose only label match is the
            // CALLER ITSELF -- `.init()` calling some other `init` -- takes the
            // elif and DOES produce a raw_call, which is then resolved
            // cross-file. Reading this as the engine's "a self-target emits
            // nothing" rule cost 4 raw_calls per file on zls and was caught by
            // the parity harness, not by reading either side.
            let resolved = match &tgt {
                Some(t) if t != caller_nid => Some(t.clone()),
                _ => None,
            };
            match resolved {
                Some(tgt) => {
                    let pair = (caller_nid.to_string(), tgt.clone());
                    if ctx.seen_call_pairs.insert(pair) {
                        let line = node.start_position().row + 1;
                        ctx.add_edge(caller_nid, &tgt, "calls", line);
                    }
                }
                None => {
                    if !callee.is_empty() {
                        let line = node.start_position().row + 1;
                        ctx.raw_calls.push(vec![
                            ("caller_nid", Val::S(caller_nid.to_string())),
                            ("callee", Val::S(callee)),
                            ("is_member_call", Val::B(is_member_call)),
                            ("source_file", Val::S(ctx.str_path.to_string())),
                            ("source_location", Val::S(format!("L{line}"))),
                        ]);
                    }
                }
            }
        }
    }
    for child in children(node) {
        walk_calls(ctx, child, caller_nid)?;
    }
    Ok(())
}

pub fn walk_zig<'py>(
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
    // `_make_id(str(path))` -- the WHOLE path, not the stem.
    let file_nid = make_id_ascii(&[path]).ok_or("non_ascii_path")?;

    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_zig::LANGUAGE.into())
        .map_err(|_| "grammar_load_failed")?;
    let tree = parser.parse(source, None).ok_or("parse_failed")?;
    let root = tree.root_node();
    // `extract_zig` itself does NOT check this and walks an errored tree anyway.
    // Deferring is still correct: Python then produces its own result, which is
    // the authoritative one. Handling it here would mean reproducing
    // tree-sitter's error recovery, which is the one thing this design refuses
    // to guess at.
    if root.has_error() {
        return Err("parse_error");
    }

    let mut ctx = Ctx {
        src: source,
        str_path: path,
        stem,
        file_nid: file_nid.clone(),
        nodes: Vec::new(),
        edges: Vec::new(),
        raw_calls: Vec::new(),
        seen_ids: HashSet::new(),
        function_bodies: Vec::new(),
        seen_call_pairs: HashSet::new(),
    };

    let file_label = path.rsplit('/').next().unwrap_or(path).to_string();
    ctx.add_node(&file_nid, &file_label, 1);

    walk(&mut ctx, root, None)?;

    // `Node` is a cheap Copy handle into `tree`, which outlives this scope, so
    // the bodies are carried directly rather than re-found by byte range -- no
    // assumption about the body's node KIND, which Go's re-find has to make.
    let bodies: Vec<(String, Node)> = ctx.function_bodies.clone();
    for (caller_nid, body) in bodies {
        walk_calls(&mut ctx, body, &caller_nid)?;
    }

    // `imports_from` survives a missing target; every other relation does not.
    let clean: Vec<&EdgeRow> = ctx
        .edges
        .iter()
        .filter(|e| {
            ctx.seen_ids.contains(&e.source)
                && (ctx.seen_ids.contains(&e.target) || e.relation == "imports_from")
        })
        .collect();

    let out = PyDict::new(py);
    let nodes = PyList::empty(py);
    for n in &ctx.nodes {
        nodes
            .append(emit::node_to_py(py, n, false, false).map_err(|_| "py_error")?)
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

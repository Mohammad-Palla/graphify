//! Bash: a BESPOKE walker, not on the shared engine.
//!
//! `extract_bash` does not go through `_extract_generic` at all -- it has its own
//! two-pass walk, its own node metadata (`{"language": "bash", "kind": …}` on
//! every node), and a synthetic `__entry` node that owns every top-level call.
//! None of that fits `EngineConfig`, so this is shaped like `js/` and `py/`.
//!
//! # What defers, and why
//!
//! Every `source` / `.` command and every `.sh` script invocation resolves
//! through the FILESYSTEM -- `Path.resolve()`, `is_file()`, `os.path.normpath`,
//! a `var_bases` table built from top-level assignments, and a `_within_tree`
//! traversal guard that exists because a corpus is attacker-controllable. That
//! is ~250 lines of path policy whose failures would be silent, and the
//! `bash_sources` it produces feeds a cross-file resolver.
//!
//! So a file containing either construct defers, whole. Measured over the
//! corpora, 10% of shell files contain a `source`/`.` command, so the cost is
//! small and it is paid only by the files that actually need it -- the same
//! "defer with a reason" rule the rest of the kernel follows, rather than a
//! reimplementation of path resolution that would be wrong in ways no parity run
//! could distinguish from a corpus that happens not to contain the shape.

use std::collections::HashSet;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tree_sitter::{Node, Parser};

use crate::engine::R;
use crate::ids::{file_stem, make_id_ascii};
use crate::js::ast::{children, text_checked};
use crate::js::emit::{self, EdgeRow, NodeRow, RawCall, Val};
use crate::Outcome;

const SOURCE_COMMANDS: &[&str] = &["source", "."];
const SCRIPT_RUNNERS: &[&str] = &["bash", "sh", "zsh", "ksh", "dash"];
/// Parent kinds that make a contained command part of a substitution rather than
/// a real invocation. Token-level filtering misses these: `$(build)` exposes
/// `build` as a child command whose name token has no metacharacters -- only the
/// PARENT does.
const EXPANSION_PARENTS: &[&str] = &["command_substitution", "process_substitution"];

struct Ctx<'a> {
    src: &'a [u8],
    str_path: &'a str,
    stem: String,
    file_nid: String,
    entry_nid: String,
    nodes: Vec<NodeRow>,
    edges: Vec<EdgeRow>,
    raw_calls: Vec<RawCall>,
    seen_ids: HashSet<String>,
    raw_seen: HashSet<(String, String)>,
    defined_functions: HashSet<String>,
    function_bodies: Vec<(String, usize, usize)>,
}

impl<'a> Ctx<'a> {
    fn text(&self, node: Node) -> R<&'a str> {
        text_checked(node, self.src).ok_or("invalid_utf8_text")
    }

    /// Every bash node carries `metadata = {"language": "bash", "kind": …}`.
    fn add_node(&mut self, nid: &str, label: &str, line: usize, kind: &'static str) {
        if nid.is_empty() || !self.seen_ids.insert(nid.to_string()) {
            return;
        }
        self.nodes.push(NodeRow {
            id: nid.to_string(),
            fields: vec![
                ("label", Val::S(label.to_string())),
                ("file_type", Val::Static("code")),
                ("source_file", Val::S(self.str_path.to_string())),
                ("source_location", Val::S(format!("L{line}"))),
                (
                    "metadata",
                    Val::Meta(vec![
                        ("language".to_string(), Val::Static("bash")),
                        ("kind".to_string(), Val::Static(kind)),
                    ]),
                ),
            ],
        });
    }

    /// `add_edge`, with the self-loop and empty-endpoint guard the Python applies
    /// BEFORE building the dict.
    fn add_edge(
        &mut self,
        src: &str,
        tgt: &str,
        relation: &'static str,
        line: usize,
        context: Option<&'static str>,
    ) {
        if src.is_empty() || tgt.is_empty() || src == tgt {
            return;
        }
        let mut fields = vec![
            ("confidence", Val::Static("EXTRACTED")),
            ("source_file", Val::S(self.str_path.to_string())),
            ("source_location", Val::S(format!("L{line}"))),
            ("weight", Val::F(1.0)),
        ];
        if let Some(c) = context {
            fields.push(("context", Val::Static(c)));
        }
        self.edges.push(EdgeRow {
            source: src.to_string(),
            target: tgt.to_string(),
            relation,
            fields,
        });
    }

    fn mkid(&self, parts: &[&str]) -> R<String> {
        make_id_ascii(parts).ok_or("non_ascii_id")
    }
}

/// `literal`: the token's text with one layer of matching quotes removed, or
/// None when it holds a shell metacharacter.
fn literal(ctx: &Ctx, node: Node) -> R<Option<String>> {
    let raw = ctx.text(node)?.trim();
    if raw.is_empty() {
        return Ok(None);
    }
    let bytes = raw.as_bytes();
    let raw = if (bytes[0] == b'\'' || bytes[0] == b'"') && bytes[bytes.len() - 1] == bytes[0] {
        &raw[1..raw.len() - 1]
    } else {
        raw
    };
    // `$(` and `<(` are redundant with `$` and `<`, and are in the Python's list
    // anyway; keeping the same set means the same answers.
    for token in ["$", "`", "$(", "<(", ">", "|", ";", "&"] {
        if raw.contains(token) {
            return Ok(None);
        }
    }
    Ok(Some(raw.to_string()))
}

fn is_inside_expansion(node: Node) -> bool {
    let mut parent = node.parent();
    while let Some(p) = parent {
        if EXPANSION_PARENTS.contains(&p.kind()) {
            return true;
        }
        parent = p.parent();
    }
    false
}

/// Bare `$(fn)` at script level and process substitutions stay suppressed
/// (#2141); value capture via `x=$(fn)` IS a real invocation (#2978).
fn call_in_command_allowed(cmd_node: Node) -> bool {
    let mut parent = cmd_node.parent();
    let mut saw_command_substitution = false;
    while let Some(p) = parent {
        if p.kind() == "process_substitution" {
            return false;
        }
        if p.kind() == "command_substitution" {
            saw_command_substitution = true;
        }
        if p.kind() == "variable_assignment" && saw_command_substitution {
            return true;
        }
        parent = p.parent();
    }
    !saw_command_substitution
}

/// The `word` child of a `function_definition`.
fn func_name(ctx: &Ctx, node: Node) -> R<Option<String>> {
    for child in children(node) {
        if child.kind() == "word" {
            return literal(ctx, child);
        }
    }
    Ok(None)
}

fn prescan_functions(ctx: &mut Ctx, node: Node) -> R<()> {
    if node.kind() == "function_definition" {
        if let Some(name) = func_name(ctx, node)? {
            ctx.defined_functions.insert(name);
        }
    }
    for child in children(node) {
        prescan_functions(ctx, child)?;
    }
    Ok(())
}

fn walk<'tree>(ctx: &mut Ctx, node: Node<'tree>, parent_nid: &str) -> R<()> {
    let t = node.kind();

    if t == "function_definition" {
        if let Some(name) = func_name(ctx, node)? {
            let fn_nid = ctx.mkid(&[&ctx.stem.clone(), &name])?;
            let line = node.start_position().row + 1;
            ctx.add_node(&fn_nid, &format!("{name}()"), line, "bash_function");
            ctx.add_edge(parent_nid, &fn_nid, "defines", line, None);
            ctx.defined_functions.insert(name);
            let body = children(node)
                .into_iter()
                .find(|c| c.kind() == "compound_statement");
            // Pushed even when there is no body: the Python appends
            // `(fn_nid, None)` and `walk_calls` returns immediately on it.
            match body {
                Some(b) => {
                    ctx.function_bodies
                        .push((fn_nid.clone(), b.start_byte(), b.end_byte()));
                    // Recurse so NESTED definitions are discovered and get their
                    // own `function_bodies` entry.
                    walk(ctx, b, &fn_nid)?;
                }
                None => ctx.function_bodies.push((fn_nid.clone(), usize::MAX, usize::MAX)),
            }
        }
        return Ok(());
    }

    if t == "command" {
        if is_inside_expansion(node) {
            return Ok(());
        }
        let mut cmd_name_node = node.child_by_field_name("name");
        if cmd_name_node.is_none() {
            cmd_name_node = children(node).into_iter().next();
        }
        let cmd_name_node = match cmd_name_node {
            Some(n) => n,
            None => return Ok(()),
        };
        let cmd = literal(ctx, cmd_name_node)?;
        let args: Vec<Node> = children(node)
            .into_iter()
            .filter(|c| {
                matches!(c.kind(), "word" | "string" | "concatenation") && c.id() != cmd_name_node.id()
            })
            .collect();
        let is_source = cmd
            .as_deref()
            .map(|c| SOURCE_COMMANDS.contains(&c) && !ctx.defined_functions.contains(c))
            .unwrap_or(false);
        if is_source {
            // Every branch below this point in the Python resolves a path
            // against the filesystem. See the module comment.
            if !args.is_empty() {
                return Err("bash_source_command");
            }
            return Ok(());
        }
        if let Some(cmd) = cmd {
            if !ctx.defined_functions.contains(&cmd) {
                let mut raw = if cmd.ends_with(".sh") { Some(cmd.clone()) } else { None };
                if SCRIPT_RUNNERS.contains(&cmd.as_str()) && !args.is_empty() {
                    raw = literal(ctx, args[0])?;
                }
                if raw.map(|r| r.ends_with(".sh")).unwrap_or(false) {
                    // `resolved.is_file()` plus a cwd-relative rewrite.
                    return Err("bash_script_invocation");
                }
            }
        }
        return Ok(());
    }

    if t == "declaration_command" {
        // `export`/`declare`/`readonly VAR=value`, at PROGRAM level only.
        if node.parent().map(|p| p.kind() == "program").unwrap_or(false) {
            for child in children(node) {
                if child.kind() != "variable_assignment" {
                    continue;
                }
                if let Some(var_node) = child.child_by_field_name("name") {
                    let var = ctx.text(var_node)?.trim().to_string();
                    if !var.is_empty() {
                        let var_nid = ctx.mkid(&[&ctx.stem.clone(), &var])?;
                        let line = child.start_position().row + 1;
                        ctx.add_node(&var_nid, &var, line, "code");
                        let f = ctx.file_nid.clone();
                        ctx.add_edge(&f, &var_nid, "defines", line, None);
                    }
                }
            }
        }
        return Ok(());
    }

    for child in children(node) {
        walk(ctx, child, parent_nid)?;
    }
    Ok(())
}

fn walk_calls<'tree>(
    ctx: &mut Ctx,
    body: Node<'tree>,
    func_nid: &str,
    seen_calls: &mut HashSet<(String, String)>,
) -> R<()> {
    for child in children(body) {
        // A nested definition's body is walked separately, so its calls are not
        // attributed to the enclosing scope.
        if child.kind() == "function_definition" {
            continue;
        }
        if child.kind() == "command" && call_in_command_allowed(child) {
            let mut cmd_name_node = child.child_by_field_name("name");
            if cmd_name_node.is_none() {
                cmd_name_node = children(child).into_iter().next();
            }
            if let Some(cmd_name_node) = cmd_name_node {
                if let Some(name) = literal(ctx, cmd_name_node)? {
                    // A DEFINED function wins. There is deliberately no skip-list
                    // of external commands: one would produce false negatives
                    // whenever a script defines a function shadowing an external
                    // (`install`, `find`).
                    if ctx.defined_functions.contains(&name) {
                        let tgt = ctx.mkid(&[&ctx.stem.clone(), &name])?;
                        let key = (func_nid.to_string(), tgt.clone());
                        if !tgt.is_empty() && seen_calls.insert(key) {
                            let line = child.start_position().row + 1;
                            ctx.add_edge(func_nid, &tgt, "calls", line, Some("call"));
                        }
                    } else if !SOURCE_COMMANDS.contains(&name.as_str())
                        && !SCRIPT_RUNNERS.contains(&name.as_str())
                    {
                        // Not defined here -- it may come from a sourced library.
                        // A genuine external command matches nothing in the
                        // cross-file resolver and yields no edge, so this cannot
                        // over-connect the graph (#2141).
                        let raw_key = (func_nid.to_string(), name.clone());
                        if ctx.raw_seen.insert(raw_key) {
                            let line = child.start_position().row + 1;
                            ctx.raw_calls.push(vec![
                                ("language", Val::Static("bash")),
                                ("callee", Val::S(name)),
                                ("caller_nid", Val::S(func_nid.to_string())),
                                ("source_file", Val::S(ctx.str_path.to_string())),
                                ("source_location", Val::S(format!("L{line}"))),
                            ]);
                        }
                    }
                }
            }
        }
        walk_calls(ctx, child, func_nid, seen_calls)?;
    }
    Ok(())
}

pub fn walk_bash<'py>(
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
    let entry_nid = format!("{file_nid}__entry");

    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_bash::LANGUAGE.into())
        .map_err(|_| "grammar_load_failed")?;
    let tree = parser.parse(source, None).ok_or("parse_failed")?;
    let root = tree.root_node();
    // `extract_bash` has no `parse_errors` block -- it catches the exception and
    // returns an error dict -- but the tree it walks after a recovery is still
    // Python's, so a file with any error defers for the same reason as elsewhere.
    if root.has_error() {
        return Err("parse_error");
    }

    let mut ctx = Ctx {
        src: source,
        str_path: path,
        stem,
        file_nid: file_nid.clone(),
        entry_nid: entry_nid.clone(),
        nodes: Vec::new(),
        edges: Vec::new(),
        raw_calls: Vec::new(),
        seen_ids: HashSet::new(),
        raw_seen: HashSet::new(),
        defined_functions: HashSet::new(),
        function_bodies: Vec::new(),
    };

    let file_label = path.rsplit('/').next().unwrap_or(path).to_string();
    // `file_nid` is fully path-derived and never produced by `_make_id(stem,
    // name)`, so the `__entry` suffix cannot collide with a function node.
    ctx.add_node(&file_nid, &file_label, 1, "file");
    ctx.add_node(&entry_nid, &format!("{file_label} script"), 1, "bash_entrypoint");
    ctx.add_edge(&file_nid, &entry_nid, "contains", 1, None);

    // Pre-pass: every defined name, so the `source` handler can detect a
    // user-defined function shadowing `source` regardless of definition order.
    prescan_functions(&mut ctx, root)?;
    // `var_bases` is deliberately NOT built: every one of its consumers is a
    // filesystem path resolution that defers above.

    walk(&mut ctx, root, &file_nid.clone())?;

    // Second pass. Top-level calls belong to the ENTRY node, not the file.
    let mut top_seen: HashSet<(String, String)> = HashSet::new();
    walk_calls(&mut ctx, root, &entry_nid.clone(), &mut top_seen)?;
    let bodies = ctx.function_bodies.clone();
    for (fn_nid, start, end) in bodies {
        if start == usize::MAX {
            continue; // a definition with no compound_statement body
        }
        let body = find_node(root, start, end).ok_or("body_not_found")?;
        let mut seen: HashSet<(String, String)> = HashSet::new();
        walk_calls(&mut ctx, body, &fn_nid, &mut seen)?;
    }

    let out = PyDict::new(py);
    let nodes = PyList::empty(py);
    for n in &ctx.nodes {
        nodes
            .append(emit::node_to_py(py, n, false, false).map_err(|_| "py_error")?)
            .map_err(|_| "py_error")?;
    }
    let edges = PyList::empty(py);
    for e in &ctx.edges {
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
    // Always present and always empty: a file that would have populated it
    // defers, so an empty list here is the whole truth for a native file.
    out.set_item("bash_sources", PyList::empty(py))
        .map_err(|_| "py_error")?;
    Ok(out)
}

/// Re-find a node by byte range. `function_bodies` stores ranges rather than
/// `Node`s because the borrow checker will not let a `Node<'tree>` live in a
/// struct that is also mutably borrowed by the walk -- the same reason
/// `engine::Ctx` keys `method_scopes` on a range.
fn find_node<'tree>(root: Node<'tree>, start: usize, end: usize) -> Option<Node<'tree>> {
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        if n.start_byte() == start && n.end_byte() == end && n.kind() == "compound_statement" {
            return Some(n);
        }
        if n.start_byte() <= start && n.end_byte() >= end {
            stack.extend(children(n));
        }
    }
    None
}

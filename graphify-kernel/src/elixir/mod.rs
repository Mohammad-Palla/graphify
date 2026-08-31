//! Elixir: a BESPOKE walker, like `zig/`, `bash/`, `go/` and `rust/`.
//!
//! # The shape of the language, and why the walk looks odd
//!
//! Elixir has almost no syntax. `defmodule`, `def`, `alias`, `import` and `use`
//! are not keywords -- they are ordinary MACRO CALLS, so tree-sitter-elixir
//! parses every one of them as a plain `call` node whose first `identifier`
//! child happens to spell `defmodule`. That is why `walk` dispatches on the
//! TEXT of a child rather than on a node kind, and why there is no
//! `class_types` / `function_types` config this could have been driven by.
//!
//! # Parse ceiling
//!
//! 100.0% over 871 files (elixir, phoenix, ecto) -- joint best of every language
//! ported here, alongside Scala. No parse-error floor at all.

use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tree_sitter::{Node, Parser};

use crate::engine::R;
use crate::ids::{file_stem, make_id_ascii};
use crate::js::ast::{children, text_checked};
use crate::js::emit::{self, EdgeRow, NodeRow, RawCall, Val};
use crate::py::helpers::BUILTIN_GLOBALS;
use crate::Outcome;

/// `_IMPORT_KEYWORDS`.
const IMPORT_KEYWORDS: &[&str] = &["alias", "import", "require", "use"];

/// `_SKIP_KEYWORDS`: a macro call that is a DECLARATION or a control-flow form,
/// never a function call. Reached only in the call pass.
///
/// Note this set is wider than the one `walk` dispatches on: `defmacro`,
/// `defstruct`, `defprotocol`, `defimpl`, `defguard` and the six control-flow
/// forms are skipped as callees without ever being declarations here.
const SKIP_KEYWORDS: &[&str] = &[
    "def", "defp", "defmodule", "defmacro", "defmacrop", "defstruct", "defprotocol", "defimpl",
    "defguard", "alias", "import", "require", "use", "if", "unless", "case", "cond", "with", "for",
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
    label_to_nid: HashMap<String, String>,
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

    fn add_edge(
        &mut self,
        src: &str,
        tgt: &str,
        relation: &'static str,
        line: usize,
        context: Option<&'static str>,
    ) {
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
}

/// `_get_alias_text`: the first `alias` child's text, for `defmodule Foo.Bar do`.
fn get_alias_text(ctx: &Ctx, node: Node) -> R<Option<String>> {
    for child in children(node) {
        if child.kind() == "alias" {
            return Ok(Some(ctx.text(child)?.to_string()));
        }
    }
    Ok(None)
}

/// `_get_alias_modules`: every module named by an `alias`/`import`/`require`/`use`
/// argument.
///
/// Two forms. `alias Foo.Bar` is a single `alias` node. `alias Foo.{Bar, Baz}`
/// is a `dot` node holding the base alias and a trailing `tuple` of members,
/// which expands to `["Foo.Bar", "Foo.Baz"]`.
///
/// Both arms RETURN, so only the first `alias`-or-`dot` child is ever consulted.
/// The `dot` arm returns the node's whole text when it has no usable tuple, so a
/// shape this does not understand still yields one module rather than none.
fn get_alias_modules(ctx: &Ctx, node: Node) -> R<Vec<String>> {
    for child in children(node) {
        if child.kind() == "alias" {
            return Ok(vec![ctx.text(child)?.to_string()]);
        }
        if child.kind() == "dot" {
            let mut base: Option<String> = None;
            let mut tuple_node: Option<Node> = None;
            for sub in children(child) {
                if sub.kind() == "alias" && base.is_none() {
                    // FIRST alias only -- `base is None` guards it.
                    base = Some(ctx.text(sub)?.to_string());
                } else if sub.kind() == "tuple" {
                    // LAST tuple: the Python keeps assigning.
                    tuple_node = Some(sub);
                }
            }
            if let (Some(base), Some(tn)) = (&base, tuple_node) {
                let mut members = Vec::new();
                for m in children(tn) {
                    if m.kind() == "alias" {
                        members.push(format!("{base}.{}", ctx.text(m)?));
                    }
                }
                if !members.is_empty() {
                    return Ok(members);
                }
            }
            return Ok(vec![ctx.text(child)?.to_string()]);
        }
    }
    Ok(Vec::new())
}

/// The `def`/`defp` head, unwrapping `when` guards.
///
/// `def f(x) when is_list(x)` wraps the head in a `binary_operator`, and without
/// unwrapping it a function whose ONLY clause carries `when` is dropped
/// entirely (#3111).
///
/// One subtlety reproduced literally: the `call` arm does NOT break the outer
/// loop, so a later argument can still overwrite the name -- only the bare
/// `identifier` arm breaks.
fn def_name(ctx: &Ctx, arguments: Node) -> R<Option<String>> {
    let mut func_name: Option<String> = None;
    for child in children(arguments) {
        let mut child = child;
        while child.kind() == "binary_operator" {
            let head = children(child)
                .into_iter()
                .find(|sub| matches!(sub.kind(), "call" | "identifier" | "binary_operator"));
            match head {
                Some(h) => child = h,
                None => break,
            }
        }
        if child.kind() == "call" {
            for sub in children(child) {
                if sub.kind() == "identifier" {
                    func_name = Some(ctx.text(sub)?.to_string());
                    break;
                }
            }
            // No outer break here -- deliberate, see the doc comment.
        } else if child.kind() == "identifier" {
            func_name = Some(ctx.text(child)?.to_string());
            break;
        }
    }
    Ok(func_name)
}

fn walk<'tree>(
    ctx: &mut Ctx<'_, 'tree>,
    node: Node<'tree>,
    parent_module_nid: Option<&str>,
) -> R<()> {
    if node.kind() != "call" {
        for child in children(node) {
            walk(ctx, child, parent_module_nid)?;
        }
        return Ok(());
    }

    // The LAST child of each kind wins: the Python's loop keeps assigning.
    let mut identifier_node: Option<Node> = None;
    let mut arguments_node: Option<Node> = None;
    let mut do_block_node: Option<Node> = None;
    for child in children(node) {
        match child.kind() {
            "identifier" => identifier_node = Some(child),
            "arguments" => arguments_node = Some(child),
            "do_block" => do_block_node = Some(child),
            _ => {}
        }
    }

    let identifier_node = match identifier_node {
        Some(n) => n,
        None => {
            for child in children(node) {
                walk(ctx, child, parent_module_nid)?;
            }
            return Ok(());
        }
    };

    let keyword = ctx.text(identifier_node)?.to_string();
    let line = node.start_position().row + 1;

    if keyword == "defmodule" {
        let module_name = match arguments_node {
            Some(a) => get_alias_text(ctx, a)?,
            None => None,
        };
        // An unnamed `defmodule` returns WITHOUT recursing -- its whole body is
        // dropped, not walked at file scope.
        let module_name = match module_name {
            Some(m) if !m.is_empty() => m,
            _ => return Ok(()),
        };
        let module_nid = ctx.mkid(&[&ctx.stem.clone(), &module_name])?;
        ctx.add_node(&module_nid, &module_name, line);
        let f = ctx.file_nid.clone();
        ctx.add_edge(&f, &module_nid, "contains", line, None);
        if let Some(db) = do_block_node {
            for child in children(db) {
                walk(ctx, child, Some(&module_nid))?;
            }
        }
        return Ok(());
    }

    if keyword == "def" || keyword == "defp" {
        let func_name = match arguments_node {
            Some(a) => def_name(ctx, a)?,
            None => None,
        };
        let func_name = match func_name {
            Some(f) if !f.is_empty() => f,
            _ => return Ok(()),
        };
        // A top-level `def` is contained by the FILE node, and the id is minted
        // from the file nid -- not from the stem, unlike the module branch.
        let container = parent_module_nid.unwrap_or(&ctx.file_nid).to_string();
        let func_nid = ctx.mkid(&[&container, &func_name])?;
        ctx.add_node(&func_nid, &format!("{func_name}()"), line);
        match parent_module_nid {
            Some(p) => ctx.add_edge(p, &func_nid, "method", line, None),
            None => {
                let f = ctx.file_nid.clone();
                ctx.add_edge(&f, &func_nid, "contains", line, None);
            }
        }
        if let Some(db) = do_block_node {
            ctx.function_bodies.push((func_nid, db));
        }
        return Ok(());
    }

    if IMPORT_KEYWORDS.contains(&keyword.as_str()) {
        if let Some(a) = arguments_node {
            for module_name in get_alias_modules(ctx, a)? {
                let tgt = ctx.mkid(&[&module_name])?;
                let f = ctx.file_nid.clone();
                ctx.add_edge(&f, &tgt, "imports", line, Some("import"));
            }
            return Ok(());
        }
        // Falls through when the keyword matched but there are no arguments.
    }

    for child in children(node) {
        walk(ctx, child, parent_module_nid)?;
    }
    Ok(())
}

fn walk_calls<'tree>(
    ctx: &mut Ctx<'_, 'tree>,
    node: Node<'tree>,
    caller_nid: &str,
) -> R<()> {
    if node.kind() != "call" {
        for child in children(node) {
            walk_calls(ctx, child, caller_nid)?;
        }
        return Ok(());
    }

    // Only the FIRST `identifier` child is tested against the skip set; the
    // `break` makes a later one irrelevant.
    for child in children(node) {
        if child.kind() == "identifier" {
            let kw = ctx.text(child)?;
            if SKIP_KEYWORDS.contains(&kw) {
                for c in children(node) {
                    walk_calls(ctx, c, caller_nid)?;
                }
                return Ok(());
            }
            break;
        }
    }

    // Whichever comes FIRST in child order decides: a `dot` makes it a member
    // call, a bare `identifier` a local one.
    let mut callee_name: Option<String> = None;
    let mut is_member_call = false;
    for child in children(node) {
        if child.kind() == "dot" {
            is_member_call = true;
            let dot_text = ctx.text(child)?;
            let trimmed = dot_text.trim_end_matches('.');
            callee_name = Some(trimmed.rsplit('.').next().unwrap_or("").to_string());
            break;
        }
        if child.kind() == "identifier" {
            callee_name = Some(ctx.text(child)?.to_string());
            break;
        }
    }

    if let Some(callee) = callee_name {
        if !callee.is_empty() && !BUILTIN_GLOBALS.contains(&callee.as_str()) {
            let line = node.start_position().row + 1;
            let tgt = ctx.label_to_nid.get(&callee).cloned();
            // The `else` hangs off `if tgt_nid and tgt_nid != caller_nid`, so a
            // call resolving to the CALLER ITSELF still produces a raw_call --
            // the same shape that cost Zig 4 raw_calls per file.
            match tgt {
                Some(t) if t != caller_nid => {
                    let pair = (caller_nid.to_string(), t.clone());
                    if ctx.seen_call_pairs.insert(pair) {
                        ctx.add_edge(caller_nid, &t, "calls", line, Some("call"));
                    }
                }
                _ => {
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

    // Recursion is unconditional: a call's arguments contain more calls.
    for child in children(node) {
        walk_calls(ctx, child, caller_nid)?;
    }
    Ok(())
}

pub fn walk_elixir<'py>(
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
        .set_language(&tree_sitter_elixir::LANGUAGE.into())
        .map_err(|_| "grammar_load_failed")?;
    let tree = parser.parse(source, None).ok_or("parse_failed")?;
    let root = tree.root_node();
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
        label_to_nid: HashMap::new(),
        seen_call_pairs: HashSet::new(),
    };

    let file_label = path.rsplit('/').next().unwrap_or(path).to_string();
    ctx.add_node(&file_nid, &file_label, 1);

    walk(&mut ctx, root, None)?;

    // `label.strip("()").lstrip(".")`, built AFTER the whole declaration walk.
    // A later node with the same normalised label OVERWRITES an earlier one --
    // dict assignment, not `setdefault`.
    let pairs: Vec<(String, String)> = ctx
        .nodes
        .iter()
        .filter_map(|n| match n.fields.first() {
            Some((_, Val::S(label))) => {
                let norm = label
                    .trim_matches(|c| c == '(' || c == ')')
                    .trim_start_matches('.')
                    .to_string();
                Some((norm, n.id.clone()))
            }
            _ => None,
        })
        .collect();
    for (norm, id) in pairs {
        ctx.label_to_nid.insert(norm, id);
    }

    let bodies: Vec<(String, Node)> = ctx.function_bodies.clone();
    for (caller_nid, body) in bodies {
        walk_calls(&mut ctx, body, &caller_nid)?;
    }

    // `imports` survives a missing target here, where Zig's exemption is
    // `imports_from` -- the relation name differs per language and copying the
    // wrong one silently drops every import edge.
    let clean: Vec<&EdgeRow> = ctx
        .edges
        .iter()
        .filter(|e| {
            ctx.seen_ids.contains(&e.source)
                && (ctx.seen_ids.contains(&e.target) || e.relation == "imports")
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
    // INTS, not floats -- see `Val::I`.
    out.set_item("input_tokens", 0i64).map_err(|_| "py_error")?;
    out.set_item("output_tokens", 0i64).map_err(|_| "py_error")?;
    Ok(out)
}

//! Rust: a BESPOKE walker, like `bash/` and `go/`.
//!
//! `extract_rust` threads a `parent_impl_nid` rather than a class node -- an
//! `impl` block is not a declaration, so its methods hang off the type it
//! implements FOR, and the same type can be reopened in several blocks. Nothing
//! in `EngineConfig` expresses that.
//!
//! Touches no filesystem, so every file the grammar parses is handled.

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

/// `_RUST_TRAIT_METHOD_BLOCKLIST`: method names so common across traits that a
/// bare cross-file lookup on them is noise rather than evidence.
///
/// Applied only to the raw_call (cross-file) arm -- an in-file EXTRACTED match
/// still wins. Deliberately Rust-local rather than added to the shared
/// `BUILTIN_GLOBALS`, which Go's comment explains from the other side: putting
/// `new` there would kill every in-file `Type::new()` edge in Rust.
const TRAIT_METHOD_BLOCKLIST: &[&str] = &[
    "new", "default", "parse", "from_str", "now", "clone", "into", "from", "to_string",
    "to_owned", "len", "is_empty", "iter", "next", "build", "start", "run", "init", "app", "get",
    "set", "push", "pop", "insert", "remove", "contains", "collect", "map", "filter", "unwrap",
    "expect", "ok", "err", "some", "none", "send", "recv", "lock", "read", "write",
];

const TYPE_WRAPPERS: &[&str] = &[
    "reference_type",
    "pointer_type",
    "array_type",
    "tuple_type",
    "slice_type",
];

/// The type-node kinds the struct-field and enum-variant scans accept when there
/// is no `type` field to read.
const TYPE_NODES: &[&str] = &[
    "type_identifier",
    "generic_type",
    "scoped_type_identifier",
    "reference_type",
    "primitive_type",
    "tuple_type",
    "array_type",
];

struct Ctx<'a> {
    src: &'a [u8],
    str_path: &'a str,
    stem: String,
    file_nid: String,
    nodes: Vec<NodeRow>,
    edges: Vec<EdgeRow>,
    raw_calls: Vec<RawCall>,
    seen_ids: HashSet<String>,
    function_bodies: Vec<(String, usize, usize)>,
    label_to_nid: HashMap<String, String>,
    seen_call_pairs: HashSet<(String, String)>,
}

impl<'a> Ctx<'a> {
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

    fn ensure_named_node(&mut self, name: &str, _line: usize) -> R<String> {
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
                    ("origin_file", Val::S(self.str_path.to_string())),
                ],
            });
        }
        Ok(bare)
    }
}

/// `_rust_collect_type_refs`. A `scoped_type_identifier` collapses to its LAST
/// segment (`std::io::Error` -> `Error`), unlike Go's `qualified_type`, which
/// keeps the qualifier.
fn collect_type_refs(
    ctx: &Ctx,
    node: Option<Node>,
    generic: bool,
    out: &mut Vec<(String, bool)>,
) -> R<()> {
    let node = match node {
        Some(n) => n,
        None => return Ok(()),
    };
    match node.kind() {
        "primitive_type" => return Ok(()),
        "type_identifier" => {
            let text = ctx.text(node)?;
            if !text.is_empty() {
                out.push((text.to_string(), generic));
            }
            return Ok(());
        }
        "scoped_type_identifier" => {
            let raw = ctx.text(node)?;
            let text = raw.rsplit("::").next().unwrap_or(raw);
            if !text.is_empty() {
                out.push((text.to_string(), generic));
            }
            return Ok(());
        }
        "generic_type" => {
            let mut name_node = node.child_by_field_name("type");
            if name_node.is_none() {
                name_node = children(node)
                    .into_iter()
                    .find(|c| matches!(c.kind(), "type_identifier" | "scoped_type_identifier"));
            }
            if let Some(name_node) = name_node {
                let raw = ctx.text(name_node)?;
                let text = raw.rsplit("::").next().unwrap_or(raw);
                if !text.is_empty() {
                    out.push((text.to_string(), generic));
                }
            }
            for c in children(node) {
                if c.kind() == "type_arguments" {
                    for arg in children(c) {
                        if arg.is_named() {
                            collect_type_refs(ctx, Some(arg), true, out)?;
                        }
                    }
                }
            }
            return Ok(());
        }
        k if TYPE_WRAPPERS.contains(&k) => {
            for c in children(node) {
                if c.is_named() {
                    collect_type_refs(ctx, Some(c), generic, out)?;
                }
            }
            return Ok(());
        }
        _ => {}
    }
    if node.is_named() {
        for c in children(node) {
            if c.is_named() {
                collect_type_refs(ctx, Some(c), generic, out)?;
            }
        }
    }
    Ok(())
}

fn emit_type_refs(
    ctx: &mut Ctx,
    owner: &str,
    type_node: Option<Node>,
    type_ctx: &'static str,
    line: usize,
) -> R<()> {
    let mut refs: Vec<(String, bool)> = Vec::new();
    collect_type_refs(ctx, type_node, false, &mut refs)?;
    for (ref_name, generic) in refs {
        let c = if generic { "generic_arg" } else { type_ctx };
        let tgt = ctx.ensure_named_node(&ref_name, line)?;
        if tgt != owner {
            ctx.add_edge(owner, &tgt, "references", line, Some(c));
        }
    }
    Ok(())
}

fn emit_param_return_refs(ctx: &mut Ctx, func_node: Node, func_nid: &str, line: usize) -> R<()> {
    if let Some(params) = func_node.child_by_field_name("parameters") {
        for p in children(params) {
            if p.kind() != "parameter" {
                continue;
            }
            emit_type_refs(
                ctx,
                func_nid,
                p.child_by_field_name("type"),
                "parameter_type",
                line,
            )?;
        }
    }
    if let Some(return_type) = func_node.child_by_field_name("return_type") {
        emit_type_refs(ctx, func_nid, Some(return_type), "return_type", line)?;
    }
    Ok(())
}

/// The FIRST collected ref is the supertrait / implemented trait; every later
/// one is a generic argument of it.
fn emit_first_is_parent(
    ctx: &mut Ctx,
    owner: &str,
    node: Option<Node>,
    parent_rel: &'static str,
    line: usize,
) -> R<()> {
    let mut refs: Vec<(String, bool)> = Vec::new();
    collect_type_refs(ctx, node, false, &mut refs)?;
    for (idx, (ref_name, _generic)) in refs.into_iter().enumerate() {
        let tgt = ctx.ensure_named_node(&ref_name, line)?;
        if tgt == owner {
            continue;
        }
        if idx == 0 {
            ctx.add_edge(owner, &tgt, parent_rel, line, None);
        } else {
            ctx.add_edge(owner, &tgt, "references", line, Some("generic_arg"));
        }
    }
    Ok(())
}

fn walk<'tree>(ctx: &mut Ctx, node: Node<'tree>, parent_impl_nid: Option<&str>) -> R<()> {
    match node.kind() {
        "function_item" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let func_name = ctx.text(name_node)?.to_string();
                let line = node.start_position().row + 1;
                let func_nid = match parent_impl_nid {
                    Some(p) => {
                        let nid = ctx.mkid(&[p, &func_name])?;
                        ctx.add_node(&nid, &format!(".{func_name}()"), line);
                        ctx.add_edge(p, &nid, "method", line, None);
                        nid
                    }
                    None => {
                        let nid = ctx.mkid(&[&ctx.stem.clone(), &func_name])?;
                        ctx.add_node(&nid, &format!("{func_name}()"), line);
                        let f = ctx.file_nid.clone();
                        ctx.add_edge(&f, &nid, "contains", line, None);
                        nid
                    }
                };
                emit_param_return_refs(ctx, node, &func_nid, line)?;
                if let Some(body) = node.child_by_field_name("body") {
                    ctx.function_bodies
                        .push((func_nid, body.start_byte(), body.end_byte()));
                }
            }
            return Ok(());
        }
        "struct_item" | "enum_item" | "trait_item" => {
            let name_node = match node.child_by_field_name("name") {
                Some(n) => n,
                None => return Ok(()),
            };
            let item_name = ctx.text(name_node)?.to_string();
            let line = node.start_position().row + 1;
            let item_nid = ctx.mkid(&[&ctx.stem.clone(), &item_name])?;
            ctx.add_node(&item_nid, &item_name, line);
            let f = ctx.file_nid.clone();
            ctx.add_edge(&f, &item_nid, "contains", line, None);

            if node.kind() == "trait_item" {
                for c in children(node) {
                    if c.kind() != "trait_bounds" {
                        continue;
                    }
                    for sub in children(c) {
                        if sub.is_named() {
                            // Per BOUND, so each supertrait's first ref is an
                            // `inherits` -- the index resets for every bound.
                            emit_first_is_parent(ctx, &item_nid, Some(sub), "inherits", line)?;
                        }
                    }
                }
            }
            if node.kind() == "struct_item" {
                for c in children(node) {
                    if c.kind() != "field_declaration_list" {
                        continue;
                    }
                    for field in children(c) {
                        if field.kind() != "field_declaration" {
                            continue;
                        }
                        let mut type_node = field.child_by_field_name("type");
                        if type_node.is_none() {
                            type_node = children(field).into_iter().find(|fc| {
                                matches!(
                                    fc.kind(),
                                    "type_identifier"
                                        | "generic_type"
                                        | "scoped_type_identifier"
                                        | "reference_type"
                                        | "primitive_type"
                                )
                            });
                        }
                        let fline = field.start_position().row + 1;
                        emit_type_refs(ctx, &item_nid, type_node, "field", fline)?;
                    }
                }
                // A tuple struct (`struct Wrapper(pub Logger, Config);`) nests
                // its positional types directly under
                // `ordered_field_declaration_list` with no `field_declaration`
                // wrapper, so without this branch its references vanish.
                for c in children(node) {
                    if c.kind() != "ordered_field_declaration_list" {
                        continue;
                    }
                    let fline = c.start_position().row + 1;
                    for tc in children(c) {
                        if !TYPE_NODES.contains(&tc.kind()) {
                            continue;
                        }
                        emit_type_refs(ctx, &item_nid, Some(tc), "field", fline)?;
                    }
                }
            }
            if node.kind() == "enum_item" {
                // A variant's payload nests under `enum_variant_list` ->
                // `enum_variant` -> `ordered_field_declaration_list` (tuple
                // variant) or `field_declaration_list` (struct variant).
                for c in children(node) {
                    if c.kind() != "enum_variant_list" {
                        continue;
                    }
                    for variant in children(c) {
                        if variant.kind() != "enum_variant" {
                            continue;
                        }
                        let vline = variant.start_position().row + 1;
                        for vc in children(variant) {
                            if vc.kind() == "ordered_field_declaration_list" {
                                for tc in children(vc) {
                                    if TYPE_NODES.contains(&tc.kind()) {
                                        emit_type_refs(ctx, &item_nid, Some(tc), "field", vline)?;
                                    }
                                }
                            } else if vc.kind() == "field_declaration_list" {
                                for field in children(vc) {
                                    if field.kind() != "field_declaration" {
                                        continue;
                                    }
                                    let fline = field.start_position().row + 1;
                                    emit_type_refs(
                                        ctx,
                                        &item_nid,
                                        field.child_by_field_name("type"),
                                        "field",
                                        fline,
                                    )?;
                                }
                            }
                        }
                    }
                }
            }
            return Ok(());
        }
        "impl_item" => {
            let type_node = node.child_by_field_name("type");
            let trait_node = node.child_by_field_name("trait");
            let line = node.start_position().row + 1;
            let mut impl_nid: Option<String> = None;
            if let Some(type_node) = type_node {
                // The WHOLE type text, `impl Foo<Bar>` included: the Python does
                // not reduce it, so `Foo<Bar>` and `Foo` are different nodes.
                let type_name = ctx.text(type_node)?.trim().to_string();
                let nid = ctx.mkid(&[&ctx.stem.clone(), &type_name])?;
                ctx.add_node(&nid, &type_name, line);
                impl_nid = Some(nid);
            }
            if let (Some(trait_node), Some(nid)) = (trait_node, impl_nid.clone()) {
                emit_first_is_parent(ctx, &nid, Some(trait_node), "implements", line)?;
            }
            if let Some(body) = node.child_by_field_name("body") {
                for child in children(body) {
                    walk(ctx, child, impl_nid.as_deref())?;
                }
            }
            return Ok(());
        }
        "use_declaration" => {
            if let Some(arg) = node.child_by_field_name("argument") {
                let raw = ctx.text(arg)?;
                // `split("{")[0].rstrip(":").rstrip("*").rstrip(":")` -- three
                // strips in that order, which is not the same as stripping the
                // union: `a::*` loses `*` then the remaining `::`.
                let clean = raw.split('{').next().unwrap_or(raw);
                let clean = clean.trim_end_matches(':');
                let clean = clean.trim_end_matches('*');
                let clean = clean.trim_end_matches(':');
                let module_name = clean.rsplit("::").next().unwrap_or(clean).trim();
                if !module_name.is_empty() {
                    let tgt = ctx.mkid(&[module_name])?;
                    let f = ctx.file_nid.clone();
                    ctx.add_edge(&f, &tgt, "imports_from", line_of(node), Some("import"));
                }
            }
            return Ok(());
        }
        _ => {}
    }
    // The default recurse DROPS `parent_impl_nid`, so a function nested inside
    // anything other than an `impl` body is file-scoped.
    for child in children(node) {
        walk(ctx, child, None)?;
    }
    Ok(())
}

fn line_of(node: Node) -> usize {
    node.start_position().row + 1
}

fn walk_calls<'tree>(ctx: &mut Ctx, node: Node<'tree>, caller_nid: &str) -> R<()> {
    if node.kind() == "function_item" {
        return Ok(());
    }
    if node.kind() == "call_expression" {
        let func_node = node.child_by_field_name("function");
        let mut callee_name: Option<String> = None;
        let mut is_member_call = false;
        let mut is_scoped_call = false;
        if let Some(func_node) = func_node {
            match func_node.kind() {
                "identifier" => callee_name = Some(ctx.text(func_node)?.to_string()),
                "field_expression" => {
                    is_member_call = true;
                    if let Some(field) = func_node.child_by_field_name("field") {
                        callee_name = Some(ctx.text(field)?.to_string());
                    }
                }
                "scoped_identifier" => {
                    // `Type::method()` still allows an in-file EXTRACTED match,
                    // but never cross-file resolution: a bare last-segment lookup
                    // ignores crate boundaries and invents INFERRED edges (#908).
                    is_scoped_call = true;
                    if let Some(name) = func_node.child_by_field_name("name") {
                        callee_name = Some(ctx.text(name)?.to_string());
                    }
                }
                _ => {}
            }
        }
        if let Some(callee) = callee_name {
            if !BUILTIN_GLOBALS.contains(&callee.as_str()) {
                let tgt_nid = ctx.label_to_nid.get(&callee).cloned();
                let line = node.start_position().row + 1;
                match tgt_nid {
                    Some(tgt) if tgt != caller_nid => {
                        let pair = (caller_nid.to_string(), tgt.clone());
                        if ctx.seen_call_pairs.insert(pair) {
                            let sf = ctx.str_path.to_string();
                            ctx.edges.push(EdgeRow {
                                source: caller_nid.to_string(),
                                target: tgt,
                                relation: "calls",
                                fields: vec![
                                    ("context", Val::Static("call")),
                                    ("confidence", Val::Static("EXTRACTED")),
                                    ("source_file", Val::S(sf)),
                                    ("source_location", Val::S(format!("L{line}"))),
                                    ("weight", Val::F(1.0)),
                                ],
                            });
                        }
                    }
                    _ => {
                        if !is_scoped_call
                            && !TRAIT_METHOD_BLOCKLIST.contains(&callee.to_lowercase().as_str())
                        {
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
    }
    for child in children(node) {
        walk_calls(ctx, child, caller_nid)?;
    }
    Ok(())
}

pub fn walk_rust<'py>(
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

fn find_node<'tree>(root: Node<'tree>, start: usize, end: usize) -> Option<Node<'tree>> {
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        if n.start_byte() == start && n.end_byte() == end && n.kind() == "block" {
            return Some(n);
        }
        if n.start_byte() <= start && n.end_byte() >= end {
            stack.extend(children(n));
        }
    }
    None
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
        .set_language(&tree_sitter_rust::LANGUAGE.into())
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

    for n in &ctx.nodes {
        let mut label = String::new();
        for (k, v) in &n.fields {
            if *k == "label" {
                if let Val::S(s) = v {
                    label = s.clone();
                }
            }
        }
        let normalised = label.trim_matches(|c| c == '(' || c == ')').trim_start_matches('.');
        ctx.label_to_nid.insert(normalised.to_string(), n.id.clone());
    }

    let bodies = ctx.function_bodies.clone();
    for (caller_nid, start, end) in bodies {
        let body = find_node(root, start, end).ok_or("body_not_found")?;
        walk_calls(&mut ctx, body, &caller_nid)?;
    }

    let mut clean: Vec<&EdgeRow> = Vec::new();
    for e in &ctx.edges {
        let target_ok =
            ctx.seen_ids.contains(&e.target) || matches!(e.relation, "imports" | "imports_from");
        if ctx.seen_ids.contains(&e.source) && target_ok {
            clean.push(e);
        }
    }

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

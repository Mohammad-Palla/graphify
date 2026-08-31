//! Julia: a BESPOKE walker.
//!
//! # Two things that would look like bugs in any other walker here
//!
//! 1. **`calls` edges point at ids that may not exist.** `walk_calls` mints
//!    `_make_id(stem, callee_name)` unconditionally -- no `ensure_named_node`,
//!    no stub -- and `extract_julia` does NOT filter its edges. So a call to a
//!    function defined in another file emits an edge to a node this file never
//!    creates, and the corpus-level pass is left to resolve or prune it. Adding
//!    the filter every other bespoke walker has would delete real edges.
//! 2. **The module branch parents to `file_nid`, not `scope_nid`.** Every other
//!    declaration in this walker uses `scope_nid`, so a nested `module` inside a
//!    `module` is still `defines`-linked from the FILE. Reproduced as written.
//!
//! # Parse ceiling
//!
//! 80.4% over 1,106 files. The JuliaLang/julia repo itself is the floor at 77.5%
//! (it carries a lot of deliberately exotic parser-test source); Flux and
//! DataFrames, which look like ordinary Julia packages, are 95.8% and 97.3%.

use std::collections::HashSet;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tree_sitter::{Node, Parser};

use crate::engine::R;
use crate::ids::{file_stem, make_id_ascii};
use crate::js::ast::{children, text_checked};
use crate::js::emit::{self, EdgeRow, NodeRow, Val};
use crate::Outcome;

struct Ctx<'a, 'tree> {
    src: &'a [u8],
    str_path: &'a str,
    stem: String,
    file_nid: String,
    nodes: Vec<NodeRow>,
    edges: Vec<EdgeRow>,
    seen_ids: HashSet<String>,
    /// `(func_nid, body)`. For a `function_definition` the stored node is the
    /// DEFINITION itself, not its body -- the call pass then skips the
    /// `signature` child by hand. For a short function it is the RHS only.
    function_bodies: Vec<(String, Node<'tree>)>,
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

    /// `_semantic_reference_edge`, whose key order is NOT `add_edge`'s:
    /// `context` sits THIRD, right after `relation`, and there is no optional
    /// slot. The order reaches the exported JSON, so this is its own emitter.
    fn add_reference_edge(&mut self, src: &str, tgt: &str, context: &'static str, line: usize) {
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

    /// A SOURCELESS stub for a name not defined in this file, so the
    /// corpus-level rewire can collapse it onto the real definition. A SOURCED
    /// stub would make `_disambiguate_colliding_node_ids` bake this file's path
    /// into the id and block the rewire -- the #1402 phantom-duplicate bug.
    fn ensure_named_node(&mut self, name: &str) -> R<String> {
        let scoped = self.mkid(&[&self.stem.clone(), name])?;
        if self.seen_ids.contains(&scoped) {
            return Ok(scoped);
        }
        let bare = self.mkid(&[name])?;
        if self.seen_ids.insert(bare.clone()) {
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

/// `(type_name, supertype_name)` from a `type_head`.
///
/// A bare declaration (`Foo`) exposes an `identifier`. A subtyping declaration
/// (`Foo <: Bar`) wraps both names in a `binary_expression`, and there the FIRST
/// identifier is the type and the LAST is the supertype -- which are the same
/// node when there is only one, so the `len >= 2` guard is what stops `Foo` from
/// inheriting itself.
fn type_head_names(ctx: &Ctx, type_head: Node) -> R<(Option<String>, Option<String>)> {
    let bin_expr = children(type_head)
        .into_iter()
        .find(|c| c.kind() == "binary_expression");
    if let Some(be) = bin_expr {
        let ids: Vec<Node> = children(be)
            .into_iter()
            .filter(|c| c.kind() == "identifier")
            .collect();
        if ids.is_empty() {
            return Ok((None, None));
        }
        let name = ctx.text(ids[0])?.to_string();
        let super_name = if ids.len() >= 2 {
            Some(ctx.text(ids[ids.len() - 1])?.to_string())
        } else {
            None
        };
        return Ok((Some(name), super_name));
    }
    let name_node = children(type_head)
        .into_iter()
        .find(|c| c.kind() == "identifier");
    Ok((
        match name_node {
            Some(n) => Some(ctx.text(n)?.to_string()),
            None => None,
        },
        None,
    ))
}

/// The function name inside a `signature`: `signature > call_expression >
/// identifier`, where the identifier must be the call's FIRST child.
fn func_name_from_signature(ctx: &Ctx, sig: Node) -> R<Option<String>> {
    for child in children(sig) {
        if child.kind() == "call_expression" {
            if let Some(callee) = child.child(0) {
                if callee.kind() == "identifier" {
                    return Ok(Some(ctx.text(callee)?.to_string()));
                }
            }
        }
    }
    Ok(None)
}

/// `identifier` (`Foo`), `scoped_identifier` (`Base.Threads`) or `import_path`
/// (relative `..Sibling`) -> the module name.
fn mod_name(ctx: &Ctx, n: Node) -> R<Option<String>> {
    if n.kind() == "import_path" {
        let ids: Vec<Node> = children(n)
            .into_iter()
            .filter(|c| c.kind() == "identifier")
            .collect();
        return Ok(match ids.last() {
            Some(last) => Some(ctx.text(*last)?.to_string()),
            None => None,
        });
    }
    if matches!(n.kind(), "identifier" | "scoped_identifier") {
        return Ok(Some(ctx.text(n)?.to_string()));
    }
    Ok(None)
}

fn walk_calls<'tree>(ctx: &mut Ctx<'_, 'tree>, node: Node<'tree>, func_nid: &str) -> R<()> {
    let t = node.kind();
    if matches!(t, "function_definition" | "short_function_definition") {
        return Ok(());
    }
    if t == "call_expression" {
        if let Some(callee) = node.child(0) {
            let line = node.start_position().row + 1;
            if callee.kind() == "identifier" {
                let callee_name = ctx.text(callee)?.to_string();
                // NOT `ensure_named_node`: the target id is minted blind and may
                // name no node in this file. See the module doc.
                let target = ctx.mkid(&[&ctx.stem.clone(), &callee_name])?;
                ctx.add_edge(func_nid, &target, "calls", line, Some("call"));
            } else if callee.kind() == "field_expression" {
                // `len(children) >= 3` -- `obj.method` has object, `.`, method.
                let kids = children(callee);
                if kids.len() >= 3 {
                    let method_name = ctx.text(kids[kids.len() - 1])?.to_string();
                    let target = ctx.mkid(&[&ctx.stem.clone(), &method_name])?;
                    ctx.add_edge(func_nid, &target, "calls", line, Some("call"));
                }
            }
        }
    }
    for child in children(node) {
        walk_calls(ctx, child, func_nid)?;
    }
    Ok(())
}

fn walk<'tree>(ctx: &mut Ctx<'_, 'tree>, node: Node<'tree>, scope_nid: &str) -> R<()> {
    let t = node.kind();

    if t == "module_definition" {
        let name_node = children(node).into_iter().find(|c| c.kind() == "identifier");
        if let Some(nn) = name_node {
            let name = ctx.text(nn)?.to_string();
            let mod_nid = ctx.mkid(&[&ctx.stem.clone(), &name])?;
            let line = node.start_position().row + 1;
            ctx.add_node(&mod_nid, &name, line);
            // `file_nid`, not `scope_nid` -- see the module doc.
            let f = ctx.file_nid.clone();
            ctx.add_edge(&f, &mod_nid, "defines", line, None);
            for child in children(node) {
                walk(ctx, child, &mod_nid)?;
            }
        }
        return Ok(());
    }

    if t == "struct_definition" {
        // `struct` and `mutable struct` are both `struct_definition`.
        let type_head = children(node).into_iter().find(|c| c.kind() == "type_head");
        let type_head = match type_head {
            Some(th) => th,
            None => return Ok(()),
        };
        let (struct_name, super_name) = type_head_names(ctx, type_head)?;
        let struct_name = match struct_name {
            Some(s) if !s.is_empty() => s,
            _ => return Ok(()),
        };
        let struct_nid = ctx.mkid(&[&ctx.stem.clone(), &struct_name])?;
        let line = node.start_position().row + 1;
        ctx.add_node(&struct_nid, &struct_name, line);
        ctx.add_edge(scope_nid, &struct_nid, "defines", line, None);
        if let Some(sn) = super_name {
            if !sn.is_empty() {
                let base = ctx.ensure_named_node(&sn)?;
                ctx.add_edge(&struct_nid, &base, "inherits", line, None);
            }
        }
        // `name::Type` lowers to a `typed_expression` child of the struct, not of
        // a field list.
        for child in children(node) {
            if child.kind() != "typed_expression" {
                continue;
            }
            let type_ids: Vec<Node> = children(child)
                .into_iter()
                .filter(|c| c.kind() == "identifier")
                .collect();
            if type_ids.len() >= 2 {
                let field_line = child.start_position().row + 1;
                let type_name = ctx.text(type_ids[type_ids.len() - 1])?.to_string();
                let type_nid = ctx.ensure_named_node(&type_name)?;
                ctx.add_reference_edge(&struct_nid, &type_nid, "field", field_line);
            }
        }
        return Ok(());
    }

    if t == "abstract_definition" {
        // Abstract types are the backbone of Julia's dispatch hierarchies, so an
        // intermediate `abstract type Foo <: Bar end` vanishing used to break the
        // whole inheritance chain and lose the type node with it.
        let type_head = children(node).into_iter().find(|c| c.kind() == "type_head");
        if let Some(th) = type_head {
            let (abs_name, super_name) = type_head_names(ctx, th)?;
            if let Some(abs_name) = abs_name {
                if !abs_name.is_empty() {
                    let abs_nid = ctx.mkid(&[&ctx.stem.clone(), &abs_name])?;
                    let line = node.start_position().row + 1;
                    ctx.add_node(&abs_nid, &abs_name, line);
                    ctx.add_edge(scope_nid, &abs_nid, "defines", line, None);
                    if let Some(sn) = super_name {
                        if !sn.is_empty() {
                            let base = ctx.ensure_named_node(&sn)?;
                            ctx.add_edge(&abs_nid, &base, "inherits", line, None);
                        }
                    }
                }
            }
        }
        return Ok(());
    }

    if t == "function_definition" {
        let sig = children(node).into_iter().find(|c| c.kind() == "signature");
        if let Some(sig) = sig {
            let func_name = func_name_from_signature(ctx, sig)?;
            if let Some(func_name) = func_name {
                if !func_name.is_empty() {
                    let func_nid = ctx.mkid(&[&ctx.stem.clone(), &func_name])?;
                    let line = node.start_position().row + 1;
                    ctx.add_node(&func_nid, &format!("{func_name}()"), line);
                    ctx.add_edge(scope_nid, &func_nid, "defines", line, None);
                    // The DEFINITION node, not its body.
                    ctx.function_bodies.push((func_nid, node));
                }
            }
        }
        return Ok(());
    }

    if t == "assignment" {
        // The short form `foo(x) = expr`.
        let lhs = node.child(0);
        if let Some(lhs) = lhs {
            if lhs.kind() == "call_expression" {
                if let Some(callee) = lhs.child(0) {
                    if callee.kind() == "identifier" {
                        let func_name = ctx.text(callee)?.to_string();
                        let func_nid = ctx.mkid(&[&ctx.stem.clone(), &func_name])?;
                        let line = node.start_position().row + 1;
                        ctx.add_node(&func_nid, &format!("{func_name}()"), line);
                        ctx.add_edge(scope_nid, &func_nid, "defines", line, None);
                        // ONLY the RHS: walking the LHS would find the
                        // definition's own `call_expression` and emit a self-loop.
                        let kids = children(node);
                        if kids.len() >= 3 {
                            ctx.function_bodies
                                .push((func_nid, kids[kids.len() - 1]));
                        }
                    }
                }
            }
        }
        return Ok(());
    }

    if matches!(t, "using_statement" | "import_statement") {
        let line = node.start_position().row + 1;
        for child in children(node) {
            let name = if matches!(
                child.kind(),
                "identifier" | "scoped_identifier" | "import_path"
            ) {
                mod_name(ctx, child)?
            } else if child.kind() == "selected_import" {
                // `import Base.Threads: nthreads` -- the PACKAGE is the first
                // named child and may itself be scoped or a relative path.
                let pkg = children(child).into_iter().find(|c| {
                    matches!(c.kind(), "identifier" | "scoped_identifier" | "import_path")
                });
                match pkg {
                    Some(p) => mod_name(ctx, p)?,
                    None => None,
                }
            } else {
                None
            };
            if let Some(name) = name {
                if !name.is_empty() {
                    let imp_nid = ctx.mkid(&[&name])?;
                    // A SOURCED node, unlike every other cross-file target here.
                    ctx.add_node(&imp_nid, &name, line);
                    ctx.add_edge(scope_nid, &imp_nid, "imports", line, Some("import"));
                }
            }
        }
        return Ok(());
    }

    for child in children(node) {
        walk(ctx, child, scope_nid)?;
    }
    Ok(())
}

pub fn walk_julia<'py>(
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
        .set_language(&tree_sitter_julia::LANGUAGE.into())
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
        seen_ids: HashSet::new(),
        function_bodies: Vec::new(),
    };

    let file_label = path.rsplit('/').next().unwrap_or(path).to_string();
    ctx.add_node(&file_nid, &file_label, 1);

    walk(&mut ctx, root, &file_nid.clone())?;

    let bodies: Vec<(String, Node)> = ctx.function_bodies.clone();
    for (func_nid, body) in bodies {
        if body.kind() == "function_definition" {
            // Walk the children directly: passing the definition itself would hit
            // the boundary check and return immediately. The `signature` child is
            // skipped because it holds the function's OWN call_expression, which
            // would become a self-loop.
            for child in children(body) {
                if child.kind() != "signature" {
                    walk_calls(&mut ctx, child, &func_nid)?;
                }
            }
        } else {
            walk_calls(&mut ctx, body, &func_nid)?;
        }
    }

    // NO edge cleaning -- see the module doc.
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
    out.set_item("nodes", nodes).map_err(|_| "py_error")?;
    out.set_item("edges", edges).map_err(|_| "py_error")?;
    Ok(out)
}

//! Fortran: a BESPOKE walker.
//!
//! # Case
//!
//! Fortran is case-insensitive, so EVERY name here is lowercased before it
//! reaches an id or a label. Missing one produces two nodes for one procedure
//! and no edge between them.
//!
//! # The C preprocessor stays in Python
//!
//! A capital-F extension (`.F90`, `.F`) conventionally requires `cpp` expansion
//! before parsing, and `extract_fortran` shells out to run it. The kernel does
//! not and must not spawn a subprocess, so `extract_fortran` computes the source
//! bytes FIRST and hands them to `try_extract` as `source_override` -- the same
//! mechanism Vue SFCs use. The kernel therefore always walks exactly the bytes
//! Python would have parsed, preprocessed or not, and neither side runs `cpp`
//! twice.
//!
//! # Parse ceiling: read this before trusting the aggregate
//!
//! The headline number over the three corpora is 12.5%, and it is MEANINGLESS.
//! Split by extension:
//!
//! ```text
//! .f90   509 files   99.6% clean     free-form, modern Fortran
//! .f    3581 files    0.0% clean     FIXED-FORM FORTRAN 77
//! ```
//!
//! tree-sitter-fortran cannot parse fixed-form source at all, and LAPACK -- one
//! of the three corpora -- is 3,581 fixed-form files. Rejecting Fortran on the
//! 12.5% would have been rejecting it on a corpus-selection artefact. Free-form
//! Fortran, which is what anything written since roughly 1995 looks like, is
//! among the highest ceilings measured here. Fixed-form files defer on
//! `has_error` at zero risk.

use std::collections::HashSet;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tree_sitter::{Node, Parser};

use crate::engine::R;
use crate::ids::{file_stem, make_id_ascii};
use crate::js::ast::{children, text_checked};
use crate::js::emit::{self, EdgeRow, NodeRow, Val};
use crate::Outcome;

/// `walk_calls`'s boundary set: a nested scope has its own `scope_bodies` entry.
const SCOPE_KINDS: &[&str] = &[
    "subroutine",
    "function",
    "module",
    "program",
    "internal_procedures",
];

/// The statement-header kinds skipped when walking a scope for calls -- the
/// header holds the procedure's OWN name and would self-loop.
const STMT_HEADERS: &[&str] = &[
    "subroutine_statement",
    "function_statement",
    "program_statement",
    "module_statement",
];

struct Ctx<'a, 'tree> {
    src: &'a [u8],
    str_path: &'a str,
    stem: String,
    file_nid: String,
    nodes: Vec<NodeRow>,
    edges: Vec<EdgeRow>,
    seen_ids: HashSet<String>,
    scope_bodies: Vec<(String, Node<'tree>)>,
}

impl<'a, 'tree> Ctx<'a, 'tree> {
    fn text(&self, node: Node) -> R<&'a str> {
        text_checked(node, self.src).ok_or("invalid_utf8_text")
    }

    /// Every Fortran name goes through this: `_read_text(...).lower()`.
    ///
    /// Python's `str.lower()` is full Unicode lowercasing; `to_lowercase` is the
    /// same algorithm. An identifier that lowercases differently would then have
    /// to survive `make_id_ascii`, which rejects non-ASCII and defers.
    fn text_lower(&self, node: Node) -> R<String> {
        Ok(self.text(node)?.to_lowercase())
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

/// `_fortran_name`: the first `name`-or-`identifier` child, lowercased.
fn fortran_name(ctx: &Ctx, stmt: Node) -> R<Option<String>> {
    for child in children(stmt) {
        if matches!(child.kind(), "name" | "identifier") {
            return Ok(Some(ctx.text_lower(child)?));
        }
    }
    Ok(None)
}

/// `references[parameter_type]` / `references[return_type]` for a procedure,
/// derived from its `variable_declaration` SIBLINGS.
///
/// Fortran declares parameter types in the body, not the signature, so a
/// declaration has to be matched back to the parameter list by NAME. Only
/// `derived_type` declarations count -- an intrinsic `real :: x` names no
/// user-defined type.
fn emit_signature_refs(ctx: &mut Ctx, scope: Node, fn_nid: &str, is_function: bool) -> R<()> {
    let stmt_kind = if is_function {
        "function_statement"
    } else {
        "subroutine_statement"
    };
    let stmt = match children(scope).into_iter().find(|c| c.kind() == stmt_kind) {
        Some(s) => s,
        None => return Ok(()),
    };

    let mut param_names: HashSet<String> = HashSet::new();
    if let Some(params) = children(stmt).into_iter().find(|c| c.kind() == "parameters") {
        for c in children(params) {
            if c.kind() == "identifier" {
                param_names.insert(ctx.text_lower(c)?);
            }
        }
    }

    let mut result_name: Option<String> = None;
    if is_function {
        let result_node = children(stmt)
            .into_iter()
            .find(|c| c.kind() == "function_result");
        match result_node {
            Some(rn) => {
                if let Some(res_id) = children(rn).into_iter().find(|c| c.kind() == "identifier") {
                    result_name = Some(ctx.text_lower(res_id)?);
                }
            }
            // An implicit result variable shares the function's own name.
            None => result_name = fortran_name(ctx, stmt)?,
        }
    }

    for child in children(scope) {
        if child.kind() != "variable_declaration" {
            continue;
        }
        let derived = children(child).into_iter().find(|c| c.kind() == "derived_type");
        let derived = match derived {
            Some(d) => d,
            None => continue,
        };
        let type_name_node = children(derived)
            .into_iter()
            .find(|c| c.kind() == "type_name");
        let type_name = match type_name_node {
            Some(t) => ctx.text_lower(t)?,
            None => continue,
        };
        for var in children(child) {
            if var.kind() != "identifier" {
                continue;
            }
            let var_name = ctx.text_lower(var)?;
            let var_line = var.start_position().row + 1;
            if param_names.contains(&var_name) {
                let tgt = ctx.ensure_named_node(&type_name)?;
                if tgt != fn_nid {
                    ctx.add_edge(fn_nid, &tgt, "references", var_line, Some("parameter_type"));
                }
            } else if is_function && Some(&var_name) == result_name.as_ref() {
                let tgt = ctx.ensure_named_node(&type_name)?;
                if tgt != fn_nid {
                    ctx.add_edge(fn_nid, &tgt, "references", var_line, Some("return_type"));
                }
            }
        }
    }
    Ok(())
}

fn walk_calls<'tree>(ctx: &mut Ctx<'_, 'tree>, node: Node<'tree>, scope_nid: &str) -> R<()> {
    let t = node.kind();
    if SCOPE_KINDS.contains(&t) {
        return Ok(());
    }
    if t == "subroutine_call" {
        // `call FOO(args)`. Emitted unconditionally -- the target may name no
        // node in this file, and the corpus pass resolves it.
        if let Some(nn) = children(node).into_iter().find(|c| c.kind() == "identifier") {
            let callee = ctx.text_lower(nn)?;
            let target = ctx.mkid(&[&ctx.stem.clone(), &callee])?;
            let line = node.start_position().row + 1;
            ctx.add_edge(scope_nid, &target, "calls", line, Some("call"));
        }
    } else if t == "call_expression" {
        // `x = compute(args)`. Fortran spells a function call and an ARRAY INDEX
        // identically, so this arm emits ONLY when the callee already names a
        // node in this file -- an array variable produces no matching node. The
        // whole call pass runs after the declaration walk, so `seen_ids` is
        // complete by now and the guard is meaningful.
        if let Some(nn) = children(node).into_iter().find(|c| c.kind() == "identifier") {
            let callee = ctx.text_lower(nn)?;
            let target = ctx.mkid(&[&ctx.stem.clone(), &callee])?;
            if ctx.seen_ids.contains(&target) && target != scope_nid {
                let line = node.start_position().row + 1;
                ctx.add_edge(scope_nid, &target, "calls", line, Some("call"));
            }
        }
    }
    for child in children(node) {
        walk_calls(ctx, child, scope_nid)?;
    }
    Ok(())
}

fn walk<'tree>(ctx: &mut Ctx<'_, 'tree>, node: Node<'tree>, scope_nid: &str) -> R<()> {
    let t = node.kind();

    if t == "program" || t == "module" {
        let stmt_kind = if t == "program" {
            "program_statement"
        } else {
            "module_statement"
        };
        let stmt = children(node).into_iter().find(|c| c.kind() == stmt_kind);
        let name = match stmt {
            Some(s) => fortran_name(ctx, s)?,
            None => None,
        };
        if let Some(name) = name {
            if !name.is_empty() {
                let nid = ctx.mkid(&[&ctx.stem.clone(), &name])?;
                let line = node.start_position().row + 1;
                ctx.add_node(&nid, &name, line);
                let f = ctx.file_nid.clone();
                ctx.add_edge(&f, &nid, "defines", line, None);
                // A `program` has executable statements and so registers a body;
                // a `module` does NOT -- its procedures register their own.
                if t == "program" {
                    ctx.scope_bodies.push((nid.clone(), node));
                }
                for child in children(node) {
                    walk(ctx, child, &nid)?;
                }
            }
        }
        return Ok(());
    }

    if t == "internal_procedures" {
        // A transparent wrapper: procedures inside a module live under it, and
        // the enclosing scope must be threaded through unchanged.
        for child in children(node) {
            walk(ctx, child, scope_nid)?;
        }
        return Ok(());
    }

    if t == "derived_type_definition" {
        let stmt = children(node)
            .into_iter()
            .find(|c| c.kind() == "derived_type_statement");
        if let Some(stmt) = stmt {
            let name_node = children(stmt).into_iter().find(|c| c.kind() == "type_name");
            if let Some(nn) = name_node {
                let type_name = ctx.text_lower(nn)?;
                let type_nid = ctx.mkid(&[&ctx.stem.clone(), &type_name])?;
                let line = node.start_position().row + 1;
                ctx.add_node(&type_nid, &type_name, line);
                ctx.add_edge(scope_nid, &type_nid, "defines", line, None);
            }
        }
        return Ok(());
    }

    if t == "subroutine" || t == "function" {
        let is_function = t == "function";
        let stmt_kind = if is_function {
            "function_statement"
        } else {
            "subroutine_statement"
        };
        let stmt = children(node).into_iter().find(|c| c.kind() == stmt_kind);
        let name = match stmt {
            Some(s) => fortran_name(ctx, s)?,
            None => None,
        };
        if let Some(name) = name {
            if !name.is_empty() {
                let nid = ctx.mkid(&[&ctx.stem.clone(), &name])?;
                let line = node.start_position().row + 1;
                ctx.add_node(&nid, &format!("{name}()"), line);
                ctx.add_edge(scope_nid, &nid, "defines", line, None);
                ctx.scope_bodies.push((nid.clone(), node));
                emit_signature_refs(ctx, node, &nid, is_function)?;
                for child in children(node) {
                    walk(ctx, child, &nid)?;
                }
            }
        }
        return Ok(());
    }

    if t == "use_statement" {
        let line = node.start_position().row + 1;
        let name_node = children(node)
            .into_iter()
            .find(|c| matches!(c.kind(), "module_name" | "name" | "identifier"));
        if let Some(nn) = name_node {
            let mod_name = ctx.text_lower(nn)?;
            let imp_nid = ctx.mkid(&[&mod_name])?;
            // A SOURCED node, so a `use`d module is not a sourceless stub.
            ctx.add_node(&imp_nid, &mod_name, line);
            // `context="use"`, not `"import"`.
            ctx.add_edge(scope_nid, &imp_nid, "imports", line, Some("use"));
        }
        return Ok(());
    }

    for child in children(node) {
        walk(ctx, child, scope_nid)?;
    }
    Ok(())
}

pub fn walk_fortran<'py>(
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
        .set_language(&tree_sitter_fortran::LANGUAGE.into())
        .map_err(|_| "grammar_load_failed")?;
    let tree = parser.parse(source, None).ok_or("parse_failed")?;
    let root = tree.root_node();
    // Every fixed-form FORTRAN 77 file lands here. See the module doc.
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
        scope_bodies: Vec::new(),
    };

    let file_label = path.rsplit('/').next().unwrap_or(path).to_string();
    ctx.add_node(&file_nid, &file_label, 1);

    walk(&mut ctx, root, &file_nid.clone())?;

    let bodies: Vec<(String, Node)> = ctx.scope_bodies.clone();
    for (scope_nid, body) in bodies {
        for child in children(body) {
            if !STMT_HEADERS.contains(&child.kind()) {
                walk_calls(&mut ctx, child, &scope_nid)?;
            }
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
    for e in &ctx.edges {
        edges
            .append(emit::edge_to_py(py, e).map_err(|_| "py_error")?)
            .map_err(|_| "py_error")?;
    }
    out.set_item("nodes", nodes).map_err(|_| "py_error")?;
    out.set_item("edges", edges).map_err(|_| "py_error")?;
    Ok(out)
}

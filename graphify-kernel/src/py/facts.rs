//! Symbol-resolution facts for `_collect_python_facts_one_file`, from the
//! phase-2 parse.
//!
//! `augment.collect_python` is 4.2s of django's phase 3 and parses every Python
//! file for a SECOND time (the cross-file pass was the third). Unlike that one it
//! runs in a pool, so fusing it is not automatically a win: the parse and walk
//! move to phase 2 where they are already paid for, but
//! `_resolve_python_module_path` and the submodule `is_file()` probes move from
//! four workers into the SERIAL parent.
//!
//! Measured before building, over django's 2,929 files run serially:
//!
//! ```text
//! total             6.39s
//!   parse           2.26s  (35.4%)  -> phase 2, already paid
//!   module resolve  0.28s  ( 4.4%)  -> the serial parent   [10,176 calls]
//!   walk + rest     3.84s  (60.1%)  -> phase 2, native
//! ```
//!
//! 95.6% moves to where it is free and 4.4% moves to where it costs, so the trade
//! is worth making. It would not have been obvious either way without measuring,
//! and the shape differs from the cross-file pass precisely because that one was
//! already serial.
//!
//! So this emits the parsed material and nothing resolved: each
//! `from ... import ...` as `(level, module_name, line, [(imported, local)])`, and
//! each top-level function's calls as `(source_id, [(callee, line)])`. Every
//! filesystem decision stays in Python, as it does for the import resolver.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tree_sitter::Node;

use crate::ids::make_id_ascii;
use crate::js::ast::children;
use super::{Ctx, R};

pub struct ImportFact {
    pub level: usize,
    pub module_name: String,
    pub line: usize,
    /// `(imported_name, local_name)` in child order.
    pub names: Vec<(String, String)>,
}

pub struct UseFact {
    /// The top-level function's NAME, not its node id.
    ///
    /// The id is `_make_id(_file_stem(path), name)`, and `_file_stem` is derived
    /// from the ABSOLUTE path -- so baking it in here embeds the scan root's slug
    /// in the payload, which rides into the per-file AST cache. That cache is
    /// portable by construction: replaying it under a different root must not
    /// replay the first root's ids (#2257). `test_cache.py::
    /// test_warm_cache_from_another_root_does_not_leak_that_root` caught exactly
    /// that. The parent mints the id from the path it is actually looking at.
    pub source_name: String,
    /// `(callee identifier, line)` in walk order.
    pub calls: Vec<(String, usize)>,
}

pub struct Facts {
    pub imports: Vec<ImportFact>,
    pub uses: Vec<UseFact>,
}

/// `_python_import_from_module`: `(level, module_name)`, or None.
fn import_from_module(ctx: &Ctx, node: Node) -> R<Option<(usize, String)>> {
    let mut level = 0usize;
    let mut module_name = String::new();
    for child in children(node) {
        if child.kind() == "import" {
            break;
        }
        if child.kind() == "relative_import" {
            let raw = ctx.text(child)?;
            let stripped = raw.trim_start_matches('.');
            level = raw.len() - stripped.len();
            if !stripped.is_empty() {
                module_name = stripped.to_string();
            }
            // A `dotted_name` inside overrides the text-derived remainder.
            for sub in children(child) {
                if sub.kind() == "dotted_name" {
                    module_name = ctx.text(sub)?.to_string();
                }
            }
        } else if child.kind() == "dotted_name" {
            module_name = ctx.text(child)?.to_string();
        }
    }
    if level == 0 && module_name.is_empty() {
        return Ok(None);
    }
    Ok(Some((level, module_name)))
}

/// `_python_imported_names`: the names after the `import` keyword.
///
/// A bare `dotted_name` binds its LAST segment (`from p import a.b` is not legal
/// Python, but the grammar admits it and the Python takes `split(".")[-1]`), and
/// an `aliased_import` binds its alias, or the same last segment when absent.
fn imported_names(ctx: &Ctx, node: Node) -> R<Vec<(String, String)>> {
    let mut out = Vec::new();
    let mut past_import = false;
    for child in children(node) {
        if child.kind() == "import" {
            past_import = true;
            continue;
        }
        if !past_import {
            continue;
        }
        match child.kind() {
            "dotted_name" => {
                let name = ctx.text(child)?.to_string();
                let local = name.rsplit('.').next().unwrap_or(&name).to_string();
                out.push((name, local));
            }
            "aliased_import" => {
                let Some(name_node) = child.child_by_field_name("name") else {
                    continue;
                };
                let name = ctx.text(name_node)?.to_string();
                let local = match child.child_by_field_name("alias") {
                    Some(a) => ctx.text(a)?.to_string(),
                    None => name.rsplit('.').next().unwrap_or(&name).to_string(),
                };
                out.push((name, local));
            }
            _ => {}
        }
    }
    Ok(out)
}

/// `_walk_python_tree` is pre-order over EVERY node, so an import nested in a
/// function or class body is collected too.
fn collect_imports(ctx: &Ctx, node: Node, out: &mut Vec<ImportFact>) -> R<()> {
    if node.kind() == "import_from_statement" {
        if let Some((level, module_name)) = import_from_module(ctx, node)? {
            out.push(ImportFact {
                level,
                module_name,
                line: node.start_position().row + 1,
                names: imported_names(ctx, node)?,
            });
        }
    }
    for child in children(node) {
        collect_imports(ctx, child, out)?;
    }
    Ok(())
}

/// `_python_call_identifier`: a `call` whose `function` is a bare identifier.
fn collect_calls(ctx: &Ctx, node: Node, out: &mut Vec<(String, usize)>) -> R<()> {
    if node.kind() == "call" {
        if let Some(f) = node.child_by_field_name("function") {
            if f.kind() == "identifier" {
                out.push((ctx.text(f)?.to_string(), node.start_position().row + 1));
            }
        }
    }
    for child in children(node) {
        collect_calls(ctx, child, out)?;
    }
    Ok(())
}

pub fn collect(ctx: &Ctx, root: Node) -> R<Facts> {
    let mut imports = Vec::new();
    collect_imports(ctx, root, &mut imports)?;

    // `_python_top_level_function_bodies`: DIRECT children of the module only, so
    // a method or a nested def contributes no `uses` facts.
    let mut uses = Vec::new();
    for node in children(root) {
        if node.kind() != "function_definition" {
            continue;
        }
        let (Some(name_node), Some(body)) = (
            node.child_by_field_name("name"),
            node.child_by_field_name("body"),
        ) else {
            continue;
        };
        let source_name = ctx.text(name_node)?.to_string();
        // The id the parent will mint must be reachable, so refuse a name whose
        // id recipe we cannot reproduce rather than emitting one Python would
        // normalise differently.
        make_id_ascii(&[&ctx.stem, &source_name]).ok_or("non_ascii_id")?;
        let mut calls = Vec::new();
        collect_calls(ctx, body, &mut calls)?;
        uses.push(UseFact { source_name, calls });
    }
    Ok(Facts { imports, uses })
}

pub fn to_py<'py>(py: Python<'py>, f: &Facts) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let imports = PyList::empty(py);
    for i in &f.imports {
        let names = PyList::empty(py);
        for (a, b) in &i.names {
            names.append((a, b))?;
        }
        imports.append((i.level, &i.module_name, i.line, names))?;
    }
    out.set_item("imports", imports)?;
    let uses = PyList::empty(py);
    for u in &f.uses {
        let calls = PyList::empty(py);
        for (name, line) in &u.calls {
            calls.append((name, *line))?;
        }
        uses.append((&u.source_name, calls))?;
    }
    out.set_item("uses", uses)?;
    Ok(out)
}

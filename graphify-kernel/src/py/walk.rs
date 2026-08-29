//! The declaration walk: the Python-live branches of `_extract_generic`'s `walk`.
//!
//! Branch order matters and is preserved: imports, classes, functions,
//! `decorated_definition`, `ERROR`, then the default recurse. Each of the first
//! five `return`s, so a node matching an earlier arm never reaches a later one.
//!
//! `_PYTHON_CONFIG` supplies `class_types={"class_definition"}`,
//! `function_types={"function_definition"}`,
//! `import_types={"import_statement","import_from_statement"}`, `name_field="name"`,
//! `body_field="body"`, and EMPTY `name_fallback_child_types` /
//! `body_fallback_child_types` -- so `_find_body` is just the `body` field and the
//! name-fallback loops are dead. They are written out below anyway, as empty-set
//! membership tests would be, so the shape stays comparable to the Python.

use tree_sitter::Node;

use crate::js::ast::{children, line_of};
use super::helpers::{self, DECORATOR_NOISE};
use super::imports;
use super::{Ctx, R};

/// `_find_body(node, config)` for `_PYTHON_CONFIG`: the `body` field, with no
/// fallback child types to scan.
fn find_body(node: Node) -> Option<Node> {
    node.child_by_field_name("body")
}

pub fn walk<'tree>(
    ctx: &mut Ctx<'_, 'tree>,
    node: Node<'tree>,
    parent_class_nid: Option<&str>,
) -> R<()> {
    let t = node.kind();

    // ── Import types ────────────────────────────────────────────────────────
    if matches!(t, "import_statement" | "import_from_statement") {
        imports::import_python(ctx, node)?;
        // `_import_python` returns None, so the `imported_modules` module-node
        // branch is dead for Python (only Swift's handler returns pairs). And
        // `export_statement` is not a Python node kind, so the fall-through
        // there cannot fire either.
        return Ok(());
    }

    // ── Class types ─────────────────────────────────────────────────────────
    if t == "class_definition" {
        let Some(name_node) = node.child_by_field_name("name") else {
            return Ok(());
        };
        let class_name = ctx.text(name_node)?.to_string();
        // `_make_id(stem, ".".join(namespace_stack), class_name)`. namespace_stack
        // is permanently empty for Python and `make_id_ascii` drops empty parts,
        // exactly as Python's `if p` filter does.
        let class_nid = ctx.mkid(&[&ctx.stem.clone(), "", &class_name])?;
        let line = line_of(node);
        ctx.add_node(&class_nid, &class_name, line);
        // A class is callable (constructor)...
        ctx.callable_def_nids.insert(class_nid.clone());
        // ...but only via its constructor, which the indirect_call guard excludes
        // to avoid false edges on `select(Model)` (#2137).
        ctx.callable_class_nids.insert(class_nid.clone());

        // A nested class is contained by its ENCLOSING type, not the file
        // (#2040). The `!= class_nid` guard avoids a self-loop when same-name
        // nesting collides ids, since class ids omit the enclosing type name.
        match parent_class_nid {
            Some(p) if p != class_nid => {
                let p = p.to_string();
                ctx.add_edge(&p, &class_nid, "contains", line);
            }
            _ => {
                let f = ctx.file_nid.clone();
                ctx.add_edge(&f, &class_nid, "contains", line);
            }
        }

        // Python-specific: inheritance.
        if let Some(args) = node.child_by_field_name("superclasses") {
            for arg in children(args) {
                if arg.kind() != "identifier" {
                    continue;
                }
                let base = ctx.text(arg)?.to_string();
                let base_nid = ctx.ensure_named_node(&base, line)?;
                ctx.add_edge(&class_nid, &base_nid, "inherits", line);
            }
        }

        if let Some(body) = find_body(node) {
            for child in children(body) {
                walk(ctx, child, Some(&class_nid))?;
            }
        }
        return Ok(());
    }

    // ── Function types ──────────────────────────────────────────────────────
    if t == "function_definition" {
        let func_name = match node.child_by_field_name("name") {
            Some(n) => ctx.text(n)?.to_string(),
            None => return Ok(()),
        };
        if func_name.is_empty() {
            return Ok(()); // `if not func_name`
        }
        // `sanitize_symbol_name_fn` is None for Python, so sanitized == raw.
        // A name that normalizes to nothing would collapse `_make_id(prefix, name)`
        // onto the absolute-path-derived prefix, leaking the scan path and
        // colliding with the file/class node (#1899).
        if !ctx.normalizes_to_something(&func_name)? {
            return Ok(());
        }
        let line = line_of(node);
        let func_nid = match parent_class_nid {
            Some(p) => {
                let nid = ctx.mkid(&[p, &func_name])?;
                ctx.add_node(&nid, &format!(".{func_name}()"), line);
                let p = p.to_string();
                ctx.add_edge(&p, &nid, "method", line);
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
        ctx.callable_def_nids.insert(func_nid.clone());
        let bound = helpers::local_bound_names(ctx, node)?;
        ctx.local_bound_names.insert(func_nid.clone(), bound);

        let params_node = node.child_by_field_name("parameters");
        for (ref_name, is_generic) in helpers::collect_param_refs(ctx, params_node)? {
            let context = if is_generic { "generic_arg" } else { "parameter_type" };
            let target_nid = ctx.ensure_named_node(&ref_name, line)?;
            if target_nid != func_nid {
                ctx.add_semantic_reference_edge(&func_nid, &target_nid, context, line);
            }
        }
        if let Some(return_type_node) = node.child_by_field_name("return_type") {
            let mut return_refs = Vec::new();
            helpers::collect_type_refs(ctx, Some(return_type_node), false, &mut return_refs)?;
            for (ref_name, is_generic) in return_refs {
                let context = if is_generic { "generic_arg" } else { "return_type" };
                let target_nid = ctx.ensure_named_node(&ref_name, line)?;
                if target_nid != func_nid {
                    ctx.add_semantic_reference_edge(&func_nid, &target_nid, context, line);
                }
            }
        }

        // The function branch does NOT recurse into the body: a nested `def` or
        // `class` gets no node of its own. Its calls are reached later, by
        // `walk_calls` over this body.
        if let Some(body) = find_body(node) {
            ctx.function_bodies.push((func_nid.clone(), body));
        }
        return Ok(());
    }

    // ── decorated_definition ────────────────────────────────────────────────
    // A transparent wrapper: the default recurse below would clear
    // parent_class_nid and emit the inner method with a class-unqualified id,
    // diverging from the class-qualified id the rationale walker uses for the
    // same method's docstring and orphaning the docstring node (#1050).
    if t == "decorated_definition" {
        // Emit the same shape TS/JS emits in `_ts_emit_decorator_edges`: a
        // `references` edge (context="decorator") from the decorated symbol to
        // each decorator, so `affected <decorator>` reports the functions it
        // wraps (#2154). The owner ids mirror the definition branches verbatim,
        // so the edge lands on the node the walk is about to create.
        let inner = node.child_by_field_name("definition");
        let inner_name = match inner {
            Some(i) => match i.child_by_field_name("name") {
                Some(n) => Some(ctx.text(n)?.to_string()),
                None => None,
            },
            None => None,
        };
        if let (Some(inner), Some(inner_name)) = (inner, inner_name) {
            // A name that normalizes to nothing is skipped by the definition
            // branches (#1899), so an edge to it would dangle.
            if !inner_name.is_empty() && ctx.normalizes_to_something(&inner_name)? {
                let owner_nid = if inner.kind() == "class_definition" {
                    ctx.mkid(&[&ctx.stem.clone(), "", &inner_name])?
                } else if let Some(p) = parent_class_nid {
                    ctx.mkid(&[p, &inner_name])?
                } else {
                    ctx.mkid(&[&ctx.stem.clone(), &inner_name])?
                };
                for child in children(node) {
                    if child.kind() != "decorator" {
                        continue;
                    }
                    let Some(deco_name) = helpers::decorator_name(ctx, child)? else {
                        continue;
                    };
                    // Builtin/stdlib decorators are ambient vocabulary: no stub
                    // nodes, no false rewires onto same-named local definitions.
                    if deco_name.is_empty() || DECORATOR_NOISE.contains(&deco_name.as_str()) {
                        continue;
                    }
                    let deco_line = line_of(child);
                    let target = ctx.ensure_named_node(&deco_name, deco_line)?;
                    if target != owner_nid {
                        ctx.add_edge_ctx(&owner_nid, &target, "references", deco_line, "decorator");
                    }
                }
            }
        }
        for child in children(node) {
            walk(ctx, child, parent_class_nid)?;
        }
        return Ok(());
    }

    // ── ERROR ───────────────────────────────────────────────────────────────
    // Unreachable while the walker defers on `root.has_error()` -- a tree with no
    // error cannot contain an ERROR node. Written out so the transliteration
    // stays complete: if that deferral is ever narrowed, the enclosing class
    // linkage must survive a parse artifact (#2551) rather than being dropped by
    // the default recurse.
    if t == "ERROR" {
        for child in children(node) {
            walk(ctx, child, parent_class_nid)?;
        }
        return Ok(());
    }

    // ── Default: recurse, dropping the class link ───────────────────────────
    for child in children(node) {
        walk(ctx, child, None)?;
    }
    Ok(())
}

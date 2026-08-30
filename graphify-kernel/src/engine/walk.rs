//! The shared declaration walk: `_extract_generic`'s inner `walk`, with every
//! `_is_<lang>` guard replaced by a [`LangHooks`] call at the same position.
//!
//! Branch ORDER is Python's and is load-bearing: `walk` is a flat if/elif chain
//! and several node kinds match more than one arm. A `field_declaration` inside
//! a class body is claimed by `before_function` BEFORE the default recurse; a
//! `decorated_definition` is a transparent wrapper that must keep
//! `parent_class_nid`, while the default recurse deliberately DROPS it.

use tree_sitter::Node;

use super::{has, Ctx, Handled, R};
use crate::js::ast::children;

/// `_find_body(node, config)`: the `body_field`, else the first child whose kind
/// is in `body_fallback_child_types`.
fn find_body<'tree>(ctx: &Ctx<'_, 'tree>, node: Node<'tree>) -> Option<Node<'tree>> {
    if let Some(b) = node.child_by_field_name(ctx.cfg.body_field) {
        return Some(b);
    }
    children(node)
        .into_iter()
        .find(|c| has(ctx.cfg.body_fallback_child_types, c.kind()))
}

/// The `name_field`, else the first child in `name_fallback_child_types`.
fn find_name<'tree>(ctx: &Ctx<'_, 'tree>, node: Node<'tree>) -> Option<Node<'tree>> {
    if let Some(n) = node.child_by_field_name(ctx.cfg.name_field) {
        return Some(n);
    }
    children(node)
        .into_iter()
        .find(|c| has(ctx.cfg.name_fallback_child_types, c.kind()))
}

pub fn walk<'tree>(
    ctx: &mut Ctx<'_, 'tree>,
    node: Node<'tree>,
    parent_class_nid: Option<&str>,
) -> R<()> {
    let t = node.kind();
    let hooks = ctx.cfg.hooks;

    // ── import_types ────────────────────────────────────────────────────────
    if has(ctx.cfg.import_types, t) {
        hooks.import_handler(ctx, node)?;
        // The `export_statement` fall-through below the handler is JS-only, and
        // `js/` is not driven by this engine, so this arm always returns.
        return Ok(());
    }

    // ── class_types ─────────────────────────────────────────────────────────
    if has(ctx.cfg.class_types, t) {
        let name_node = match find_name(ctx, node) {
            Some(n) => n,
            None => return Ok(()),
        };
        let class_name = ctx.text(name_node)?.to_string();
        // `_make_id(stem, ".".join(namespace_stack), class_name)`. Outside C# the
        // middle part is "" and `make_id` drops it.
        let class_nid = ctx.mkid(&[&ctx.stem.clone(), &ctx.ns(), &class_name])?;
        let line = node.start_position().row + 1;
        let metadata = hooks.class_metadata(ctx, node, parent_class_nid)?;
        ctx.add_node_meta(&class_nid, &class_name, line, metadata);
        // A class is callable (via its constructor), but ONLY via it (#2137).
        ctx.callable_def_nids.insert(class_nid.clone());
        ctx.callable_class_nids.insert(class_nid.clone());
        // A nested type is contained by its ENCLOSING type, not the file
        // (#2040). The `!= class_nid` guard avoids a self-loop when same-name
        // nesting collides ids, since class ids omit the enclosing type name.
        match parent_class_nid {
            Some(p) if p != class_nid => ctx.add_edge(p, &class_nid, "contains", line),
            _ => {
                let f = ctx.file_nid.clone();
                ctx.add_edge(&f, &class_nid, "contains", line);
            }
        }

        hooks.on_class(ctx, node, &class_nid, &class_name, line)?;

        if let Some(body) = find_body(ctx, node) {
            for child in children(body) {
                walk(ctx, child, Some(&class_nid))?;
            }
        }
        return Ok(());
    }

    // ── declarations between the class and function branches ────────────────
    if hooks.before_function(ctx, node, parent_class_nid)? == Handled::Yes {
        return Ok(());
    }

    // ── function_types ──────────────────────────────────────────────────────
    if has(ctx.cfg.function_types, t) {
        // Swift's `deinit` / `subscript` have no name field and are resolved
        // before the generic fallback; a language without them never matches.
        let func_name: Option<String> = match t {
            "deinit_declaration" => Some("deinit".to_string()),
            "subscript_declaration" => Some("subscript".to_string()),
            // C/C++ style: the name is the innermost identifier of the
            // `declarator`, not a `name` field. An elif in the Python, so it
            // REPLACES the name-field lookup rather than falling back to it --
            // a declaration with no declarator yields None and is skipped.
            _ if ctx.cfg.resolve_function_name.is_some() => {
                let resolve = ctx.cfg.resolve_function_name.unwrap();
                match node.child_by_field_name("declarator") {
                    Some(d) => resolve(ctx, d)?,
                    None => None,
                }
            }
            _ => match find_name(ctx, node) {
                Some(n) => Some(ctx.text(n)?.to_string()),
                None => None,
            },
        };
        let func_name = match func_name {
            Some(n) if !n.is_empty() => n,
            _ => return Ok(()),
        };
        // `sanitize_symbol_name_fn` is unset for every language on this engine
        // so far; when one needs it, it becomes a hook and this comment goes.
        if !ctx.normalizes_to_something(&func_name)? {
            return Ok(());
        }

        let line = node.start_position().row + 1;
        let parens = ctx.cfg.function_label_parens;
        let func_nid = match parent_class_nid {
            Some(p) => {
                let nid = ctx.mkid(&[p, &func_name])?;
                let label = if parens {
                    format!(".{func_name}()")
                } else {
                    format!(".{func_name}")
                };
                ctx.add_node(&nid, &label, line);
                ctx.add_edge(p, &nid, "method", line);
                nid
            }
            None => {
                let nid = ctx.mkid(&[&ctx.stem.clone(), &func_name])?;
                let label = if parens {
                    format!("{func_name}()")
                } else {
                    func_name.clone()
                };
                ctx.add_node(&nid, &label, line);
                let f = ctx.file_nid.clone();
                ctx.add_edge(&f, &nid, "contains", line);
                nid
            }
        };
        ctx.callable_def_nids.insert(func_nid.clone());

        hooks.on_function(ctx, node, &func_nid, &func_name, line, parent_class_nid)?;

        if let Some(body) = find_body(ctx, node) {
            if let Some(p) = parent_class_nid {
                ctx.method_scopes
                    .insert((body.start_byte(), body.end_byte()), (node, p.to_string()));
            }
            ctx.function_bodies.push((func_nid.clone(), body));
        }
        return Ok(());
    }

    // ── the trailing per-language extra walk ────────────────────────────────
    if hooks.extra_walk(ctx, node, parent_class_nid)? == Handled::Yes {
        return Ok(());
    }

    // `decorated_definition` is Python-only and `t == "ERROR"` is unreachable:
    // every walker on this engine defers the whole file on `root.has_error()`,
    // which is true when any descendant is ERROR or MISSING.

    // ── Default: recurse, DROPPING parent_class_nid ─────────────────────────
    // An unknown wrapper usually IS a scope boundary, so the enclosing class
    // linkage is deliberately not threaded through it.
    for child in children(node) {
        walk(ctx, child, None)?;
    }
    Ok(())
}

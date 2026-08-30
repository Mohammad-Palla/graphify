//! The declaration walk: the Java-live slice of `engine.py::_extract_generic`'s
//! inner `walk`, plus `_java_extra_walk`.
//!
//! Branch order matters and is Python's, not a tidied one: `walk` is a flat
//! if/elif chain and several node kinds match more than one arm. In particular a
//! `field_declaration` inside a class body is claimed by the Java field branch
//! BEFORE the default recurse, and an `enum_constant` by the extra walk.
//!
//! The `_JAVA_CONFIG` sets are `match` arms rather than runtime sets because for
//! one language they are constants. The sets they come from:
//!
//! ```text
//! class_types     annotation_type_declaration, class_declaration,
//!                 enum_declaration, interface_declaration, record_declaration
//! function_types  constructor_declaration, method_declaration
//! import_types    import_declaration
//! ```

use tree_sitter::Node;

use super::helpers::{self, Role};
use super::imports;
use super::{Ctx, R};
use crate::js::ast::children;

fn is_class_type(kind: &str) -> bool {
    matches!(
        kind,
        "annotation_type_declaration"
            | "class_declaration"
            | "enum_declaration"
            | "interface_declaration"
            | "record_declaration"
    )
}

fn is_function_type(kind: &str) -> bool {
    matches!(kind, "constructor_declaration" | "method_declaration")
}

/// `_emit_java_parent`: resolve a base type to a node, minting a SOURCELESS stub
/// when it is not already known, then link it.
///
/// Note this is NOT `ensure_named_node`: the stub it mints carries no
/// `origin_file` key, and the scoped id it probes is `_make_id(stem, base)`
/// rather than `_make_id(stem, "", base)`. Those produce the same string here
/// (`make_id` drops empty parts) but the node SHAPES differ, so the two emitters
/// are kept apart.
fn emit_java_parent(
    ctx: &mut Ctx,
    class_nid: &str,
    base_name: &str,
    rel: &'static str,
    at_line: usize,
) -> R<()> {
    if base_name.is_empty() {
        return Ok(());
    }
    let base_nid = ctx.ensure_parent_node(base_name)?;
    ctx.add_edge(class_nid, &base_nid, rel, at_line);
    Ok(())
}

/// `_emit_java_parent_type`: the first `type` role becomes the parent link, every
/// `generic_arg` becomes a `references` edge.
fn emit_java_parent_type(
    ctx: &mut Ctx,
    class_nid: &str,
    type_node: Option<Node>,
    rel: &'static str,
    at_line: usize,
) -> R<()> {
    let mut refs = Vec::new();
    helpers::collect_type_refs(ctx, type_node, false, &mut refs, None, false)?;
    let refs: Vec<(String, Role)> = refs.into_iter().map(|(n, r)| (n.to_string(), r)).collect();
    let mut parent_emitted = false;
    for (ref_name, role) in refs {
        if role == Role::Type && !parent_emitted {
            emit_java_parent(ctx, class_nid, &ref_name, rel, at_line)?;
            parent_emitted = true;
        } else if role == Role::GenericArg {
            let target = ctx.ensure_named_node(&ref_name, at_line)?;
            if target != class_nid {
                ctx.add_edge_ctx(class_nid, &target, "references", at_line, "generic_arg");
            }
        }
    }
    Ok(())
}

/// Emit `references` edges for every type in `type_node`, mapping the `type`
/// role to `type_ctx` and `generic_arg` to `"generic_arg"`. Shared by the field,
/// parameter, return-type and record-component branches, which differ only in
/// that one context string.
fn emit_type_refs(
    ctx: &mut Ctx,
    owner_nid: &str,
    type_node: Option<Node>,
    type_ctx: &'static str,
    line: usize,
    preserve_qualified: bool,
) -> R<()> {
    let mut refs = Vec::new();
    helpers::collect_type_refs(ctx, type_node, false, &mut refs, None, preserve_qualified)?;
    let refs: Vec<(String, Role)> = refs.into_iter().map(|(n, r)| (n.to_string(), r)).collect();
    for (ref_name, role) in refs {
        let c = if role == Role::GenericArg { "generic_arg" } else { type_ctx };
        let target = ctx.ensure_named_node(&ref_name, line)?;
        if target != owner_nid {
            ctx.add_edge_ctx(owner_nid, &target, "references", line, c);
        }
    }
    Ok(())
}

/// The annotation block shared by the class and function branches.
///
/// The two differ in ONE way, and it is not obvious: at class level the
/// dotted-name substitution is `if "." in anno_raw and _is_java: anno_name =
/// anno_raw`, while at function level it is written inline as
/// `anno_raw if "." in anno_raw else anno_name`. Both reduce to the same choice
/// for Java, so one helper serves both -- stated here because the asymmetry in
/// the Python invites "fixing" one to match the other.
fn emit_annotations(ctx: &mut Ctx, owner_nid: &str, decl: Node, line: usize) -> R<()> {
    let mut targets: std::collections::HashSet<String> = std::collections::HashSet::new();
    let names: Vec<(String, String)> = helpers::annotation_names(ctx, decl)?
        .into_iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
    for (anno_name, anno_raw) in names {
        let chosen = if anno_raw.contains('.') { &anno_raw } else { &anno_name };
        let target = ctx.ensure_named_node(chosen, line)?;
        if target != owner_nid && targets.insert(target.clone()) {
            ctx.add_edge_ctx(owner_nid, &target, "references", line, "attribute");
        }
    }
    let lits: Vec<String> = helpers::annotation_class_literal_refs(ctx, decl)?
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    for ref_name in lits {
        let target = ctx.ensure_named_node(&ref_name, line)?;
        if target != owner_nid && targets.insert(target.clone()) {
            ctx.add_edge_ctx(owner_nid, &target, "references", line, "attribute");
        }
    }
    Ok(())
}

pub fn walk<'tree>(
    ctx: &mut Ctx<'_, 'tree>,
    node: Node<'tree>,
    parent_class_nid: Option<&str>,
) -> R<()> {
    let t = node.kind();

    // ── import_types ────────────────────────────────────────────────────────
    // `export_statement` is JS-only, so the re-export fall-through below the
    // handler in Python is unreachable for Java: this arm always returns.
    if t == "import_declaration" {
        imports::import_java(ctx, node)?;
        return Ok(());
    }

    // ── class_types ─────────────────────────────────────────────────────────
    if is_class_type(t) {
        let name_node = match node.child_by_field_name("name") {
            Some(n) => n,
            // `name_fallback_child_types` is empty for Java, so a missing name
            // field ends the branch.
            None => return Ok(()),
        };
        let class_name = ctx.text(name_node)?.to_string();
        let class_nid = ctx.mkid(&[&ctx.stem.clone(), "", &class_name])?;
        let line = node.start_position().row + 1;
        ctx.add_node(&class_nid, &class_name, line);
        ctx.callable_def_nids.insert(class_nid.clone());
        ctx.callable_class_nids.insert(class_nid.clone());
        match parent_class_nid {
            Some(p) if p != class_nid => ctx.add_edge(p, &class_nid, "contains", line),
            _ => {
                let f = ctx.file_nid.clone();
                ctx.add_edge(&f, &class_nid, "contains", line);
            }
        }

        // extends
        if let Some(sup) = node.child_by_field_name("superclass") {
            if let Some(sub) = children(sup).into_iter().find(|c| c.is_named()) {
                emit_java_parent_type(ctx, &class_nid, Some(sub), "inherits", line)?;
            }
        }
        // implements
        if let Some(ifs) = node.child_by_field_name("interfaces") {
            for sub in children(ifs) {
                if sub.kind() != "type_list" {
                    continue;
                }
                for tid in children(sub) {
                    if tid.is_named() {
                        emit_java_parent_type(ctx, &class_nid, Some(tid), "implements", line)?;
                    }
                }
            }
        }
        // interface extends
        if t == "interface_declaration" {
            for child in children(node) {
                if child.kind() != "extends_interfaces" {
                    continue;
                }
                for sub in children(child) {
                    if sub.kind() != "type_list" {
                        continue;
                    }
                    for tid in children(sub) {
                        if tid.is_named() {
                            emit_java_parent_type(ctx, &class_nid, Some(tid), "inherits", line)?;
                        }
                    }
                }
            }
        }

        emit_annotations(ctx, &class_nid, node, line)?;

        // record components
        if t == "record_declaration" {
            if let Some(components) = node.child_by_field_name("parameters") {
                for component in children(components) {
                    let type_node = match component.kind() {
                        "formal_parameter" => component.child_by_field_name("type"),
                        "spread_parameter" => children(component).into_iter().find(|c| {
                            c.is_named() && !matches!(c.kind(), "modifiers" | "variable_declarator")
                        }),
                        _ => continue,
                    };
                    let component_line = component.start_position().row + 1;
                    emit_type_refs(ctx, &class_nid, type_node, "field", component_line, false)?;
                }
            }
        }

        // `_find_body(node, config)` is `child_by_field_name("body")`:
        // `body_fallback_child_types` is empty for Java. `ruby_segments` is
        // empty, so the namespace push/pop around this loop is a no-op.
        if let Some(body) = node.child_by_field_name("body") {
            for child in children(body) {
                walk(ctx, child, Some(&class_nid))?;
            }
        }
        return Ok(());
    }

    // ── Java: field_declaration ─────────────────────────────────────────────
    if t == "field_declaration" {
        if let Some(parent) = parent_class_nid {
            let parent = parent.to_string();
            if let Some(type_node) = node.child_by_field_name("type") {
                if let Some(receiver_type) = helpers::receiver_type_name(ctx, Some(type_node))? {
                    let receiver_type = receiver_type.to_string();
                    let names: Vec<String> = helpers::declarator_names(ctx, node)?
                        .into_iter()
                        .map(|s| s.to_string())
                        .collect();
                    let fields = ctx.java_field_types.entry(parent.clone()).or_default();
                    for field_name in names {
                        fields.insert(field_name, receiver_type.clone());
                    }
                }
                let line = node.start_position().row + 1;
                emit_type_refs(ctx, &parent, Some(type_node), "field", line, false)?;
            }
            // Python returns from inside `if type_node is not None`, so a field
            // with no type field falls through to the branches below. For Java
            // that means the default recurse -- reproduced by only returning here.
            if node.child_by_field_name("type").is_some() {
                return Ok(());
            }
        }
    }

    // ── Java: annotation_type_element_declaration ───────────────────────────
    if t == "annotation_type_element_declaration" {
        if let Some(parent) = parent_class_nid {
            let parent = parent.to_string();
            let line = node.start_position().row + 1;
            // `preserve_qualified=True` here and nowhere else in the Java path.
            emit_type_refs(
                ctx,
                &parent,
                node.child_by_field_name("type"),
                "return_type",
                line,
                true,
            )?;
            return Ok(());
        }
    }

    // ── function_types ──────────────────────────────────────────────────────
    if is_function_type(t) {
        // `resolve_function_name_fn` and `name_fallback_child_types` are unset
        // for Java, so the name is exactly the `name` field.
        let func_name = match node.child_by_field_name("name") {
            Some(n) => ctx.text(n)?.to_string(),
            None => return Ok(()),
        };
        if func_name.is_empty() {
            return Ok(());
        }
        // `sanitize_symbol_name_fn` is None for Java, so sanitized == raw.
        // #1899: a name that normalizes to nothing would collapse the id onto
        // its path-derived prefix.
        if !ctx.normalizes_to_something(&func_name)? {
            return Ok(());
        }
        let line = node.start_position().row + 1;
        let func_nid = match parent_class_nid {
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
        ctx.callable_def_nids.insert(func_nid.clone());

        if let Some(params_node) = node.child_by_field_name("parameters") {
            for p in children(params_node) {
                if p.kind() != "formal_parameter" {
                    continue;
                }
                emit_type_refs(
                    ctx,
                    &func_nid,
                    p.child_by_field_name("type"),
                    "parameter_type",
                    line,
                    false,
                )?;
            }
        }
        // A `method_declaration`'s `type` field is its RETURN type. A
        // `constructor_declaration` has no `type` field, so this is a no-op there.
        if let Some(return_node) = node.child_by_field_name("type") {
            emit_type_refs(ctx, &func_nid, Some(return_node), "return_type", line, false)?;
        }
        emit_annotations(ctx, &func_nid, node, line)?;

        if let Some(body) = node.child_by_field_name("body") {
            if let Some(p) = parent_class_nid {
                ctx.java_method_scopes
                    .insert((body.start_byte(), body.end_byte()), (node, p.to_string()));
            }
            ctx.function_bodies.push((func_nid.clone(), body));
        }
        return Ok(());
    }

    // ── `_java_extra_walk`: enum_constant ───────────────────────────────────
    if t == "enum_constant" {
        if let Some(parent) = parent_class_nid {
            let parent = parent.to_string();
            let name_node = match node.child_by_field_name("name") {
                Some(n) => n,
                // Python returns True (handled) even with no name, so the node is
                // consumed rather than recursed into.
                None => return Ok(()),
            };
            let const_name = ctx.text(name_node)?.to_string();
            let line = node.start_position().row + 1;
            let const_nid = ctx.mkid(&[&parent, &const_name])?;
            ctx.add_node(&const_nid, &const_name, line);
            ctx.add_edge(&parent, &const_nid, "case_of", line);
            // Anonymous-body constants (`MONDAY { void greet(){} }`).
            for child in children(node) {
                if child.kind() == "class_body" {
                    for member in children(child) {
                        walk(ctx, member, Some(&const_nid))?;
                    }
                }
            }
            return Ok(());
        }
    }

    // `t == "ERROR"` is unreachable: `extract` defers the whole file on
    // `root.has_error()`, which is true when any descendant is ERROR or MISSING.

    // ── Default: recurse, DROPPING parent_class_nid ─────────────────────────
    for child in children(node) {
        walk(ctx, child, None)?;
    }
    Ok(())
}

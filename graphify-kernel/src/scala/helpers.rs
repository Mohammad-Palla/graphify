//! `_scala_collect_type_refs`, reproduced.

use tree_sitter::Node;

use crate::engine::{Ctx, R};
use crate::js::ast::children;

/// Walk a Scala type expression, appending `(name, role)`.
///
/// `role` is `"generic_arg"` once the walk is inside a `type_arguments` list and
/// `"type"` otherwise -- the caller turns that into the edge's `context`.
///
/// Three shapes, matching the Python branch for branch:
/// * `type_identifier` -- emit and STOP.
/// * `generic_type` -- emit the base (from the `type` field, else the first
///   `type_identifier` child), then recurse into `type_arguments` with
///   `generic = true`.
/// * the wrapper set -- recurse into every NAMED child, carrying `generic`
///   through unchanged.
///
/// Anything else is dropped, which is why a `self_type`'s binder identifier
/// contributes nothing: `identifier` is not `type_identifier`.
pub fn collect_type_refs(
    ctx: &Ctx,
    node: Option<Node>,
    generic: bool,
    out: &mut Vec<(String, &'static str)>,
) -> R<()> {
    let node = match node {
        Some(n) => n,
        None => return Ok(()),
    };
    let role = if generic { "generic_arg" } else { "type" };
    match node.kind() {
        "type_identifier" => {
            let text = ctx.text(node)?;
            if !text.is_empty() {
                out.push((text.to_string(), role));
            }
        }
        "generic_type" => {
            let base = node
                .child_by_field_name("type")
                .or_else(|| children(node).into_iter().find(|c| c.kind() == "type_identifier"));
            // The Python re-checks the kind even when it came from the field:
            // `if base is not None and base.type == "type_identifier"`.
            if let Some(b) = base {
                if b.kind() == "type_identifier" {
                    let text = ctx.text(b)?;
                    if !text.is_empty() {
                        out.push((text.to_string(), role));
                    }
                }
            }
            for c in children(node) {
                if c.kind() != "type_arguments" {
                    continue;
                }
                for arg in children(c) {
                    if arg.is_named() {
                        collect_type_refs(ctx, Some(arg), true, out)?;
                    }
                }
            }
        }
        "compound_type" | "infix_type" | "function_type" | "tuple_type" | "annotated_type"
        | "projected_type" => {
            for c in children(node) {
                if c.is_named() {
                    collect_type_refs(ctx, Some(c), generic, out)?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// `"generic_arg" if role == "generic_arg" else <otherwise>`, the phrasing every
/// Scala call site uses.
pub fn ctx_for(role: &str, otherwise: &'static str) -> &'static str {
    if role == "generic_arg" {
        "generic_arg"
    } else {
        otherwise
    }
}

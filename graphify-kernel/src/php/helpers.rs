//! The `_php_*` helpers from `engine.py`.

use tree_sitter::Node;

use crate::engine::{Ctx, R};
use crate::js::ast::children;

/// The type-expression node kinds a property / parameter declaration may carry.
pub const TYPE_NODES: &[&str] = &[
    "named_type",
    "primitive_type",
    "nullable_type",
    "union_type",
    "intersection_type",
    "optional_type",
];

/// `_php_name_text`: the UNQUALIFIED tail of a `name` / `qualified_name`.
///
/// PHP namespaces use a backslash separator, so `\App\Models\User` reduces to
/// `User`. Returns None for the empty tail, which is what the Python's `or None`
/// does.
pub fn name_text(ctx: &Ctx, node: Option<Node>) -> R<Option<String>> {
    let node = match node {
        Some(n) => n,
        None => return Ok(None),
    };
    let raw = ctx.text(node)?;
    let tail = raw.rsplit('\\').next().unwrap_or(raw);
    Ok(if tail.is_empty() { None } else { Some(tail.to_string()) })
}

/// `_php_collect_type_refs`: `(name, is_generic_arg)` per named type.
pub fn collect_type_refs(
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
        "named_type" => {
            // The FIRST name-ish child only, and then return -- a `named_type`
            // wrapping more than one name is not walked further.
            for c in children(node) {
                if matches!(c.kind(), "name" | "qualified_name") {
                    if let Some(text) = name_text(ctx, Some(c))? {
                        out.push((text, generic));
                    }
                    return Ok(());
                }
            }
            return Ok(());
        }
        "name" | "qualified_name" => {
            if let Some(text) = name_text(ctx, Some(node))? {
                out.push((text, generic));
            }
            return Ok(());
        }
        "nullable_type" | "union_type" | "intersection_type" | "optional_type" => {
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

/// `_php_class_const_scope`: the class named by a `Foo::BAR` / `Foo::$bar`.
///
/// Note this returns the RAW text, qualifier included -- unlike `name_text`.
/// The callers look the result up in `label_to_nid_ci`, and a qualified name
/// simply misses.
pub fn class_const_scope(ctx: &Ctx, n: Node) -> R<Option<String>> {
    let mut scope = n.child_by_field_name("scope");
    if scope.is_none() {
        scope = children(n)
            .into_iter()
            .find(|c| c.is_named() && matches!(c.kind(), "name" | "qualified_name" | "identifier"));
    }
    match scope {
        Some(s) => Ok(Some(ctx.text(s)?.to_string())),
        None => Ok(None),
    }
}

/// `_php_method_return_type_node`: the type node sitting AFTER
/// `formal_parameters`. PHP has no `return_type` field, so it is positional.
pub fn method_return_type_node<'tree>(method_node: Node<'tree>) -> Option<Node<'tree>> {
    let mut saw_params = false;
    for c in children(method_node) {
        if c.kind() == "formal_parameters" {
            saw_params = true;
            continue;
        }
        if saw_params && c.is_named() && c.kind() != "compound_statement" {
            if TYPE_NODES.contains(&c.kind()) {
                return Some(c);
            }
        }
    }
    None
}

/// The first `string_content` inside a call's first `argument`, as
/// `config('a.b')` needs. None when the first argument is not a literal string.
pub fn first_string_argument(ctx: &Ctx, node: Node) -> R<Option<String>> {
    let args_node = match node.child_by_field_name("arguments") {
        Some(a) => a,
        None => return Ok(None),
    };
    for arg in children(args_node) {
        if arg.kind() != "argument" {
            continue;
        }
        for inner in children(arg) {
            if inner.kind() != "string" {
                continue;
            }
            for sc in children(inner) {
                if sc.kind() == "string_content" {
                    return Ok(Some(ctx.text(sc)?.to_string()));
                }
            }
            break;
        }
        // The Python's `if first_key: break` -- a non-string first argument
        // keeps scanning the remaining arguments.
    }
    Ok(None)
}

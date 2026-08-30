//! The `_kotlin_*` helpers from `engine.py`.
//!
//! Several of them accept TWO node kinds for the same thing (`identifier` and
//! `simple_identifier`, `import` and `import_header`). That is not defensive
//! coding: PyPI's `tree_sitter_kotlin` and the older JVM-binding forks name
//! these differently, and the Python accepts both so the extractor works across
//! grammar generations (#2526). The native side must accept exactly the same
//! pair or it would silently handle a different set of files.

use tree_sitter::Node;

use crate::engine::{Ctx, R};
use crate::js::ast::children;

/// A plain identifier, under either grammar generation's name.
pub const IDENTS: &[&str] = &["simple_identifier", "identifier"];

/// `_kotlin_user_type_name`: the head identifier of a `user_type`, generics
/// dropped.
pub fn user_type_name(ctx: &Ctx, node: Option<Node>) -> R<Option<String>> {
    let node = match node {
        Some(n) => n,
        None => return Ok(None),
    };
    for c in children(node) {
        if matches!(c.kind(), "type_identifier" | "identifier") {
            let text = ctx.text(c)?;
            return Ok(if text.is_empty() { None } else { Some(text.to_string()) });
        }
        if c.kind() == "simple_user_type" {
            for sub in children(c) {
                if matches!(sub.kind(), "identifier" | "type_identifier") {
                    let text = ctx.text(sub)?;
                    return Ok(if text.is_empty() { None } else { Some(text.to_string()) });
                }
            }
            return Ok(None);
        }
    }
    Ok(None)
}

fn is_builtin(name: &str) -> bool {
    super::consts::BUILTIN_TYPES.contains(&name) || crate::java::consts::BUILTIN_TYPES.contains(&name)
}

/// `_kotlin_collect_type_refs`: `(name, is_generic_arg)` per referenced type.
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
    if matches!(node.kind(), "integral_literal" | "boolean_literal") {
        return Ok(());
    }
    if node.kind() == "user_type" {
        for c in children(node) {
            if matches!(c.kind(), "identifier" | "type_identifier") {
                let text = ctx.text(c)?;
                if !text.is_empty() && !is_builtin(text) {
                    out.push((text.to_string(), generic));
                }
                break;
            }
            if c.kind() == "simple_user_type" {
                for sub in children(c) {
                    if matches!(sub.kind(), "identifier" | "type_identifier") {
                        let text = ctx.text(sub)?;
                        if !text.is_empty() && !is_builtin(text) {
                            out.push((text.to_string(), generic));
                        }
                        break;
                    }
                }
                break;
            }
        }
        for c in children(node) {
            if c.kind() != "type_arguments" {
                continue;
            }
            for arg in children(c) {
                if arg.kind() == "type_projection" {
                    for sub in children(arg) {
                        if sub.is_named() {
                            collect_type_refs(ctx, Some(sub), true, out)?;
                        }
                    }
                } else if arg.is_named() {
                    collect_type_refs(ctx, Some(arg), true, out)?;
                }
            }
        }
        return Ok(());
    }
    // A BARE identifier node emits directly. Missing this arm sent the walk
    // deeper instead: `crossinline transform: suspend (value: T) -> R?` yielded
    // `T` where the Python yields `value`, on 26 files across ktor and
    // kotlinx.coroutines.
    if matches!(node.kind(), "identifier" | "type_identifier") {
        let text = ctx.text(node)?;
        if !text.is_empty() && !is_builtin(text) {
            out.push((text.to_string(), generic));
        }
        return Ok(());
    }
    // Listed explicitly in the Python, and equivalent to the generic tail below
    // -- kept as its own arm so the two stay in step if either changes.
    if matches!(node.kind(), "nullable_type" | "parenthesized_type" | "type_reference") {
        for c in children(node) {
            if c.is_named() {
                collect_type_refs(ctx, Some(c), generic, out)?;
            }
        }
        return Ok(());
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

/// `_kotlin_property_type_node`.
pub fn property_type_node<'tree>(property_node: Node<'tree>) -> Option<Node<'tree>> {
    for c in children(property_node) {
        if c.kind() == "variable_declaration" {
            for sub in children(c) {
                if matches!(sub.kind(), "user_type" | "nullable_type" | "type_reference") {
                    return Some(sub);
                }
            }
        }
        if matches!(c.kind(), "user_type" | "nullable_type" | "type_reference") {
            return Some(c);
        }
    }
    None
}

/// `_kotlin_function_return_type_node`: the type after the `:` that FOLLOWS the
/// parameter list. Positional -- there is no `return_type` field.
pub fn function_return_type_node<'tree>(func_node: Node<'tree>) -> Option<Node<'tree>> {
    let mut saw_params = false;
    let mut saw_colon = false;
    for c in children(func_node) {
        if c.kind() == "function_value_parameters" {
            saw_params = true;
            continue;
        }
        if saw_params && c.kind() == ":" {
            saw_colon = true;
            continue;
        }
        if saw_colon && c.is_named() {
            return Some(c);
        }
    }
    None
}

/// `_kotlin_package_name`: the file's dotted package FQN, or None.
pub fn package_name(ctx: &Ctx, root: Node) -> R<Option<String>> {
    for child in children(root) {
        if child.kind() != "package_header" {
            continue;
        }
        for c in children(child) {
            if matches!(c.kind(), "qualified_identifier" | "identifier") {
                let pkg = ctx.text(c)?.trim();
                return Ok(if pkg.is_empty() { None } else { Some(pkg.to_string()) });
            }
        }
        return Ok(None);
    }
    Ok(None)
}

/// `_kotlin_nav_identifier_segments`: flatten a navigation chain into its dotted
/// segments, or None when ANY segment is not a plain identifier.
///
/// None matters: a receiver that is an expression, a call, `this` or a string
/// literal must never read as a qualified name (#2550). Older grammars with a
/// different navigation shape also bail here, preserving their behaviour.
pub fn nav_identifier_segments(ctx: &Ctx, nav: Node) -> R<Option<Vec<String>>> {
    let mut segments: Vec<String> = Vec::new();
    let mut node = Some(nav);
    while let Some(n) = node {
        if n.kind() != "navigation_expression" {
            break;
        }
        let named: Vec<Node> = children(n).into_iter().filter(|c| c.is_named()).collect();
        // Grammar 1.1.0's shape is `<receiver> "." <identifier>`, the dot unnamed.
        if named.len() != 2 {
            return Ok(None);
        }
        let (head, tail) = (named[0], named[1]);
        if !IDENTS.contains(&tail.kind()) {
            return Ok(None);
        }
        segments.push(ctx.text(tail)?.to_string());
        node = Some(head);
    }
    let node = match node {
        Some(n) if IDENTS.contains(&n.kind()) => n,
        _ => return Ok(None),
    };
    segments.push(ctx.text(node)?.to_string());
    segments.reverse();
    Ok(Some(segments))
}

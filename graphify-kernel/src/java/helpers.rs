//! The `_java_*` helpers from `engine.py`, transliterated.
//!
//! Every function here mirrors one Python function of the same name, and the
//! comments name it. Where the Rust reads differently from the Python the reason
//! is stated -- those are the places a future reader would otherwise "fix" back
//! into a divergence.

use std::collections::HashSet;

use tree_sitter::Node;

use super::consts::{BUILTIN_TYPES, TYPE_PARAMETER_SCOPES};
use crate::engine::{Ctx, R};
use crate::js::ast::children;

/// `role` in Python's `(name, role)` tuples. An enum rather than a `&str`
/// because only these two values are ever produced and every consumer branches
/// on which one it is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    Type,
    GenericArg,
}

impl Role {
    fn of(generic: bool) -> Role {
        if generic {
            Role::GenericArg
        } else {
            Role::Type
        }
    }
}

// These two ARE sorted -- `consts.rs` is generated with `sorted()` -- so a
// binary search is sound. It is not sound by inspection though, and the same
// assumption about `BUILTIN_GLOBALS` (which is grouped by language, not sorted)
// cost 64 DIVERGENT files, so the invariant is asserted rather than trusted.
fn is_builtin(name: &str) -> bool {
    debug_assert!(BUILTIN_TYPES.windows(2).all(|w| w[0] < w[1]));
    BUILTIN_TYPES.binary_search(&name).is_ok()
}

fn is_type_param_scope(kind: &str) -> bool {
    debug_assert!(TYPE_PARAMETER_SCOPES.windows(2).all(|w| w[0] < w[1]));
    TYPE_PARAMETER_SCOPES.binary_search(&kind).is_ok()
}

#[cfg(test)]
mod sorted_tests {
    use super::{BUILTIN_TYPES, TYPE_PARAMETER_SCOPES};

    #[test]
    fn generated_constant_arrays_are_sorted() {
        assert!(BUILTIN_TYPES.windows(2).all(|w| w[0] < w[1]));
        assert!(TYPE_PARAMETER_SCOPES.windows(2).all(|w| w[0] < w[1]));
    }
}

/// `_java_type_parameters_in_scope`: the type-parameter names visible from
/// `node`, by walking `parent` to the root.
pub fn type_parameters_in_scope<'a>(ctx: &Ctx<'a, '_>, node: Node) -> R<HashSet<&'a str>> {
    let mut names = HashSet::new();
    let mut scope = Some(node);
    while let Some(s) = scope {
        if is_type_param_scope(s.kind()) {
            if let Some(params) = s.child_by_field_name("type_parameters") {
                for param in children(params) {
                    if param.kind() != "type_parameter" {
                        continue;
                    }
                    if let Some(n) = children(param).into_iter().find(|c| c.kind() == "type_identifier") {
                        names.insert(ctx.text(n)?);
                    }
                }
            }
        }
        scope = s.parent();
    }
    Ok(names)
}

/// `raw.rsplit(".", 1)[-1]` -- the segment after the last dot, or the whole
/// string when there is none. Python's `rsplit(sep, 1)[-1]` never fails and
/// never returns the pre-dot part, which `split_once` would.
fn last_segment(raw: &str) -> &str {
    match raw.rsplit_once('.') {
        Some((_, tail)) => tail,
        None => raw,
    }
}

/// `_java_collect_type_refs`. Appends `(name, role)` in Python's order.
///
/// `skip` is threaded rather than recomputed per level, matching Python: the
/// default is computed ONCE at the top call from the node it was given, so a
/// nested type argument is filtered against the OUTER scope's type parameters,
/// not its own. Recomputing it per level would be the more obvious reading and
/// would quietly change which names survive.
pub fn collect_type_refs<'a>(
    ctx: &Ctx<'a, '_>,
    node: Option<Node>,
    generic: bool,
    out: &mut Vec<(&'a str, Role)>,
    skip: Option<&HashSet<&'a str>>,
    preserve_qualified: bool,
) -> R<()> {
    let node = match node {
        Some(n) => n,
        None => return Ok(()),
    };
    // `if skip is None: skip = _java_type_parameters_in_scope(node, source)`.
    let owned;
    let skip = match skip {
        Some(s) => s,
        None => {
            owned = type_parameters_in_scope(ctx, node)?;
            &owned
        }
    };

    match node.kind() {
        "integral_type" | "floating_point_type" | "boolean_type" | "void_type" => Ok(()),
        "type_identifier" => {
            let name = ctx.text(node)?;
            if !name.is_empty() && !skip.contains(name) && !is_builtin(name) {
                out.push((name, Role::of(generic)));
            }
            Ok(())
        }
        "scoped_type_identifier" => {
            let raw = ctx.text(node)?;
            let simple = last_segment(raw);
            let text = if preserve_qualified { raw } else { simple };
            if !text.is_empty() && !is_builtin(simple) {
                out.push((text, Role::of(generic)));
            }
            Ok(())
        }
        "generic_type" => {
            // Two separate passes over the children, exactly as Python does:
            // the first `break`s after the base type, the second visits every
            // `type_arguments`. Fusing them into one loop would reorder `out`.
            for c in children(node) {
                if matches!(c.kind(), "type_identifier" | "scoped_type_identifier") {
                    let raw = ctx.text(c)?;
                    let simple = last_segment(raw);
                    let scoped = c.kind() == "scoped_type_identifier";
                    let text = if preserve_qualified && scoped { raw } else { simple };
                    if !text.is_empty()
                        && !is_builtin(simple)
                        && (scoped || !skip.contains(simple))
                    {
                        out.push((text, Role::of(generic)));
                    }
                    break;
                }
            }
            for c in children(node) {
                if c.kind() == "type_arguments" {
                    for arg in children(c) {
                        if arg.is_named() {
                            collect_type_refs(ctx, Some(arg), true, out, Some(skip), preserve_qualified)?;
                        }
                    }
                }
            }
            Ok(())
        }
        "array_type" => {
            for c in children(node) {
                if c.is_named() {
                    collect_type_refs(ctx, Some(c), generic, out, Some(skip), preserve_qualified)?;
                }
            }
            Ok(())
        }
        _ => {
            if node.is_named() {
                for c in children(node) {
                    if c.is_named() {
                        collect_type_refs(ctx, Some(c), generic, out, Some(skip), preserve_qualified)?;
                    }
                }
            }
            Ok(())
        }
    }
}

/// `_java_receiver_type_name`.
pub fn receiver_type_name<'a>(ctx: &Ctx<'a, '_>, type_node: Option<Node>) -> R<Option<&'a str>> {
    let type_node = match type_node {
        Some(n) => n,
        None => return Ok(None),
    };
    let name = match type_node.kind() {
        "type_identifier" => ctx.text(type_node)?,
        "scoped_type_identifier" => last_segment(ctx.text(type_node)?),
        "generic_type" => {
            let base = children(type_node)
                .into_iter()
                .find(|c| matches!(c.kind(), "type_identifier" | "scoped_type_identifier"));
            return receiver_type_name(ctx, base);
        }
        _ => return Ok(None),
    };
    if name.is_empty() || is_builtin(name) {
        return Ok(None);
    }
    // Python recomputes the scope set from `type_node` here; it is NOT the
    // caller's `skip`.
    if type_parameters_in_scope(ctx, type_node)?.contains(name) {
        return Ok(None);
    }
    Ok(Some(name))
}

/// `_java_declarator_names`.
pub fn declarator_names<'a>(ctx: &Ctx<'a, '_>, decl: Node) -> R<Vec<&'a str>> {
    let mut names = Vec::new();
    for child in children(decl) {
        if child.kind() != "variable_declarator" {
            continue;
        }
        if let Some(n) = child.child_by_field_name("name") {
            let name = ctx.text(n)?;
            if !name.is_empty() {
                names.push(name);
            }
        }
    }
    Ok(names)
}

/// `_java_annotation_nodes`: the annotations on a declaration's `modifiers`
/// child. Only the FIRST `modifiers` child is read, as in Python.
pub fn annotation_nodes<'tree>(decl: Node<'tree>) -> Vec<Node<'tree>> {
    let modifiers = match children(decl).into_iter().find(|c| c.kind() == "modifiers") {
        Some(m) => m,
        None => return Vec::new(),
    };
    children(modifiers)
        .into_iter()
        .filter(|c| matches!(c.kind(), "marker_annotation" | "annotation"))
        .collect()
}

/// `_java_annotation_names`: `(simple, raw)` pairs.
pub fn annotation_names<'a>(ctx: &Ctx<'a, '_>, decl: Node) -> R<Vec<(&'a str, &'a str)>> {
    let mut names = Vec::new();
    for anno in annotation_nodes(decl) {
        let name_node = anno.child_by_field_name("name").or_else(|| {
            children(anno).into_iter().find(|s| {
                matches!(s.kind(), "identifier" | "scoped_identifier" | "type_identifier")
            })
        });
        if let Some(n) = name_node {
            let raw = ctx.text(n)?;
            let text = last_segment(raw);
            if !text.is_empty() {
                names.push((text, raw));
            }
        }
    }
    Ok(names)
}

/// `_java_annotation_class_literal_refs`.
///
/// The stack is popped from the END and extended in child order, so it visits
/// children in REVERSE. That ordering reaches `out` unchanged, so it is
/// reproduced rather than tidied into a queue.
pub fn annotation_class_literal_refs<'a>(ctx: &Ctx<'a, '_>, decl: Node) -> R<Vec<&'a str>> {
    let mut names = Vec::new();
    for anno in annotation_nodes(decl) {
        let arguments = match anno.child_by_field_name("arguments") {
            Some(a) => a,
            None => continue,
        };
        let mut stack = vec![arguments];
        while let Some(current) = stack.pop() {
            if current.kind() == "class_literal" {
                let type_node = children(current).into_iter().find(|c| c.is_named());
                let mut refs = Vec::new();
                collect_type_refs(ctx, type_node, false, &mut refs, None, true)?;
                names.extend(refs.into_iter().map(|(n, _)| n));
                continue;
            }
            stack.extend(children(current).into_iter().filter(|c| c.is_named()));
        }
    }
    Ok(names)
}

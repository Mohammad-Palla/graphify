//! The `_cpp_*` and `_get_cpp_func_name` helpers from `engine.py`.

use tree_sitter::Node;

use crate::engine::{Ctx, R};
use crate::js::ast::children;

/// `_C_PRIMITIVE_TYPE_NODES`, shared with C.
pub const PRIMITIVE_TYPE_NODES: &[&str] = &[
    "primitive_type",
    "sized_type_specifier",
    "auto",
    "placeholder_type_specifier",
];

/// Wrappers `_cpp_collect_type_refs` descends through. Note `type_descriptor`
/// leads and there is no `ref_type` -- this list is NOT C's, despite the overlap.
const TYPE_WRAPPERS: &[&str] = &[
    "type_descriptor",
    "pointer_declarator",
    "reference_declarator",
    "array_declarator",
    "type_qualifier",
    "abstract_pointer_declarator",
    "abstract_reference_declarator",
    "abstract_array_declarator",
];

/// `_cpp_collect_type_refs`: `(name, is_generic_arg)` per named type.
///
/// Wider than C's: a `qualified_identifier` collapses to its tail (`std::string`
/// -> `string`) and a `template_type` yields its base name plus every argument
/// as a `generic_arg`.
pub fn collect_type_refs(
    ctx: &Ctx,
    node: Option<Node>,
    generic: bool,
    out: &mut Vec<(String, bool)>,
) -> R<()> {
    let node = match node {
        Some(n) if !PRIMITIVE_TYPE_NODES.contains(&n.kind()) => n,
        _ => return Ok(()),
    };
    match node.kind() {
        "type_identifier" => {
            let text = ctx.text(node)?;
            if !text.is_empty() {
                out.push((text.to_string(), generic));
            }
        }
        "qualified_identifier" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                collect_type_refs(ctx, Some(name_node), generic, out)?;
            }
        }
        "template_type" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let text = ctx.text(name_node)?;
                if !text.is_empty() {
                    out.push((text.to_string(), generic));
                }
            }
            if let Some(args_node) = node.child_by_field_name("arguments") {
                for c in children(args_node) {
                    if c.is_named() {
                        collect_type_refs(ctx, Some(c), true, out)?;
                    }
                }
            }
        }
        k if TYPE_WRAPPERS.contains(&k) => {
            for c in children(node) {
                if c.is_named() {
                    collect_type_refs(ctx, Some(c), generic, out)?;
                }
            }
        }
        // Anything else contributes nothing, with no default recursion.
        _ => {}
    }
    Ok(())
}

/// `_get_cpp_func_name`: unwrap a declarator to the name it declares.
///
/// A `qualified_identifier` is returned WHOLE, qualifier included. That looks
/// like an oversight and is load-bearing: an out-of-class definition
/// (`void Foo::bar() {}`) keeps `Foo::`, so `_make_id(stem, "Foo::bar")`
/// normalizes to the same id as the in-class member `_make_id(class_nid,
/// "bar")` -- the declaration in Foo.h and the definition in Foo.cpp collapse
/// onto ONE method node instead of two (#1547).
pub fn func_name(ctx: &Ctx, node: Node) -> R<Option<String>> {
    match node.kind() {
        "identifier" | "field_identifier" | "destructor_name" | "operator_name"
        | "qualified_identifier" => {
            return Ok(Some(ctx.text(node)?.to_string()));
        }
        _ => {}
    }
    if let Some(decl) = node.child_by_field_name("declarator") {
        return func_name(ctx, decl);
    }
    for child in children(node) {
        if child.kind() == "identifier" {
            return Ok(Some(ctx.text(child)?.to_string()));
        }
    }
    Ok(None)
}

/// `_cpp_declarator_name`: the bare variable name, or None.
///
/// None for anything that is not a plain named local -- an array, a function
/// pointer, a structured binding -- so the type table never records a guess.
pub fn declarator_name(ctx: &Ctx, node: Node) -> R<Option<String>> {
    declarator_name_src(ctx.src, node)
}

/// The same function against raw source bytes, for a walker that has its own
/// `Ctx` rather than the engine's.
///
/// `objc/` shares this grammar's declarator shapes exactly -- `_cpp_declarator_name`
/// is what `extract_objc` itself calls -- so it must share the code too, not a
/// copy that can drift.
pub fn declarator_name_src(src: &[u8], node: Node) -> R<Option<String>> {
    if node.kind() == "identifier" {
        return Ok(Some(
            crate::js::ast::text_checked(node, src)
                .ok_or("invalid_utf8_text")?
                .to_string(),
        ));
    }
    if matches!(
        node.kind(),
        "pointer_declarator" | "reference_declarator" | "init_declarator"
    ) {
        let mut inner = node.child_by_field_name("declarator");
        if inner.is_none() {
            inner = children(node).into_iter().find(|c| {
                matches!(
                    c.kind(),
                    "identifier" | "pointer_declarator" | "reference_declarator"
                )
            });
        }
        if let Some(inner) = inner {
            return declarator_name_src(src, inner);
        }
    }
    Ok(None)
}

/// `_cpp_local_var_types`: `var -> ClassName` from one function body.
///
/// PRECISION over recall throughout: only a class-like type with exactly ONE
/// named declarator is recorded. A built-in (`int x`), an ambiguous
/// multi-declarator line (`Foo a, b;`) or an un-nameable declarator contributes
/// nothing. First binding wins, and the table is FILE-scoped -- a later body's
/// `Foo f;` must not clobber an earlier one.
pub fn local_var_types(
    ctx: &Ctx,
    body_node: Node,
    table: &mut std::collections::HashMap<String, String>,
) -> R<()> {
    let mut stack = vec![body_node];
    while let Some(n) = stack.pop() {
        if matches!(n.kind(), "function_definition" | "lambda_expression") && n.id() != body_node.id()
        {
            // A nested function or lambda has its own scope; its locals would
            // pollute this body's table.
            continue;
        }
        if n.kind() == "declaration" {
            if let Some(type_node) = n.child_by_field_name("type") {
                if matches!(type_node.kind(), "type_identifier" | "qualified_identifier") {
                    let raw = ctx.text(type_node)?;
                    let type_name = raw.rsplit("::").next().unwrap_or(raw).trim();
                    let declarators: Vec<Node> = children(n)
                        .into_iter()
                        .filter(|c| {
                            matches!(
                                c.kind(),
                                "identifier"
                                    | "pointer_declarator"
                                    | "reference_declarator"
                                    | "init_declarator"
                            )
                        })
                        .collect();
                    let upper = type_name
                        .chars()
                        .next()
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false);
                    if !type_name.is_empty() && upper && declarators.len() == 1 {
                        if !type_name.is_ascii() {
                            return Err("non_ascii_id");
                        }
                        if let Some(var) = declarator_name(ctx, declarators[0])? {
                            table.entry(var).or_insert_with(|| type_name.to_string());
                        }
                    }
                }
            }
        }
        stack.extend(children(n));
    }
    Ok(())
}

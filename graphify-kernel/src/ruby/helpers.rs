//! The `_ruby_*` helpers from `engine.py`.

use std::collections::HashMap;

use tree_sitter::Node;

use crate::engine::{Ctx, R};
use crate::js::ast::children;

/// `_ruby_const_last_name`: `A::B::C` -> `C`.
pub fn const_last_name(ctx: &Ctx, node: Node) -> R<String> {
    if node.kind() == "constant" {
        return Ok(ctx.text(node)?.to_string());
    }
    if node.kind() == "scope_resolution" {
        let consts: Vec<Node> = children(node)
            .into_iter()
            .filter(|c| c.kind() == "constant")
            .collect();
        if let Some(last) = consts.last() {
            return Ok(ctx.text(*last)?.to_string());
        }
    }
    Ok(String::new())
}

/// `_ruby_const_full_name`: `A::B::C` kept WHOLE.
pub fn const_full_name(ctx: &Ctx, node: Node) -> R<String> {
    if !matches!(node.kind(), "constant" | "scope_resolution") {
        return Ok(String::new());
    }
    Ok(ctx.text(node)?.trim().to_string())
}

/// `_ruby_new_class_name`: `ClassName` when `node` is `ClassName.new(...)`.
///
/// Only a bare capitalized constant receiver counts; namespaced (`A::B.new`)
/// and dynamic receivers are ignored so the binding stays unambiguous.
fn new_class_name(ctx: &Ctx, node: Option<Node>) -> R<Option<String>> {
    let node = match node {
        Some(n) if n.kind() == "call" => n,
        _ => return Ok(None),
    };
    let recv = node.child_by_field_name("receiver");
    let meth = node.child_by_field_name("method");
    let (recv, meth) = match (recv, meth) {
        (Some(r), Some(m)) => (r, m),
        _ => return Ok(None),
    };
    if recv.kind() != "constant" || ctx.text(meth)? != "new" {
        return Ok(None);
    }
    Ok(Some(ctx.text(recv)?.to_string()))
}

/// `_ruby_local_class_bindings`: `local_var -> ClassName` for `var =
/// ClassName.new` within one method body, not descending into nested methods.
///
/// 100%-confidence contract: a variable assigned more than once, or to anything
/// other than a single `Constant.new`, maps to None (ambiguous) so callers never
/// resolve it. Only the certain single-binding case carries a type -- which is
/// why the value type is `Option<String>` and not `String`: a poisoned name is
/// PRESENT with no type, and that is different from being absent.
pub fn local_class_bindings(ctx: &Ctx, body_node: Node) -> R<HashMap<String, Option<String>>> {
    let mut bindings: HashMap<String, Option<String>> = HashMap::new();
    visit(ctx, body_node, &mut bindings)?;
    Ok(bindings)
}

fn visit(ctx: &Ctx, n: Node, bindings: &mut HashMap<String, Option<String>>) -> R<()> {
    for child in children(n) {
        if matches!(child.kind(), "method" | "singleton_method") {
            continue; // a nested method has its own scope
        }
        if child.kind() == "assignment" {
            let left = child.child_by_field_name("left");
            let right = child.child_by_field_name("right");
            if let Some(left) = left {
                if left.kind() == "identifier" {
                    let var = ctx.text(left)?.to_string();
                    let cls = new_class_name(ctx, right)?;
                    match cls {
                        // Assigned to something untypable: poison it if it was
                        // typed, but do not INTRODUCE it.
                        None => {
                            if bindings.contains_key(&var) {
                                bindings.insert(var.clone(), None);
                            }
                        }
                        Some(cls) => match bindings.get(&var) {
                            Some(existing) => {
                                if existing.as_deref() != Some(cls.as_str()) {
                                    bindings.insert(var.clone(), None);
                                }
                            }
                            None => {
                                bindings.insert(var.clone(), Some(cls));
                            }
                        },
                    }
                }
            }
        }
        visit(ctx, child, bindings)?;
    }
    Ok(())
}

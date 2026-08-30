//! `_java_method_receiver_types` and `_java_lambda_parameters`.
//!
//! The call WALK itself is `engine::calls`; what is left here is the one thing
//! that is genuinely Java-shaped -- building the `name -> declared type` table a
//! method body sees, which the engine calls once per method before any body is
//! walked.
//!

use std::collections::{HashMap, HashSet};

use tree_sitter::Node;

use super::helpers;
use crate::engine::{Ctx, RecvTable, R};
use crate::js::ast::children;
use crate::js::emit::{RawCall, Val};
use crate::py::helpers::BUILTIN_GLOBALS;

/// `callee_name not in _LANGUAGE_BUILTIN_GLOBALS`.
///
/// A LINEAR scan, deliberately: `BUILTIN_GLOBALS` is written in the Python
/// source's grouped-by-language order, not sorted, so `binary_search` silently
/// misses most of it. That mistake here reported 64 of 264 gson files DIVERGENT
/// -- `parseInt`, `set` and `next` are all in the set, so every call to one of
/// them produced a `raw_call` Python had filtered out.
fn is_builtin_global(name: &str) -> bool {
    BUILTIN_GLOBALS.contains(&name)
}

/// `_java_method_receiver_types`: the `name -> declared type` table one method
/// body sees.
///
/// Parameters shadow fields; a conflicting local declaration makes the name
/// AMBIGUOUS and removes it entirely, because raw call facts carry no lexical
/// scope and a wrong type binds the call to the wrong method.
pub fn method_receiver_types(
    ctx: &Ctx,
    method_node: Node,
    field_types: &HashMap<String, String>,
) -> R<RecvTable> {
    let mut method_types: HashMap<String, String> = HashMap::new();
    let mut ambiguous: HashSet<String> = HashSet::new();

    fn bind(
        method_types: &mut HashMap<String, String>,
        ambiguous: &mut HashSet<String>,
        name: &str,
        type_name: Option<&str>,
    ) {
        let type_name = match type_name {
            Some(t) if !t.is_empty() => t,
            _ => return,
        };
        if name.is_empty() || ambiguous.contains(name) {
            return;
        }
        match method_types.get(name) {
            Some(prev) if prev != type_name => {
                method_types.remove(name);
                ambiguous.insert(name.to_string());
            }
            _ => {
                method_types.insert(name.to_string(), type_name.to_string());
            }
        }
    }

    if let Some(params) = method_node.child_by_field_name("parameters") {
        for param in children(params) {
            if !matches!(param.kind(), "formal_parameter" | "spread_parameter") {
                continue;
            }
            let type_name = helpers::receiver_type_name(ctx, param.child_by_field_name("type"))?;
            if let Some(name_node) = param.child_by_field_name("name") {
                bind(&mut method_types, &mut ambiguous, ctx.text(name_node)?, type_name);
            }
        }
    }

    // Reverse-DFS via pop(), as in Python. Order does not reach the output --
    // the table is a map -- but the `continue` that skips nested type bodies
    // does depend on visiting a node before its children, which pop() preserves.
    let body = method_node.child_by_field_name("body");
    let mut stack: Vec<Node> = match body {
        Some(b) => children(b).into_iter().collect(),
        None => Vec::new(),
    };
    while let Some(node) = stack.pop() {
        if matches!(
            node.kind(),
            "class_declaration"
                | "class_body"
                | "interface_declaration"
                | "record_declaration"
                | "enum_declaration"
                | "annotation_type_declaration"
        ) {
            continue;
        }
        if node.kind() == "lambda_expression" {
            for (name, type_name) in lambda_parameters(ctx, node)? {
                let conflicts = match (field_types.get(name), type_name) {
                    (None, _) => false,
                    (Some(f), Some(t)) => f != t,
                    (Some(_), None) => true,
                };
                if type_name.is_none() || conflicts {
                    method_types.remove(name);
                    ambiguous.insert(name.to_string());
                } else {
                    bind(&mut method_types, &mut ambiguous, name, type_name);
                }
            }
        }
        if node.kind() == "local_variable_declaration" {
            let type_name = helpers::receiver_type_name(ctx, node.child_by_field_name("type"))?;
            for name in helpers::declarator_names(ctx, node)? {
                let conflicts = match (field_types.get(name), type_name) {
                    (None, _) => false,
                    (Some(f), Some(t)) => f != t,
                    (Some(_), None) => true,
                };
                if conflicts {
                    method_types.remove(name);
                    ambiguous.insert(name.to_string());
                } else {
                    bind(&mut method_types, &mut ambiguous, name, type_name);
                }
            }
        }
        stack.extend(children(node));
    }

    let mut table = field_types.clone();
    table.extend(method_types);
    for name in &ambiguous {
        table.remove(name);
    }
    for (name, type_name) in field_types {
        table.insert(format!("this.{name}"), type_name.clone());
    }
    // Flat: a Java binding is method-wide, so the call offset never enters the
    // lookup. C#'s is positional -- see `RecvTable`.
    Ok(RecvTable::Flat(table))
}

/// `_java_lambda_parameters`.
fn lambda_parameters<'a>(
    ctx: &Ctx<'a, '_>,
    lambda_node: Node,
) -> R<Vec<(&'a str, Option<&'a str>)>> {
    let parameters = match lambda_node.child_by_field_name("parameters") {
        Some(p) => p,
        None => return Ok(Vec::new()),
    };
    if parameters.kind() == "identifier" {
        return Ok(vec![(ctx.text(parameters)?, None)]);
    }
    if parameters.kind() == "inferred_parameters" {
        let mut out = Vec::new();
        for child in children(parameters) {
            if child.kind() == "identifier" {
                out.push((ctx.text(child)?, None));
            }
        }
        return Ok(out);
    }
    let mut bindings = Vec::new();
    for parameter in children(parameters) {
        if !matches!(parameter.kind(), "formal_parameter" | "spread_parameter") {
            continue;
        }
        if let Some(name_node) = parameter.child_by_field_name("name") {
            let ty = helpers::receiver_type_name(ctx, parameter.child_by_field_name("type"))?;
            bindings.push((ctx.text(name_node)?, ty));
        }
    }
    Ok(bindings)
}

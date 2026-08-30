//! The call-graph pass: the Java-live slice of `_extract_generic`'s inner
//! `walk_calls`, plus `_java_method_receiver_types`.
//!
//! `_JAVA_CONFIG` leaves `call_accessor_node_types` empty, so the whole generic
//! accessor model (`call_accessor_field` / `call_accessor_object_field`) is dead
//! for Java: the `elif _is_java` branch reads `name` / `object` / `type`
//! directly. That is why the generic path is absent here rather than ported.
//!
//! # Java defers every member call
//!
//! `_java_defer = (_is_java and is_member_call)` -- unconditionally, with no
//! receiver or capitalization test, unlike Python's and C#'s narrower rules. So
//! `a.b()` NEVER becomes a `calls` edge from this walker; it becomes a
//! `raw_call` tagged `lang="java"` with the receiver's declared type attached,
//! and `_resolve_java_member_calls` binds it in phase 3 where the type table is
//! corpus-wide. Only an unqualified `b()` or a `new Foo()` can resolve here.

use std::collections::{HashMap, HashSet};

use tree_sitter::Node;

use super::helpers;
use super::{Ctx, R};
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
) -> R<HashMap<String, String>> {
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
    Ok(table)
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

/// `raw.split("<", 1)[0].strip()` then `rsplit(".", 1)[-1]`.
fn constructed_type_name(raw: &str) -> &str {
    let base = raw.split_once('<').map(|(a, _)| a).unwrap_or(raw).trim();
    base.rsplit_once('.').map(|(_, t)| t).unwrap_or(base)
}

pub fn walk_calls<'tree>(
    ctx: &mut Ctx<'_, 'tree>,
    node: Node<'tree>,
    caller_nid: &str,
    receiver_types: &HashMap<String, String>,
) -> R<()> {
    // `function_boundary_types`. The JS-only descend into untracked closures does
    // not apply, so this is an unconditional stop: a nested method's calls are
    // attributed to that method, which has its own entry in `function_bodies`.
    if matches!(node.kind(), "constructor_declaration" | "method_declaration") {
        return Ok(());
    }

    if matches!(node.kind(), "method_invocation" | "object_creation_expression") {
        let mut callee_name: Option<&str> = None;
        let mut is_member_call = false;
        let mut member_receiver: Option<String> = None;

        if node.kind() == "object_creation_expression" {
            // `new Foo(...)`: the constructed type is the `type` field, not
            // `name`, so the generic path misses it (#1373).
            if let Some(type_node) = node.child_by_field_name("type") {
                let raw = ctx.text(type_node)?;
                let name = constructed_type_name(raw);
                if !name.is_empty() {
                    callee_name = Some(name);
                }
            }
        } else {
            if let Some(name_node) = node.child_by_field_name("name") {
                callee_name = Some(ctx.text(name_node)?);
            }
            if let Some(receiver) = node.child_by_field_name("object") {
                is_member_call = true;
                match receiver.kind() {
                    "identifier" => member_receiver = Some(ctx.text(receiver)?.to_string()),
                    "this" => member_receiver = Some("this".to_string()),
                    "field_access" => {
                        let owner = receiver.child_by_field_name("object");
                        let field = receiver.child_by_field_name("field");
                        if let (Some(o), Some(f)) = (owner, field) {
                            if o.kind() == "this" {
                                member_receiver = Some(format!("this.{}", ctx.text(f)?));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        if let Some(callee) = callee_name {
            if !is_builtin_global(callee) {
                // `_java_defer`: ANY member call defers, with no receiver test.
                let tgt_nid = if is_member_call {
                    None
                } else {
                    ctx.label_to_nid.get(callee).cloned()
                };
                match tgt_nid {
                    Some(tgt) if tgt != caller_nid => {
                        let pair = (caller_nid.to_string(), tgt.clone());
                        if ctx.seen_call_pairs.insert(pair) {
                            let line = node.start_position().row + 1;
                            let sf = ctx.str_path.to_string();
                            ctx.edges.push(crate::js::emit::EdgeRow {
                                source: caller_nid.to_string(),
                                target: tgt,
                                relation: "calls",
                                fields: vec![
                                    ("context", Val::Static("call")),
                                    ("confidence", Val::Static("EXTRACTED")),
                                    ("source_file", Val::S(sf)),
                                    ("source_location", Val::S(format!("L{line}"))),
                                    ("weight", Val::F(1.0)),
                                ],
                            });
                        }
                    }
                    Some(_) => { /* tgt == caller_nid: Python emits nothing */ }
                    None => {
                        let line = node.start_position().row + 1;
                        // `receiver_types` is keyed by the enclosing body; the
                        // lookup key is the raw receiver text, and `this.field`
                        // receivers were stamped into the table under that exact
                        // dotted form by `method_receiver_types`.
                        let receiver_type = member_receiver
                            .as_deref()
                            .and_then(|r| receiver_types.get(r))
                            .cloned();
                        // Key order is Python's dict-literal order, and it
                        // reaches the pickled result, so it is not free to vary:
                        // `receiver` is present even when None (Python writes the
                        // key unconditionally), `lang` follows, and
                        // `receiver_type` is added ONLY when a type was found.
                        let mut rc: RawCall = vec![
                            ("caller_nid", Val::S(caller_nid.to_string())),
                            ("callee", Val::S(callee.to_string())),
                            ("is_member_call", Val::B(is_member_call)),
                            ("source_file", Val::S(ctx.str_path.to_string())),
                            ("source_location", Val::S(format!("L{line}"))),
                            (
                                "receiver",
                                match &member_receiver {
                                    Some(r) => Val::S(r.clone()),
                                    None => Val::None,
                                },
                            ),
                            ("lang", Val::Static("java")),
                        ];
                        if let Some(rt) = receiver_type {
                            rc.push(("receiver_type", Val::S(rt)));
                        }
                        ctx.raw_calls.push(rc);
                    }
                }
            }
        }
    }

    // The indirect-dispatch block is `if _is_python`, so Java skips it.
    for child in children(node) {
        walk_calls(ctx, child, caller_nid, receiver_types)?;
    }
    Ok(())
}

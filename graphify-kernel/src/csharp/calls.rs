//! The C# call branch and its positional receiver table.

use std::collections::HashMap;

use tree_sitter::Node;

use crate::engine::{CallInfo, Ctx, RecvTable, R};
use crate::js::ast::children;

use super::helpers::{read_type_name, receiver_type_name};

/// Type declarations a method body may CONTAIN but whose members belong to
/// their own scope, not this method's.
const NESTED_TYPE_DECLARATIONS: &[&str] = &[
    "class_declaration",
    "struct_declaration",
    "interface_declaration",
    "record_declaration",
    "enum_declaration",
];

/// `_csharp_method_receiver_types`: the SCOPED bindings visible to one method
/// (#2299, #2472).
///
/// Positional rather than flat, which is the whole point: a name can be bound
/// several times in one method -- a parameter, a `var` local in an inner block,
/// a pattern binding in one arm of an `if` -- and the binding that applies to a
/// call is the innermost one whose range contains it. The old method-wide rule
/// let one untypable `out var x` wipe a same-named typed binding in a sibling
/// scope.
///
/// Scope ranges are deliberately conservative: a pattern binding spans its whole
/// enclosing block, which is over-wide, but over-wide only ever produces TIES,
/// and a tie drops the edge rather than guessing.
pub fn method_receiver_types(
    ctx: &Ctx,
    method_node: Node,
    field_types: &HashMap<String, String>,
) -> R<RecvTable> {
    let mut bindings: HashMap<String, Vec<(usize, usize, Option<String>)>> = HashMap::new();
    let mut field_poisoned: Vec<String> = Vec::new();

    // `bind`, inlined as a closure would need two mutable borrows.
    macro_rules! bind {
        ($name:expr, $ty:expr, $scope:expr) => {{
            let name: String = $name;
            let ty: Option<String> = $ty;
            let scope: Node = $scope;
            if !name.is_empty() {
                // `field_types.get(name) not in (None, type_name)`: a local that
                // disagrees with a same-named field/property drops the name.
                if let Some(f) = field_types.get(&name) {
                    if Some(f) != ty.as_ref() {
                        field_poisoned.push(name.clone());
                    }
                }
                bindings.entry(name).or_default().push((
                    scope.start_byte(),
                    scope.end_byte(),
                    ty,
                ));
            }
        }};
    }
    macro_rules! bind_parameter {
        ($param:expr, $scope:expr) => {{
            let param: Node = $param;
            if let Some(name_node) = param.child_by_field_name("name") {
                bind!(
                    ctx.text(name_node)?.to_string(),
                    receiver_type_name(ctx, param.child_by_field_name("type"))?,
                    $scope
                );
            }
        }};
    }

    let body = method_node.child_by_field_name("body");
    // Parameters scope to the BODY: a parameter and an (illegal) same-named
    // top-level local share one C# declaration space, and equal ranges tie at
    // the call site -- drop, never a guess.
    let param_scope = body.unwrap_or(method_node);
    if let Some(params) = method_node.child_by_field_name("parameters") {
        for param in children(params) {
            if param.kind() == "parameter" {
                bind_parameter!(param, param_scope);
            }
        }
    }

    let mut stack: Vec<(Node, Node)> = match body {
        Some(b) => children(b).into_iter().map(|c| (c, param_scope)).collect(),
        None => Vec::new(),
    };
    while let Some((node, scope)) = stack.pop() {
        if NESTED_TYPE_DECLARATIONS.contains(&node.kind()) {
            continue;
        }
        match node.kind() {
            "lambda_expression" => {
                // A lambda parameter is visible exactly inside the lambda: typed
                // binds its type there, untyped (`x => ...`) binds None so calls
                // on it stay unstamped WITHOUT wiping an outer binding (#2472).
                if let Some(lam_params) = node.child_by_field_name("parameters") {
                    if lam_params.kind() == "implicit_parameter" {
                        bind!(ctx.text(lam_params)?.to_string(), None, node);
                    } else {
                        for param in children(lam_params) {
                            match param.kind() {
                                "parameter" => bind_parameter!(param, node),
                                "implicit_parameter" => {
                                    bind!(ctx.text(param)?.to_string(), None, node)
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            "local_function_statement" => {
                if let Some(lf_params) = node.child_by_field_name("parameters") {
                    for param in children(lf_params) {
                        if param.kind() == "parameter" {
                            bind_parameter!(param, node);
                        }
                    }
                }
            }
            "local_declaration_statement" => {
                if let Some(vd) = children(node)
                    .into_iter()
                    .find(|c| c.kind() == "variable_declaration")
                {
                    let declared = receiver_type_name(ctx, vd.child_by_field_name("type"))?;
                    for declarator in children(vd) {
                        if declarator.kind() != "variable_declarator" {
                            continue;
                        }
                        let name_node = declarator.child_by_field_name("name").or_else(|| {
                            children(declarator)
                                .into_iter()
                                .find(|g| g.kind() == "identifier")
                        });
                        let name_node = match name_node {
                            Some(n) => n,
                            None => continue,
                        };
                        let mut type_name = declared.clone();
                        if type_name.is_none() {
                            // `var v = new T()` -- recover T from the creation.
                            for g in children(declarator) {
                                if g.kind() == "object_creation_expression" {
                                    type_name =
                                        receiver_type_name(ctx, g.child_by_field_name("type"))?;
                                    break;
                                }
                            }
                        }
                        bind!(ctx.text(name_node)?.to_string(), type_name, scope);
                    }
                }
            }
            "declaration_expression" | "declaration_pattern" => {
                // #2346: inline-declared receivers. `out Sect s` is a
                // declaration_expression; `is Leaf lf`, `case Twig tw:` and a
                // switch-arm `Stem st =>` are declaration_patterns. `out var v`
                // yields None and stays untypable inside that block ONLY.
                if let Some(name_node) = node.child_by_field_name("name") {
                    if name_node.kind() == "identifier" {
                        bind!(
                            ctx.text(name_node)?.to_string(),
                            receiver_type_name(ctx, node.child_by_field_name("type"))?,
                            scope
                        );
                    }
                }
            }
            _ => {}
        }
        let child_scope = if matches!(
            node.kind(),
            "block" | "lambda_expression" | "local_function_statement"
        ) {
            node
        } else {
            scope
        };
        stack.extend(children(node).into_iter().map(|c| (c, child_scope)));
    }

    let mut base = field_types.clone();
    for name in &field_poisoned {
        base.remove(name);
        bindings.remove(name);
    }
    Ok(RecvTable::Scoped { bindings, base })
}

/// The `object_creation_expression` arm: `new Foo(...)` names the CONSTRUCTED
/// type, which the invocation path never sees (#1373's C# twin).
///
/// A qualifier written in source is kept so `_resolve_csharp_qualified_calls`
/// can pick one of several same-named `Cache` classes instead of hitting the
/// ambiguity guard. Target-typed `new()` parses as
/// `implicit_object_creation_expression` and is deliberately not in `call_types`.
pub fn object_creation_info(ctx: &Ctx, node: Node) -> R<CallInfo> {
    let mut info = CallInfo::default();
    if let Some((name, qualified, qualifier)) =
        read_type_name(ctx, node.child_by_field_name("type"))?
    {
        if !name.is_empty() {
            info.callee_name = Some(name);
            if qualified && !qualifier.is_empty() {
                info.qualified_prefix = Some(qualifier);
            }
        }
    }
    Ok(info)
}

/// The callee/receiver half of the `invocation_expression` arm.
///
/// Capturing the receiver is what lets `_resolve_csharp_member_calls` bind the
/// call to the receiver's DECLARED type; without it a bare method name matched
/// any same-named method in the corpus, mis-resolving `_server.Save()` to an
/// unrelated `Cache.Save()` (#1609).
pub fn invocation_info(ctx: &Ctx, node: Node, fn_node: Option<Node>) -> R<CallInfo> {
    let mut info = CallInfo::default();
    match fn_node {
        Some(f) if f.kind() == "member_access_expression" => {
            let mname = f.child_by_field_name("name");
            let recv = f.child_by_field_name("expression");
            if let Some(mname) = mname {
                info.callee_name = Some(ctx.text(mname)?.to_string());
                info.is_member_call = true;
                if let Some(recv) = recv {
                    match recv.kind() {
                        "identifier" => {
                            info.member_receiver = Some(ctx.text(recv)?.to_string())
                        }
                        "this" | "this_expression" => {
                            info.member_receiver = Some("this".to_string())
                        }
                        // `base.M()`: resolved against the caller's single
                        // resolvable base class in the cross-file pass.
                        "base" | "base_expression" => {
                            info.member_receiver = Some("base".to_string())
                        }
                        "member_access_expression" => {
                            // `this.field.M()` is typed exactly like a bare
                            // `field.M()`; any other chained receiver stays
                            // untyped (the resolver bails rather than guessing).
                            let inner = recv.child_by_field_name("expression");
                            let fname = recv.child_by_field_name("name");
                            if let (Some(inner), Some(fname)) = (inner, fname) {
                                if matches!(inner.kind(), "this" | "this_expression")
                                    && fname.kind() == "identifier"
                                {
                                    info.member_receiver = Some(ctx.text(fname)?.to_string());
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        Some(f) if f.kind() == "identifier" => {
            info.callee_name = Some(ctx.text(f)?.to_string());
        }
        _ => {
            // Fallback: the original name-field / first-named-child scan.
            if let Some(name_node) = node.child_by_field_name("name") {
                info.callee_name = Some(ctx.text(name_node)?.to_string());
            } else {
                for child in children(node) {
                    if !child.is_named() {
                        continue;
                    }
                    let raw = ctx.text(child)?;
                    if raw.contains('.') {
                        let parts: Vec<&str> = raw.split('.').collect();
                        info.callee_name = Some(parts[parts.len() - 1].to_string());
                        info.is_member_call = true;
                        if parts.len() == 2 && !parts[0].is_empty() {
                            info.member_receiver = Some(parts[0].to_string());
                        }
                    } else {
                        info.callee_name = Some(raw.to_string());
                    }
                    break;
                }
            }
        }
    }
    Ok(info)
}

//! The call-graph pass: `walk_calls`, the indirect-dispatch emitters, and the
//! module-level dispatch-table scan -- the Python-live slice of each.
//!
//! Emission ORDER is part of the output, not an implementation detail: nodes and
//! edges are compared byte-for-byte, so the branches below fire in exactly the
//! sequence the Python does (call handling, then dispatch tables, then
//! assignment/return references, then the recursion).

use std::collections::HashSet;

use tree_sitter::Node;

use crate::js::ast::{children, line_of};
use crate::js::emit::{EdgeRow, Val};
use super::helpers::{self, BUILTIN_GLOBALS};
use super::{Ctx, R};

/// `local_bound_names.get(caller_nid, frozenset()) | extra_locals`.
fn enclosing_locals(ctx: &Ctx, caller_nid: &str, extra: &HashSet<String>) -> HashSet<String> {
    let mut out = ctx
        .local_bound_names
        .get(caller_nid)
        .cloned()
        .unwrap_or_default();
    out.extend(extra.iter().cloned());
    out
}

/// `_emit_indirect_by_name`: resolve a name referenced AS A VALUE to a real
/// callable def and emit one INFERRED `indirect_call`, deferring an unknown or
/// foreign name to the cross-file resolver.
///
/// Scope filtering is the CALLER's job -- an identifier reference must reject
/// param/local shadows, whereas a `getattr(obj, "x")` string names an ATTRIBUTE
/// and is never shadowed, so that path calls straight in here.
fn emit_indirect_by_name(
    ctx: &mut Ctx,
    ident_name: &str,
    loc_line: usize,
    scope_nid: &str,
    context: &'static str,
) -> R<()> {
    let ref_nid = ctx.label_to_nid.get(ident_name).cloned();
    // Defer to the cross-file resolver when the name is not defined in this file,
    // or resolves to an import-surfaced FOREIGN symbol whose definition (and
    // callability) lives elsewhere. That pass applies the single-definition
    // god-node guard plus the GLOBAL callable-target check, so a foreign
    // non-callable still produces no edge.
    let defer = match &ref_nid {
        None => true,
        Some(nid) => {
            !ctx.callable_def_nids.contains(nid)
                && ctx.nid_to_sf.get(nid).map(String::as_str).unwrap_or("") != ctx.str_path
        }
    };
    if defer {
        ctx.raw_calls.push(vec![
            ("caller_nid", Val::S(scope_nid.to_string())),
            ("callee", Val::S(ident_name.to_string())),
            ("is_member_call", Val::B(false)),
            ("indirect", Val::B(true)),
            ("context", Val::Static(context)),
            ("source_file", Val::S(ctx.str_path.to_string())),
            ("source_location", Val::S(format!("L{loc_line}"))),
        ]);
        return Ok(());
    }
    let ref_nid = ref_nid.expect("defer covers the None case");
    // Self-ref, or a same-named LOCAL non-callable data node -- no edge.
    if ref_nid == scope_nid || !ctx.callable_def_nids.contains(&ref_nid) {
        return Ok(());
    }
    // A class referenced as a value (`select(Model)`, an exception tuple) is a
    // descriptor, not an invocation (#2137).
    if ctx.callable_class_nids.contains(&ref_nid) {
        return Ok(());
    }
    let pair = (scope_nid.to_string(), ref_nid.clone());
    if ctx.seen_call_pairs.contains(&pair) {
        return Ok(()); // already a direct call to this target
    }
    if ctx.seen_indirect_pairs.contains(&pair) {
        return Ok(());
    }
    ctx.seen_indirect_pairs.insert(pair);
    ctx.edges.push(EdgeRow {
        source: scope_nid.to_string(),
        target: ref_nid,
        relation: "indirect_call",
        fields: vec![
            ("context", Val::Static(context)),
            ("confidence", Val::Static("INFERRED")),
            // 0.85 = "strong inference" on the extraction-spec rubric: the symbol
            // link is direct, but that it is ever INVOKED is the inference (#2813).
            ("confidence_score", Val::F(0.85)),
            ("source_file", Val::S(ctx.str_path.to_string())),
            ("source_location", Val::S(format!("L{loc_line}"))),
            ("weight", Val::F(1.0)),
        ],
    });
    Ok(())
}

/// `_emit_indirect_ref`: the identifier form, with the shadow guard.
fn emit_indirect_ref(
    ctx: &mut Ctx,
    ident: Option<Node>,
    scope_nid: &str,
    locals: &HashSet<String>,
    context: &'static str,
) -> R<()> {
    let Some(ident) = ident else { return Ok(()) };
    if !matches!(ident.kind(), "identifier" | "shorthand_property_identifier") {
        return Ok(());
    }
    let ident_name = ctx.text(ident)?.to_string();
    // Shadowing: a param / local binding names a local value, not the module fn.
    if locals.contains(&ident_name) || ident_name == "self" || ident_name == "cls" {
        return Ok(());
    }
    // `js_external_imports` is only ever populated for the JS family, so the
    // external-import shadow test below it in Python is vacuously false here.
    emit_indirect_by_name(ctx, &ident_name, line_of(ident), scope_nid, context)
}

/// `_python_dispatch_value_idents`: dict VALUES (never keys), and the elements of
/// a list/set/tuple. Nested collections are reached by the caller's recursion.
fn dispatch_value_idents(coll: Node) -> Vec<Node> {
    let mut out = Vec::new();
    if coll.kind() == "dictionary" {
        for pair in children(coll) {
            if pair.kind() != "pair" {
                continue;
            }
            if let Some(val) = pair.child_by_field_name("value") {
                if val.kind() == "identifier" {
                    out.push(val);
                }
            }
        }
    } else {
        for el in children(coll) {
            if el.kind() == "identifier" {
                out.push(el);
            }
        }
    }
    out
}

/// `_python_ref_value_idents`: a bare name on an assignment RHS or a return, or
/// the elements of a bare unpack. A collection LITERAL is a dispatch table
/// reached by the normal recursion, so it is not handled here.
fn ref_value_idents(value: Option<Node>) -> Vec<Node> {
    let Some(value) = value else { return Vec::new() };
    if value.kind() == "identifier" {
        return vec![value];
    }
    if value.kind() == "expression_list" {
        return children(value)
            .into_iter()
            .filter(|c| c.kind() == "identifier")
            .collect();
    }
    Vec::new()
}

/// `_getattr_ref_name`: `getattr(obj, "name")` with a PLAIN string literal, as
/// `(name, string_node)`. A dynamic name -- a variable, an f-string, a
/// concatenation -- is not statically resolvable and yields None, as do the
/// 1-arg form and `obj.getattr(...)` (a method, not the builtin).
fn getattr_ref_name<'a, 'tree>(
    ctx: &Ctx<'a, 'tree>,
    call_node: Node<'tree>,
) -> R<Option<(String, Node<'tree>)>> {
    let Some(fnode) = call_node.child_by_field_name("function") else {
        return Ok(None);
    };
    if fnode.kind() != "identifier" || ctx.text(fnode)? != "getattr" {
        return Ok(None);
    }
    let Some(args) = call_node.child_by_field_name("arguments") else {
        return Ok(None);
    };
    let positional: Vec<Node> = children(args)
        .into_iter()
        .filter(|c| c.is_named() && !matches!(c.kind(), "keyword_argument" | "comment"))
        .collect();
    if positional.len() < 2 {
        return Ok(None);
    }
    let name_node = positional[1];
    if name_node.kind() != "string"
        || children(name_node).iter().any(|ch| ch.kind() == "interpolation")
    {
        return Ok(None);
    }
    let Some(content) = children(name_node)
        .into_iter()
        .find(|ch| ch.kind() == "string_content")
    else {
        return Ok(None); // empty string "" -- no attribute name
    };
    Ok(Some((ctx.text(content)?.to_string(), name_node)))
}

pub fn walk_calls<'tree>(
    ctx: &mut Ctx<'_, 'tree>,
    node: Node<'tree>,
    caller_nid: &str,
    extra_locals: &HashSet<String>,
) -> R<()> {
    // `function_boundary_types` = {"function_definition"}. The JS descend-into-
    // untracked-closures arm is guarded by `_is_js_family` and is dead here.
    if node.kind() == "function_definition" {
        return Ok(());
    }

    if node.kind() == "call" {
        let mut callee_name: Option<String> = None;
        let mut is_member_call = false;
        let mut member_receiver: Option<String> = None;
        let mut is_this_field_call = false;

        if let Some(func_node) = node.child_by_field_name("function") {
            if func_node.kind() == "identifier" {
                callee_name = Some(ctx.text(func_node)?.to_string());
            } else if func_node.kind() == "attribute" {
                is_member_call = true;
                if let Some(attr) = func_node.child_by_field_name("attribute") {
                    callee_name = Some(ctx.text(attr)?.to_string());
                }
                // Capture a simple-identifier receiver so cross-file member-call
                // resolution can resolve qualified class-method calls (#1446).
                let obj = func_node.child_by_field_name("object");
                match obj {
                    Some(o) if o.kind() == "identifier" => {
                        member_receiver = Some(ctx.text(o)?.to_string());
                    }
                    Some(o) if o.kind() == "call" => {
                        // `super().method()` has a call node as its receiver.
                        // Preserve it as a known intra-class receiver instead of
                        // treating it as unresolved.
                        if let Some(rf) = o.child_by_field_name("function") {
                            if rf.kind() == "identifier" && ctx.text(rf)? == "super" {
                                member_receiver = Some("super".to_string());
                            }
                        }
                    }
                    Some(o) if o.kind() == "attribute" => {
                        // `this.field.method()` (#1316). `this` is not a Python
                        // node kind, so this arm cannot fire for Python -- kept
                        // because it is reached through config fields Python
                        // sets, not through an `_is_js_family` guard.
                        if let Some(inner_obj) = o.child_by_field_name("object") {
                            if inner_obj.kind() == "this" {
                                if let Some(inner_prop) = o.child_by_field_name("attribute") {
                                    member_receiver = Some(ctx.text(inner_prop)?.to_string());
                                    is_this_field_call = true;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            } else {
                callee_name = Some(ctx.text(func_node)?.to_string());
            }
        }

        if let Some(name) = callee_name.clone() {
            if !name.is_empty() && !BUILTIN_GLOBALS.contains(&name.as_str()) {
                // Python member calls defer to receiver-based resolution unless
                // the receiver is known to stay in the current class. Falling
                // back to a bare method name for an unresolved or lowercase
                // receiver (`d.get()`) can bind to an unrelated module function
                // and inflate it into a god node (#2417). Note that a member call
                // with NO captured receiver defers too: Python tests
                // `member_receiver not in {...}`, and `None` is not in the set.
                let python_defer = is_member_call
                    && !matches!(
                        member_receiver.as_deref(),
                        Some("self") | Some("cls") | Some("super")
                    );
                let upper_receiver = member_receiver
                    .as_deref()
                    .and_then(|r| r.chars().next())
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false);
                let tgt_nid = if python_defer
                    || (is_member_call
                        && member_receiver.is_some()
                        && (upper_receiver || is_this_field_call))
                {
                    None
                } else {
                    ctx.label_to_nid.get(&name).cloned()
                };

                match tgt_nid {
                    Some(tgt) if tgt != caller_nid => {
                        let pair = (caller_nid.to_string(), tgt.clone());
                        if !ctx.seen_call_pairs.contains(&pair) {
                            ctx.seen_call_pairs.insert(pair);
                            let line = line_of(node);
                            ctx.edges.push(EdgeRow {
                                source: caller_nid.to_string(),
                                target: tgt,
                                relation: "calls",
                                fields: vec![
                                    ("context", Val::Static("call")),
                                    ("confidence", Val::Static("EXTRACTED")),
                                    ("source_file", Val::S(ctx.str_path.to_string())),
                                    ("source_location", Val::S(format!("L{line}"))),
                                    ("weight", Val::F(1.0)),
                                ],
                            });
                        }
                    }
                    Some(_) => {} // tgt == caller_nid: no edge, and no raw_call
                    None => {
                        // Callee not in this file -- save for cross-file resolution.
                        let line = line_of(node);
                        ctx.raw_calls.push(vec![
                            ("caller_nid", Val::S(caller_nid.to_string())),
                            ("callee", Val::S(name.clone())),
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
                        ]);
                    }
                }
            }
        }

        // Indirect dispatch: a function passed BY NAME as a call argument is a
        // real dependency the callee-only scan above cannot see. Emitted as a
        // distinct `indirect_call` so strict `calls` queries stay precise.
        if let Some(args_node) = node.child_by_field_name("arguments") {
            let locals = enclosing_locals(ctx, caller_nid, extra_locals);
            for arg in children(args_node) {
                if arg.kind() == "identifier" {
                    emit_indirect_ref(ctx, Some(arg), caller_nid, &locals, "argument")?;
                } else if arg.kind() == "keyword_argument" {
                    let v = arg.child_by_field_name("value");
                    emit_indirect_ref(ctx, v, caller_nid, &locals, "argument")?;
                }
            }
        }
        // Reflective dispatch: `getattr(obj, "handler")` names a callable by
        // string literal (#1566 slice 3). A string is an ATTRIBUTE name, never
        // shadowed by a param/local, so it bypasses the identifier shadow guard.
        if let Some((ref_name, loc)) = getattr_ref_name(ctx, node)? {
            emit_indirect_by_name(ctx, &ref_name, line_of(loc), caller_nid, "getattr")?;
        }
    }

    // `static_prop_types`, `helper_fn_names` and `container_bind_methods` are all
    // empty for `_PYTHON_CONFIG`, so the three branches between here and the
    // dispatch tables are dead.

    // Dispatch tables (#1566): a function listed as a value in a collection
    // literal inside this body is an indirect dependency of the enclosing function.
    if matches!(node.kind(), "dictionary" | "list" | "set" | "tuple") {
        let locals = enclosing_locals(ctx, caller_nid, extra_locals);
        for ident in dispatch_value_idents(node) {
            emit_indirect_ref(ctx, Some(ident), caller_nid, &locals, "collection")?;
        }
    }

    // Assignment / return references (#1566 slice 2). The VALUE side only -- the
    // assignment TARGET is a new local binding, not a reference.
    if node.kind() == "assignment" {
        let locals = enclosing_locals(ctx, caller_nid, extra_locals);
        for ident in ref_value_idents(node.child_by_field_name("right")) {
            emit_indirect_ref(ctx, Some(ident), caller_nid, &locals, "assignment")?;
        }
    } else if node.kind() == "return_statement" {
        let locals = enclosing_locals(ctx, caller_nid, extra_locals);
        let value = children(node).into_iter().find(|c| c.is_named());
        for ident in ref_value_idents(value) {
            emit_indirect_ref(ctx, Some(ident), caller_nid, &locals, "return")?;
        }
    }

    for child in children(node) {
        walk_calls(ctx, child, caller_nid, extra_locals)?;
    }
    Ok(())
}

/// Module-level dispatch tables (#1566): a function listed as a value in a
/// TOP-LEVEL collection literal (a route / handler registry) is an indirect
/// dependency of the FILE. Function and class bodies are walked separately, so
/// this scan stops at their boundaries -- it must not re-attribute a method's
/// local table to the file.
pub fn scan_module_dispatch<'tree>(ctx: &mut Ctx<'_, 'tree>, root: Node<'tree>) -> R<()> {
    let module_bound = helpers::module_bound_names(ctx, root)?;
    scan_one(ctx, root, &module_bound)
}

fn scan_one<'tree>(
    ctx: &mut Ctx<'_, 'tree>,
    n: Node<'tree>,
    module_bound: &HashSet<String>,
) -> R<()> {
    if matches!(n.kind(), "function_definition" | "class_definition") {
        return Ok(());
    }
    let file_nid = ctx.file_nid.clone();
    if matches!(n.kind(), "dictionary" | "list" | "set" | "tuple") {
        for ident in dispatch_value_idents(n) {
            emit_indirect_ref(ctx, Some(ident), &file_nid, module_bound, "collection")?;
        }
    } else if n.kind() == "assignment" {
        // Module-level alias / re-export: `CALLBACK = handler`.
        for ident in ref_value_idents(n.child_by_field_name("right")) {
            emit_indirect_ref(ctx, Some(ident), &file_nid, module_bound, "assignment")?;
        }
    } else if n.kind() == "call" {
        // Module-level reflective dispatch: `HANDLER = getattr(mod, "handler")`.
        if let Some((ref_name, loc)) = getattr_ref_name(ctx, n)? {
            emit_indirect_by_name(ctx, &ref_name, line_of(loc), &file_nid, "getattr")?;
        }
    }
    for c in children(n) {
        scan_one(ctx, c, module_bound)?;
    }
    Ok(())
}

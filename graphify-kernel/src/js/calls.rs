//! The call-graph pass: `walk_calls`, the indirect-reference emitters, and the
//! module-level dispatch-table scan.
//!
//! For TypeScript the whole `walk_calls` dispatch reduces to the *generic* arm --
//! `static_prop_types`, `helper_fn_names` and `container_bind_methods` are empty
//! in `_TS_CONFIG`, and every other arm is gated on another language. What
//! remains is: descend through untracked closures, record requires, classify a
//! call's callee and receiver, resolve it against this file's label map or defer
//! it to the corpus pass as a `raw_call`, and capture by-name callback references.

use std::collections::HashSet;

use tree_sitter::Node;

use super::ast::{children, line_of};
use super::emit::{EdgeRow, Val};
use super::walk::is_call_type;
use super::{imports, pat, Ctx, R};

/// `config.function_boundary_types` for TS.
fn is_boundary(kind: &str) -> bool {
    matches!(
        kind,
        "function_declaration"
            | "generator_function_declaration"
            | "arrow_function"
            | "method_definition"
            | "function_expression"
            | "generator_function"
    )
}

pub fn walk_calls<'t>(
    ctx: &mut Ctx<'_, 't>,
    node: Node<'t>,
    caller_nid: &str,
    extra_locals: &HashSet<String>,
) -> R<()> {
    if is_boundary(node.kind()) {
        // #1630 Pattern B / #2575: an inline closure or a nested NAMED function
        // that is not separately tracked would otherwise lose every call inside
        // it at this boundary. Tracked bodies are entered via their own
        // `function_bodies` entry, so the guard prevents double-walking.
        if pat::is_descend_type(node.kind()) {
            if let Some(body) = node.child_by_field_name("body") {
                if !ctx.tracked_body_ids.contains(&body.id()) {
                    let mut closure_locals = extra_locals.clone();
                    closure_locals.extend(pat::local_bound_names(node, ctx.src));
                    for child in children(node) {
                        walk_calls(ctx, child, caller_nid, &closure_locals)?;
                    }
                }
            }
        }
        return Ok(());
    }

    if matches!(node.kind(), "lexical_declaration" | "variable_declaration") {
        imports::require_imports_js(ctx, node, caller_nid)?;
    }

    if is_call_type(node.kind()) {
        if imports::dynamic_import_js(ctx, node, caller_nid)? {
            for child in children(node) {
                walk_calls(ctx, child, caller_nid, extra_locals)?;
            }
            return Ok(());
        }

        let mut callee_name: Option<String> = None;
        let mut is_member_call = false;
        let mut is_this_field_call = false;
        let mut member_receiver: Option<String> = None;

        let mut func_node = node.child_by_field_name("function");
        if func_node.is_none() && node.kind() == "new_expression" {
            func_node = node.child_by_field_name("constructor");
        }
        if let Some(fnode) = func_node {
            match fnode.kind() {
                "identifier" => callee_name = Some(ctx.text(fnode)?.to_string()),
                "member_expression" => {
                    is_member_call = true;
                    if let Some(attr) = fnode.child_by_field_name("property") {
                        callee_name = Some(ctx.text(attr)?.to_string());
                    }
                    // A simple-identifier receiver (`ClassName.method()`) is kept
                    // so cross-file resolution can bind qualified class-method
                    // calls (#1446). Chained receivers are skipped UNLESS the
                    // chain is `this.field.method()` (#1316).
                    let obj = fnode.child_by_field_name("object");
                    match obj {
                        Some(o) if o.kind() == "identifier" => {
                            member_receiver = Some(ctx.text(o)?.to_string());
                        }
                        Some(o) if o.kind() == "member_expression" => {
                            if let Some(inner_obj) = o.child_by_field_name("object") {
                                if inner_obj.kind() == "this" {
                                    if let Some(inner_prop) = o.child_by_field_name("property") {
                                        member_receiver =
                                            Some(ctx.text(inner_prop)?.to_string());
                                        is_this_field_call = true;
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                // `f()()`, `(await x)()`, a parenthesized callee: Python reads the
                // node's whole text as the callee name.
                _ => callee_name = Some(ctx.text(fnode)?.to_string()),
            }
        }

        let named_callee = callee_name
            .as_deref()
            .filter(|c| !c.is_empty() && !pat::is_builtin_global(c));
        if let Some(callee) = named_callee.map(|s| s.to_string()) {
            // A capitalized or `this.field` receiver defers to the receiver-typed
            // cross-file resolver: a bare method-name match ignores the receiver's
            // declared type and mis-binds to an unrelated same-named method.
            // `member_receiver[:1].isupper()`. Python's `str.isupper` and Rust's
            // `char::is_uppercase` agree on ASCII and *mostly* agree beyond it, so
            // a non-ASCII receiver defers rather than betting on the overlap --
            // getting it wrong flips a whole call between a `calls` edge and a
            // `raw_call`, which is a silent graph difference, not an error.
            let receiver_upper = match member_receiver.as_deref() {
                None => false,
                Some(r) => match r.chars().next() {
                    None => false,
                    Some(c) if !c.is_ascii() => return Err("non_ascii_receiver"),
                    Some(c) => c.is_ascii_uppercase(),
                },
            };
            let defers =
                is_member_call && member_receiver.is_some() && (receiver_upper || is_this_field_call);
            let tgt_nid = if defers {
                None
            } else {
                ctx.label_to_nid.get(&callee).cloned()
            };
            match tgt_nid {
                Some(tgt) if tgt != caller_nid => {
                    let pair = (caller_nid.to_string(), tgt.clone());
                    if ctx.seen_call_pairs.insert(pair) {
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
                Some(_) => {} // self-call: no edge, and no raw_call either
                None => {
                    let line = line_of(node);
                    let mut entry = vec![
                        ("caller_nid", Val::S(caller_nid.to_string())),
                        ("callee", Val::S(callee.clone())),
                        ("is_member_call", Val::B(is_member_call)),
                        ("source_file", Val::S(ctx.str_path.to_string())),
                        ("source_location", Val::S(format!("L{line}"))),
                    ];
                    // Python writes `"receiver": swift_receiver or member_receiver`:
                    // the key is always present, with an explicit None when no
                    // receiver was captured.
                    entry.push((
                        "receiver",
                        match &member_receiver {
                            Some(r) => Val::S(r.clone()),
                            None => Val::None,
                        },
                    ));
                    ctx.raw_calls.push(entry);
                }
            }
        }

        // A callback passed BY NAME (`arr.map(fn)`, `setTimeout(fn)`). Positional
        // identifier arguments only: an inline arrow is a direct definition, not a
        // by-name reference. Runs regardless of whether the callee resolved.
        if let Some(args_node) = node.child_by_field_name("arguments") {
            let mut enclosing = ctx
                .local_bound_names
                .get(caller_nid)
                .cloned()
                .unwrap_or_default();
            enclosing.extend(extra_locals.iter().cloned());
            for arg in children(args_node) {
                if arg.kind() == "identifier" {
                    emit_indirect_ref(ctx, arg, caller_nid, &enclosing, "argument")?;
                }
            }
        }
    }

    if matches!(node.kind(), "object" | "array") {
        let mut enclosing = ctx
            .local_bound_names
            .get(caller_nid)
            .cloned()
            .unwrap_or_default();
        enclosing.extend(extra_locals.iter().cloned());
        for ident in pat::dispatch_value_idents(node) {
            emit_indirect_ref(ctx, ident, caller_nid, &enclosing, "collection")?;
        }
    }

    // `catch (e)` binds `e` for the rest of THIS subtree only, so the extended set
    // is passed down rather than assigned into the caller's.
    let mut extra_owned: Option<HashSet<String>> = None;
    if node.kind() == "catch_clause" {
        if let Some(param) = node.child_by_field_name("parameter") {
            let mut caught = HashSet::new();
            pat::collect_pattern_idents(param, ctx.src, &mut caught);
            let mut merged = extra_locals.clone();
            merged.extend(caught);
            extra_owned = Some(merged);
        }
    }
    let effective = extra_owned.as_ref().unwrap_or(extra_locals);

    for child in children(node) {
        walk_calls(ctx, child, caller_nid, effective)?;
    }
    Ok(())
}

/// `_emit_indirect_ref`.
fn emit_indirect_ref(
    ctx: &mut Ctx,
    ident: Node,
    scope_nid: &str,
    enclosing_locals: &HashSet<String>,
    context: &'static str,
) -> R<()> {
    if !matches!(
        ident.kind(),
        "identifier" | "shorthand_property_identifier"
    ) {
        return Ok(());
    }
    let name = ctx.text(ident)?.to_string();
    // A param / local binding names a local value, not the module function.
    if enclosing_locals.contains(&name) || name == "self" || name == "cls" {
        return Ok(());
    }
    // An import from outside the corpus binds the name for the whole module, so it
    // shadows in every scope.
    if ctx.js_external_imports.contains(&name) {
        return Ok(());
    }
    emit_indirect_by_name(ctx, &name, ident, scope_nid, context)
}

/// `_emit_indirect_by_name`.
fn emit_indirect_by_name(
    ctx: &mut Ctx,
    name: &str,
    loc_node: Node,
    scope_nid: &str,
    context: &'static str,
) -> R<()> {
    let line = line_of(loc_node);
    let ref_nid = ctx.label_to_nid.get(name).cloned();
    // Defer to the cross-file resolver when the name is not defined in this file,
    // or resolves to an import-surfaced FOREIGN symbol whose definition (and
    // callability) lives elsewhere.
    let foreign = match &ref_nid {
        None => true,
        Some(r) => {
            !ctx.callable_def_nids.contains(r)
                && ctx.nid_to_sf.get(r).map(String::as_str).unwrap_or("") != ctx.str_path
        }
    };
    if foreign {
        ctx.raw_calls.push(vec![
            ("caller_nid", Val::S(scope_nid.to_string())),
            ("callee", Val::S(name.to_string())),
            ("is_member_call", Val::B(false)),
            ("indirect", Val::B(true)),
            ("context", Val::Static(context)),
            ("source_file", Val::S(ctx.str_path.to_string())),
            ("source_location", Val::S(format!("L{line}"))),
        ]);
        return Ok(());
    }
    let ref_nid = ref_nid.ok_or("indirect_ref_missing")?;
    if ref_nid == scope_nid || !ctx.callable_def_nids.contains(&ref_nid) {
        return Ok(()); // self-ref, or a same-named LOCAL non-callable data node
    }
    // A class referenced as a VALUE is a descriptor, not an invocation (#2137).
    if ctx.callable_class_nids.contains(&ref_nid) {
        return Ok(());
    }
    if ctx
        .seen_call_pairs
        .contains(&(scope_nid.to_string(), ref_nid.clone()))
    {
        return Ok(()); // already a direct call to this target
    }
    if !ctx
        .seen_indirect_pairs
        .insert((scope_nid.to_string(), ref_nid.clone()))
    {
        return Ok(());
    }
    ctx.edges.push(EdgeRow {
        source: scope_nid.to_string(),
        target: ref_nid,
        relation: "indirect_call",
        fields: vec![
            ("context", Val::Static(context)),
            ("confidence", Val::Static("INFERRED")),
            // 0.85 = "strong inference": the symbol link is direct, but that it is
            // ever INVOKED is the inference (#2813).
            ("confidence_score", Val::F(0.85)),
            ("source_file", Val::S(ctx.str_path.to_string())),
            ("source_location", Val::S(format!("L{line}"))),
            ("weight", Val::F(1.0)),
        ],
    });
    Ok(())
}

/// The module-level dispatch-table scan (#1566): a function listed as a value in
/// a TOP-LEVEL object/array literal, or passed by name to a top-level call, is an
/// indirect dependency of the FILE.
pub fn scan_module_dispatch(ctx: &mut Ctx, root: Node) -> R<()> {
    let js_module_bound = pat::module_bound_names(root, ctx.src);
    scan(ctx, root, &js_module_bound)
}

fn scan(ctx: &mut Ctx, n: Node, bound: &HashSet<String>) -> R<()> {
    if pat::is_scope_boundary(n.kind()) {
        return Ok(()); // function / class bodies are walked separately
    }
    let file_nid = ctx.file_nid.clone();
    if matches!(n.kind(), "object" | "array") {
        for ident in pat::dispatch_value_idents(n) {
            emit_indirect_ref(ctx, ident, &file_nid, bound, "collection")?;
        }
    } else if is_call_type(n.kind()) {
        if let Some(margs) = n.child_by_field_name("arguments") {
            for marg in children(margs) {
                if marg.kind() == "identifier" {
                    emit_indirect_ref(ctx, marg, &file_nid, bound, "argument")?;
                }
            }
        }
    }
    for c in children(n) {
        scan(ctx, c, bound)?;
    }
    Ok(())
}

//! The shared call-graph pass: `_extract_generic`'s inner `walk_calls`.
//!
//! Three hook points, at the three places the Python has an `_is_<lang>` guard
//! that changes an OUTCOME rather than just adding an edge: extracting the
//! callee and receiver, deciding whether to defer, and stamping extra keys on
//! the raw_call.

use tree_sitter::Node;

use super::{has, CallInfo, Ctx, RecvTable, R};
use crate::js::ast::children;
use crate::js::emit::{EdgeRow, RawCall, Val};
use crate::py::helpers::BUILTIN_GLOBALS;

/// `callee_name not in _LANGUAGE_BUILTIN_GLOBALS`. Linear, deliberately -- see
/// `engine::has`.
fn is_builtin_global(name: &str) -> bool {
    BUILTIN_GLOBALS.contains(&name)
}

/// The generic accessor path, for a language whose `call_info` returns `None`.
///
/// Mirrors the `if func_node:` block: an `identifier` callee is read directly,
/// an accessor node yields the method name from `call_accessor_field` and a
/// simple-identifier receiver from `call_accessor_object_field`, and anything
/// else is read whole (which is how Java's `name` field would work if it went
/// through here).
fn generic_call_info(ctx: &Ctx, node: Node) -> R<CallInfo> {
    let mut info = CallInfo::default();
    let mut func_node = node.child_by_field_name(ctx.cfg.call_function_field);
    if func_node.is_none() && node.kind() == "new_expression" {
        func_node = node.child_by_field_name("constructor");
    }
    let func_node = match func_node {
        Some(f) => f,
        None => return Ok(info),
    };
    if func_node.kind() == "identifier" {
        info.callee_name = Some(ctx.text(func_node)?.to_string());
    } else if has(ctx.cfg.call_accessor_node_types, func_node.kind()) {
        info.is_member_call = true;
        if !ctx.cfg.call_accessor_field.is_empty() {
            if let Some(attr) = func_node.child_by_field_name(ctx.cfg.call_accessor_field) {
                info.callee_name = Some(ctx.text(attr)?.to_string());
            }
        }
        if !ctx.cfg.call_accessor_object_field.is_empty() {
            // Only a SIMPLE identifier receiver is captured (#1446); a chained
            // `a.b.method()` is left unresolved unless it is `this.field.m()`
            // (#1316), which the inner-accessor arm below recovers.
            let obj = func_node.child_by_field_name(ctx.cfg.call_accessor_object_field);
            if let Some(o) = obj {
                if o.kind() == "identifier" {
                    info.member_receiver = Some(ctx.text(o)?.to_string());
                } else if has(ctx.cfg.call_accessor_node_types, o.kind()) {
                    if let Some(inner) = o.child_by_field_name(ctx.cfg.call_accessor_object_field) {
                        if inner.kind() == "this" {
                            if let Some(prop) = o.child_by_field_name(ctx.cfg.call_accessor_field) {
                                info.member_receiver = Some(ctx.text(prop)?.to_string());
                                info.is_this_field_call = true;
                            }
                        }
                    }
                }
            }
        }
    } else {
        // "Try reading the node directly (e.g. Java name field is the callee)."
        info.callee_name = Some(ctx.text(func_node)?.to_string());
    }
    Ok(info)
}

pub fn walk_calls<'tree>(
    ctx: &mut Ctx<'_, 'tree>,
    node: Node<'tree>,
    caller_nid: &str,
    receiver_types: &RecvTable,
) -> R<()> {
    let hooks = ctx.cfg.hooks;

    // A nested function is its own caller; it has its own `function_bodies`
    // entry. The JS-only descend into untracked closures does not apply here.
    if has(ctx.cfg.function_boundary_types, node.kind()) {
        return Ok(());
    }

    if has(ctx.cfg.call_types, node.kind()) {
        let info = match hooks.call_info(ctx, node, caller_nid)? {
            Some(i) => i,
            None => generic_call_info(ctx, node)?,
        };

        if let Some(callee) = info.callee_name.clone() {
            if !is_builtin_global(&callee) {
                // The defer decision. `tgt_nid = None` routes the call to
                // cross-file receiver-typed resolution instead of binding it to
                // a bare name here.
                let generic_defer = info.is_member_call
                    && info
                        .member_receiver
                        .as_deref()
                        .map(|r| {
                            r.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                                || info.is_this_field_call
                        })
                        .unwrap_or(false);
                let tgt_nid = if hooks.defers(&info) || generic_defer {
                    None
                } else {
                    let looked_up = ctx.label_to_nid.get(&callee).cloned();
                    hooks.refine_target(ctx, &info, looked_up)
                };

                match tgt_nid {
                    Some(tgt) if tgt != caller_nid => {
                        let pair = (caller_nid.to_string(), tgt.clone());
                        if ctx.seen_call_pairs.insert(pair) {
                            let line = node.start_position().row + 1;
                            let sf = ctx.str_path.to_string();
                            ctx.edges.push(EdgeRow {
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
                    // `tgt_nid == caller_nid` emits nothing at all: the Python
                    // falls out of the `if` without reaching the raw_call arm.
                    Some(_) => {}
                    None => {
                        let line = node.start_position().row + 1;
                        // Key order is Python's dict-literal order and reaches
                        // the pickled result. `receiver` is written even when
                        // None; `swift_receiver or member_receiver` is the
                        // Python expression.
                        let receiver = info
                            .swift_receiver
                            .clone()
                            .or_else(|| info.member_receiver.clone());
                        let mut rc: RawCall = vec![
                            ("caller_nid", Val::S(caller_nid.to_string())),
                            ("callee", Val::S(callee.clone())),
                            ("is_member_call", Val::B(info.is_member_call)),
                            ("source_file", Val::S(ctx.str_path.to_string())),
                            ("source_location", Val::S(format!("L{line}"))),
                            (
                                "receiver",
                                match &receiver {
                                    Some(r) => Val::S(r.clone()),
                                    None => Val::None,
                                },
                            ),
                        ];
                        rc.extend(hooks.raw_call_extra(ctx, node, &info, receiver_types));
                        ctx.raw_calls.push(rc);
                    }
                }
            }
        }
        // Runs whatever the defer decision was, and even for a builtin-global
        // callee: the Python's helper / container-binding blocks sit after the
        // whole `if callee_name ...` chain, not inside it.
        hooks.after_call(ctx, node, caller_nid, &info)?;
    }

    hooks.walk_calls_extra(ctx, node, caller_nid)?;

    // The indirect-dispatch block is `if _is_python`, and `py/` is not on this
    // engine, so there is nothing between the call branch and the recursion.
    for child in children(node) {
        walk_calls(ctx, child, caller_nid, receiver_types)?;
    }
    Ok(())
}

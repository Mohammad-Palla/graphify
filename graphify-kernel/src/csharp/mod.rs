//! C# on the shared engine.
//!
//! The most hook-heavy language of the ten, and deliberately the second one
//! ported: it is the only user of `class_metadata`, the only one with a
//! POSITIONAL receiver table, and the only one that pushes `namespace_stack` /
//! `scope_stack`. If the engine's hook points were wrong anywhere, C# is where
//! it would show -- which is the point of doing it at language 2 rather than 7.
//!
//! Three things the engine grew to hold it, all of which are `_extract_generic`
//! behaviour that Java simply never reached:
//!
//!   * `Ctx::namespace_stack` / `scope_stack`, which `add_node` folds into every
//!     node's metadata and every id minted from the file stem;
//!   * `Val::Meta` / `Val::List` plus `engine::meta::sanitize`, because C# is the
//!     first language to emit a `metadata` block at all;
//!   * `RecvTable::Scoped`, because a C# receiver's type depends on WHERE in the
//!     method the call is (#2472), not just on its name.

use std::collections::HashSet;

use pyo3::prelude::*;
use tree_sitter::Node;

use crate::engine::{CallInfo, Ctx, EngineConfig, Handled, LangHooks, RecvTable, R};
use crate::js::ast::children;
use crate::js::emit::Val;
use crate::Outcome;

pub mod calls;
pub mod helpers;
pub mod imports;

use helpers::{
    attribute_names, classify_base, collect_type_refs, namespace_id, namespace_name,
    pre_scan_interfaces, read_type_name, receiver_type_name, type_parameters_in_scope, Role,
    TypeRef,
};

struct CSharp;

/// The `{"ref_token": …, "qualified": True, "ref_qualifier": …}` block that every
/// C# `references` edge carries, in the Python's insertion order. The two
/// optional keys are appended only when truthy.
fn ref_metadata(name: &str, qualified: bool, qualifier: &str) -> Vec<(&'static str, Val)> {
    let mut md: Vec<(&'static str, Val)> = vec![("ref_token", Val::S(name.to_string()))];
    if qualified {
        md.push(("qualified", Val::B(true)));
    }
    if !qualifier.is_empty() {
        md.push(("ref_qualifier", Val::S(qualifier.to_string())));
    }
    md
}

/// The shape repeated by the field, property, parameter, return-type and
/// primary-constructor blocks: one `references` edge per collected type, with
/// `generic_arg` overriding the block's own context.
fn emit_type_refs(
    ctx: &mut Ctx,
    owner_nid: &str,
    refs: Vec<TypeRef>,
    type_ctx: &'static str,
    line: usize,
) -> R<()> {
    for (ref_name, role, qualified, qualifier) in refs {
        let c = if role == Role::GenericArg { "generic_arg" } else { type_ctx };
        let target = ctx.ensure_named_node(&ref_name, line)?;
        if target != owner_nid {
            let md = ref_metadata(&ref_name, qualified, &qualifier);
            ctx.add_edge_meta(owner_nid, &target, "references", line, Some(c), md);
        }
    }
    Ok(())
}

impl LangHooks for CSharp {
    fn prescan<'tree>(&self, ctx: &Ctx<'_, 'tree>, root: Node<'tree>) -> R<HashSet<String>> {
        pre_scan_interfaces(ctx, root)
    }

    fn import_handler<'tree>(&self, ctx: &mut Ctx<'_, 'tree>, node: Node<'tree>) -> R<()> {
        imports::import_csharp(ctx, node)
    }

    fn class_metadata<'tree>(
        &self,
        ctx: &Ctx<'_, 'tree>,
        node: Node<'tree>,
        parent_class_nid: Option<&str>,
    ) -> R<Vec<(&'static str, Val)>> {
        let mut md: Vec<(&'static str, Val)> = Vec::new();
        if parent_class_nid.is_some() {
            md.push(("is_nested_type", Val::B(true)));
        }
        // #2332: `partial class Foo` split across files mints one node per file
        // (the id carries the file stem). Stamp the halves so the corpus-level
        // `_merge_csharp_partial_class_nodes` pass can collapse them.
        const PARTIAL_KINDS: &[&str] = &[
            "class_declaration",
            "struct_declaration",
            "interface_declaration",
            "record_declaration",
        ];
        if PARTIAL_KINDS.contains(&node.kind()) {
            let mut is_partial = false;
            for c in children(node) {
                if c.kind() == "modifier" && ctx.text(c)? == "partial" {
                    is_partial = true;
                    break;
                }
            }
            if is_partial {
                md.push(("is_partial", Val::B(true)));
            }
        }
        Ok(md)
    }

    fn on_class<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        class_nid: &str,
        _class_name: &str,
        line: usize,
    ) -> R<()> {
        // ── inheritance / interface implementation via base_list ────────────
        let type_params = type_parameters_in_scope(ctx, node)?;
        for child in children(node) {
            if child.kind() != "base_list" {
                continue;
            }
            for sub in children(child) {
                if !matches!(sub.kind(), "identifier" | "generic_name" | "qualified_name") {
                    continue;
                }
                let (base, qualified, qualifier) = match read_type_name(ctx, Some(sub))? {
                    Some(i) => i,
                    None => continue,
                };
                if base.is_empty() || type_params.contains(&base) {
                    continue;
                }
                let base_nid = ctx.ensure_scoped_stub(&base)?;
                let relation = classify_base(&base, &ctx.prescan)?;
                let md = ref_metadata(&base, qualified, &qualifier);
                ctx.add_edge_meta(class_nid, &base_nid, relation, line, None, md);
                if sub.kind() == "generic_name" {
                    for tal in children(sub) {
                        if tal.kind() != "type_argument_list" {
                            continue;
                        }
                        for arg in children(tal) {
                            if !arg.is_named() {
                                continue;
                            }
                            let mut refs: Vec<TypeRef> = Vec::new();
                            collect_type_refs(ctx, Some(arg), true, &mut refs, Some(&type_params))?;
                            // No `!= class_nid` guard here, unlike every other
                            // C# references site. Python's omission, kept.
                            for (ref_name, _role, ref_qualified, ref_qualifier) in refs {
                                let target = ctx.ensure_named_node(&ref_name, line)?;
                                let md = ref_metadata(&ref_name, ref_qualified, &ref_qualifier);
                                ctx.add_edge_meta(
                                    class_nid,
                                    &target,
                                    "references",
                                    line,
                                    Some("generic_arg"),
                                    md,
                                );
                            }
                        }
                    }
                }
            }
        }

        // ── primary constructor (`class Foo(IBar bar)`, C# 12+) ─────────────
        // Its dependencies are declared on the type declaration itself, so
        // neither the field nor the property handler ever sees them: the
        // parameter type got no references edge, and the name was never
        // registered for receiver typing, so `bar.Baz()` lost its calls edge
        // too. The list is an UNNAMED child, so `child_by_field_name` misses it.
        if matches!(
            node.kind(),
            "class_declaration" | "record_declaration" | "struct_declaration"
        ) {
            let type_params = type_parameters_in_scope(ctx, node)?;
            for c in children(node) {
                if c.kind() != "parameter_list" {
                    continue;
                }
                for param in children(c) {
                    if param.kind() != "parameter" {
                        continue;
                    }
                    let ptype = match param.child_by_field_name("type") {
                        Some(p) => p,
                        None => continue,
                    };
                    let pname = param.child_by_field_name("name");
                    let p_line = param.start_position().row + 1;
                    // Receiver binding mirrors the field rule: Pascal-case only,
                    // and never a bare type parameter (`T item`).
                    let recv = receiver_type_name(ctx, Some(ptype))?;
                    if let (Some(pn), Some(recv)) = (pname, recv) {
                        if !type_params.contains(&recv) {
                            let key = ctx.text(pn)?.to_string();
                            ctx.field_types
                                .entry(class_nid.to_string())
                                .or_default()
                                .insert(key, recv);
                        }
                    }
                    let mut refs: Vec<TypeRef> = Vec::new();
                    collect_type_refs(ctx, Some(ptype), false, &mut refs, Some(&type_params))?;
                    emit_type_refs(ctx, class_nid, refs, "field", p_line)?;
                }
            }
        }
        Ok(())
    }

    fn before_function<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        parent_class_nid: Option<&str>,
    ) -> R<Handled> {
        let parent = match parent_class_nid {
            Some(p) => p.to_string(),
            None => return Ok(Handled::No),
        };
        match node.kind() {
            "field_declaration" => {
                let mut type_node = node.child_by_field_name("type");
                if type_node.is_none() {
                    for child in children(node) {
                        if child.kind() == "variable_declaration" {
                            type_node = child.child_by_field_name("type");
                            if type_node.is_some() {
                                break;
                            }
                        }
                    }
                }
                if let Some((type_name, _q, _qf)) = read_type_name(ctx, type_node)? {
                    let scope_for_params = type_node.unwrap_or(node);
                    let type_params = type_parameters_in_scope(ctx, scope_for_params)?;
                    if type_name.is_empty() || type_params.contains(&type_name) {
                        return Ok(Handled::Yes);
                    }
                    // Record the declared type for the receiver tables (#2299).
                    // Pascal-case only: a primitive owns no resolvable method.
                    if helpers::first_is_upper(&type_name)? {
                        let mut names: Vec<String> = Vec::new();
                        for child in children(node) {
                            if child.kind() != "variable_declaration" {
                                continue;
                            }
                            for declarator in children(child) {
                                if declarator.kind() != "variable_declarator" {
                                    continue;
                                }
                                let name_node =
                                    declarator.child_by_field_name("name").or_else(|| {
                                        children(declarator)
                                            .into_iter()
                                            .find(|g| g.kind() == "identifier")
                                    });
                                if let Some(n) = name_node {
                                    names.push(ctx.text(n)?.to_string());
                                }
                            }
                        }
                        let fields = ctx.field_types.entry(parent.clone()).or_default();
                        for name in names {
                            fields.insert(name, type_name.clone());
                        }
                    }
                    let line = node.start_position().row + 1;
                    // The WHOLE type expression, so `Box<Widget>` yields the Box
                    // field ref AND the Widget generic_arg ref.
                    let mut refs: Vec<TypeRef> = Vec::new();
                    collect_type_refs(ctx, type_node, false, &mut refs, Some(&type_params))?;
                    emit_type_refs(ctx, &parent, refs, "field", line)?;
                }
                Ok(Handled::Yes)
            }
            "property_declaration" => {
                // A property becomes a node, the way a C++ data member does.
                // FIELDS stay out: the id recipe casefolds and strips leading
                // underscores, so `_count` and `Count` normalize to one id and
                // emitting both would hide the public member behind the private
                // backing field (#3006).
                if let Some(prop_node_name) = node.child_by_field_name("name") {
                    let property_name = ctx.text(prop_node_name)?.to_string();
                    if !property_name.is_empty() {
                        let property_line = node.start_position().row + 1;
                        let property_nid = ctx.mkid(&[&parent, &property_name])?;
                        if !ctx.seen_ids.contains(&property_nid) {
                            ctx.add_node(&property_nid, &property_name, property_line);
                            ctx.add_edge_ctx(
                                &parent,
                                &property_nid,
                                "defines",
                                property_line,
                                "field",
                            );
                        }
                    }
                }
                if let Some(type_node) = node.child_by_field_name("type") {
                    // Unlike a field, a property exposes its type directly (no
                    // variable_declaration wrapper).
                    let prop_name_node = node.child_by_field_name("name");
                    let prop_type = receiver_type_name(ctx, Some(type_node))?;
                    if let (Some(pn), Some(pt)) = (prop_name_node, prop_type) {
                        let key = ctx.text(pn)?.to_string();
                        ctx.field_types.entry(parent.clone()).or_default().insert(key, pt);
                    }
                    let line = node.start_position().row + 1;
                    // No `skip` argument here, unlike the field block: the
                    // Python lets `_csharp_collect_type_refs` derive the type
                    // parameters from the type node itself.
                    let mut refs: Vec<TypeRef> = Vec::new();
                    collect_type_refs(ctx, Some(type_node), false, &mut refs, None)?;
                    emit_type_refs(ctx, &parent, refs, "field", line)?;
                }
                Ok(Handled::Yes)
            }
            _ => Ok(Handled::No),
        }
    }

    fn on_function<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        func_nid: &str,
        _func_name: &str,
        line: usize,
        _parent_class_nid: Option<&str>,
    ) -> R<()> {
        let type_params = type_parameters_in_scope(ctx, node)?;
        if let Some(params_node) = node.child_by_field_name("parameters") {
            for p in children(params_node) {
                if p.kind() != "parameter" {
                    continue;
                }
                let mut refs: Vec<TypeRef> = Vec::new();
                collect_type_refs(
                    ctx,
                    p.child_by_field_name("type"),
                    false,
                    &mut refs,
                    Some(&type_params),
                )?;
                emit_type_refs(ctx, func_nid, refs, "parameter_type", line)?;
            }
        }
        // C#'s return type is the `returns` field, not `type`.
        if let Some(return_node) = node.child_by_field_name("returns") {
            let mut refs: Vec<TypeRef> = Vec::new();
            collect_type_refs(ctx, Some(return_node), false, &mut refs, Some(&type_params))?;
            emit_type_refs(ctx, func_nid, refs, "return_type", line)?;
        }
        for (attr_name, qualified, qualifier) in attribute_names(ctx, node)? {
            let target = ctx.ensure_named_node(&attr_name, line)?;
            if target != func_nid {
                let md = ref_metadata(&attr_name, qualified, &qualifier);
                ctx.add_edge_meta(func_nid, &target, "references", line, Some("attribute"), md);
            }
        }
        Ok(())
    }

    fn extra_walk<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        parent_class_nid: Option<&str>,
    ) -> R<Handled> {
        let t = node.kind();
        if t == "enum_member_declaration" {
            let parent = match parent_class_nid {
                Some(p) => p.to_string(),
                None => return Ok(Handled::No),
            };
            // `enum_declaration` is a class_type, so the enum type was a node
            // but its members were not, leaving it a leaf: "which value does
            // this consumer branch on" had no answer.
            let name_node = match node.child_by_field_name("name") {
                Some(n) => n,
                None => return Ok(Handled::Yes),
            };
            let member_name = ctx.text(name_node)?.to_string();
            if member_name.is_empty() {
                return Ok(Handled::Yes);
            }
            let line = node.start_position().row + 1;
            let member_nid = ctx.mkid(&[&parent, &member_name])?;
            // C# is case-sensitive so `enum E { Value, value }` is legal, but
            // the id recipe casefolds -- the FIRST declaration keeps the node.
            if !ctx.seen_ids.contains(&member_nid) {
                ctx.add_node(&member_nid, &member_name, line);
                ctx.add_edge(&parent, &member_nid, "case_of", line);
            }
            return Ok(Handled::Yes);
        }

        if t == "namespace_declaration" {
            let ns_name = namespace_name(ctx, node)?;
            let mut pushed = false;
            if !ns_name.is_empty() {
                ctx.namespace_stack.push(ns_name);
                ctx.scope_stack.push(format!("s{}", node.start_byte()));
                pushed = true;
                let ns_label = ctx.ns();
                let ns_nid = namespace_id(&ns_label);
                let line = node.start_position().row + 1;
                ctx.add_node_full(
                    &ns_nid,
                    &ns_label,
                    line,
                    Some("namespace"),
                    vec![("kind", Val::Static("csharp_namespace"))],
                );
                let f = ctx.file_nid.clone();
                ctx.add_edge(&f, &ns_nid, "contains", line);
            }
            // The Python's try/finally: the pop happens even if the recursion
            // raises. Here the `?` is hoisted out so the pop is unconditional.
            let mut result = Ok(());
            if let Some(body) = node.child_by_field_name("body") {
                for child in children(body) {
                    result = crate::engine::walk::walk(ctx, child, parent_class_nid);
                    if result.is_err() {
                        break;
                    }
                }
            }
            if pushed {
                ctx.namespace_stack.pop();
                ctx.scope_stack.pop();
            }
            result?;
            return Ok(Handled::Yes);
        }

        if t == "file_scoped_namespace_declaration" {
            // `namespace Foo;` -- no body, and NOTHING is popped: the namespace
            // is in scope for the rest of the file, so the push outlives this
            // call. Returning Handled means the declaration node itself is not
            // recursed into; its siblings are walked with the stack still set.
            let ns_name = namespace_name(ctx, node)?;
            if !ns_name.is_empty() {
                ctx.namespace_stack.push(ns_name);
                ctx.scope_stack.push(format!("s{}", node.start_byte()));
                let ns_label = ctx.ns();
                let ns_nid = namespace_id(&ns_label);
                let line = node.start_position().row + 1;
                ctx.add_node_full(
                    &ns_nid,
                    &ns_label,
                    line,
                    Some("namespace"),
                    vec![("kind", Val::Static("csharp_namespace"))],
                );
                let f = ctx.file_nid.clone();
                ctx.add_edge(&f, &ns_nid, "contains", line);
            }
            return Ok(Handled::Yes);
        }

        if t.starts_with("preproc_") {
            if let Some(parent) = parent_class_nid {
                // tree-sitter wraps members guarded by `#if`/`#else` in preproc_*
                // nodes. They are conditional CONTAINERS, not ownership scopes:
                // dropping parent_class_nid here makes guarded methods look
                // file-level (#2631).
                let parent = parent.to_string();
                for child in children(node) {
                    crate::engine::walk::walk(ctx, child, Some(&parent))?;
                }
                return Ok(Handled::Yes);
            }
        }
        Ok(Handled::No)
    }

    fn call_info<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        caller_nid: &str,
    ) -> R<Option<CallInfo>> {
        if node.kind() == "object_creation_expression" {
            return Ok(Some(calls::object_creation_info(ctx, node)?));
        }
        if node.kind() != "invocation_expression" {
            return Ok(None);
        }
        let fn_node = node.child_by_field_name("function");
        let info = calls::invocation_info(ctx, node, fn_node)?;

        // #2911: a `references[generic_arg]` edge per type argument at the call
        // site (`recv.Do<T>()`, the `services.AddScoped<ISvc, Impl>()` DI shape,
        // static `Foo<IBar>()`). The property/return/parameter branches already
        // walk their declared type for the same reason; without this the type
        // arguments never became nodes and dependency edges were silently erased.
        if let Some(fn_node) = fn_node {
            let mut call_tal: Option<Node> = None;
            if fn_node.kind() == "member_access_expression" {
                if let Some(ma_name) = fn_node.child_by_field_name("name") {
                    if ma_name.kind() == "generic_name" {
                        call_tal = children(ma_name)
                            .into_iter()
                            .find(|c| c.kind() == "type_argument_list");
                    }
                }
            } else if fn_node.kind() == "generic_name" {
                call_tal = children(fn_node)
                    .into_iter()
                    .find(|c| c.kind() == "type_argument_list");
            }
            if let Some(call_tal) = call_tal {
                let call_type_params = type_parameters_in_scope(ctx, node)?;
                let call_line = node.start_position().row + 1;
                for call_arg in children(call_tal) {
                    if !call_arg.is_named() {
                        continue;
                    }
                    let mut refs: Vec<TypeRef> = Vec::new();
                    collect_type_refs(
                        ctx,
                        Some(call_arg),
                        true,
                        &mut refs,
                        Some(&call_type_params),
                    )?;
                    for (ref_name, _role, qualified, qualifier) in refs {
                        let target = ctx.ensure_named_node(&ref_name, call_line)?;
                        if target == caller_nid {
                            continue;
                        }
                        let md = ref_metadata(&ref_name, qualified, &qualifier);
                        ctx.add_edge_meta(
                            caller_nid,
                            &target,
                            "references",
                            call_line,
                            Some("generic_arg"),
                            md,
                        );
                    }
                }
            }
        }
        Ok(Some(info))
    }

    /// ANY member call with a captured receiver defers to the receiver-typed
    /// resolver: a bare method-name match ignores the receiver's declared type
    /// and mis-binds to an unrelated same-named method (#1609). Broader than
    /// Python's rule -- the receiver may be lowercase (`_server.Save()`).
    fn defers(&self, info: &CallInfo) -> bool {
        info.is_member_call && info.member_receiver.as_deref().is_some_and(|r| !r.is_empty())
    }

    fn refine_target(&self, ctx: &Ctx, info: &CallInfo, tgt: Option<String>) -> Option<String> {
        let tgt = tgt?;
        if info.qualified_prefix.is_some()
            && ctx.nid_to_sf.get(&tgt).map(|s| s.is_empty()).unwrap_or(true)
        {
            return None;
        }
        Some(tgt)
    }

    fn raw_call_extra<'tree>(
        &self,
        _ctx: &Ctx<'_, 'tree>,
        node: Node<'tree>,
        info: &CallInfo,
        receiver_types: &RecvTable,
    ) -> Vec<(&'static str, Val)> {
        let mut out: Vec<(&'static str, Val)> = vec![("lang", Val::Static("csharp"))];
        if let Some(p) = &info.qualified_prefix {
            out.push(("qualified_prefix", Val::S(p.clone())));
        }
        // Position-aware (#2472): the receiver's type comes from the binding
        // that is lexically visible AT THIS CALL, not from a method-wide map.
        if let Some(rt) = info
            .member_receiver
            .as_deref()
            .and_then(|r| receiver_types.type_of(r, node.start_byte()))
        {
            out.push(("receiver_type", Val::S(rt.to_string())));
        }
        out
    }
}

static HOOKS: CSharp = CSharp;

pub static CONFIG: EngineConfig = EngineConfig {
    language: "csharp",
    grammar: || tree_sitter_c_sharp::LANGUAGE.into(),
    class_types: &[
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "struct_declaration",
        "record_declaration",
    ],
    function_types: &["method_declaration"],
    import_types: &["using_directive"],
    // `object_creation_expression` joins the invocation node so `new Foo(...)`
    // links the constructing method to Foo, the way Java has since #1373.
    call_types: &["invocation_expression", "object_creation_expression"],
    function_boundary_types: &["method_declaration"],
    static_prop_types: &[],
    helper_fn_names: &[],
    container_bind_methods: &[],
    event_listener_properties: &[],
    name_field: "name",
    name_fallback_child_types: &[],
    body_field: "body",
    body_fallback_child_types: &["declaration_list"],
    call_function_field: "function",
    call_accessor_node_types: &["member_access_expression"],
    call_accessor_field: "name",
    call_accessor_object_field: "",
    function_label_parens: true,
    resolve_function_name: None,
    type_table_key: None,
    hooks: &HOOKS,
};

pub fn walk_csharp<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
    _res: &crate::Resolvers<'py>,
) -> PyResult<Outcome<'py>> {
    crate::engine::run(py, &CONFIG, path, source, calls::method_receiver_types, None)
}

//! Java on the shared engine: a [`EngineConfig`] plus the nine hook slots.
//!
//! Everything structural -- the class branch, the function branch, name and
//! body resolution, the call branch, the raw_call shape -- lives in
//! `engine::walk` / `engine::calls` and is shared. What is here is exactly what
//! the Python guards on `_is_java`, and nothing else.
//!
//! `_JAVA_CONFIG` leaves `static_prop_types`, `helper_fn_names`,
//! `container_bind_methods`, `event_listener_properties`, both fallback tuples
//! and `call_accessor_node_types` empty, and sets no `resolve_function_name_fn`
//! or `sanitize_symbol_name_fn`. Those appear below as empty slices / absent
//! hooks rather than as dead code.
//!
//! `_java_extra_walk` is NOT wired through `config.extra_walk_fn` (which is
//! None) -- the Python calls it from `walk` behind an `_is_java` guard. Here it
//! is the `extra_walk` hook, which is the same position.

use std::collections::HashSet;

use pyo3::prelude::*;
use tree_sitter::Node;

use crate::engine::{CallInfo, Ctx, EngineConfig, Handled, LangHooks, RecvTable, R};
use crate::js::ast::children;
use crate::js::emit::Val;
use crate::Outcome;

pub mod calls;
pub mod consts;
pub mod helpers;
pub mod imports;

use helpers::Role;

struct Java;

/// `_emit_java_parent`: resolve a base type to a node, minting a SOURCELESS stub
/// when it is not already known, then link it.
///
/// NOT `ensure_named_node`: the stub carries no `origin_file` key and the scoped
/// probe omits the empty namespace part. The two produce the same id string here
/// but different node SHAPES, so they stay separate emitters.
fn emit_parent(
    ctx: &mut Ctx,
    class_nid: &str,
    base_name: &str,
    rel: &'static str,
    at_line: usize,
) -> R<()> {
    if base_name.is_empty() {
        return Ok(());
    }
    let base_nid = ctx.ensure_parent_node(base_name)?;
    ctx.add_edge(class_nid, &base_nid, rel, at_line);
    Ok(())
}

/// `_emit_java_parent_type`: the FIRST `type` role becomes the parent link,
/// every `generic_arg` becomes a `references` edge.
fn emit_parent_type(
    ctx: &mut Ctx,
    class_nid: &str,
    type_node: Option<Node>,
    rel: &'static str,
    at_line: usize,
) -> R<()> {
    let mut refs = Vec::new();
    helpers::collect_type_refs(ctx, type_node, false, &mut refs, None, false)?;
    let refs: Vec<(String, Role)> = refs.into_iter().map(|(n, r)| (n.to_string(), r)).collect();
    let mut parent_emitted = false;
    for (ref_name, role) in refs {
        if role == Role::Type && !parent_emitted {
            emit_parent(ctx, class_nid, &ref_name, rel, at_line)?;
            parent_emitted = true;
        } else if role == Role::GenericArg {
            let target = ctx.ensure_named_node(&ref_name, at_line)?;
            if target != class_nid {
                ctx.add_edge_ctx(class_nid, &target, "references", at_line, "generic_arg");
            }
        }
    }
    Ok(())
}

/// `references` edges for every type in `type_node`, mapping the `type` role to
/// `type_ctx` and `generic_arg` to `"generic_arg"`. Shared by the field,
/// parameter, return-type and record-component blocks, which differ only in that
/// one context string.
fn emit_type_refs(
    ctx: &mut Ctx,
    owner_nid: &str,
    type_node: Option<Node>,
    type_ctx: &'static str,
    line: usize,
    preserve_qualified: bool,
) -> R<()> {
    let mut refs = Vec::new();
    helpers::collect_type_refs(ctx, type_node, false, &mut refs, None, preserve_qualified)?;
    let refs: Vec<(String, Role)> = refs.into_iter().map(|(n, r)| (n.to_string(), r)).collect();
    for (ref_name, role) in refs {
        let c = if role == Role::GenericArg { "generic_arg" } else { type_ctx };
        let target = ctx.ensure_named_node(&ref_name, line)?;
        if target != owner_nid {
            ctx.add_edge_ctx(owner_nid, &target, "references", line, c);
        }
    }
    Ok(())
}

/// The annotation block, shared by the class and function hooks -- and by Groovy.
///
/// Java's two call sites differ in ONE way and it is cosmetic: at class level
/// the dotted-name substitution reads `if "." in anno_raw and _is_java:
/// anno_name = anno_raw`, at function level `anno_raw if "." in anno_raw else
/// anno_name`. Both reduce to the same choice for Java, so one helper serves
/// both -- stated because the asymmetry invites "fixing" one to match the other.
///
/// `dotted` is where GROOVY diverges, and it is not cosmetic. The class-level
/// guard is `and _is_java`, so Groovy takes the bare `anno_name` even for
/// `@org.pkg.Foo`. The Python says why: `_resolve_java_type_references` maps
/// internal FQNs back to real nodes for Java (#2504) and "Groovy has no such
/// resolver pass, so it keeps the legacy bare-name stub". Passing `true` here
/// for Groovy would silently retarget every inline-qualified annotation edge.
pub(crate) fn emit_annotations(
    ctx: &mut Ctx,
    owner_nid: &str,
    decl: Node,
    line: usize,
    dotted: bool,
) -> R<()> {
    let mut targets: HashSet<String> = HashSet::new();
    let names: Vec<(String, String)> = helpers::annotation_names(ctx, decl)?
        .into_iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
    for (anno_name, anno_raw) in names {
        let chosen = if dotted && anno_raw.contains('.') { &anno_raw } else { &anno_name };
        let target = ctx.ensure_named_node(chosen, line)?;
        if target != owner_nid && targets.insert(target.clone()) {
            ctx.add_edge_ctx(owner_nid, &target, "references", line, "attribute");
        }
    }
    let lits: Vec<String> = helpers::annotation_class_literal_refs(ctx, decl)?
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    for ref_name in lits {
        let target = ctx.ensure_named_node(&ref_name, line)?;
        if target != owner_nid && targets.insert(target.clone()) {
            ctx.add_edge_ctx(owner_nid, &target, "references", line, "attribute");
        }
    }
    Ok(())
}

/// `extends` / `implements` / interface-`extends`, the branch guarded in the
/// Python by `config.ts_module in ("tree_sitter_java", "tree_sitter_groovy")`.
///
/// Shared with Groovy rather than copied, because it is literally the same
/// branch: a module-name guard, not an `_is_java` one. That distinction is the
/// same one that cost 241 DIVERGENT files on C, where an `_is_<lang>` grep
/// could not see a `config.ts_module in (...)` guard.
pub(crate) fn emit_inheritance(ctx: &mut Ctx, node: Node, class_nid: &str, line: usize) -> R<()> {
        // extends
        if let Some(sup) = node.child_by_field_name("superclass") {
            if let Some(sub) = children(sup).into_iter().find(|c| c.is_named()) {
                emit_parent_type(ctx, class_nid, Some(sub), "inherits", line)?;
            }
        }
        // implements
        if let Some(ifs) = node.child_by_field_name("interfaces") {
            for sub in children(ifs) {
                if sub.kind() != "type_list" {
                    continue;
                }
                for tid in children(sub) {
                    if tid.is_named() {
                        emit_parent_type(ctx, class_nid, Some(tid), "implements", line)?;
                    }
                }
            }
        }
        // interface extends
        if node.kind() == "interface_declaration" {
            for child in children(node) {
                if child.kind() != "extends_interfaces" {
                    continue;
                }
                for sub in children(child) {
                    if sub.kind() != "type_list" {
                        continue;
                    }
                    for tid in children(sub) {
                        if tid.is_named() {
                            emit_parent_type(ctx, class_nid, Some(tid), "inherits", line)?;
                        }
                    }
                }
            }
        }
    Ok(())
}

impl LangHooks for Java {
    fn import_handler<'tree>(&self, ctx: &mut Ctx<'_, 'tree>, node: Node<'tree>) -> R<()> {
        imports::import_java(ctx, node)
    }

    fn on_class<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        class_nid: &str,
        _class_name: &str,
        line: usize,
    ) -> R<()> {
        emit_inheritance(ctx, node, class_nid, line)?;

        emit_annotations(ctx, class_nid, node, line, true)?;

        // record components: the reference line is the COMPONENT's, not the
        // record's.
        if node.kind() == "record_declaration" {
            if let Some(components) = node.child_by_field_name("parameters") {
                for component in children(components) {
                    let type_node = match component.kind() {
                        "formal_parameter" => component.child_by_field_name("type"),
                        "spread_parameter" => children(component).into_iter().find(|c| {
                            c.is_named() && !matches!(c.kind(), "modifiers" | "variable_declarator")
                        }),
                        _ => continue,
                    };
                    let component_line = component.start_position().row + 1;
                    emit_type_refs(ctx, class_nid, type_node, "field", component_line, false)?;
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
                let type_node = match node.child_by_field_name("type") {
                    Some(t) => t,
                    // Python returns from INSIDE `if type_node is not None`, so a
                    // field with no type field falls through to the branches below.
                    None => return Ok(Handled::No),
                };
                if let Some(receiver_type) = helpers::receiver_type_name(ctx, Some(type_node))? {
                    let receiver_type = receiver_type.to_string();
                    let names: Vec<String> = helpers::declarator_names(ctx, node)?
                        .into_iter()
                        .map(|s| s.to_string())
                        .collect();
                    let fields = ctx.field_types.entry(parent.clone()).or_default();
                    for field_name in names {
                        fields.insert(field_name, receiver_type.clone());
                    }
                }
                let line = node.start_position().row + 1;
                emit_type_refs(ctx, &parent, Some(type_node), "field", line, false)?;
                Ok(Handled::Yes)
            }
            "annotation_type_element_declaration" => {
                let line = node.start_position().row + 1;
                // `preserve_qualified=True` here and nowhere else in the Java path.
                emit_type_refs(
                    ctx,
                    &parent,
                    node.child_by_field_name("type"),
                    "return_type",
                    line,
                    true,
                )?;
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
        if let Some(params_node) = node.child_by_field_name("parameters") {
            for p in children(params_node) {
                if p.kind() != "formal_parameter" {
                    continue;
                }
                emit_type_refs(
                    ctx,
                    func_nid,
                    p.child_by_field_name("type"),
                    "parameter_type",
                    line,
                    false,
                )?;
            }
        }
        // A `method_declaration`'s `type` field is its RETURN type; a
        // `constructor_declaration` has no `type` field, so this is a no-op there.
        if let Some(return_node) = node.child_by_field_name("type") {
            emit_type_refs(ctx, func_nid, Some(return_node), "return_type", line, false)?;
        }
        emit_annotations(ctx, func_nid, node, line, true)
    }

    fn extra_walk<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        parent_class_nid: Option<&str>,
    ) -> R<Handled> {
        // `_java_extra_walk`: enum_constant.
        if node.kind() != "enum_constant" {
            return Ok(Handled::No);
        }
        let parent = match parent_class_nid {
            Some(p) => p.to_string(),
            None => return Ok(Handled::No),
        };
        let name_node = match node.child_by_field_name("name") {
            Some(n) => n,
            // Python returns True (handled) even with no name, so the node is
            // consumed rather than recursed into.
            None => return Ok(Handled::Yes),
        };
        let const_name = ctx.text(name_node)?.to_string();
        let line = node.start_position().row + 1;
        let const_nid = ctx.mkid(&[&parent, &const_name])?;
        ctx.add_node(&const_nid, &const_name, line);
        ctx.add_edge(&parent, &const_nid, "case_of", line);
        // Anonymous-body constants (`MONDAY { void greet(){} }`): descend so the
        // body's methods are not dropped; const_nid attaches them to the constant.
        for child in children(node) {
            if child.kind() == "class_body" {
                for member in children(child) {
                    crate::engine::walk::walk(ctx, member, Some(&const_nid))?;
                }
            }
        }
        Ok(Handled::Yes)
    }

    fn call_info<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        _caller_nid: &str,
    ) -> R<Option<CallInfo>> {
        let mut info = CallInfo::default();
        if node.kind() == "object_creation_expression" {
            // #1373: the constructed type is the `type` field, not `name`.
            if let Some(type_node) = node.child_by_field_name("type") {
                let raw = ctx.text(type_node)?;
                let base = raw.split_once('<').map(|(a, _)| a).unwrap_or(raw).trim();
                let name = base.rsplit_once('.').map(|(_, t)| t).unwrap_or(base);
                if !name.is_empty() {
                    info.callee_name = Some(name.to_string());
                }
            }
        } else {
            if let Some(name_node) = node.child_by_field_name("name") {
                info.callee_name = Some(ctx.text(name_node)?.to_string());
            }
            if let Some(receiver) = node.child_by_field_name("object") {
                info.is_member_call = true;
                match receiver.kind() {
                    "identifier" => info.member_receiver = Some(ctx.text(receiver)?.to_string()),
                    "this" => info.member_receiver = Some("this".to_string()),
                    "field_access" => {
                        let owner = receiver.child_by_field_name("object");
                        let field = receiver.child_by_field_name("field");
                        if let (Some(o), Some(f)) = (owner, field) {
                            if o.kind() == "this" {
                                info.member_receiver = Some(format!("this.{}", ctx.text(f)?));
                                info.is_this_field_call = true;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(Some(info))
    }

    /// `_java_defer = (_is_java and is_member_call)` -- unconditional, with no
    /// receiver or capitalization test, unlike Python's and C#'s narrower rules.
    fn defers(&self, info: &CallInfo) -> bool {
        info.is_member_call
    }

    fn raw_call_extra<'tree>(
        &self,
        _ctx: &Ctx<'_, 'tree>,
        node: Node<'tree>,
        _caller_nid: &str,
        info: &CallInfo,
        receiver_types: &RecvTable,
    ) -> Vec<(&'static str, Val)> {
        let mut out: Vec<(&'static str, Val)> = vec![("lang", Val::Static("java"))];
        // `(receiver_types or {}).get(member_receiver or "")` -- a flat lookup.
        // The offset is passed because the accessor takes one; `RecvTable::Flat`
        // ignores it.
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

static HOOKS: Java = Java;

pub static CONFIG: EngineConfig = EngineConfig {
    language: "java",
    grammar: || tree_sitter_java::LANGUAGE.into(),
    class_types: &[
        "annotation_type_declaration",
        "class_declaration",
        "enum_declaration",
        "interface_declaration",
        "record_declaration",
    ],
    function_types: &["constructor_declaration", "method_declaration"],
    import_types: &["import_declaration"],
    call_types: &["method_invocation", "object_creation_expression"],
    function_boundary_types: &["constructor_declaration", "method_declaration"],
    static_prop_types: &[],
    helper_fn_names: &[],
    container_bind_methods: &[],
    event_listener_properties: &[],
    name_field: "name",
    name_fallback_child_types: &[],
    body_field: "body",
    body_fallback_child_types: &[],
    call_function_field: "name",
    call_accessor_node_types: &[],
    call_accessor_field: "attribute",
    call_accessor_object_field: "",
    function_label_parens: true,
    resolve_function_name: None,
    sanitize_symbol_name: None,
    type_table_key: None,
    hooks: &HOOKS,
};

pub fn walk_java<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
    _res: &crate::Resolvers<'py>,
) -> PyResult<Outcome<'py>> {
    crate::engine::run(py, &CONFIG, path, source, calls::method_receiver_types, None)
}

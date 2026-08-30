//! C++ on the shared engine.
//!
//! Shares C's `#include` handler and the `("tree_sitter_c", "tree_sitter_cpp")`
//! parameter/return-type block, and nothing else -- `_cpp_collect_type_refs` is
//! materially wider than `_c_collect_type_refs` (qualified tails, template
//! arguments), and C++ has classes where C has none.
//!
//! It is the first language to need the two guard positions that live OUTSIDE
//! `walk` / `walk_calls`: `before_calls`, where `_cpp_local_var_types` builds the
//! file's `var -> ClassName` table from every function body, and
//! `EngineConfig::type_table_key`, which emits it as `cpp_type_table`. Ruby and
//! Swift use the same two positions, so they are shared rather than C++-shaped.

use std::collections::HashMap;

use pyo3::prelude::*;
use tree_sitter::Node;

use crate::engine::{CallInfo, Ctx, EngineConfig, Handled, LangHooks, RecvTable, R};
use crate::js::ast::children;
use crate::js::emit::Val;
use crate::Outcome;

pub mod helpers;

use helpers::{collect_type_refs, func_name};

struct Cpp;

/// One `references` edge per collected type, `generic_arg` overriding the
/// block's own context.
fn emit_type_refs(
    ctx: &mut Ctx,
    owner_nid: &str,
    type_node: Option<Node>,
    type_ctx: &'static str,
    line: usize,
) -> R<()> {
    let mut refs: Vec<(String, bool)> = Vec::new();
    collect_type_refs(ctx, type_node, false, &mut refs)?;
    for (ref_name, generic) in refs {
        let c = if generic { "generic_arg" } else { type_ctx };
        let target = ctx.ensure_named_node(&ref_name, line)?;
        if target != owner_nid {
            ctx.add_edge_ctx(owner_nid, &target, "references", line, c);
        }
    }
    Ok(())
}

impl LangHooks for Cpp {
    fn import_handler<'tree>(&self, ctx: &mut Ctx<'_, 'tree>, node: Node<'tree>) -> R<()> {
        crate::c::imports::import_c(ctx, node)
    }

    /// Inheritance via `base_class_clause`. Multiple bases are siblings
    /// separated by `,` tokens; `access_specifier` and `virtual` are skipped by
    /// not matching any arm.
    fn on_class<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        class_nid: &str,
        _class_name: &str,
        line: usize,
    ) -> R<()> {
        for child in children(node) {
            if child.kind() != "base_class_clause" {
                continue;
            }
            for sub in children(child) {
                let mut template_args_node: Option<Node> = None;
                let base = match sub.kind() {
                    "type_identifier" => ctx.text(sub)?.to_string(),
                    "qualified_identifier" => {
                        // The unqualified tail, so `std::vector` matches a
                        // `vector` node id if one exists; the full text if the
                        // grammar gives no `name` field.
                        match sub.child_by_field_name("name") {
                            Some(tail) => ctx.text(tail)?.to_string(),
                            None => ctx.text(sub)?.to_string(),
                        }
                    }
                    "template_type" => {
                        let b = match sub.child_by_field_name("name") {
                            Some(tname) => ctx.text(tname)?.to_string(),
                            None => ctx.text(sub)?.to_string(),
                        };
                        template_args_node = sub.child_by_field_name("arguments");
                        b
                    }
                    _ => continue,
                };
                if base.is_empty() {
                    continue;
                }
                let base_nid = ctx.ensure_named_node(&base, line)?;
                ctx.add_edge(class_nid, &base_nid, "inherits", line);
                // `class Car : public Base<Dep>` references Dep.
                if let Some(args) = template_args_node {
                    let mut arg_refs: Vec<(String, bool)> = Vec::new();
                    for arg in children(args) {
                        if arg.is_named() {
                            collect_type_refs(ctx, Some(arg), true, &mut arg_refs)?;
                        }
                    }
                    for (ref_name, _generic) in arg_refs {
                        let target = ctx.ensure_named_node(&ref_name, line)?;
                        if target != class_nid {
                            ctx.add_edge_ctx(
                                class_nid,
                                &target,
                                "references",
                                line,
                                "generic_arg",
                            );
                        }
                    }
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
        if node.kind() != "field_declaration" {
            return Ok(Handled::No);
        }
        let parent = match parent_class_nid {
            Some(p) => p.to_string(),
            None => return Ok(Handled::No),
        };
        // A `field_declaration` carrying a `function_declarator` is a member
        // FUNCTION prototype, not a data member.
        let decls: Vec<Node> = node
            .children_by_field_name("declarator", &mut node.walk())
            .collect();
        let is_method = decls.iter().any(|d| {
            d.kind() == "function_declarator"
                || (matches!(d.kind(), "pointer_declarator" | "reference_declarator")
                    && children(*d).into_iter().any(|c| c.kind() == "function_declarator"))
        });
        let type_node = node.child_by_field_name("type");
        // #2876: a nested type (`class Inner { … };` inside a class body) is a
        // field_declaration whose `type` field IS the class_specifier. Returning
        // here used to drop Inner and everything it declares, silently and with
        // no parse error. Walk it as a class; the declarator loop below still
        // runs, since `class Inner { } inst;` declares a member alongside it.
        let is_nested_type = match type_node {
            Some(t) => {
                crate::engine::has(ctx.cfg.class_types, t.kind())
                    && t.child_by_field_name("body").is_some()
            }
            None => false,
        };
        if is_nested_type {
            crate::engine::walk::walk(ctx, type_node.unwrap(), parent_class_nid)?;
        }
        if !is_method && !is_nested_type {
            if type_node.is_some() {
                let line = node.start_position().row + 1;
                emit_type_refs(ctx, &parent, type_node, "field", line)?;
            }
        }
        // A node per data member. `children_by_field_name` visits only the
        // declarator children -- iterating all children would pick up the type
        // node and yield the type's name instead of the field's.
        for decl in decls {
            if let Some(name) = func_name(ctx, decl)? {
                let line = decl.start_position().row + 1;
                let field_nid = ctx.mkid(&[&parent, &name])?;
                ctx.add_node(&field_nid, &name, line);
                ctx.add_edge_ctx(&parent, &field_nid, "defines", line, "field");
            }
        }
        Ok(Handled::Yes)
    }

    /// The `("tree_sitter_c", "tree_sitter_cpp")` block, with C++'s collector.
    fn on_function<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        func_nid: &str,
        _func_name: &str,
        line: usize,
        _parent_class_nid: Option<&str>,
    ) -> R<()> {
        emit_type_refs(ctx, func_nid, node.child_by_field_name("type"), "return_type", line)?;
        let mut decl = node.child_by_field_name("declarator");
        while let Some(d) = decl {
            if !matches!(d.kind(), "pointer_declarator" | "reference_declarator") {
                break;
            }
            decl = d.child_by_field_name("declarator");
        }
        let decl = match decl {
            Some(d) if d.kind() == "function_declarator" => d,
            _ => return Ok(()),
        };
        if let Some(params_node) = decl.child_by_field_name("parameters") {
            for p in children(params_node) {
                if p.kind() != "parameter_declaration" {
                    continue;
                }
                let ptype = match p.child_by_field_name("type") {
                    Some(t) => t,
                    None => continue,
                };
                emit_type_refs(ctx, func_nid, Some(ptype), "parameter_type", line)?;
            }
        }
        Ok(())
    }

    fn call_info<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        _caller_nid: &str,
    ) -> R<Option<CallInfo>> {
        let mut info = CallInfo::default();
        let func_node = match node.child_by_field_name(ctx.cfg.call_function_field) {
            Some(f) => f,
            None => return Ok(Some(info)),
        };
        match func_node.kind() {
            "identifier" => info.callee_name = Some(ctx.text(func_node)?.to_string()),
            "field_expression" => {
                // `f.bar()` / `f->bar()` / `this->bar()`: the receiver is the
                // `argument` field and the callee the `field` (#1547). Only a
                // simple identifier (or `this`) is captured; a chained receiver
                // is left to bail in the cross-file pass.
                info.is_member_call = true;
                if let Some(name) = func_node.child_by_field_name("field") {
                    info.callee_name = Some(ctx.text(name)?.to_string());
                }
                if let Some(obj) = func_node.child_by_field_name("argument") {
                    match obj.kind() {
                        "identifier" => info.member_receiver = Some(ctx.text(obj)?.to_string()),
                        "this" => info.member_receiver = Some("this".to_string()),
                        _ => {}
                    }
                }
            }
            "qualified_identifier" => {
                // `Foo::bar()`: the scope IS the receiver type, named explicitly
                // in source, so it is EXTRACTED rather than inferred.
                info.is_member_call = true;
                if let Some(name) = func_node.child_by_field_name("name") {
                    info.callee_name = Some(ctx.text(name)?.to_string());
                }
                if let Some(scope) = func_node.child_by_field_name("scope") {
                    info.member_receiver = Some(ctx.text(scope)?.to_string());
                }
            }
            _ => {}
        }
        Ok(Some(info))
    }

    /// `_cpp_local_var_types` over every function body, file-scoped.
    fn before_calls<'tree>(&self, ctx: &mut Ctx<'_, 'tree>) -> R<()> {
        let bodies: Vec<Node> = ctx.function_bodies.iter().map(|(_, b)| *b).collect();
        let mut table = std::mem::take(&mut ctx.type_table);
        let result = (|| {
            for body in bodies {
                helpers::local_var_types(ctx, body, &mut table)?;
            }
            Ok(())
        })();
        ctx.type_table = table;
        result
    }

    /// The C++ raw_call carries only a language tag: a `.h` file routes to
    /// `extract_cpp` or `extract_objc` by CONTENT, and both resolvers see `.h`
    /// in their suffix sets, so a `source_file` suffix cannot separate them. The
    /// receiver's type is resolved later from `cpp_type_table`, not stamped here.
    fn raw_call_extra<'tree>(
        &self,
        _ctx: &Ctx<'_, 'tree>,
        _node: Node<'tree>,
        _info: &CallInfo,
        _receiver_types: &RecvTable,
    ) -> Vec<(&'static str, Val)> {
        vec![("lang", Val::Static("cpp"))]
    }
}

static HOOKS: Cpp = Cpp;

pub static CONFIG: EngineConfig = EngineConfig {
    language: "cpp",
    grammar: || tree_sitter_cpp::LANGUAGE.into(),
    class_types: &["class_specifier", "struct_specifier"],
    function_types: &["function_definition"],
    import_types: &["preproc_include"],
    call_types: &["call_expression"],
    function_boundary_types: &["function_definition"],
    static_prop_types: &[],
    helper_fn_names: &[],
    container_bind_methods: &[],
    event_listener_properties: &[],
    name_field: "name",
    name_fallback_child_types: &[],
    body_field: "body",
    body_fallback_child_types: &[],
    call_function_field: "function",
    call_accessor_node_types: &["field_expression", "qualified_identifier"],
    call_accessor_field: "field",
    call_accessor_object_field: "",
    function_label_parens: true,
    resolve_function_name: Some(|ctx, node| helpers::func_name(ctx, node)),
    type_table_key: Some("cpp_type_table"),
    hooks: &HOOKS,
};

pub fn walk_cpp<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
    res: &crate::Resolvers<'py>,
) -> PyResult<Outcome<'py>> {
    crate::engine::run(
        py,
        &CONFIG,
        path,
        source,
        crate::engine::no_receiver_types,
        Some(&res.c),
    )
}

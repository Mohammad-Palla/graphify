//! Kotlin on the shared engine, and the last language ported.
//!
//! It uses more of the engine than any other: eleven of the sixteen hook
//! positions, plus `initializer_nodes`. Three of those positions exist only
//! because Kotlin reached them:
//!
//!   * `on_function_body` -- `object : Foo { ... }` anonymous objects live
//!     INSIDE a function body, which the function branch never recurses into, so
//!     their members and every call in them got no nodes at all (#2347).
//!   * `result_extra` -- the declared `package` qualifies every node in the
//!     file, and the import-target and qualified-call resolvers key their
//!     per-package symbol indexes off it (#2526/#2550).
//!   * `Ctx::initializer_nodes` -- `val repo = createRepo()` is a call that
//!     lives in no function (#2565). Swift will use the same slot.
//!
//! Several node-kind lists here accept TWO names for one thing (`identifier` and
//! `simple_identifier`). That is grammar-generation skew, not defensiveness --
//! see `helpers`.

use pyo3::prelude::*;
use tree_sitter::Node;

use crate::engine::{CallInfo, Ctx, EngineConfig, Handled, LangHooks, RecvTable, R};
use crate::js::ast::children;
use crate::js::emit::Val;
use crate::Outcome;

pub mod consts;
pub mod helpers;
pub mod imports;

use helpers::{collect_type_refs, user_type_name, IDENTS};

struct Kotlin;

fn emit_type_refs(
    ctx: &mut Ctx,
    owner: &str,
    type_node: Option<Node>,
    type_ctx: &'static str,
    line: usize,
) -> R<()> {
    let mut refs: Vec<(String, bool)> = Vec::new();
    collect_type_refs(ctx, type_node, false, &mut refs)?;
    for (ref_name, generic) in refs {
        let c = if generic { "generic_arg" } else { type_ctx };
        let tgt = ctx.ensure_named_node(&ref_name, line)?;
        if tgt != owner {
            ctx.add_edge_ctx(owner, &tgt, "references", line, c);
        }
    }
    Ok(())
}

/// The `user_type` a `delegation_specifier` names, plus whether it is a
/// superclass (`constructor_invocation`) or an interface.
fn delegation_target<'tree>(spec: Node<'tree>) -> (&'static str, Option<Node<'tree>>) {
    let mut relation = "implements";
    for sub in children(spec) {
        if sub.kind() == "constructor_invocation" {
            relation = "inherits";
            return (
                relation,
                children(sub).into_iter().find(|i| i.kind() == "user_type"),
            );
        }
        if sub.kind() == "user_type" {
            return (relation, Some(sub));
        }
        // `class Foo : Bar by baz` wraps the delegated interface in an
        // `explicit_delegation`; take its first `user_type` so the implements
        // edge (and generic-arg recovery) still fire.
        if sub.kind() == "explicit_delegation" {
            return (
                relation,
                children(sub).into_iter().find(|i| i.kind() == "user_type"),
            );
        }
    }
    (relation, None)
}

impl LangHooks for Kotlin {
    fn import_handler<'tree>(&self, ctx: &mut Ctx<'_, 'tree>, node: Node<'tree>) -> R<()> {
        imports::import_kotlin(ctx, node)
    }

    fn on_class<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        class_nid: &str,
        _class_name: &str,
        line: usize,
    ) -> R<()> {
        for child in children(node) {
            if child.kind() != "delegation_specifiers" {
                continue;
            }
            for spec in children(child) {
                if spec.kind() != "delegation_specifier" {
                    continue;
                }
                let (relation, user_type_node) = delegation_target(spec);
                let user_type_node = match user_type_node {
                    Some(u) => u,
                    None => continue,
                };
                let base = match user_type_name(ctx, Some(user_type_node))? {
                    Some(b) if !b.is_empty() => b,
                    _ => continue,
                };
                let base_nid = ctx.ensure_named_node(&base, line)?;
                ctx.add_edge(class_nid, &base_nid, relation, line);
                for arg_child in children(user_type_node) {
                    if arg_child.kind() != "type_arguments" {
                        continue;
                    }
                    for arg in children(arg_child) {
                        if arg.kind() != "type_projection" {
                            continue;
                        }
                        for inner in children(arg) {
                            if !inner.is_named() {
                                continue;
                            }
                            let mut refs: Vec<(String, bool)> = Vec::new();
                            collect_type_refs(ctx, Some(inner), true, &mut refs)?;
                            // No `!= class_nid` guard here, unlike the sibling
                            // emitters. Python's omission, kept.
                            for (ref_name, _g) in refs {
                                let tgt = ctx.ensure_named_node(&ref_name, line)?;
                                ctx.add_edge_ctx(class_nid, &tgt, "references", line, "generic_arg");
                            }
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
        if node.kind() != "property_declaration" {
            return Ok(Handled::No);
        }
        // Field-type references stay CLASS-gated: a top-level property keeps its
        // pre-#2565 no-references behaviour.
        if let Some(parent) = parent_class_nid {
            let parent = parent.to_string();
            if let Some(type_node) = helpers::property_type_node(node) {
                let line = node.start_position().row + 1;
                emit_type_refs(ctx, &parent, Some(type_node), "field", line)?;
            }
        }
        // #2565: seed the initializer so the call pass collects its calls. The
        // WHOLE expression, not just `call_types`, so nested argument calls
        // (`HttpClient(base())`) and lambda bodies are reached; a literal
        // initializer (`val plain = 5`) contains no call and yields nothing.
        // The explicit type, if any, sits inside `variable_declaration` BEFORE
        // the `=`, so post-`=` named children are only the initializer.
        let owner_nid = parent_class_nid
            .map(|p| p.to_string())
            .unwrap_or_else(|| ctx.file_nid.clone());
        let mut seen_eq = false;
        for child in children(node) {
            if !child.is_named() {
                seen_eq = seen_eq || child.kind() == "=";
                continue;
            }
            if seen_eq {
                ctx.initializer_nodes.push((owner_nid.clone(), child));
            } else if child.kind() == "property_delegate" {
                // `by lazy { ... }`, or any delegate.
                for sub in children(child) {
                    if sub.is_named() {
                        ctx.initializer_nodes.push((owner_nid.clone(), sub));
                    }
                }
            }
        }
        Ok(Handled::Yes)
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
        let params_container = children(node)
            .into_iter()
            .find(|c| c.kind() == "function_value_parameters");
        if let Some(params_container) = params_container {
            for p in children(params_container) {
                if p.kind() != "parameter" {
                    continue;
                }
                let param_type_node = children(p).into_iter().find(|sub| {
                    matches!(sub.kind(), "user_type" | "nullable_type" | "type_reference")
                });
                emit_type_refs(ctx, func_nid, param_type_node, "parameter_type", line)?;
            }
        }
        if let Some(return_type_node) = helpers::function_return_type_node(node) {
            emit_type_refs(ctx, func_nid, Some(return_type_node), "return_type", line)?;
        }
        Ok(())
    }

    /// #2347: `object : Foo { ... }` anonymous objects inside a function body.
    ///
    /// The function branch never recurses into the body and `object_literal` is
    /// not a class type, so the literal's members -- and every call inside them
    /// -- got no nodes at all. Scanned WITHOUT crossing a nested
    /// `function_declaration` boundary (a local fun's literals are not this
    /// function's) and without descending into a literal already found.
    fn on_function_body<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        _node: Node<'tree>,
        func_nid: &str,
        body: Node<'tree>,
    ) -> R<()> {
        let mut literals: Vec<Node> = Vec::new();
        let mut stack: Vec<Node> = children(body);
        while let Some(n) = stack.pop() {
            if n.kind() == "function_declaration" {
                continue;
            }
            if n.kind() == "object_literal" {
                literals.push(n);
                continue;
            }
            stack.extend(children(n));
        }
        literals.sort_by_key(|n| n.start_byte());
        for lit in literals {
            let lit_line = lit.start_position().row + 1;
            let mut lit_bases: Vec<(String, &'static str)> = Vec::new();
            for child in children(lit) {
                if child.kind() != "delegation_specifiers" {
                    continue;
                }
                for spec in children(child) {
                    if spec.kind() != "delegation_specifier" {
                        continue;
                    }
                    let (relation, user_type_node) = delegation_target(spec);
                    if let Some(base) = user_type_name(ctx, user_type_node)? {
                        if !base.is_empty() {
                            lit_bases.push((base, relation));
                        }
                    }
                }
            }
            let obj_label = match lit_bases.first() {
                Some((b, _)) => b.clone(),
                None => format!("object@L{lit_line}"),
            };
            let obj_nid = ctx.mkid(&[func_nid, &format!("object:{obj_label}"), &format!("L{lit_line}")])?;
            ctx.add_node(&obj_nid, &obj_label, lit_line);
            ctx.add_edge(func_nid, &obj_nid, "contains", lit_line);
            ctx.callable_def_nids.insert(obj_nid.clone());
            ctx.callable_class_nids.insert(obj_nid.clone());
            for (base, relation) in lit_bases {
                let base_nid = ctx.ensure_named_node(&base, lit_line)?;
                if base_nid != obj_nid {
                    ctx.add_edge(&obj_nid, &base_nid, relation, lit_line);
                }
            }
            if let Some(lit_body) = children(lit).into_iter().find(|c| c.kind() == "class_body") {
                for child in children(lit_body) {
                    crate::engine::walk::walk(ctx, child, Some(&obj_nid))?;
                }
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
        // `enum_entry` (#1700's Kotlin half).
        if node.kind() == "enum_entry" {
            let parent = match parent_class_nid {
                Some(p) => p.to_string(),
                None => return Ok(Handled::No),
            };
            let name_node = children(node)
                .into_iter()
                .find(|c| IDENTS.contains(&c.kind()));
            let name_node = match name_node {
                Some(n) => n,
                None => return Ok(Handled::Yes),
            };
            let const_name = ctx.text(name_node)?.to_string();
            let line = node.start_position().row + 1;
            let const_nid = ctx.mkid(&[&parent, &const_name])?;
            ctx.add_node(&const_nid, &const_name, line);
            ctx.add_edge(&parent, &const_nid, "case_of", line);
            for child in children(node) {
                if child.kind() == "class_body" {
                    for member in children(child) {
                        crate::engine::walk::walk(ctx, member, Some(&const_nid))?;
                    }
                }
            }
            return Ok(Handled::Yes);
        }
        // A `companion object` is a transparent scope: its members belong to the
        // ENCLOSING class. Recursing into the `class_body`'s children directly
        // matters -- a bare `class_body` would itself default-recurse and drop
        // the parent link, leaving companion `fun`s file-level.
        if node.kind() == "companion_object" {
            for child in children(node) {
                if child.kind() == "class_body" {
                    for member in children(child) {
                        crate::engine::walk::walk(ctx, member, parent_class_nid)?;
                    }
                } else {
                    crate::engine::walk::walk(ctx, child, parent_class_nid)?;
                }
            }
            return Ok(Handled::Yes);
        }
        Ok(Handled::No)
    }

    fn call_info<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        _caller_nid: &str,
    ) -> R<Option<CallInfo>> {
        let mut info = CallInfo::default();
        let first = match children(node).into_iter().next() {
            Some(f) => f,
            None => return Ok(Some(info)),
        };
        if IDENTS.contains(&first.kind()) {
            info.callee_name = Some(ctx.text(first)?.to_string());
        } else if first.kind() == "navigation_expression" {
            info.is_member_call = true;
            for child in children(first).into_iter().rev() {
                if IDENTS.contains(&child.kind()) {
                    info.callee_name = Some(ctx.text(child)?.to_string());
                    break;
                }
            }
            // #2550: `com.example.Foo.bar()` is a NESTED navigation chain, and
            // the last identifier alone (`bar`) rarely matches in-file, so the
            // call was dropped -- the shared cross-file pass skips member calls.
            // When EVERY segment is a plain identifier and there are >= 3 (a real
            // dotted FQN, not `recv.method()`), stamp the prefix.
            //
            // `member_receiver` is deliberately NOT set: an uppercase receiver
            // would trip the capitalized-receiver deferral and regress in-file
            // `Foo.bar()` resolution.
            if let Some(segments) = helpers::nav_identifier_segments(ctx, first)? {
                if segments.len() >= 3 {
                    info.qualified_prefix = Some(segments[..segments.len() - 1].join("."));
                }
            }
        }
        Ok(Some(info))
    }

    fn raw_call_extra<'tree>(
        &self,
        _ctx: &Ctx<'_, 'tree>,
        _node: Node<'tree>,
        _caller_nid: &str,
        info: &CallInfo,
        _receiver_types: &RecvTable,
    ) -> Vec<(&'static str, Val)> {
        match &info.qualified_prefix {
            Some(p) => vec![
                ("lang", Val::Static("kotlin")),
                ("qualified_prefix", Val::S(p.clone())),
            ],
            None => Vec::new(),
        }
    }

    fn result_extra<'tree>(
        &self,
        ctx: &Ctx<'_, 'tree>,
        root: Node<'tree>,
    ) -> R<Vec<(&'static str, Val)>> {
        Ok(match helpers::package_name(ctx, root)? {
            Some(pkg) => vec![("kotlin_package", Val::S(pkg))],
            None => Vec::new(),
        })
    }
}

static HOOKS: Kotlin = Kotlin;

pub static CONFIG: EngineConfig = EngineConfig {
    language: "kotlin",
    grammar: || tree_sitter_kotlin_ng::LANGUAGE.into(),
    class_types: &["class_declaration", "object_declaration"],
    function_types: &["function_declaration"],
    // Grammar 1.1.0 names the import node `import`; older forks use
    // `import_header`. Both, so the walker works across generations (#2526).
    import_types: &["import_header", "import"],
    call_types: &["call_expression"],
    function_boundary_types: &["function_declaration"],
    static_prop_types: &[],
    helper_fn_names: &[],
    container_bind_methods: &[],
    event_listener_properties: &[],
    name_field: "name",
    name_fallback_child_types: &["simple_identifier", "identifier"],
    body_field: "body",
    body_fallback_child_types: &["function_body", "class_body", "enum_class_body"],
    call_function_field: "",
    call_accessor_node_types: &["navigation_expression"],
    call_accessor_field: "",
    call_accessor_object_field: "",
    function_label_parens: true,
    resolve_function_name: None,
    sanitize_symbol_name: None,
    type_table_key: None,
    hooks: &HOOKS,
};

pub fn walk_kotlin<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
    _res: &crate::Resolvers<'py>,
) -> PyResult<Outcome<'py>> {
    crate::engine::run(
        py,
        &CONFIG,
        path,
        source,
        crate::engine::no_receiver_types,
        None,
    )
}

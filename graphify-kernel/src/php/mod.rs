//! PHP on the shared engine.
//!
//! The first language whose `LanguageConfig` uses the four FRAMEWORK fields --
//! `static_prop_types`, `helper_fn_names`, `container_bind_methods`,
//! `event_listener_properties` -- which encode Laravel conventions rather than
//! language syntax. Java, C#, C and C++ all leave them empty, which is why
//! `test_one_language_config_per_routed_grammar` could assert they were empty
//! for every routed grammar until now.
//!
//! Those four fields are why PHP needed three more hook positions, all of them
//! places the Python has a guard and the engine did not:
//!
//!   * `after_call` -- the `config('a.b')` helper edge and the
//!     `$this->app->bind(A::class, B::class)` container binding both need the
//!     resolved `callee_name`, so they sit after the call is classified.
//!   * `walk_calls_extra` -- `Foo::$bar` and `Foo::BAR` are not calls, so their
//!     edges are emitted at the body level of `walk_calls`.
//!   * `after_calls` -- a `$listen` array names classes that may be declared
//!     later in the file, so those edges are harvested during the walk and
//!     resolved once every node exists.
//!
//! All four framework edges carry `confidence_score` and no `context`, a key
//! order `add_edge` does not produce -- hence `Ctx::add_edge_scored`.

use pyo3::prelude::*;
use tree_sitter::Node;

use crate::engine::{has, CallInfo, Ctx, EngineConfig, Handled, LangHooks, R};
use crate::js::ast::children;
use crate::Outcome;

pub mod helpers;
pub mod imports;

use helpers::{class_const_scope, collect_type_refs, name_text};

struct Php;

fn emit_type_refs(
    ctx: &mut Ctx,
    owner_nid: &str,
    type_node: Option<Node>,
    type_ctx: &'static str,
    line: usize,
) -> R<Vec<(String, bool)>> {
    let mut refs: Vec<(String, bool)> = Vec::new();
    collect_type_refs(ctx, type_node, false, &mut refs)?;
    for (ref_name, generic) in &refs {
        let c = if *generic { "generic_arg" } else { type_ctx };
        let target = ctx.ensure_named_node(ref_name, line)?;
        if target != owner_nid {
            ctx.add_edge_ctx(owner_nid, &target, "references", line, c);
        }
    }
    Ok(refs)
}

/// The `$listen = [Event::class => [Listener::class]]` harvest.
///
/// Returns whether the property was CONSUMED. A `property_declaration` that is
/// not one of the listener properties falls through to the type-reference
/// branch, which is why this reports rather than emitting `Handled` itself.
fn harvest_event_listeners(ctx: &mut Ctx, node: Node) -> R<bool> {
    let mut handled = false;
    for element in children(node) {
        if element.kind() != "property_element" {
            continue;
        }
        let mut prop_name: Option<String> = None;
        let mut array_node: Option<Node> = None;
        for c in children(element) {
            if c.kind() == "variable_name" {
                for sc in children(c) {
                    if sc.kind() == "name" {
                        prop_name = Some(ctx.text(sc)?.to_string());
                        break;
                    }
                }
            } else if c.kind() == "array_creation_expression" {
                array_node = Some(c);
            }
        }
        let (prop_name, array_node) = match (prop_name, array_node) {
            (Some(p), Some(a)) if has(ctx.cfg.event_listener_properties, &p) => (p, a),
            _ => continue,
        };
        let _ = prop_name;
        handled = true;
        for entry in children(array_node) {
            if entry.kind() != "array_element_initializer" {
                continue;
            }
            let mut event_cls: Option<String> = None;
            let mut listener_arr: Option<Node> = None;
            for sub in children(entry) {
                if sub.kind() == "class_constant_access_expression" && event_cls.is_none() {
                    for sc in children(sub) {
                        if sc.is_named() && matches!(sc.kind(), "name" | "qualified_name") {
                            event_cls = Some(ctx.text(sc)?.to_string());
                            break;
                        }
                    }
                } else if sub.kind() == "array_creation_expression" {
                    listener_arr = Some(sub);
                }
            }
            let (event_cls, listener_arr) = match (event_cls, listener_arr) {
                (Some(e), Some(l)) if !e.is_empty() => (e, l),
                _ => continue,
            };
            for listener_entry in children(listener_arr) {
                if listener_entry.kind() != "array_element_initializer" {
                    continue;
                }
                for item in children(listener_entry) {
                    if item.kind() != "class_constant_access_expression" {
                        continue;
                    }
                    for sc in children(item) {
                        if sc.is_named() && matches!(sc.kind(), "name" | "qualified_name") {
                            let listener_cls = ctx.text(sc)?.to_string();
                            let line_no = item.start_position().row + 1;
                            ctx.pending_listen_edges
                                .push((event_cls.clone(), listener_cls, line_no));
                            break;
                        }
                    }
                    break;
                }
            }
        }
    }
    Ok(handled)
}

impl LangHooks for Php {
    fn import_handler<'tree>(&self, ctx: &mut Ctx<'_, 'tree>, node: Node<'tree>) -> R<()> {
        imports::import_php(ctx, node)
    }

    /// `extends` -> inherits, `implements` -> implements, `use` -> mixes_in.
    ///
    /// The line is the CLAUSE's, not the class's -- unlike every other language
    /// on this engine, which uses the class declaration's line throughout.
    fn on_class<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        class_nid: &str,
        _class_name: &str,
        _line: usize,
    ) -> R<()> {
        let mut emit = |ctx: &mut Ctx, base: Option<String>, rel: &'static str, at: usize| -> R<()> {
            let base = match base {
                Some(b) if !b.is_empty() => b,
                _ => return Ok(()),
            };
            let base_nid = ctx.ensure_parent_node(&base)?;
            ctx.add_edge(class_nid, &base_nid, rel, at);
            Ok(())
        };
        for child in children(node) {
            let rel = match child.kind() {
                "base_clause" => "inherits",
                "class_interface_clause" => "implements",
                _ => continue,
            };
            let at = child.start_position().row + 1;
            for sub in children(child) {
                if matches!(sub.kind(), "name" | "qualified_name") {
                    let base = name_text(ctx, Some(sub))?;
                    emit(ctx, base, rel, at)?;
                }
            }
        }
        // Traits, from `use Foo;` inside the class body.
        let mut body = node.child_by_field_name("body");
        if body.is_none() {
            body = children(node).into_iter().find(|c| c.kind() == "declaration_list");
        }
        if let Some(body) = body {
            for member in children(body) {
                if member.kind() != "use_declaration" {
                    continue;
                }
                let at = member.start_position().row + 1;
                for sub in children(member) {
                    if matches!(sub.kind(), "name" | "qualified_name") {
                        let base = name_text(ctx, Some(sub))?;
                        emit(ctx, base, "mixes_in", at)?;
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
        let parent = match parent_class_nid {
            Some(p) => p.to_string(),
            None => return Ok(Handled::No),
        };
        // The listener harvest comes FIRST and only consumes the node when it
        // actually matched a listener property; anything else falls through to
        // the type-reference branch below.
        if harvest_event_listeners(ctx, node)? {
            return Ok(Handled::Yes);
        }
        for c in children(node) {
            if !helpers::TYPE_NODES.contains(&c.kind()) {
                continue;
            }
            let line = node.start_position().row + 1;
            emit_type_refs(ctx, &parent, Some(c), "field", line)?;
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
        parent_class_nid: Option<&str>,
    ) -> R<()> {
        let params_container = children(node)
            .into_iter()
            .find(|c| c.kind() == "formal_parameters");
        if let Some(params_container) = params_container {
            for p in children(params_container) {
                // PHP 8 constructor property promotion parses a promoted
                // parameter as `property_promotion_parameter`, not
                // `simple_parameter`; its type sits in the same shape, and it
                // additionally declares a class FIELD.
                let is_promoted = match p.kind() {
                    "simple_parameter" => false,
                    "property_promotion_parameter" => true,
                    _ => continue,
                };
                let type_node = children(p)
                    .into_iter()
                    .find(|sub| helpers::TYPE_NODES.contains(&sub.kind()));
                let mut refs: Vec<(String, bool)> = Vec::new();
                collect_type_refs(ctx, type_node, false, &mut refs)?;
                // The promoted-parameter FIELD edge is emitted inside this loop,
                // immediately after each parameter_type edge -- not in a second
                // pass. Two passes produce the same edges in a different ORDER,
                // which the per-file parity harness canonicalizes away and the
                // exported JSON does not: 33 Symfony files diverged on exactly
                // that.
                for (ref_name, generic) in refs {
                    let c = if generic { "generic_arg" } else { "parameter_type" };
                    let target = ctx.ensure_named_node(&ref_name, line)?;
                    if target != func_nid {
                        ctx.add_edge_ctx(func_nid, &target, "references", line, c);
                    }
                    if is_promoted {
                        if let Some(parent) = parent_class_nid {
                            if target != parent {
                                let fctx = if generic { "generic_arg" } else { "field" };
                                ctx.add_edge_ctx(parent, &target, "references", line, fctx);
                            }
                        }
                    }
                }
            }
        }
        if let Some(return_node) = helpers::method_return_type_node(node) {
            emit_type_refs(ctx, func_nid, Some(return_node), "return_type", line)?;
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
        match node.kind() {
            "function_call_expression" => {
                if let Some(func_node) = node.child_by_field_name("function") {
                    info.callee_name = Some(ctx.text(func_node)?.to_string());
                }
            }
            "scoped_call_expression" => {
                // A static call `Helper::format()` names the CLASS, not the
                // method -- the Python's choice, kept: the class is the thing
                // the graph can resolve.
                if let Some(scope_node) = node.child_by_field_name("scope") {
                    info.callee_name = Some(ctx.text(scope_node)?.to_string());
                }
            }
            _ => {
                // `member_call_expression` ($obj->method()) and, because it is
                // also in `call_types`, `class_constant_access_expression`.
                info.is_member_call = true;
                if let Some(name_node) = node.child_by_field_name("name") {
                    info.callee_name = Some(ctx.text(name_node)?.to_string());
                }
            }
        }
        Ok(Some(info))
    }

    fn after_call<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        caller_nid: &str,
        info: &CallInfo,
    ) -> R<()> {
        let callee = match info.callee_name.as_deref() {
            Some(c) => c,
            None => return Ok(()),
        };
        // `config('foo.bar')` -> a `uses_config` edge to the "foo" config file.
        if has(ctx.cfg.helper_fn_names, callee) {
            if let Some(first_key) = helpers::first_string_argument(ctx, node)? {
                let segment = first_key.split('.').next().unwrap_or("");
                let lower = segment.to_lowercase();
                let tgt = ctx
                    .label_to_nid_ci
                    .get(&lower)
                    .or_else(|| ctx.label_to_nid_ci.get(&format!("{lower}.php")))
                    .cloned();
                if let Some(tgt) = tgt {
                    if tgt != caller_nid {
                        let line = node.start_position().row + 1;
                        ctx.add_edge_scored(caller_nid, &tgt, format!("uses_{callee}"), line);
                    }
                }
            }
        }
        // `$this->app->bind(Foo::class, Bar::class)` -> Foo bound_to Bar.
        if node.kind() == "member_call_expression" && has(ctx.cfg.container_bind_methods, callee) {
            let mut class_args: Vec<String> = Vec::new();
            if let Some(args_node) = node.child_by_field_name("arguments") {
                'outer: for arg in children(args_node) {
                    if arg.kind() != "argument" {
                        continue;
                    }
                    // The `break` is INSIDE the match in the Python: an
                    // argument whose first child is not a `::class` keeps being
                    // scanned. Breaking unconditionally here dropped the
                    // `bound_to` edge in Laravel's ContextualAttributeBindingTest,
                    // where the contract argument is wrapped.
                    for inner in children(arg) {
                        if inner.kind() == "class_constant_access_expression" {
                            if let Some(cls) = class_const_scope(ctx, inner)? {
                                class_args.push(cls);
                            }
                            break;
                        }
                    }
                    if class_args.len() >= 2 {
                        break 'outer;
                    }
                }
            }
            if class_args.len() == 2 {
                let contract = ctx.label_to_nid_ci.get(&class_args[0].to_lowercase()).cloned();
                let implementation =
                    ctx.label_to_nid_ci.get(&class_args[1].to_lowercase()).cloned();
                if let (Some(c), Some(i)) = (contract, implementation) {
                    if c != i {
                        let line = node.start_position().row + 1;
                        ctx.add_edge_scored(&c, &i, "bound_to".to_string(), line);
                    }
                }
            }
        }
        Ok(())
    }

    fn walk_calls_extra<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        caller_nid: &str,
    ) -> R<()> {
        // `Foo::$bar` -> uses_static_prop.
        if has(ctx.cfg.static_prop_types, node.kind()) {
            let mut scope_node = node.child_by_field_name("scope");
            if scope_node.is_none() {
                scope_node = children(node).into_iter().find(|c| {
                    c.is_named() && matches!(c.kind(), "name" | "qualified_name" | "identifier")
                });
            }
            if let Some(scope_node) = scope_node {
                let class_name = ctx.text(scope_node)?.to_lowercase();
                if let Some(tgt) = ctx.label_to_nid_ci.get(&class_name).cloned() {
                    if tgt != caller_nid {
                        let line = node.start_position().row + 1;
                        ctx.add_edge_scored(
                            caller_nid,
                            &tgt,
                            "uses_static_prop".to_string(),
                            line,
                        );
                    }
                }
            }
        }
        // `Foo::BAR` -> references_constant.
        if node.kind() == "class_constant_access_expression" {
            if let Some(class_name) = class_const_scope(ctx, node)? {
                if let Some(tgt) = ctx.label_to_nid_ci.get(&class_name.to_lowercase()).cloned() {
                    if tgt != caller_nid {
                        let line = node.start_position().row + 1;
                        ctx.add_edge_scored(
                            caller_nid,
                            &tgt,
                            "references_constant".to_string(),
                            line,
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// The `$listen` edges, resolved once every node in the file exists.
    ///
    /// Deduped on `(event, listener)` -- a PAIR, not the `(src, tgt, relation)`
    /// triple the other three framework edges use. Kept separate for that
    /// reason; merging it into `seen_rel_triples` would be equivalent only by
    /// accident.
    fn after_calls<'tree>(&self, ctx: &mut Ctx<'_, 'tree>) -> R<()> {
        let pending = std::mem::take(&mut ctx.pending_listen_edges);
        let mut seen: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
        for (event_name, listener_name, line) in pending {
            let event_nid = ctx.label_to_nid_ci.get(&event_name.to_lowercase()).cloned();
            let listener_nid = ctx.label_to_nid_ci.get(&listener_name.to_lowercase()).cloned();
            let (event_nid, listener_nid) = match (event_nid, listener_nid) {
                (Some(e), Some(l)) if e != l => (e, l),
                _ => continue,
            };
            if !seen.insert((event_nid.clone(), listener_nid.clone())) {
                continue;
            }
            // Not `add_edge_scored`: this one is deduped on the pair above and
            // must NOT consult `seen_rel_triples`.
            ctx.seen_rel_triples.remove(&(
                event_nid.clone(),
                listener_nid.clone(),
                "listened_by".to_string(),
            ));
            ctx.add_edge_scored(&event_nid, &listener_nid, "listened_by".to_string(), line);
        }
        Ok(())
    }
}

static HOOKS: Php = Php;

pub static CONFIG: EngineConfig = EngineConfig {
    language: "php",
    grammar: || tree_sitter_php::LANGUAGE_PHP.into(),
    class_types: &["class_declaration"],
    function_types: &["function_definition", "method_declaration"],
    import_types: &["namespace_use_clause"],
    call_types: &[
        "function_call_expression",
        "member_call_expression",
        "scoped_call_expression",
        "class_constant_access_expression",
    ],
    function_boundary_types: &["function_definition", "method_declaration"],
    static_prop_types: &["scoped_property_access_expression"],
    helper_fn_names: &["config"],
    container_bind_methods: &["bind", "singleton", "scoped", "instance"],
    event_listener_properties: &["listen", "subscribe"],
    name_field: "name",
    name_fallback_child_types: &["name"],
    body_field: "body",
    body_fallback_child_types: &["declaration_list", "compound_statement"],
    call_function_field: "function",
    call_accessor_node_types: &["member_call_expression"],
    call_accessor_field: "name",
    call_accessor_object_field: "",
    function_label_parens: true,
    resolve_function_name: None,
    type_table_key: None,
    hooks: &HOOKS,
};

pub fn walk_php<'py>(
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

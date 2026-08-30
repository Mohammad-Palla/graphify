//! Ruby on the shared engine.
//!
//! Four things Ruby needs that no earlier language did, all of them real
//! `LanguageConfig` behaviour rather than new invention:
//!
//!   * `sanitize_symbol_name`. A trailing `!`, `?` or `=` is part of a Ruby
//!     method name and would normalize away entirely, taking the method with it
//!     (#3077). The ID uses the encoded form; the LABEL keeps the raw one.
//!   * `qualify_class_name` + `Ctx::scope_segments`. `module Billing; class
//!     Invoice` labels `Billing::Invoice`, and a compact `class
//!     Billing::Invoice` splits into the same two segments, so both declaration
//!     styles converge on ONE label (#2302).
//!   * `Ctx::caller_var_types`. Ruby's receiver table is per-CALLER, where
//!     Java's is per-method and C++'s per-file.
//!   * `Ctx::deferred_raw_calls`. `include`/`extend` mixins are found in a class
//!     body, before the call pass has produced any raw_calls, and the Python
//!     appends them at the very end.
//!
//! `_RUBY_CONFIG.import_types` is EMPTY: `require` is an ordinary method call,
//! so there is no import branch at all.

use std::collections::HashMap;

use pyo3::prelude::*;
use tree_sitter::Node;

use crate::engine::{CallInfo, Ctx, EngineConfig, Handled, LangHooks, RecvTable, R};
use crate::js::ast::children;
use crate::js::emit::Val;
use crate::Outcome;

pub mod helpers;

use helpers::{const_full_name, const_last_name};

struct Ruby;

/// `(receiver, method)` pairs whose result is a class definition (#1640).
const CLASS_FACTORIES: &[(&str, &str)] = &[("Struct", "new"), ("Class", "new"), ("Data", "define")];

impl LangHooks for Ruby {
    /// `class_name.split("::")`, joined onto the enclosing scope.
    fn qualify_class_name(&self, ctx: &Ctx, name: &str) -> R<(String, Vec<String>)> {
        let segments: Vec<String> = name.split("::").map(|s| s.to_string()).collect();
        let mut all = ctx.scope_segments.clone();
        all.extend(segments.clone());
        Ok((all.join("::"), segments))
    }

    fn on_class<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        class_nid: &str,
        _class_name: &str,
        line: usize,
    ) -> R<()> {
        // `class Dog < Animal` puts the base in the `superclass` field.
        if let Some(sup) = node.child_by_field_name("superclass") {
            let mut base = String::new();
            for sub in children(sup) {
                if sub.kind() == "constant" {
                    base = ctx.text(sub)?.to_string();
                    break;
                }
                if sub.kind() == "scope_resolution" {
                    base = const_last_name(ctx, sub)?;
                    break;
                }
            }
            if !base.is_empty() {
                let base_nid = ctx.ensure_named_node(&base, line)?;
                ctx.add_edge(class_nid, &base_nid, "inherits", line);
            }
        }

        // `include` / `extend` / `prepend <Const>` in the body -> a `mixes_in`
        // edge to the module (#1668). The module usually lives in another file,
        // so resolution is deferred to the cross-file Ruby resolver. Only bare or
        // namespaced CONSTANT arguments count; `extend self` is skipped.
        let body = find_body(ctx, node);
        let body = match body {
            Some(b) => b,
            None => return Ok(()),
        };
        for stmt in children(body) {
            if stmt.kind() != "call" || stmt.child_by_field_name("receiver").is_some() {
                continue;
            }
            let m = match stmt.child_by_field_name("method") {
                Some(m) => m,
                None => continue,
            };
            if !matches!(ctx.text(m)?, "include" | "extend" | "prepend") {
                continue;
            }
            let args = match stmt.child_by_field_name("arguments") {
                Some(a) => a,
                None => continue,
            };
            let sline = stmt.start_position().row + 1;
            for arg in children(args) {
                if !matches!(arg.kind(), "constant" | "scope_resolution") {
                    continue;
                }
                // The FULL path, not the last segment: truncating
                // `ActiveSupport::Concern` to `Concern` fabricated edges to any
                // local module of that name (#2302).
                let module = const_full_name(ctx, arg)?;
                if module.is_empty() {
                    continue;
                }
                ctx.deferred_raw_calls.push(vec![
                    ("caller_nid", Val::S(class_nid.to_string())),
                    ("callee", Val::S(module)),
                    ("is_mixin", Val::B(true)),
                    ("source_file", Val::S(ctx.str_path.to_string())),
                    ("source_location", Val::S(format!("L{sline}"))),
                ]);
            }
        }
        Ok(())
    }

    /// `_ruby_extra_walk`: a constant assigned `Struct.new(...)`,
    /// `Class.new(Super)` or `Data.define(...)` DEFINES a class named after the
    /// constant (#1640).
    fn extra_walk<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        _parent_class_nid: Option<&str>,
    ) -> R<Handled> {
        if node.kind() != "assignment" {
            return Ok(Handled::No);
        }
        let (left, right) = match (
            node.child_by_field_name("left"),
            node.child_by_field_name("right"),
        ) {
            (Some(l), Some(r)) if l.kind() == "constant" && r.kind() == "call" => (l, r),
            _ => return Ok(Handled::No),
        };
        let (recv, meth) = match (
            right.child_by_field_name("receiver"),
            right.child_by_field_name("method"),
        ) {
            (Some(rc), Some(m)) if rc.kind() == "constant" => (rc, m),
            _ => return Ok(Handled::No),
        };
        let recv_text = ctx.text(recv)?.to_string();
        let meth_text = ctx.text(meth)?;
        if !CLASS_FACTORIES
            .iter()
            .any(|(r, m)| *r == recv_text && *m == meth_text)
        {
            return Ok(Handled::No);
        }
        let const_name = ctx.text(left)?.to_string();
        if const_name.is_empty() {
            return Ok(Handled::No);
        }
        // Qualified against the enclosing scope, mirroring the class branch:
        // `module Billing; Invoice = Struct.new` labels `Billing::Invoice`.
        let const_segments: Vec<String> = const_name.split("::").map(|s| s.to_string()).collect();
        let mut all = ctx.scope_segments.clone();
        all.extend(const_segments.clone());
        let const_name = all.join("::");
        let line = node.start_position().row + 1;
        let class_nid = ctx.mkid(&[&ctx.stem.clone(), &const_name])?;
        ctx.add_node(&class_nid, &const_name, line);
        ctx.callable_def_nids.insert(class_nid.clone());
        ctx.callable_class_nids.insert(class_nid.clone());
        // Containment always hangs off the FILE node here, unlike the class
        // branch, which prefers an enclosing type.
        let f = ctx.file_nid.clone();
        ctx.add_edge(&f, &class_nid, "contains", line);

        // `Class.new(Super)`: the first positional constant argument is the base.
        if recv_text == "Class" {
            if let Some(args) = children(right).into_iter().find(|c| c.kind() == "argument_list") {
                for arg in children(args) {
                    if !matches!(arg.kind(), "constant" | "scope_resolution") {
                        continue;
                    }
                    let base = const_last_name(ctx, arg)?;
                    if !base.is_empty() {
                        // `ensure_named_node`'s shape, including `origin_file`,
                        // so `_disambiguate_colliding_node_ids` can tell this
                        // file's unresolved reference from another file's.
                        let base_nid = ctx.ensure_named_node(&base, line)?;
                        ctx.add_edge(&class_nid, &base_nid, "inherits", line);
                    }
                    break;
                }
            }
        }

        // Recurse the do/brace block so block-defined methods attach to the
        // class. Without it the default recurse resets the parent to None and
        // every method hangs off the file with a dot-less label.
        let block = children(right)
            .into_iter()
            .find(|c| matches!(c.kind(), "do_block" | "block"));
        if let Some(block) = block {
            let body = children(block)
                .into_iter()
                .find(|c| c.kind() == "body_statement")
                .unwrap_or(block);
            let n = const_segments.len();
            ctx.scope_segments.extend(const_segments);
            let mut result = Ok(());
            for child in children(body) {
                result = crate::engine::walk::walk(ctx, child, Some(&class_nid));
                if result.is_err() {
                    break;
                }
            }
            let keep = ctx.scope_segments.len() - n;
            ctx.scope_segments.truncate(keep);
            result?;
        }
        Ok(Handled::Yes)
    }

    /// Ruby's `call` node carries `receiver` and `method` as DIRECT fields --
    /// there is no intermediate accessor node, so the generic accessor model
    /// does not apply.
    fn call_info<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        _caller_nid: &str,
    ) -> R<Option<CallInfo>> {
        let mut info = CallInfo::default();
        if let Some(meth) = node.child_by_field_name("method") {
            info.callee_name = Some(ctx.text(meth)?.to_string());
        }
        if let Some(recv) = node.child_by_field_name("receiver") {
            info.is_member_call = true;
            match recv.kind() {
                "identifier" | "constant" => {
                    info.member_receiver = Some(ctx.text(recv)?.to_string())
                }
                "scope_resolution" => {
                    // The WHOLE constant path. Truncating to the last segment
                    // bound `ActiveRecord::Base.transaction` to whatever single
                    // class named `Base` the corpus defined -- the god-node guard
                    // catches an ambiguous match, not a unique-but-wrong one
                    // (#3078).
                    let full = const_full_name(ctx, recv)?;
                    info.member_receiver = if full.is_empty() { None } else { Some(full) };
                }
                _ => {}
            }
        }
        Ok(Some(info))
    }

    /// `_ruby_local_class_bindings` per function body, keyed by CALLER.
    fn before_calls<'tree>(&self, ctx: &mut Ctx<'_, 'tree>) -> R<()> {
        let bodies: Vec<(String, Node)> = ctx.function_bodies.clone();
        for (caller_nid, body) in bodies {
            let table = helpers::local_class_bindings(ctx, body)?;
            ctx.caller_var_types.insert(caller_nid, table);
        }
        Ok(())
    }

    /// The receiver's inferred type, when unambiguously known.
    ///
    /// The key is written even when the lookup MISSES: the Python is
    /// `rc_entry["receiver_type"] = ruby_var_types.get(caller_nid, {}).get(
    /// member_receiver)`, so a member call with a receiver always carries the
    /// key, and its value is None when nothing typed it.
    fn raw_call_extra<'tree>(
        &self,
        ctx: &Ctx<'_, 'tree>,
        _node: Node<'tree>,
        caller_nid: &str,
        info: &CallInfo,
        _receiver_types: &RecvTable,
    ) -> Vec<(&'static str, Val)> {
        let recv = match info.member_receiver.as_deref() {
            Some(r) if !r.is_empty() => r,
            _ => return Vec::new(),
        };
        let ty = ctx
            .caller_var_types
            .get(caller_nid)
            .and_then(|t| t.get(recv))
            .cloned()
            .flatten();
        vec![(
            "receiver_type",
            match ty {
                Some(t) => Val::S(t),
                None => Val::None,
            },
        )]
    }
}

/// `_find_body` for the mixin scan. A copy rather than a call into
/// `engine::walk`, which keeps it private -- and the config's fallback list is
/// the only thing it needs.
fn find_body<'tree>(ctx: &Ctx<'_, 'tree>, node: Node<'tree>) -> Option<Node<'tree>> {
    if let Some(b) = node.child_by_field_name(ctx.cfg.body_field) {
        return Some(b);
    }
    children(node)
        .into_iter()
        .find(|c| crate::engine::has(ctx.cfg.body_fallback_child_types, c.kind()))
}

/// `_ruby_sanitize_method_name`: encode a trailing `!`, `?` or `=` (#3077).
fn sanitize_symbol_name(name: &str) -> String {
    if name.is_empty() {
        return name.to_string();
    }
    if let Some(base) = name.strip_suffix('!') {
        return format!("{base}_bang");
    }
    if let Some(base) = name.strip_suffix('?') {
        return format!("{base}_pred");
    }
    if let Some(base) = name.strip_suffix('=') {
        return format!("{base}_eq");
    }
    name.to_string()
}

static HOOKS: Ruby = Ruby;

pub static CONFIG: EngineConfig = EngineConfig {
    language: "ruby",
    grammar: || tree_sitter_ruby::LANGUAGE.into(),
    // `module Foo` is a container node just like `class Foo`, so it gets a node
    // and its methods attach via `method` (#1640). Without it a plain utility
    // module produced no node and its methods hung off the file.
    class_types: &["class", "module"],
    function_types: &["method", "singleton_method"],
    import_types: &[],
    call_types: &["call"],
    function_boundary_types: &["method", "singleton_method"],
    static_prop_types: &[],
    helper_fn_names: &[],
    container_bind_methods: &[],
    event_listener_properties: &[],
    name_field: "name",
    name_fallback_child_types: &["constant", "scope_resolution", "identifier"],
    body_field: "body",
    body_fallback_child_types: &["body_statement"],
    call_function_field: "method",
    call_accessor_node_types: &[],
    call_accessor_field: "attribute",
    call_accessor_object_field: "",
    function_label_parens: true,
    resolve_function_name: None,
    sanitize_symbol_name: Some(sanitize_symbol_name),
    type_table_key: None,
    hooks: &HOOKS,
};

pub fn walk_ruby<'py>(
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

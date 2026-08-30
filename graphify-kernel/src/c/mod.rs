//! C on the shared engine.
//!
//! The smallest language in the set, and deliberately the one after C#: it has
//! ZERO `_is_c` guards in `walk` / `walk_calls`, so it is the config plus one
//! import handler and nothing else. If a pure-config language works end to end,
//! that is the cheapest possible port and it de-risks the six that follow; if it
//! does not, the failure implicates the engine with nothing else in play.
//!
//! Two things it needs that Java and C# did not, both of which are genuinely
//! `_C_CONFIG` fields rather than new behaviour:
//!
//!   * `resolve_function_name_fn`. C names a function by unwrapping its
//!     `declarator` to the innermost identifier -- `int *(*sig(int))(void)` has
//!     no `name` field to read.
//!   * a filesystem seam. `#include "foo.h"` resolves through `Path.resolve()`,
//!     which the walker asks Python to do (`engine::PathResolver`).
//!
//! `_C_CONFIG.class_types` is EMPTY -- C has no class-like construct at all, so
//! `parent_class_nid` is None for the whole file and the class branch never
//! fires. Every function is therefore file-scoped.

use pyo3::prelude::*;
use tree_sitter::Node;

use crate::engine::{Ctx, EngineConfig, LangHooks, R};
use crate::js::ast::children;
use crate::Outcome;

pub mod imports;

struct C;

/// `_get_c_func_name`: unwrap the declarator to the innermost identifier.
///
/// Recursion is on the `declarator` FIELD first and only then on the children,
/// which is what makes a pointer or array declarator collapse to its name.
fn resolve_function_name(ctx: &Ctx, node: Node) -> R<Option<String>> {
    if node.kind() == "identifier" {
        return Ok(Some(ctx.text(node)?.to_string()));
    }
    if let Some(decl) = node.child_by_field_name("declarator") {
        return resolve_function_name(ctx, decl);
    }
    for child in children(node) {
        if child.kind() == "identifier" {
            return Ok(Some(ctx.text(child)?.to_string()));
        }
    }
    Ok(None)
}

/// `_C_PRIMITIVE_TYPE_NODES`: node kinds that are never a user-defined type.
const PRIMITIVE_TYPE_NODES: &[&str] = &[
    "primitive_type",
    "sized_type_specifier",
    "auto",
    "placeholder_type_specifier",
];

/// Declarator wrappers `_c_collect_type_refs` descends through.
const DECLARATOR_WRAPPERS: &[&str] = &[
    "pointer_declarator",
    "reference_declarator",
    "array_declarator",
    "type_qualifier",
    "type_descriptor",
    "abstract_pointer_declarator",
    "abstract_reference_declarator",
    "abstract_array_declarator",
];

/// `_c_collect_type_refs`: the user-defined types named by a C type expression.
///
/// Note what it does NOT do: an unrecognised node kind is dropped whole, with no
/// default recursion. That is the Python's shape, and it is why a `struct foo *`
/// (a `struct_specifier`) contributes nothing here.
fn collect_type_refs(
    ctx: &Ctx,
    node: Option<Node>,
    generic: bool,
    out: &mut Vec<(String, bool)>,
) -> R<()> {
    let node = match node {
        Some(n) if !PRIMITIVE_TYPE_NODES.contains(&n.kind()) => n,
        _ => return Ok(()),
    };
    if node.kind() == "type_identifier" {
        let text = ctx.text(node)?;
        if !text.is_empty() {
            out.push((text.to_string(), generic));
        }
        return Ok(());
    }
    if DECLARATOR_WRAPPERS.contains(&node.kind()) {
        for c in children(node) {
            if c.is_named() {
                collect_type_refs(ctx, Some(c), generic, out)?;
            }
        }
    }
    Ok(())
}

/// One `references` edge per collected type, `generic_arg` overriding the
/// block's own context.
fn emit_type_refs(
    ctx: &mut Ctx,
    func_nid: &str,
    type_node: Option<Node>,
    type_ctx: &'static str,
    line: usize,
) -> R<()> {
    let mut refs: Vec<(String, bool)> = Vec::new();
    collect_type_refs(ctx, type_node, false, &mut refs)?;
    for (ref_name, generic) in refs {
        let c = if generic { "generic_arg" } else { type_ctx };
        let target = ctx.ensure_named_node(&ref_name, line)?;
        if target != func_nid {
            ctx.add_edge_ctx(func_nid, &target, "references", line, c);
        }
    }
    Ok(())
}

impl LangHooks for C {
    fn import_handler<'tree>(&self, ctx: &mut Ctx<'_, 'tree>, node: Node<'tree>) -> R<()> {
        imports::import_c(ctx, node)
    }

    /// The `config.ts_module in ("tree_sitter_c", "tree_sitter_cpp")` block.
    ///
    /// Guarded on the MODULE, not on `_is_c`, which is why an inventory of
    /// `_is_<lang>` guards reported C as needing no hooks at all and the first
    /// parity run reported 241 of 367 files DIVERGENT. The harness found it on
    /// the first attempt; reading the guard list did not.
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
        // The `function_declarator` may be wrapped in pointer/reference
        // declarators (`static void *worker(void *arg)`).
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
}

static HOOKS: C = C;

pub static CONFIG: EngineConfig = EngineConfig {
    language: "c",
    grammar: || tree_sitter_c::LANGUAGE.into(),
    class_types: &[],
    function_types: &["function_definition"],
    import_types: &["preproc_include"],
    call_types: &["call_expression"],
    function_boundary_types: &["function_definition"],
    name_field: "name",
    name_fallback_child_types: &[],
    body_field: "body",
    body_fallback_child_types: &[],
    call_function_field: "function",
    call_accessor_node_types: &["field_expression"],
    call_accessor_field: "field",
    call_accessor_object_field: "",
    function_label_parens: true,
    resolve_function_name: Some(resolve_function_name),
    hooks: &HOOKS,
};

pub fn walk_c<'py>(
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

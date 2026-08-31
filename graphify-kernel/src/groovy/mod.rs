//! Groovy, on the shared engine — Java's shared branch, and nothing else.
//!
//! # The whole point of this module is what it does NOT do
//!
//! Groovy looks like Java to `_extract_generic` and it is a trap. An inventory
//! of `_is_groovy` finds ONE line -- the definition at `engine.py:2915` -- and
//! that variable is then never used. The real guard is
//! `config.ts_module in ("tree_sitter_java", "tree_sitter_groovy")`, invisible to
//! an `_is_<lang>` grep. This is the same shape that put 241 of 367 C files
//! DIVERGENT on their first run, so it is spelled out rather than rediscovered.
//!
//! Everything else Java does is behind `_is_java` and Groovy must NOT get it:
//! parameter types, return types, function-level annotations (`engine.py:4326`),
//! field/property declarations (3936, 3957), the enclosing-class link (4640),
//! and the whole Java call-resolution path (5389, 5515, 5591). Implementing
//! Java's other hooks here would produce extra edges on every Groovy file --
//! plausible-looking ones, which is the failure mode this project keeps
//! meeting.
//!
//! So: two hooks, both delegating to `java/`, and one of them with a flag.
//!
//! # Spock
//!
//! `extract_groovy` runs `_is_spock_file` / `_extract_spock_fallback` AFTER
//! `_extract_generic` returns, so it sits above the seam and needs nothing here
//! -- exactly as `_augment_cpp_string_tests` does for C++. The kernel returns
//! the `_extract_generic` result and Python layers the fallback on top of it.

use pyo3::prelude::*;
use tree_sitter::Node;

use crate::engine::{Ctx, EngineConfig, LangHooks, R};
use crate::Outcome;

struct Groovy;

impl LangHooks for Groovy {
    /// `_GROOVY_CONFIG.import_handler` IS `_import_java` -- the same function
    /// object, not a Groovy variant of it.
    fn import_handler<'tree>(&self, ctx: &mut Ctx<'_, 'tree>, node: Node<'tree>) -> R<()> {
        crate::java::imports::import_java(ctx, node)
    }

    fn on_class<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        class_nid: &str,
        _class_name: &str,
        line: usize,
    ) -> R<()> {
        crate::java::emit_inheritance(ctx, node, class_nid, line)?;
        // `dotted: false`. The class-level substitution is guarded `if "." in
        // anno_raw and _is_java`, so `@org.pkg.Foo` stays the bare `Foo` for
        // Groovy. The Python explains why: `_resolve_java_type_references` maps
        // internal FQNs back to real nodes for Java (#2504) and Groovy has no
        // such resolver pass, so it keeps the legacy bare-name stub. Passing
        // `true` here would retarget every inline-qualified annotation edge.
        crate::java::emit_annotations(ctx, class_nid, node, line, false)?;
        // NO record components: `record_declaration` is not in Groovy's
        // `class_types`, so the branch is unreachable rather than suppressed.
        Ok(())
    }
}

static HOOKS: Groovy = Groovy;

pub static CONFIG: EngineConfig = EngineConfig {
    language: "groovy",
    grammar: || tree_sitter_groovy::LANGUAGE.into(),
    class_types: &["class_declaration", "interface_declaration"],
    function_types: &["method_declaration", "constructor_declaration"],
    import_types: &["import_declaration"],
    call_types: &["method_invocation"],
    function_boundary_types: &["method_declaration", "constructor_declaration"],
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
    // `_GROOVY_CONFIG` does not set this, so it is the LanguageConfig default
    // `"attribute"`, not `""`. Unreachable while `call_accessor_node_types` is
    // empty, but `test_engine_configs_match_their_language_config` compares it.
    call_accessor_field: "attribute",
    call_accessor_object_field: "",
    function_label_parens: true,
    resolve_function_name: None,
    sanitize_symbol_name: None,
    type_table_key: None,
    hooks: &HOOKS,
};

pub fn walk_groovy<'py>(
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

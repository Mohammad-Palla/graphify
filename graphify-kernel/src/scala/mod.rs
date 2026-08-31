//! Scala, on the shared engine.
//!
//! Five guard sites in `engine.py`, all `_is_scala` (no module-name guard, so
//! the `_is_<lang>` inventory is complete here -- unlike C and Groovy):
//!
//! ```text
//! 3600  on_class         extend clause; FIRST base inherits, the rest mixes_in
//! 4105  before_function  val_definition / var_definition field types, FALLS THROUGH
//! 4131  before_function  self_type -> `requires`, and RETURNS
//! 4546  on_function      parameter types + return_type
//! 5192  call_info        identifier / field_expression
//! ```
//!
//! Grammar note worth keeping: Scala's parse ceiling is **100%** on
//! typelevel/cats and akka/akka, the only language ported here with no parse-error
//! floor at all. That is why it was ported and Groovy was not.

pub mod helpers;

use pyo3::prelude::*;
use tree_sitter::Node;

use crate::engine::{CallInfo, Ctx, EngineConfig, Handled, LangHooks, R};
use crate::js::ast::children;
use crate::js::emit::{EdgeRow, Val};
use crate::Outcome;

struct Scala;

impl LangHooks for Scala {
    /// `_import_scala`.
    ///
    /// Pure string work -- no resolver. `import a.b.{C, D}` takes the FIRST
    /// `stable_id`/`identifier` child, keeps the last dotted segment, strips
    /// `{`, `}` and spaces, and skips the wildcard `_`. The `break` is
    /// unconditional, so only the first such child is ever considered.
    fn import_handler<'tree>(&self, ctx: &mut Ctx<'_, 'tree>, node: Node<'tree>) -> R<()> {
        for child in children(node) {
            if !matches!(child.kind(), "stable_id" | "identifier") {
                continue;
            }
            let raw = ctx.text(child)?;
            let module_name = raw
                .rsplit('.')
                .next()
                .unwrap_or("")
                .trim_matches(|c| c == '{' || c == '}' || c == ' ');
            if !module_name.is_empty() && module_name != "_" {
                let tgt = crate::ids::make_id_ascii(&[module_name]).ok_or("non_ascii_id")?;
                let line = node.start_position().row + 1;
                let file_nid = ctx.file_nid.clone();
                let src_file = ctx.str_path.to_string();
                // NOTE: no `confidence_score` here. `_import_scala`'s literal
                // omits it, unlike `_import_lua`'s. Key order and key SET are
                // both part of byte-identical.
                ctx.edges.push(EdgeRow {
                    source: file_nid,
                    target: tgt,
                    relation: "imports",
                    fields: vec![
                        ("context", Val::Static("import")),
                        ("confidence", Val::Static("EXTRACTED")),
                        ("source_file", Val::S(src_file)),
                        ("source_location", Val::S(format!("L{line}"))),
                        ("weight", Val::F(1.0)),
                    ],
                });
            }
            // The Python breaks out of the loop whether or not it emitted.
            break;
        }
        Ok(())
    }

    /// `extend` (first base `inherits`, the rest `mixes_in`) plus
    /// `class_parameters` as constructor-as-field references.
    fn on_class<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        class_nid: &str,
        _class_name: &str,
        _line: usize,
    ) -> R<()> {
        // The `extend` FIELD first, then a positional scan for `extends_clause`.
        let extend = node
            .child_by_field_name("extend")
            .or_else(|| children(node).into_iter().find(|c| c.kind() == "extends_clause"));

        if let Some(extend) = extend {
            let mut bases: Vec<(String, usize)> = Vec::new();
            for c in children(extend) {
                match c.kind() {
                    "type_identifier" => {
                        bases.push((ctx.text(c)?.to_string(), c.start_position().row + 1));
                    }
                    "generic_type" => {
                        let base = c.child_by_field_name("type").or_else(|| {
                            children(c).into_iter().find(|sc| sc.kind() == "type_identifier")
                        });
                        if let Some(b) = base {
                            // The line is the GENERIC's, not the base's -- the
                            // Python uses `c.start_point`, not `base.start_point`.
                            bases.push((ctx.text(b)?.to_string(), c.start_position().row + 1));
                        }
                    }
                    _ => {}
                }
            }
            for (idx, (base_name, base_line)) in bases.into_iter().enumerate() {
                // First base is the superclass; every later one is a mixin.
                let rel = if idx == 0 { "inherits" } else { "mixes_in" };
                let base_nid = ctx.ensure_named_node(&base_name, base_line)?;
                if base_nid != class_nid {
                    ctx.add_edge(class_nid, &base_nid, rel, base_line);
                }
            }
        }

        for c in children(node) {
            if c.kind() != "class_parameters" {
                continue;
            }
            for cp in children(c) {
                if cp.kind() != "class_parameter" {
                    continue;
                }
                let ptype = match cp.child_by_field_name("type") {
                    Some(t) => t,
                    None => continue,
                };
                let cp_line = cp.start_position().row + 1;
                let mut refs = Vec::new();
                helpers::collect_type_refs(ctx, Some(ptype), false, &mut refs)?;
                for (ref_name, role) in refs {
                    let context = helpers::ctx_for(role, "field");
                    let target = ctx.ensure_named_node(&ref_name, cp_line)?;
                    if target != class_nid {
                        ctx.add_edge_ctx(class_nid, &target, "references", cp_line, context);
                    }
                }
            }
        }
        Ok(())
    }

    /// Two Python branches share this hook position, and they differ in whether
    /// they RETURN:
    ///
    /// * `val_definition` / `var_definition` emits field references and then
    ///   falls through, deliberately -- the comment says "so any call
    ///   expressions in the initializer get walked". Returning `Handled::Yes`
    ///   here would silently drop every call inside a field initializer.
    /// * `self_type` emits `requires` edges and RETURNS.
    fn before_function<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        parent_class_nid: Option<&str>,
    ) -> R<Handled> {
        let parent = match parent_class_nid {
            Some(p) if !p.is_empty() => p.to_string(),
            _ => return Ok(Handled::No),
        };
        let t = node.kind();

        if matches!(t, "val_definition" | "var_definition") {
            if let Some(type_node) = node.child_by_field_name("type") {
                let line = node.start_position().row + 1;
                let mut refs = Vec::new();
                helpers::collect_type_refs(ctx, Some(type_node), false, &mut refs)?;
                for (ref_name, role) in refs {
                    let context = helpers::ctx_for(role, "field");
                    let target = ctx.ensure_named_node(&ref_name, line)?;
                    if target != parent {
                        ctx.add_edge_ctx(&parent, &target, "references", line, context);
                    }
                }
            }
            // Falls through -- see the doc comment.
            return Ok(Handled::No);
        }

        if t == "self_type" {
            // `self_type` carries no field names, so the type is found
            // POSITIONALLY: named[0] is the binder, named[1] the type. `self =>`
            // binds a name with no type, so fewer than two named children
            // correctly yields nothing rather than misreading the binder.
            let named: Vec<Node> = children(node).into_iter().filter(|c| c.is_named()).collect();
            if named.len() >= 2 {
                let line = node.start_position().row + 1;
                let mut refs = Vec::new();
                helpers::collect_type_refs(ctx, Some(named[1]), false, &mut refs)?;
                for (ref_name, _role) in refs {
                    let target = ctx.ensure_named_node(&ref_name, line)?;
                    if target != parent {
                        // `requires`, with NO context -- a structural
                        // precondition, not a mixin or a reference.
                        ctx.add_edge(&parent, &target, "requires", line);
                    }
                }
            }
            return Ok(Handled::Yes);
        }

        Ok(Handled::No)
    }

    /// Parameter types and the return type.
    fn on_function<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        func_nid: &str,
        _func_name: &str,
        line: usize,
        _parent_class_nid: Option<&str>,
    ) -> R<()> {
        // `parameters` is an UNNAMED child, so it is found by scanning rather
        // than with `child_by_field_name`.
        let params = children(node).into_iter().find(|c| c.kind() == "parameters");
        if let Some(params) = params {
            for p in children(params) {
                if p.kind() != "parameter" {
                    continue;
                }
                let ptype = match p.child_by_field_name("type") {
                    Some(t) => t,
                    None => continue,
                };
                let mut refs = Vec::new();
                helpers::collect_type_refs(ctx, Some(ptype), false, &mut refs)?;
                for (ref_name, role) in refs {
                    let context = helpers::ctx_for(role, "parameter_type");
                    let target = ctx.ensure_named_node(&ref_name, line)?;
                    if target != func_nid {
                        ctx.add_edge_ctx(func_nid, &target, "references", line, context);
                    }
                }
            }
        }
        if let Some(return_node) = node.child_by_field_name("return_type") {
            let mut refs = Vec::new();
            helpers::collect_type_refs(ctx, Some(return_node), false, &mut refs)?;
            for (ref_name, role) in refs {
                let context = helpers::ctx_for(role, "return_type");
                let target = ctx.ensure_named_node(&ref_name, line)?;
                if target != func_nid {
                    ctx.add_edge_ctx(func_nid, &target, "references", line, context);
                }
            }
        }
        Ok(())
    }

    /// The callee is the call node's FIRST child, named or not.
    fn call_info<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        _caller_nid: &str,
    ) -> R<Option<CallInfo>> {
        let mut info = CallInfo::default();
        // `node.children[0]` -- the raw child list, so an unnamed token counts.
        let first = node.child(0);
        if let Some(first) = first {
            match first.kind() {
                "identifier" => {
                    info.callee_name = Some(ctx.text(first)?.to_string());
                }
                "field_expression" => {
                    info.is_member_call = true;
                    if let Some(field) = first.child_by_field_name("field") {
                        info.callee_name = Some(ctx.text(field)?.to_string());
                    } else {
                        // Last identifier wins: the Python walks `reversed(...)`.
                        for child in children(first).into_iter().rev() {
                            if child.kind() == "identifier" {
                                info.callee_name = Some(ctx.text(child)?.to_string());
                                break;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(Some(info))
    }
}

static HOOKS: Scala = Scala;

pub static CONFIG: EngineConfig = EngineConfig {
    language: "scala",
    grammar: || tree_sitter_scala::LANGUAGE.into(),
    class_types: &["class_definition", "object_definition"],
    function_types: &["function_definition"],
    import_types: &["import_declaration"],
    call_types: &["call_expression"],
    function_boundary_types: &["function_definition"],
    static_prop_types: &[],
    helper_fn_names: &[],
    container_bind_methods: &[],
    event_listener_properties: &[],
    name_field: "name",
    name_fallback_child_types: &["identifier"],
    body_field: "body",
    body_fallback_child_types: &["template_body"],
    // `call_function_field=""` -- unset. The callee comes from `call_info`.
    call_function_field: "",
    call_accessor_node_types: &["field_expression"],
    call_accessor_field: "field",
    call_accessor_object_field: "",
    function_label_parens: true,
    resolve_function_name: None,
    sanitize_symbol_name: None,
    type_table_key: None,
    hooks: &HOOKS,
};

pub fn walk_scala<'py>(
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

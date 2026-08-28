//! The declaration walk: `engine.py::_extract_generic`'s inner `walk`, plus
//! `_js_extra_walk`, `_ts_extra_walk`, `_scan_js_nested_function_declarations`
//! and `_ts_receiver_type_table`.
//!
//! The dispatch sets come from `_TS_CONFIG` / `_JS_CONFIG` and are `match` arms
//! parameterized by [`Dialect`] rather than runtime sets, because for one file
//! they are constants:
//!
//! ```text
//!                 TypeScript                        JavaScript
//! class_types     abstract_class_declaration,       class_declaration
//!                 class_declaration, enum_declaration,
//!                 interface_declaration, type_alias_declaration
//! function_types  function_declaration,             function_declaration,
//!                 generator_function_declaration,   generator_function_declaration,
//!                 method_definition,                method_definition
//!                 method_signature
//! import_types    export_statement, import_statement            (same)
//! call_types      call_expression, new_expression                (same)
//! ```
//!
//! The TypeScript-only kinds do not exist in the JavaScript grammar, so keying on
//! the dialect is redundant *today* -- and is done anyway, because "that kind
//! cannot appear here" is an assumption about a third-party grammar's shape that
//! a version bump can quietly falsify, and the failure would be an extra node
//! rather than an error.
//!
//! `static_prop_types`, `helper_fn_names`, `container_bind_methods` and
//! `event_listener_properties` are all empty for TypeScript, and
//! `name_fallback_child_types` / `body_fallback_child_types` are empty tuples, so
//! `_find_body` is exactly `child_by_field_name("body")` and the name fallback
//! loop never runs. Those branches are therefore absent below rather than ported
//! as dead code -- if a future config gives TypeScript any of them, the walker
//! must be updated, which is why they are named here.

use std::collections::HashSet;

use tree_sitter::Node;

use super::ast::{children, line_of};
use super::imports;
use super::pat;
use super::{Ctx, R};

/// Which member of the JS family a file is, i.e. which `LanguageConfig` the
/// Python path would have used.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    /// `_TS_CONFIG` / `_TSX_CONFIG`.
    TypeScript,
    /// `_JS_CONFIG`.
    JavaScript,
}

fn is_class_type(kind: &str, dialect: Dialect) -> bool {
    match kind {
        "class_declaration" => true,
        "abstract_class_declaration"
        | "interface_declaration"
        | "enum_declaration"
        | "type_alias_declaration" => dialect == Dialect::TypeScript,
        _ => false,
    }
}

fn is_function_type(kind: &str, dialect: Dialect) -> bool {
    match kind {
        "function_declaration" | "generator_function_declaration" | "method_definition" => true,
        "method_signature" => dialect == Dialect::TypeScript,
        _ => false,
    }
}

pub fn is_call_type(kind: &str) -> bool {
    matches!(kind, "call_expression" | "new_expression")
}

/// `walk(node, parent_class_nid)`.
pub fn walk<'t>(ctx: &mut Ctx<'_, 't>, node: Node<'t>, parent_class_nid: Option<&str>) -> R<()> {
    let t = node.kind();

    // ── Import types ────────────────────────────────────────────────────────
    if matches!(t, "import_statement" | "export_statement") {
        imports::import_js(ctx, node)?;
        // `_import_js` returns None for JS/TS, so the `type=module` node branch
        // (Swift's module-naming import handler) never fires here.
        if t == "export_statement" {
            // Only a re-export (a `from` clause, i.e. a string child) stops the
            // walk. `export const x = 1` / `export class C {}` must fall through
            // to its children.
            let has_source = children(node).iter().any(|c| c.kind() == "string");
            if !has_source {
                for child in children(node) {
                    walk(ctx, child, parent_class_nid)?;
                }
            }
        }
        return Ok(());
    }

    // ── Class types ─────────────────────────────────────────────────────────
    if is_class_type(t, ctx.dialect) {
        let Some(name_node) = node.child_by_field_name("name") else {
            return Ok(()); // `name_fallback_child_types` is empty for TS
        };
        let class_name = ctx.text(name_node)?.to_string();
        // `_make_id(stem, ".".join(namespace_stack), class_name)` -- the middle
        // part is always "" for TS (only the C# handler pushes a namespace) and
        // `make_id` filters empty parts before joining.
        let class_nid = ctx.mkid(&[&ctx.stem.clone(), "", &class_name])?;
        let line = line_of(node);
        ctx.add_node(&class_nid, &class_name, line);
        ctx.callable_def_nids.insert(class_nid.clone()); // callable via constructor
        ctx.callable_class_nids.insert(class_nid.clone()); // ...but ONLY that (#2137)
        match parent_class_nid {
            // The `!= class_nid` guard avoids a self-loop when same-name nesting
            // collides ids, since class ids omit the enclosing type name.
            Some(p) if p != class_nid => ctx.add_edge(p, &class_nid, "contains", line),
            _ => {
                let file_nid = ctx.file_nid.clone();
                ctx.add_edge(&file_nid, &class_nid, "contains", line)
            }
        }

        // Decorators mint `references` edges through `ensure_named_node`, which
        // creates SOURCELESS stub nodes for the corpus-level rewire. Measured at
        // 1 such edge across 900 Bun `.ts` files, so the coverage cost of
        // deferring is negligible against the risk of getting the stub shape and
        // the member-attribution rules (class vs method vs field) subtly wrong.
        if has_decorator(node) {
            return Err("decorator");
        }
        if let Some(parent) = node.parent() {
            if parent.kind() == "export_statement"
                && children(parent).iter().any(|c| c.kind() == "decorator")
            {
                return Err("decorator");
            }
        }

        if let Some(body) = node.child_by_field_name("body") {
            for child in children(body) {
                walk(ctx, child, Some(&class_nid))?;
            }
        }
        return Ok(());
    }

    // ── Function types ──────────────────────────────────────────────────────
    if is_function_type(t, ctx.dialect) {
        let func_name = match node.child_by_field_name("name") {
            Some(n) => ctx.text(n)?.to_string(),
            None => return Ok(()),
        };
        if func_name.is_empty() {
            return Ok(());
        }
        // `sanitize_symbol_name_fn` is None for TS, so sanitized == func_name.
        if !ctx.normalizes_to_something(&func_name)? {
            return Ok(());
        }
        let line = line_of(node);
        let func_nid = match parent_class_nid {
            Some(p) => {
                let nid = ctx.mkid(&[p, &func_name])?;
                ctx.add_node(&nid, &format!(".{func_name}()"), line);
                ctx.add_edge(p, &nid, "method", line);
                nid
            }
            None => {
                let nid = ctx.mkid(&[&ctx.stem.clone(), &func_name])?;
                ctx.add_node(&nid, &format!("{func_name}()"), line);
                let file_nid = ctx.file_nid.clone();
                ctx.add_edge(&file_nid, &nid, "contains", line);
                nid
            }
        };
        ctx.callable_def_nids.insert(func_nid.clone());
        let bound = pat::local_bound_names(node, ctx.src);
        ctx.local_bound_names.insert(func_nid.clone(), bound);

        // Constructor parameter properties (`constructor(private svc: Svc)`) type
        // the receiver for `this.svc.m()`. Inserted BEFORE the receiver-table
        // pass, so they win on a name clash.
        if func_name == "constructor" {
            if let Some(params) = node.child_by_field_name("parameters") {
                for p in children(params) {
                    if p.kind() != "required_parameter" {
                        continue;
                    }
                    let has_modifier = children(p)
                        .iter()
                        .any(|c| matches!(c.kind(), "accessibility_modifier" | "readonly"));
                    if !has_modifier {
                        continue;
                    }
                    let (Some(name_n), Some(type_n)) = (
                        p.child_by_field_name("pattern"),
                        p.child_by_field_name("type"),
                    ) else {
                        continue;
                    };
                    let pname = ctx.text(name_n)?.to_string();
                    for tc in children(type_n) {
                        if tc.kind() == "type_identifier" {
                            let ptype = ctx.text(tc)?.to_string();
                            if !pname.is_empty() && !ptype.is_empty() {
                                ctx.type_table.insert_if_absent(&pname, &ptype);
                            }
                            break;
                        }
                    }
                }
            }
        }

        let body = node.child_by_field_name("body");
        if let Some(body) = body {
            method_assignments_in_body(ctx, body, parent_class_nid, &func_nid)?;
            ctx.function_bodies.push((func_nid.clone(), body));
            scan_nested_function_declarations(ctx, body, &func_nid)?;
        }
        return Ok(());
    }

    // ── Extra walks ─────────────────────────────────────────────────────────
    if js_extra_walk(ctx, node, parent_class_nid)? {
        return Ok(());
    }
    // `_ts_extra_walk` is gated on `_is_typescript`, not `_is_js_family`: enum
    // members and `namespace` containers have no JavaScript spelling.
    if ctx.dialect == Dialect::TypeScript && ts_extra_walk(ctx, node, parent_class_nid)? {
        return Ok(());
    }

    // A Python-only construct in the TS grammar's namespace; kept because the
    // Python branch is language-agnostic and a `decorated_definition` node would
    // otherwise fall to the default recurse with parent_class_nid=None.
    if t == "decorated_definition" {
        for child in children(node) {
            walk(ctx, child, parent_class_nid)?;
        }
        return Ok(());
    }

    // `t == "ERROR"` recurses in Python; unreachable here because a tree with any
    // error node defers the whole file before the walk starts.

    // Default: recurse. NOTE parent_class_nid is dropped to None, so a class
    // nested inside an `if` inside a class body is file-contained, not
    // type-contained -- matching Python exactly.
    for child in children(node) {
        walk(ctx, child, None)?;
    }
    Ok(())
}

fn has_decorator(node: Node) -> bool {
    if node.kind() == "decorator" {
        return true;
    }
    children(node).iter().any(|c| has_decorator(*c))
}

/// The `this.m = fn` / `obj.m = fn` scan Python runs over a function body before
/// tracking the body itself (#2552 / #1630 lineage).
fn method_assignments_in_body<'t>(
    ctx: &mut Ctx<'_, 't>,
    body: Node<'t>,
    parent_class_nid: Option<&str>,
    func_nid: &str,
) -> R<()> {
    let function_owner_nid = parent_class_nid.unwrap_or(func_nid).to_string();

    // `const o = {}; o.m = fn` only counts when `o` is a direct object-literal
    // binding in THIS body -- the scope check that keeps a bare-named phantom
    // god node from being minted (#1077).
    let mut object_bindings: Vec<(String, Node)> = Vec::new();
    for stmt in children(body) {
        if !matches!(stmt.kind(), "lexical_declaration" | "variable_declaration") {
            continue;
        }
        for declarator in children(stmt) {
            if declarator.kind() != "variable_declarator" {
                continue;
            }
            let name = declarator.child_by_field_name("name");
            let value = declarator.child_by_field_name("value");
            if let (Some(n), Some(v)) = (name, value) {
                if n.kind() == "identifier" && v.kind() == "object" {
                    // Python keys a DICT by name, so a second `const o = {}` in the
                    // same body replaces the first -- and the owner node's line
                    // number comes from whichever declarator won. Appending here
                    // and taking the first match would emit the earlier line.
                    let name_s = ctx.text(n)?.to_string();
                    match object_bindings.iter_mut().find(|(k, _)| *k == name_s) {
                        Some(slot) => slot.1 = declarator,
                        None => object_bindings.push((name_s, declarator)),
                    }
                }
            }
        }
    }

    let mut contained_owners: HashSet<String> = HashSet::new();
    for stmt in children(body) {
        if stmt.kind() != "expression_statement" {
            continue;
        }
        let Some(assign) = children(stmt)
            .into_iter()
            .find(|c| c.kind() == "assignment_expression")
        else {
            continue;
        };
        let Some(val) = assign.child_by_field_name("right") else {
            continue;
        };
        if !pat::is_function_value(val.kind()) {
            continue;
        }
        let Some(tgt) = pat::member_assignment_target(assign.child_by_field_name("left"), ctx.src)
        else {
            continue;
        };
        let (owner_nid, m_name) = match tgt {
            pat::AssignTarget::This(member) => (function_owner_nid.clone(), member),
            pat::AssignTarget::Object { owner, member } => {
                let Some((_, declarator)) =
                    object_bindings.iter().find(|(n, _)| *n == owner).copied_pair()
                else {
                    continue;
                };
                let owner_nid = ctx.mkid(&[&function_owner_nid, &owner])?;
                let owner_line = line_of(declarator);
                ctx.add_node(&owner_nid, &owner, owner_line);
                if contained_owners.insert(owner_nid.clone()) {
                    ctx.add_edge(&function_owner_nid, &owner_nid, "contains", owner_line);
                }
                (owner_nid, member)
            }
            // `exports.x = fn` / `Foo.prototype.x = fn` inside a function body:
            // Python's `else: continue`.
            _ => continue,
        };
        let m_line = line_of(stmt);
        let m_nid = ctx.mkid(&[&owner_nid, &m_name])?;
        ctx.add_node(&m_nid, &format!(".{m_name}()"), m_line);
        ctx.add_edge(&owner_nid, &m_nid, "method", m_line);
        if let Some(m_body) = val.child_by_field_name("body") {
            ctx.function_bodies.push((m_nid, m_body));
        }
    }
    Ok(())
}

/// Helper so the `find` above can hand back an owned pair without cloning the
/// whole binding list.
trait CopiedPair<'t> {
    fn copied_pair(self) -> Option<(String, Node<'t>)>;
}
impl<'t> CopiedPair<'t> for Option<&(String, Node<'t>)> {
    fn copied_pair(self) -> Option<(String, Node<'t>)> {
        self.map(|(a, b)| (a.clone(), *b))
    }
}

/// `_scan_js_nested_function_declarations`.
pub fn scan_nested_function_declarations<'t>(
    ctx: &mut Ctx<'_, 't>,
    container: Node<'t>,
    parent_nid: &str,
) -> R<()> {
    for child in children(container) {
        if matches!(
            child.kind(),
            "function_declaration" | "generator_function_declaration"
        ) {
            let func_name = match child.child_by_field_name("name") {
                Some(n) => ctx.text(n)?.to_string(),
                None => String::new(),
            };
            if !func_name.is_empty() && ctx.normalizes_to_something(&func_name)? {
                let line = line_of(child);
                let nested_nid = ctx.mkid(&[parent_nid, &func_name])?;
                ctx.add_node(&nested_nid, &format!("{func_name}()"), line);
                ctx.add_edge(parent_nid, &nested_nid, "contains", line);
                ctx.callable_def_nids.insert(nested_nid.clone());
                let bound = pat::local_bound_names(child, ctx.src);
                ctx.local_bound_names.insert(nested_nid.clone(), bound);
                if let Some(nested_body) = child.child_by_field_name("body") {
                    ctx.function_bodies.push((nested_nid.clone(), nested_body));
                    scan_nested_function_declarations(ctx, nested_body, &nested_nid)?;
                }
            }
        } else if pat::is_function_value(child.kind()) {
            // An anonymous closure is not a node, but a `function` declared inside
            // its body still belongs to the enclosing NAMED scope.
            if let Some(b) = child.child_by_field_name("body") {
                scan_nested_function_declarations(ctx, b, parent_nid)?;
            }
        } else {
            scan_nested_function_declarations(ctx, child, parent_nid)?;
        }
    }
    Ok(())
}

/// `_js_extra_walk`. Returns whether the node was handled.
fn js_extra_walk<'t>(ctx: &mut Ctx<'_, 't>, node: Node<'t>, parent_class_nid: Option<&str>) -> R<bool> {
    // CommonJS / prototype member assignments whose value is a function.
    if node.kind() == "expression_statement" {
        if let Some(assign) = children(node)
            .into_iter()
            .find(|c| c.kind() == "assignment_expression")
        {
            if let Some(value) = assign.child_by_field_name("right") {
                if let Some(target) =
                    pat::member_assignment_target(assign.child_by_field_name("left"), ctx.src)
                {
                    let line = line_of(node);
                    if pat::is_function_value(value.kind()) {
                        let nid = match &target {
                            pat::AssignTarget::Exports(member) => {
                                let nid = ctx.mkid(&[&ctx.stem.clone(), member])?;
                                ctx.add_node(&nid, &format!("{member}()"), line);
                                let file_nid = ctx.file_nid.clone();
                                ctx.add_edge(&file_nid, &nid, "contains", line);
                                Some(nid)
                            }
                            pat::AssignTarget::Prototype { owner, member } => {
                                let owner_nid = ctx.mkid(&[&ctx.stem.clone(), owner])?;
                                let nid = ctx.mkid(&[&owner_nid, member])?;
                                ctx.add_node(&nid, &format!(".{member}()"), line);
                                ctx.add_edge(&owner_nid, &nid, "method", line);
                                Some(nid)
                            }
                            _ => None,
                        };
                        if let Some(nid) = nid {
                            ctx.callable_def_nids.insert(nid.clone());
                            let bound = pat::local_bound_names(value, ctx.src);
                            ctx.local_bound_names.insert(nid.clone(), bound);
                            if let Some(body) = value.child_by_field_name("body") {
                                ctx.function_bodies.push((nid, body));
                            }
                            return Ok(true);
                        }
                    } else if let pat::AssignTarget::Exports(member) = &target {
                        // #3035: `exports.handler = wrapper(async (req) => ...)`
                        let mut inner = Some(value);
                        while let Some(i) = inner {
                            if !matches!(i.kind(), "as_expression" | "satisfies_expression") {
                                break;
                            }
                            inner = super::ast::named_children(i).first().copied();
                        }
                        if let Some(i) = inner {
                            if matches!(i.kind(), "call_expression" | "new_expression") {
                                let mut closures = Vec::new();
                                pat::topmost_closures(i, &mut closures);
                                if !closures.is_empty() {
                                    let nid = ctx.mkid(&[&ctx.stem.clone(), member])?;
                                    ctx.add_node(&nid, &format!("{member}()"), line);
                                    let file_nid = ctx.file_nid.clone();
                                    ctx.add_edge(&file_nid, &nid, "contains", line);
                                    ctx.callable_def_nids.insert(nid.clone());
                                    for closure in closures {
                                        if let Some(body) = closure.child_by_field_name("body") {
                                            let locals =
                                                pat::local_bound_names(closure, ctx.src);
                                            ctx.closure_locals_by_body.insert(body.id(), locals);
                                            ctx.function_bodies.push((nid.clone(), body));
                                        }
                                    }
                                    return Ok(true);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Class fields whose value is a function: `class C { handler = () => {} }`.
    if let Some(pcn) = parent_class_nid {
        if matches!(node.kind(), "field_definition" | "public_field_definition") {
            let prop = node
                .child_by_field_name("property")
                .or_else(|| node.child_by_field_name("name"));
            let value = node.child_by_field_name("value");
            if let (Some(prop), Some(value)) = (prop, value) {
                if pat::is_function_value(value.kind()) {
                    let field_name = ctx.text(prop)?.to_string();
                    if !field_name.is_empty() {
                        let line = line_of(node);
                        let nid = ctx.mkid(&[pcn, &field_name])?;
                        ctx.add_node(&nid, &format!(".{field_name}()"), line);
                        ctx.add_edge(pcn, &nid, "method", line);
                        ctx.callable_def_nids.insert(nid.clone());
                        let bound = pat::local_bound_names(value, ctx.src);
                        ctx.local_bound_names.insert(nid.clone(), bound);
                        if let Some(body) = value.child_by_field_name("body") {
                            ctx.function_bodies.push((nid, body));
                        }
                        return Ok(true);
                    }
                }
            }
        }
    }

    if matches!(node.kind(), "lexical_declaration" | "variable_declaration") {
        // CJS requires emit edges but do NOT block the rest of the handling.
        let file_nid = ctx.file_nid.clone();
        let require_found = imports::require_imports_js(ctx, node, &file_nid)?;

        // Scope guard (#1077): only module-level declarations mint nodes. Without
        // it, `const x = ...` inside a callback mints a bare-named node that
        // collides across unrelated files into a phantom god node.
        let parent = node.parent();
        let is_exported = parent.map_or(false, |p| p.kind() == "export_statement");
        let is_module_level = match parent {
            None => false,
            Some(p) => {
                p.kind() == "program"
                    || (is_exported
                        && p.parent().map_or(false, |pp| pp.kind() == "program"))
            }
        };

        let mut arrow_found = false;
        let mut const_found = false;
        if node.kind() == "lexical_declaration" && is_module_level {
            for child in children(node) {
                if child.kind() != "variable_declarator" {
                    continue;
                }
                let value = child.child_by_field_name("value");
                let name_node = child.child_by_field_name("name");
                let is_exported_scalar_binding = is_exported
                    && match name_node {
                        Some(n) if n.kind() == "identifier" => {
                            ctx.normalizes_to_something(ctx.text(n)?)?
                        }
                        _ => false,
                    };
                let Some(value) = value else { continue };
                if pat::is_function_value(value.kind()) {
                    // `const f = () => {}` / `const f = function(){}`
                    if let Some(nn) = name_node {
                        let func_name = ctx.text(nn)?.to_string();
                        let line = line_of(child);
                        // A name that normalizes to nothing (minified `$`) would
                        // collapse the id onto the file stem and leak the scan
                        // path (#1899).
                        if !ctx.normalizes_to_something(&func_name)? {
                            continue;
                        }
                        let func_nid = ctx.mkid(&[&ctx.stem.clone(), &func_name])?;
                        ctx.add_node(&func_nid, &format!("{func_name}()"), line);
                        let file_nid = ctx.file_nid.clone();
                        ctx.add_edge(&file_nid, &func_nid, "contains", line);
                        ctx.callable_def_nids.insert(func_nid.clone());
                        let bound = pat::local_bound_names(value, ctx.src);
                        ctx.local_bound_names.insert(func_nid.clone(), bound);
                        if let Some(body) = value.child_by_field_name("body") {
                            ctx.function_bodies.push((func_nid.clone(), body));
                            // #2653: a `function` declared inside an arrow-defined
                            // component is otherwise never seen -- the main walk
                            // does not recurse into arrow bodies.
                            scan_nested_function_declarations(ctx, body, &func_nid)?;
                        }
                        arrow_found = true;
                    }
                } else if is_exported_scalar_binding
                    || matches!(
                        value.kind(),
                        "object"
                            | "array"
                            | "as_expression"
                            | "satisfies_expression"
                            | "call_expression"
                            | "new_expression"
                    )
                {
                    if let Some(nn) = name_node {
                        let const_name = ctx.text(nn)?.to_string();
                        let line = line_of(child);
                        let const_nid = ctx.mkid(&[&ctx.stem.clone(), &const_name])?;
                        ctx.add_node(&const_nid, &const_name, line);
                        let file_nid = ctx.file_nid.clone();
                        ctx.add_edge(&file_nid, &const_nid, "contains", line);
                        const_found = true;
                        // #2552: track each TOPMOST closure in the initializer
                        // under the const's nid, with its own locals (#2568).
                        let mut inner = Some(value);
                        while let Some(i) = inner {
                            if !matches!(i.kind(), "as_expression" | "satisfies_expression") {
                                break;
                            }
                            inner = super::ast::named_children(i).first().copied();
                        }
                        if let Some(i) = inner {
                            if matches!(i.kind(), "call_expression" | "new_expression") {
                                let mut closures = Vec::new();
                                pat::topmost_closures(i, &mut closures);
                                for closure in closures {
                                    if let Some(body) = closure.child_by_field_name("body") {
                                        let locals = pat::local_bound_names(closure, ctx.src);
                                        ctx.closure_locals_by_body.insert(body.id(), locals);
                                        ctx.function_bodies.push((const_nid.clone(), body));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if arrow_found || const_found || require_found {
            return Ok(true);
        }
    }
    Ok(false)
}

/// `_ts_extra_walk`: enum members, and the container node for a `namespace` /
/// `module`.
fn ts_extra_walk<'t>(ctx: &mut Ctx<'_, 't>, node: Node<'t>, parent_class_nid: Option<&str>) -> R<bool> {
    if let Some(pcn) = parent_class_nid {
        let parent_is_enum_body = node.parent().map_or(false, |p| p.kind() == "enum_body");
        if parent_is_enum_body
            && matches!(node.kind(), "property_identifier" | "enum_assignment")
        {
            let name_node = if node.kind() == "property_identifier" {
                Some(node)
            } else {
                node.child_by_field_name("name")
            };
            let mut member_name = String::new();
            if let Some(nn) = name_node {
                member_name = ctx.text(nn)?.to_string();
                if nn.kind() == "string" {
                    // `"Odd Name" = 7`: unquote the WHOLE text rather than reading
                    // a string_fragment -- an escape splits the string into several
                    // fragments and the first alone truncates the name.
                    member_name = member_name.trim_matches(|c| c == '\'' || c == '"' || c == '`').to_string();
                }
            }
            if !member_name.is_empty() {
                let line = line_of(node);
                let member_nid = ctx.mkid(&[pcn, &member_name])?;
                // TS is case-sensitive while the id recipe casefolds, so
                // `enum E { Value, value }` puts two legal members on one id. The
                // first declaration keeps the node rather than a second edge.
                if !ctx.seen_ids.contains(&member_nid) {
                    ctx.add_node(&member_nid, &member_name, line);
                    ctx.add_edge(pcn, &member_nid, "case_of", line);
                }
            }
            if node.kind() == "enum_assignment" {
                // Claiming the member must not swallow its initializer: descend
                // into `value` only, since `name` was read above.
                if let Some(value_node) = node.child_by_field_name("value") {
                    walk(ctx, value_node, Some(pcn))?;
                }
            }
            return Ok(true);
        }
    }

    if node.is_named() && matches!(node.kind(), "internal_module" | "module") {
        let mut name_node = node.child_by_field_name("name");
        if name_node.is_none() {
            name_node = children(node).into_iter().find(|c| {
                c.is_named() && matches!(c.kind(), "identifier" | "nested_identifier" | "string")
            });
        }
        let mut body = node.child_by_field_name("body");
        if body.is_none() {
            body = children(node)
                .into_iter()
                .find(|c| c.kind() == "statement_block");
        }
        if let Some(nn) = name_node {
            let mut ns_name = ctx.text(nn)?.to_string();
            if nn.kind() == "string" {
                ns_name = ns_name.trim_matches(|c| c == '\'' || c == '"' || c == '`').to_string();
            }
            if !ns_name.is_empty() {
                let ns_nid = ctx.mkid(&[&ctx.stem.clone(), &ns_name])?;
                let line = line_of(node);
                ctx.add_node(&ns_nid, &ns_name, line);
                let file_nid = ctx.file_nid.clone();
                ctx.add_edge(&file_nid, &ns_nid, "contains", line);
            }
        }
        if let Some(b) = body {
            for child in children(b) {
                walk(ctx, child, parent_class_nid)?;
            }
        }
        return Ok(true);
    }
    Ok(false)
}

/// `_ts_receiver_type_table`.
///
/// The traversal is a `stack.pop()` DFS that pushes children in order, so it
/// visits the LAST child's subtree first. That order is observable: the table is
/// first-binding-wins, so two same-named bindings of different types resolve to
/// whichever the reversed order reaches first. A natural front-to-back recursion
/// here would silently pick the other one.
pub fn receiver_type_table(ctx: &mut Ctx, root: Node) -> R<()> {
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        match n.kind() {
            "variable_declarator" => {
                let name_n = n.child_by_field_name("name");
                let value = n.child_by_field_name("value");
                if let (Some(name_n), Some(value)) = (name_n, value) {
                    if name_n.kind() == "identifier" && value.kind() == "new_expression" {
                        if let Some(ctor) = value.child_by_field_name("constructor") {
                            if matches!(ctor.kind(), "identifier" | "type_identifier") {
                                let name = ctx.text(name_n)?.to_string();
                                let tname = ctx.text(ctor)?.to_string();
                                if !name.is_empty() && !tname.is_empty() {
                                    ctx.type_table.insert_if_absent(&name, &tname);
                                }
                            }
                        }
                    }
                }
            }
            "required_parameter" | "optional_parameter" => {
                let pat_n = n.child_by_field_name("pattern");
                let ann = n.child_by_field_name("type");
                if let (Some(pat_n), Some(ann)) = (pat_n, ann) {
                    if pat_n.kind() == "identifier" {
                        if let Some(tname) = bare_type_ident(ctx, ann)? {
                            let name = ctx.text(pat_n)?.to_string();
                            if !name.is_empty() && !tname.is_empty() {
                                ctx.type_table.insert_if_absent(&name, &tname);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        for c in children(n) {
            stack.push(c);
        }
    }
    Ok(())
}

/// Accept only a single `type_identifier` child: an array, union, generic,
/// qualified or predefined type is skipped (precision over recall).
fn bare_type_ident(ctx: &Ctx, annotation: Node) -> R<Option<String>> {
    let kids = children(annotation);
    let idents: Vec<Node> = kids
        .iter()
        .copied()
        .filter(|c| c.kind() == "type_identifier")
        .collect();
    let others = kids
        .iter()
        .filter(|c| c.is_named() && c.kind() != "type_identifier")
        .count();
    if idents.len() == 1 && others == 0 {
        return Ok(Some(ctx.text(idents[0])?.to_string()));
    }
    Ok(None)
}

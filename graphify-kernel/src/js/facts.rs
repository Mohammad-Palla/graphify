//! Symbol-resolution facts, emitted from the SAME parse as the extraction walk.
//!
//! # Why this exists
//!
//! `_collect_js_facts_one_file` runs in phase 3 and begins by calling
//! `_parse_js_tree(path)` -- so Graphify parses every JS/TS file **twice**,
//! measured at 2.07 tree-sitter `parse()` calls per JS/TS-family file across a
//! full build. Parse is the one part of extraction that is C on both sides and
//! cannot be made faster; paying for it twice is the single largest avoidable
//! cost in the pipeline.
//!
//! CodeGraph does not have this problem, and reading its shipped source is what
//! made the shape obvious: its native kernel emits `nodes`, `edges` AND
//! `refs` (its unresolved references) from one pass into one buffer set, and its
//! entire `resolution/` tree never touches a source file again -- the only
//! `parse` calls there are `JSON.parse` on tsconfig and package.json. Its
//! resolution phase consumes what extraction already produced.
//!
//! This module is that idea applied here: the facts phase 3 needs are collected
//! during the walk that already has the tree, and phase 3 reads them instead of
//! re-parsing.
//!
//! # Fidelity, and the separate deferral axis
//!
//! Fact collection defers INDEPENDENTLY of the extraction walk. A file whose
//! nodes and edges are perfectly reproducible may still contain a construct this
//! module has no rule for; when that happens the walker still returns its
//! nodes/edges natively and simply omits the facts, and Python's collector runs
//! for that file alone. Coupling the two would mean a gap in either one costing
//! the speedup of both.
//!
//! # Ordering is the correctness argument
//!
//! `_collect_file_symbol_facts` merges results in input order and keeps the two
//! `uses` producers in separate buckets, because JS appends every file's
//! top-level-function-body `uses` before ANY file's class-member `uses`. Within a
//! file, facts are appended in `_walk_js_tree` order -- a flat pre-order DFS over
//! ALL nodes, named and unnamed. The `continue` statements in the Python loop end
//! the loop BODY, not the traversal, so an `import_statement`'s children are
//! still visited afterwards. This module reproduces that exactly.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tree_sitter::Node;

use super::ast::{children, line_of, named_children};
use super::{Ctx, R};

/// `_JS_PRIMITIVE_TYPES`: names that are never a resolvable symbol.
fn is_primitive_type(name: &str) -> bool {
    matches!(
        name,
        "string" | "number" | "boolean" | "any" | "unknown" | "void" | "never"
            | "object" | "null" | "undefined" | "bigint" | "symbol" | "this"
    )
}

/// One file's facts, in emission order. Field-for-field with
/// `_SymbolResolutionFacts`'s JS-relevant lists; `module_imports` is Python-only.
#[derive(Default)]
pub struct Facts {
    /// (name, line)
    pub declarations: Vec<(String, usize)>,
    /// (local_name, target_path, imported_name, line)
    pub imports: Vec<(String, String, String, usize)>,
    /// (alias, target_name, line)
    pub aliases: Vec<(String, String, usize)>,
    /// (exported_name, line, local_name, target_path, target_name)
    pub exports: Vec<(String, usize, Option<String>, Option<String>, Option<String>)>,
    /// (target_path, line)
    pub star_exports: Vec<(String, usize)>,
    /// (exported_name, target_path, line)
    pub namespace_exports: Vec<(String, String, usize)>,
    /// (source_id, local_name, relation, context, line)
    pub uses: Vec<(String, String, &'static str, &'static str, usize)>,
}

impl Facts {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("declarations", PyList::new(py, &self.declarations)?)?;
        d.set_item("imports", PyList::new(py, &self.imports)?)?;
        d.set_item("aliases", PyList::new(py, &self.aliases)?)?;
        let exports = PyList::empty(py);
        for (name, line, local, tpath, tname) in &self.exports {
            exports.append((name, line, local, tpath, tname))?;
        }
        d.set_item("exports", exports)?;
        d.set_item("star_exports", PyList::new(py, &self.star_exports)?)?;
        d.set_item("namespace_exports", PyList::new(py, &self.namespace_exports)?)?;
        d.set_item("uses", PyList::new(py, &self.uses)?)?;
        Ok(d)
    }
}

/// `_collect_js_facts_one_file`, minus the parse it no longer needs.
///
/// Returns `(facts, class_member_facts)` -- the two `uses` producers kept apart,
/// which is what preserves ordering when the parent merges across files.
pub fn collect(ctx: &Ctx, root: Node) -> R<(Facts, Facts)> {
    let mut facts = Facts::default();
    let mut class_members = Facts::default();
    visit(ctx, root, &mut facts, &mut class_members)?;

    // Call sites inside top-level function bodies. A SECOND traversal in Python
    // too, and appended after every other fact in this file.
    for (source_id, body) in top_level_function_bodies(ctx, root)? {
        collect_uses(ctx, body, &source_id, &mut facts)?;
    }
    Ok((facts, class_members))
}

/// The flat pre-order walk. Every node, named and unnamed, root first.
fn visit(ctx: &Ctx, node: Node, facts: &mut Facts, class_members: &mut Facts) -> R<()> {
    let kind = node.kind();
    let line = line_of(node);

    // Only a `lexical_declaration` can introduce an alias.
    if kind == "lexical_declaration" {
        for (alias, target) in lexical_aliases(ctx, node)? {
            facts.aliases.push((alias, target, line));
        }
    }

    match kind {
        "import_statement" => import_statement(ctx, node, line, facts)?,
        "export_statement" => export_statement(ctx, node, line, facts)?,
        "class_declaration" | "abstract_class_declaration" | "interface_declaration" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let class_name = ctx.text(name_node)?;
                if !class_name.is_empty() {
                    let class_nid = ctx.mkid(&[&ctx.stem.clone(), class_name])?;
                    walk_class_members(ctx, node, &class_nid, class_members)?;
                }
            }
        }
        _ => {}
    }

    // The Python loop's `continue` ends the loop body, not the traversal: an
    // import_statement's children are still visited.
    for child in children(node) {
        visit(ctx, child, facts, class_members)?;
    }
    Ok(())
}

/// `_js_lexical_aliases`, with the node-kind test already done by the caller.
fn lexical_aliases(ctx: &Ctx, node: Node) -> R<Vec<(String, String)>> {
    let mut out = Vec::new();
    for child in children(node) {
        if child.kind() != "variable_declarator" {
            continue;
        }
        let name_node = child.child_by_field_name("name");
        let value_node = child.child_by_field_name("value");
        if let (Some(n), Some(v)) = (name_node, value_node) {
            if matches!(v.kind(), "identifier" | "type_identifier") {
                out.push((ctx.text(n)?.to_string(), ctx.text(v)?.to_string()));
            }
        }
    }
    Ok(out)
}

/// `_js_module_specifier`: whitespace-trim, then quote-trim, then empty -> None.
fn module_specifier(ctx: &Ctx, node: Node) -> R<Option<String>> {
    let mut source_node = node.child_by_field_name("source");
    if source_node.is_none() {
        source_node = children(node).into_iter().find(|c| c.kind() == "string");
    }
    let Some(sn) = source_node else { return Ok(None) };
    let raw = ctx.text(sn)?.trim();
    let stripped = raw.trim_matches(|c| c == '\'' || c == '"' || c == '`');
    Ok(if stripped.is_empty() {
        None
    } else {
        Some(stripped.to_string())
    })
}

/// `_js_named_specifiers`: a nested full walk of the subtree, so `import { A as B }`
/// is found wherever the grammar puts it.
fn named_specifiers(ctx: &Ctx, node: Node, specifier_kind: &str) -> R<Vec<(String, String)>> {
    let mut pairs = Vec::new();
    fn rec(
        ctx: &Ctx,
        n: Node,
        want: &str,
        pairs: &mut Vec<(String, String)>,
    ) -> R<()> {
        if n.kind() == want {
            if let Some(name_node) = n.child_by_field_name("name") {
                let name = ctx.text(name_node)?.to_string();
                let exposed = match n.child_by_field_name("alias") {
                    Some(a) => ctx.text(a)?.to_string(),
                    None => name.clone(),
                };
                if !name.is_empty() && !exposed.is_empty() {
                    pairs.push((name, exposed));
                }
            }
        }
        for c in children(n) {
            rec(ctx, c, want, pairs)?;
        }
        Ok(())
    }
    rec(ctx, node, specifier_kind, &mut pairs)?;
    Ok(pairs)
}

/// `_js_default_import_name`: a bare identifier child of the import_clause.
fn default_import_name(ctx: &Ctx, node: Node) -> R<Option<String>> {
    for child in children(node) {
        if child.kind() != "import_clause" {
            continue;
        }
        for sub in children(child) {
            if sub.kind() == "identifier" {
                return Ok(Some(ctx.text(sub)?.to_string()));
            }
        }
    }
    Ok(None)
}

fn import_statement(ctx: &Ctx, node: Node, line: usize, facts: &mut Facts) -> R<()> {
    let Some(raw_module) = module_specifier(ctx, node)? else {
        return Ok(());
    };
    let Some(target) = ctx.res.resolve_module(&raw_module)? else {
        return Ok(());
    };
    for (imported_name, local_name) in named_specifiers(ctx, node, "import_specifier")? {
        facts
            .imports
            .push((local_name, target.clone(), imported_name, line));
    }
    if let Some(default_local) = default_import_name(ctx, node)? {
        facts
            .imports
            .push((default_local, target, "default".to_string(), line));
    }
    Ok(())
}

/// `_js_exported_declaration_names`.
fn exported_declaration_names(ctx: &Ctx, node: Node) -> R<Vec<String>> {
    let Some(declaration) = node.child_by_field_name("declaration") else {
        return Ok(Vec::new());
    };
    if declaration.kind() == "lexical_declaration" {
        return Ok(lexical_aliases(ctx, declaration)?
            .into_iter()
            .map(|(alias, _)| alias)
            .collect());
    }
    if matches!(
        declaration.kind(),
        "class_declaration"
            | "abstract_class_declaration"
            | "interface_declaration"
            | "type_alias_declaration"
            | "function_declaration"
    ) {
        if let Some(name_node) = declaration.child_by_field_name("name") {
            return Ok(vec![ctx.text(name_node)?.to_string()]);
        }
    }
    Ok(Vec::new())
}

/// `_js_default_export_name`.
fn default_export_name(ctx: &Ctx, node: Node) -> R<Option<String>> {
    if !children(node).iter().any(|c| c.kind() == "default") {
        return Ok(None);
    }
    if let Some(declaration) = node.child_by_field_name("declaration") {
        return Ok(match declaration.child_by_field_name("name") {
            Some(n) => Some(ctx.text(n)?.to_string()),
            None => None,
        });
    }
    if let Some(value) = node.child_by_field_name("value") {
        if value.kind() == "identifier" {
            return Ok(Some(ctx.text(value)?.to_string()));
        }
    }
    Ok(None)
}

fn export_statement(ctx: &Ctx, node: Node, line: usize, facts: &mut Facts) -> R<()> {
    // The declaration names this export introduces.
    for name in exported_declaration_names(ctx, node)? {
        facts.declarations.push((name, line));
    }

    let raw_module = module_specifier(ctx, node)?;
    let export_clause = children(node).into_iter().find(|c| c.kind() == "export_clause");

    if let Some(raw_module) = raw_module {
        let Some(target) = ctx.res.resolve_module(&raw_module)? else {
            return Ok(());
        };
        // `export * as NS from './m'` before the bare-star test: a namespace
        // export also has a `*` child, so testing star first would misclassify it.
        // `_js_namespace_export_name` RETURNS from inside the inner loop, so a
        // `namespace_export` child with no `identifier` under it falls through to
        // the NEXT child rather than ending the search. Breaking out
        // unconditionally here would differ on that shape.
        let mut namespace_name = None;
        'outer: for child in children(node) {
            if child.kind() != "namespace_export" {
                continue;
            }
            for sub in children(child) {
                if sub.kind() == "identifier" {
                    let t = ctx.text(sub)?;
                    namespace_name = if t.is_empty() { None } else { Some(t.to_string()) };
                    break 'outer;
                }
            }
        }
        if let Some(ns) = namespace_name {
            facts.namespace_exports.push((ns, target.clone(), line));
        } else if children(node).iter().any(|c| c.kind() == "*") {
            facts.star_exports.push((target.clone(), line));
        }
        if let Some(clause) = export_clause {
            for (original_name, exported_name) in
                named_specifiers(ctx, clause, "export_specifier")?
            {
                facts.exports.push((
                    exported_name,
                    line,
                    None,
                    Some(target.clone()),
                    Some(original_name),
                ));
            }
        }
        return Ok(());
    }

    if let Some(clause) = export_clause {
        for (local_name, exported_name) in named_specifiers(ctx, clause, "export_specifier")? {
            facts
                .exports
                .push((exported_name, line, Some(local_name), None, None));
        }
        return Ok(());
    }

    for exported_name in exported_declaration_names(ctx, node)? {
        facts
            .exports
            .push((exported_name.clone(), line, Some(exported_name), None, None));
    }

    // `export default class Foo {}` exposes the symbol under the name "default".
    if let Some(default_name) = default_export_name(ctx, node)? {
        facts
            .exports
            .push(("default".to_string(), line, Some(default_name), None, None));
    }
    Ok(())
}

/// `_js_top_level_function_bodies`: direct children of the program only.
fn top_level_function_bodies<'t>(ctx: &Ctx<'_, 't>, root: Node<'t>) -> R<Vec<(String, Node<'t>)>> {
    let mut bodies = Vec::new();
    for node in children(root) {
        if node.kind() == "function_declaration" {
            let name_node = node.child_by_field_name("name");
            let body = node.child_by_field_name("body");
            if let (Some(n), Some(b)) = (name_node, body) {
                bodies.push((ctx.mkid(&[&ctx.stem.clone(), ctx.text(n)?])?, b));
            }
            continue;
        }
        if node.kind() != "lexical_declaration" {
            continue;
        }
        for child in children(node) {
            if child.kind() != "variable_declarator" {
                continue;
            }
            let name_node = child.child_by_field_name("name");
            let value_node = child.child_by_field_name("value");
            if let (Some(n), Some(v)) = (name_node, value_node) {
                if v.kind() == "arrow_function" {
                    // NOTE: the arrow itself, not its body -- Python appends
                    // `value_node`, so the walk below covers the parameters too.
                    bodies.push((ctx.mkid(&[&ctx.stem.clone(), ctx.text(n)?])?, v));
                }
            }
        }
    }
    Ok(bodies)
}

/// `_js_call_identifier` over every node of a body.
fn collect_uses(ctx: &Ctx, node: Node, source_id: &str, facts: &mut Facts) -> R<()> {
    if node.kind() == "call_expression" {
        let mut function_node = node.child_by_field_name("function");
        if function_node.is_none() {
            function_node = named_children(node).into_iter().next();
        }
        if let Some(f) = function_node {
            if matches!(f.kind(), "identifier" | "type_identifier") {
                facts.uses.push((
                    source_id.to_string(),
                    ctx.text(f)?.to_string(),
                    "calls",
                    "call",
                    line_of(node),
                ));
            }
        }
    }
    for child in children(node) {
        collect_uses(ctx, child, source_id, facts)?;
    }
    Ok(())
}

/// `_ts_collect_type_refs`: (name, is_generic_arg) pairs from a type annotation.
fn collect_type_refs(
    ctx: &Ctx,
    node: Node,
    generic: bool,
    out: &mut Vec<(String, bool)>,
) -> R<()> {
    let t = node.kind();
    if t == "type_annotation" {
        for c in named_children(node) {
            collect_type_refs(ctx, c, generic, out)?;
        }
        return Ok(());
    }
    if matches!(t, "type_identifier" | "identifier") {
        let name = ctx.text(node)?;
        if !name.is_empty() && !is_primitive_type(name) {
            out.push((name.to_string(), generic));
        }
        return Ok(());
    }
    if t == "nested_type_identifier" {
        let text = ctx.text(node)?;
        let tail = text.rsplit('.').next().unwrap_or(text);
        if !tail.is_empty() && !is_primitive_type(tail) {
            out.push((tail.to_string(), generic));
        }
        return Ok(());
    }
    if t == "generic_type" {
        match node.child_by_field_name("name") {
            Some(name_node) => {
                let text = ctx.text(name_node)?;
                let tail = text.rsplit('.').next().unwrap_or(text);
                if !tail.is_empty() && !is_primitive_type(tail) {
                    out.push((tail.to_string(), generic));
                }
            }
            None => {
                for c in children(node) {
                    if matches!(c.kind(), "type_identifier" | "nested_type_identifier") {
                        let text = ctx.text(c)?;
                        let tail = text.rsplit('.').next().unwrap_or(text);
                        if !tail.is_empty() && !is_primitive_type(tail) {
                            out.push((tail.to_string(), generic));
                        }
                        break;
                    }
                }
            }
        }
        for c in children(node) {
            if c.kind() == "type_arguments" {
                for sub in named_children(c) {
                    collect_type_refs(ctx, sub, true, out)?;
                }
            }
        }
        return Ok(());
    }
    if node.is_named() {
        for c in named_children(node) {
            collect_type_refs(ctx, c, generic, out)?;
        }
    }
    Ok(())
}

/// `_ts_heritage_clause_entries`.
fn heritage_entries(ctx: &Ctx, clause: Node) -> R<Vec<String>> {
    let mut out = Vec::new();
    for child in named_children(clause) {
        if matches!(child.kind(), "identifier" | "type_identifier") {
            let name = ctx.text(child)?;
            if !name.is_empty() {
                out.push(name.to_string());
            }
        } else if child.kind() == "generic_type" {
            let mut name_node = child.child_by_field_name("name");
            if name_node.is_none() {
                name_node = children(child).into_iter().find(|sub| {
                    matches!(
                        sub.kind(),
                        "type_identifier" | "nested_type_identifier" | "identifier"
                    )
                });
            }
            if let Some(nn) = name_node {
                let text = ctx.text(nn)?;
                let tail = text.rsplit('.').next().unwrap_or(text);
                if !tail.is_empty() {
                    out.push(tail.to_string());
                }
            }
        } else if child.kind() == "nested_type_identifier" {
            // `implements vscode.DebugConfigurationProvider` -- a qualified base
            // is a `nested_type_identifier`, not a `type_identifier`, and only
            // the tail is the resolvable symbol. Omitting this branch silently
            // dropped every qualified `extends` / `implements` fact: 4 of the
            // first 400 Bun files, all of them vscode-API classes.
            let text = ctx.text(child)?;
            let tail = text.rsplit('.').next().unwrap_or(text);
            if !tail.is_empty() {
                out.push(tail.to_string());
            }
        }
    }
    Ok(out)
}

/// `_ts_walk_class_members`.
fn walk_class_members(ctx: &Ctx, class_node: Node, class_nid: &str, facts: &mut Facts) -> R<()> {
    for child in children(class_node) {
        if child.kind() == "class_heritage" {
            for clause in children(child) {
                let relation = match clause.kind() {
                    "extends_clause" => "inherits",
                    "implements_clause" => "implements",
                    _ => continue,
                };
                let cline = line_of(clause);
                for name in heritage_entries(ctx, clause)? {
                    facts
                        .uses
                        .push((class_nid.to_string(), name, relation, "type", cline));
                }
            }
        } else if child.kind() == "extends_type_clause" {
            // `interface A extends B, C` is an extends_type_clause, NOT a
            // class_heritage; without this branch interface inheritance is
            // dropped (#1095).
            let cline = line_of(child);
            for name in heritage_entries(ctx, child)? {
                facts
                    .uses
                    .push((class_nid.to_string(), name, "inherits", "type", cline));
            }
        }
    }

    let Some(body) = class_node.child_by_field_name("body") else {
        return Ok(());
    };
    for member in children(body) {
        let m_line = line_of(member);
        match member.kind() {
            "method_definition" | "method_signature" | "abstract_method_signature" => {
                let Some(name_node) = member.child_by_field_name("name") else {
                    continue;
                };
                let method_nid = ctx.mkid(&[class_nid, ctx.text(name_node)?])?;
                if let Some(params) = member.child_by_field_name("parameters") {
                    for p in children(params) {
                        if !matches!(p.kind(), "required_parameter" | "optional_parameter") {
                            continue;
                        }
                        let Some(type_anno) = p.child_by_field_name("type") else {
                            continue;
                        };
                        let mut refs = Vec::new();
                        collect_type_refs(ctx, type_anno, false, &mut refs)?;
                        for (name, is_generic) in refs {
                            let ctx_name = if is_generic { "generic_arg" } else { "parameter_type" };
                            facts.uses.push((
                                method_nid.clone(),
                                name,
                                "references",
                                ctx_name,
                                m_line,
                            ));
                        }
                    }
                }
                if let Some(return_type) = member.child_by_field_name("return_type") {
                    let mut refs = Vec::new();
                    collect_type_refs(ctx, return_type, false, &mut refs)?;
                    for (name, is_generic) in refs {
                        let ctx_name = if is_generic { "generic_arg" } else { "return_type" };
                        facts.uses.push((
                            method_nid.clone(),
                            name,
                            "references",
                            ctx_name,
                            m_line,
                        ));
                    }
                }
            }
            "public_field_definition" | "property_signature" => {
                let Some(type_anno) = children(member)
                    .into_iter()
                    .find(|c| c.kind() == "type_annotation")
                else {
                    continue;
                };
                let mut refs = Vec::new();
                collect_type_refs(ctx, type_anno, false, &mut refs)?;
                for (name, is_generic) in refs {
                    let ctx_name = if is_generic { "generic_arg" } else { "field" };
                    facts.uses.push((
                        class_nid.to_string(),
                        name,
                        "references",
                        ctx_name,
                        m_line,
                    ));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// `(facts, class_member_facts)` as a Python 2-tuple of dicts.
pub fn to_py<'py>(
    py: Python<'py>,
    facts: &Facts,
    class_members: &Facts,
) -> PyResult<Bound<'py, PyAny>> {
    let pair = (facts.to_py(py)?, class_members.to_py(py)?);
    Ok(pair.into_pyobject(py)?.into_any())
}

//! Objective-C: a BESPOKE walker, and the most involved one here.
//!
//! # Two source-level quirks, both reproduced
//!
//! 1. **Two macros are BLANKED before parsing.** tree-sitter-objc cannot expand
//!    `NS_ASSUME_NONNULL_BEGIN` / `_END` (argument-less, no trailing `;`), and
//!    their presence before `@interface` stops the parser emitting a
//!    `class_interface` node at all (#1475). They are overwritten with spaces of
//!    EQUAL LENGTH so byte offsets and line numbers survive. Done here rather
//!    than in the seam because it is a pure byte substitution -- unlike Fortran's
//!    `cpp` pass, which needs a subprocess and therefore stays in Python.
//! 2. **A category keys its class node off the BASE stem.** `Foo+Cat.h` declares
//!    members of the existing `Foo`, so it must not mint a second node labelled
//!    `Foo` -- that made every `[Foo ...]` receiver ambiguous and tripped the
//!    resolver's single-definition guard, destroying edges the same corpus
//!    produced fine when the members lived in `Foo.h` (#1556).
//!
//! # A determinism bug found by porting this
//!
//! `extract_objc` iterated `all_method_nids` as a SET of strings and emitted one
//! `calls` edge per match, so edge ORDER depended on Python's per-process string
//! hash seed -- measured, 4 of 183 Texture files reordered between runs with an
//! identical edge set. Edge order reaches the exported graph.json and the AST
//! cache, so two builds of the same repo could differ byte for byte. There was no
//! stable target to port against, so the Python was fixed first: it is now a LIST
//! in `nodes` insertion order, which is what this walker matches.
//!
//! # Parse ceiling
//!
//! 77.2% over 600 files. AFNetworking and SDWebImage are 91.8% / 94.2%; Texture
//! drags it to 71.6% -- it is heavily C++-flavoured Objective-C++.

use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tree_sitter::{Node, Parser};

use crate::engine::{PathResolver, R};
use crate::ids::{file_stem, make_id_ascii};
use crate::js::ast::{children, text_checked};
use crate::js::emit::{self, EdgeRow, NodeRow, RawCall, Val};
use crate::Outcome;

/// The macros blanked before parsing. See the module doc.
const BLANK_MACROS: &[&str] = &["NS_ASSUME_NONNULL_BEGIN", "NS_ASSUME_NONNULL_END"];

/// `_OBJC_STEM_PART.fullmatch`: `[A-Za-z_][A-Za-z0-9_]*`.
///
/// Hand-written rather than pulling in the `regex` crate for one pattern, as
/// `lua/` does for `require`.
fn is_stem_part(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// `_objc_category_base_stem`: `Foo+Cat` -> `Foo`.
///
/// Only the FINAL path segment is considered, and only a well-formed
/// `Name+Suffix` pair splits -- so `C++Bridge.h` and `Foo+.h` are left intact.
fn category_base_stem(stem: &str) -> String {
    let (head, sep, tail) = match stem.rfind('/') {
        Some(i) => (&stem[..i], "/", &stem[i + 1..]),
        None => ("", "", stem),
    };
    let (base, plus, suffix) = match tail.find('+') {
        Some(i) => (&tail[..i], true, &tail[i + 1..]),
        None => (tail, false, ""),
    };
    if !plus || !is_stem_part(base) || !is_stem_part(suffix) {
        return stem.to_string();
    }
    format!("{head}{sep}{base}")
}

/// `@interface/@implementation Foo (Cat)` and `Foo ()`.
///
/// The grammar emits the parentheses as ANONYMOUS children only for a category
/// or class extension; a generic class (`@interface Box<T>`) uses
/// `parameterized_arguments`, so it is not matched.
fn is_category(node: Node) -> bool {
    children(node).into_iter().any(|c| c.kind() == "(")
}

/// Children paired with their FIELD NAME, which the selector reconstruction
/// needs: `[recv kw1:a kw2:b]` marks every selector part with the field
/// `"method"`, and the receiver with `"receiver"`.
fn children_with_fields<'tree>(node: Node<'tree>) -> Vec<(Option<&'static str>, Node<'tree>)> {
    let mut out = Vec::new();
    let mut cursor = node.walk();
    if cursor.goto_first_child() {
        loop {
            out.push((cursor.field_name(), cursor.node()));
            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }
    out
}

struct Ctx<'a, 'tree> {
    src: &'a [u8],
    str_path: &'a str,
    stem: String,
    file_nid: String,
    nodes: Vec<NodeRow>,
    edges: Vec<EdgeRow>,
    raw_calls: Vec<RawCall>,
    seen_ids: HashSet<String>,
    /// `(method_nid, definition_node, container_nid)`.
    method_bodies: Vec<(String, Node<'tree>, String)>,
    /// `var -> ClassName` from `Foo *f = ...;` locals, file-scoped.
    type_table: Vec<(String, String)>,
    type_table_seen: HashSet<String>,
    /// Per-class `field -> Some(ClassName)`, with `None` as a TOMBSTONE for a
    /// conflicting redeclaration. Insertion-ordered, because the exported dict is.
    field_types: Vec<(String, Vec<(String, Option<String>)>)>,
    resolver: Option<&'a dyn PathResolver>,
}

impl<'a, 'tree> Ctx<'a, 'tree> {
    fn text(&self, node: Node) -> R<&'a str> {
        text_checked(node, self.src).ok_or("invalid_utf8_text")
    }

    fn mkid(&self, parts: &[&str]) -> R<String> {
        make_id_ascii(parts).ok_or("non_ascii_id")
    }

    fn add_node(&mut self, nid: &str, label: &str, line: usize) {
        if !self.seen_ids.insert(nid.to_string()) {
            return;
        }
        self.nodes.push(NodeRow {
            id: nid.to_string(),
            fields: vec![
                ("label", Val::S(label.to_string())),
                ("file_type", Val::Static("code")),
                ("source_file", Val::S(self.str_path.to_string())),
                ("source_location", Val::S(format!("L{line}"))),
            ],
        });
    }

    fn add_edge(
        &mut self,
        src: &str,
        tgt: &str,
        relation: &'static str,
        line: usize,
        context: Option<&'static str>,
    ) {
        let mut fields = vec![
            ("confidence", Val::Static("EXTRACTED")),
            ("source_file", Val::S(self.str_path.to_string())),
            ("source_location", Val::S(format!("L{line}"))),
            ("weight", Val::F(1.0)),
        ];
        if let Some(c) = context {
            fields.push(("context", Val::Static(c)));
        }
        self.edges.push(EdgeRow {
            source: src.to_string(),
            target: tgt.to_string(),
            relation,
            fields,
        });
    }

    /// `_semantic_reference_edge`: `context` THIRD, right after `relation`.
    fn add_reference_edge(&mut self, src: &str, tgt: &str, context: &'static str, line: usize) {
        self.edges.push(EdgeRow {
            source: src.to_string(),
            target: tgt.to_string(),
            relation: "references",
            fields: vec![
                ("context", Val::Static(context)),
                ("confidence", Val::Static("EXTRACTED")),
                ("source_file", Val::S(self.str_path.to_string())),
                ("source_location", Val::S(format!("L{line}"))),
                ("weight", Val::F(1.0)),
            ],
        });
    }

    fn ensure_named_node(&mut self, name: &str) -> R<String> {
        let scoped = self.mkid(&[&self.stem.clone(), name])?;
        if self.seen_ids.contains(&scoped) {
            return Ok(scoped);
        }
        let bare = self.mkid(&[name])?;
        if self.seen_ids.insert(bare.clone()) {
            self.nodes.push(NodeRow {
                id: bare.clone(),
                fields: vec![
                    ("label", Val::S(name.to_string())),
                    ("file_type", Val::Static("code")),
                    ("source_file", Val::Static("")),
                    ("source_location", Val::Static("")),
                    ("origin_file", Val::S(self.str_path.to_string())),
                ],
            });
        }
        Ok(bare)
    }

    /// `_record_field_type`: a conflicting redeclaration TOMBSTONES the entry
    /// (`None`) rather than picking one -- drop, don't guess.
    fn record_field_type(&mut self, cls_nid: &str, field: &str, type_name: &str) {
        if !self.field_types.iter().any(|(c, _)| c == cls_nid) {
            self.field_types.push((cls_nid.to_string(), Vec::new()));
        }
        let table = &mut self
            .field_types
            .iter_mut()
            .find(|(c, _)| c == cls_nid)
            .expect("just inserted")
            .1;
        match table.iter_mut().find(|(f, _)| f == field) {
            Some(entry) => {
                if entry.1.as_deref() != Some(type_name) {
                    entry.1 = None;
                }
            }
            None => table.push((field.to_string(), Some(type_name.to_string()))),
        }
    }
}

/// Every `type_identifier` under a property's type node, descending through
/// `generic_specifier` / `type_name`, so `NSArray<Product *>` yields BOTH
/// `NSArray` and the element type `Product` (#1475).
fn type_identifiers<'tree>(node: Node<'tree>, out: &mut Vec<Node<'tree>>) {
    if node.kind() == "type_identifier" {
        out.push(node);
        return;
    }
    for c in children(node) {
        type_identifiers(c, out);
    }
}

/// `(field, TypeName)` from a property/ivar `struct_declaration`, else None.
///
/// Precision gates (#1556): exactly ONE bare capitalized `type_identifier`
/// DIRECTLY under the declaration -- a `generic_specifier` (`NSArray<Bar *>`) or
/// `typedefed_specifier` (`id<P>`) wraps its `type_identifier` and so is
/// naturally excluded, and never types a receiver -- and exactly one
/// `struct_declarator` holding exactly one child.
fn field_decl_entry(ctx: &Ctx, sd: Node) -> R<Option<(String, String)>> {
    let kids = children(sd);
    let type_ids: Vec<Node> = kids
        .iter()
        .copied()
        .filter(|s| s.kind() == "type_identifier")
        .collect();
    let declarators: Vec<Node> = kids
        .iter()
        .copied()
        .filter(|s| s.kind() == "struct_declarator")
        .collect();
    if type_ids.len() != 1 || declarators.len() != 1 {
        return Ok(None);
    }
    let type_name = ctx.text(type_ids[0])?.trim().to_string();
    if type_name.is_empty() || !type_name.chars().next().is_some_and(|c| c.is_uppercase()) {
        return Ok(None);
    }
    // `struct_declarator` wraps ONE `pointer_declarator` (`*bar`) or `identifier`
    // (`bar`); a bitfield or array has more children -> bail.
    let inner = children(declarators[0]);
    if inner.len() != 1 {
        return Ok(None);
    }
    let field = crate::cpp::helpers::declarator_name_src(ctx.src, inner[0])?;
    match field {
        Some(f) if !f.is_empty() => Ok(Some((f, type_name))),
        _ => Ok(None),
    }
}

fn collect_instance_variables(ctx: &mut Ctx, ivars: Node, cls_nid: &str) -> R<()> {
    for iv in children(ivars) {
        if iv.kind() != "instance_variable" {
            continue;
        }
        for sd in children(iv) {
            if sd.kind() != "struct_declaration" {
                continue;
            }
            if let Some((field, type_name)) = field_decl_entry(ctx, sd)? {
                ctx.record_field_type(cls_nid, &field, &type_name);
            }
        }
    }
    Ok(())
}

/// `_objc_local_var_types`: `var -> ClassName` from `Foo *f = ...;`.
///
/// Only a capitalized bare `type_identifier` with a SINGLE named declarator is
/// recorded; first binding wins and the table is file-scoped.
fn local_var_types(ctx: &mut Ctx, body: Node) -> R<()> {
    let mut stack = vec![body];
    while let Some(n) = stack.pop() {
        if n.kind() == "method_definition" && n.id() != body.id() {
            continue;
        }
        if n.kind() == "declaration" {
            let mut type_node = n.child_by_field_name("type");
            if type_node.is_none() {
                type_node = children(n).into_iter().find(|c| c.kind() == "type_identifier");
            }
            if let Some(tn) = type_node {
                if tn.kind() == "type_identifier" {
                    let type_name = ctx.text(tn)?.trim().to_string();
                    let declarators: Vec<Node> = children(n)
                        .into_iter()
                        .filter(|c| {
                            matches!(
                                c.kind(),
                                "identifier" | "pointer_declarator" | "init_declarator"
                            )
                        })
                        .collect();
                    if !type_name.is_empty()
                        && type_name.chars().next().is_some_and(|c| c.is_uppercase())
                        && declarators.len() == 1
                    {
                        let var = crate::cpp::helpers::declarator_name_src(ctx.src, declarators[0])?;
                        if let Some(var) = var {
                            if !var.is_empty() && ctx.type_table_seen.insert(var.clone()) {
                                ctx.type_table.push((var, type_name));
                            }
                        }
                    }
                }
            }
        }
        stack.extend(children(n));
    }
    Ok(())
}

fn walk<'tree>(ctx: &mut Ctx<'_, 'tree>, node: Node<'tree>, parent_nid: Option<&str>) -> R<()> {
    let t = node.kind();
    let line = node.start_position().row + 1;

    if t == "preproc_include" {
        for child in children(node) {
            if child.kind() == "system_lib_string" {
                // `#import <Foundation/Foundation.h>`
                let raw = ctx.text(child)?.trim_matches(|c| c == '<' || c == '>');
                let module = raw
                    .rsplit('/')
                    .next()
                    .unwrap_or("")
                    .replace(".h", "");
                if !module.is_empty() {
                    let tgt = ctx.mkid(&[&module])?;
                    let f = ctx.file_nid.clone();
                    ctx.add_edge(&f, &tgt, "imports", line, Some("import"));
                }
            } else if child.kind() == "string_literal" {
                for sub in children(child) {
                    if sub.kind() != "string_content" {
                        continue;
                    }
                    let raw = ctx.text(sub)?.to_string();
                    // Resolve the quoted include to a REAL file, so the target id
                    // matches the (possibly disambiguated) id that file's node
                    // gets. A bare-stem id never survives
                    // `_disambiguate_colliding_node_ids` when a .h/.m pair exists,
                    // so the edge dangled and was dropped (#1475).
                    let resolved = match ctx.resolver {
                        Some(r) => r.resolve(&raw)?,
                        // No resolver means the seam did not supply one; deferring
                        // is the only safe answer, as the C walker does.
                        None => return Err("no_include_resolver"),
                    };
                    let f = ctx.file_nid.clone();
                    match resolved {
                        Some(real) => {
                            let tgt = ctx.mkid(&[&real])?;
                            ctx.add_edge(&f, &tgt, "imports", line, Some("import"));
                        }
                        None => {
                            let module =
                                raw.rsplit('/').next().unwrap_or("").replace(".h", "");
                            if !module.is_empty() {
                                let tgt = ctx.mkid(&[&module])?;
                                ctx.add_edge(&f, &tgt, "imports", line, Some("import"));
                            }
                        }
                    }
                }
            }
        }
        return Ok(());
    }

    if t == "module_import" {
        // `@import Foundation;` / `@import Foundation.NSString;`
        if let Some(pn) = node.child_by_field_name("path") {
            let text = ctx.text(pn)?;
            let module = text.split('.').next().unwrap_or("").trim().to_string();
            if !module.is_empty() {
                let tgt = ctx.mkid(&[&module])?;
                let f = ctx.file_nid.clone();
                ctx.add_edge(&f, &tgt, "imports", line, Some("import"));
            }
        }
        return Ok(());
    }

    if t == "class_interface" {
        let identifiers: Vec<Node> = children(node)
            .into_iter()
            .filter(|c| c.kind() == "identifier")
            .collect();
        if identifiers.is_empty() {
            for child in children(node) {
                walk(ctx, child, parent_nid)?;
            }
            return Ok(());
        }
        let name = ctx.text(identifiers[0])?.to_string();
        let cls_stem = if is_category(node) {
            category_base_stem(&ctx.stem)
        } else {
            ctx.stem.clone()
        };
        let cls_nid = ctx.mkid(&[&cls_stem, &name])?;
        ctx.add_node(&cls_nid, &name, line);
        let f = ctx.file_nid.clone();
        ctx.add_edge(&f, &cls_nid, "contains", line, None);

        // The superclass is the identifier that FOLLOWS the `:` token. The flag is
        // cleared after one match, so only the first post-colon identifier counts.
        let mut colon_seen = false;
        for child in children(node) {
            let ck = child.kind();
            if ck == ":" {
                colon_seen = true;
            } else if colon_seen && ck == "identifier" {
                let sup = ctx.text(child)?.to_string();
                let super_nid = ctx.ensure_named_node(&sup)?;
                ctx.add_edge(&cls_nid, &super_nid, "inherits", line, None);
                colon_seen = false;
            } else if ck == "parameterized_arguments" {
                // `@interface Foo : Bar <Proto1, Proto2>`
                for sub in children(child) {
                    if sub.kind() != "type_name" {
                        continue;
                    }
                    for s in children(sub) {
                        if s.kind() == "type_identifier" {
                            let pname = ctx.text(s)?.to_string();
                            let proto_nid = ctx.ensure_named_node(&pname)?;
                            ctx.add_edge(&cls_nid, &proto_nid, "implements", line, None);
                        }
                    }
                }
            } else if ck == "property_declaration" {
                let prop_line = child.start_position().row + 1;
                for sub in children(child) {
                    if sub.kind() != "struct_declaration" {
                        continue;
                    }
                    // Walk every type name in the TYPE portion, skipping the
                    // declarator (the field name), so a generic collection is not
                    // invisible. Deduped per declaration.
                    let mut seen_types: HashSet<String> = HashSet::new();
                    for s in children(sub) {
                        if matches!(s.kind(), "struct_declarator" | ";") {
                            continue;
                        }
                        let mut tis = Vec::new();
                        type_identifiers(s, &mut tis);
                        for ti in tis {
                            let tname = ctx.text(ti)?.to_string();
                            if !seen_types.insert(tname.clone()) {
                                continue;
                            }
                            let type_nid = ctx.ensure_named_node(&tname)?;
                            ctx.add_reference_edge(&cls_nid, &type_nid, "field", prop_line);
                        }
                    }
                    if let Some((field, type_name)) = field_decl_entry(ctx, sub)? {
                        ctx.record_field_type(&cls_nid, &field, &type_name);
                    }
                }
            } else if ck == "instance_variables" {
                collect_instance_variables(ctx, child, &cls_nid)?;
            } else if ck == "method_declaration" {
                walk(ctx, child, Some(&cls_nid))?;
            }
        }
        return Ok(());
    }

    if t == "class_implementation" {
        let name = {
            let mut found: Option<String> = None;
            for child in children(node) {
                if child.kind() == "identifier" {
                    found = Some(ctx.text(child)?.to_string());
                    break;
                }
            }
            found
        };
        let name = match name {
            Some(n) if !n.is_empty() => n,
            _ => {
                for child in children(node) {
                    walk(ctx, child, parent_nid)?;
                }
                return Ok(());
            }
        };
        let impl_stem = if is_category(node) {
            category_base_stem(&ctx.stem)
        } else {
            ctx.stem.clone()
        };
        let impl_nid = ctx.mkid(&[&impl_stem, &name])?;
        // The `contains` edge is emitted ONLY with the node -- a category whose
        // base class node already exists adds neither.
        if !ctx.seen_ids.contains(&impl_nid) {
            ctx.add_node(&impl_nid, &name, line);
            let f = ctx.file_nid.clone();
            ctx.add_edge(&f, &impl_nid, "contains", line, None);
        }
        for child in children(node) {
            if child.kind() == "instance_variables" {
                collect_instance_variables(ctx, child, &impl_nid)?;
            } else if child.kind() == "implementation_definition" {
                for sub in children(child) {
                    walk(ctx, sub, Some(&impl_nid))?;
                }
            }
        }
        return Ok(());
    }

    if t == "protocol_declaration" {
        let name = {
            let mut found: Option<String> = None;
            for child in children(node) {
                if child.kind() == "identifier" {
                    found = Some(ctx.text(child)?.to_string());
                    break;
                }
            }
            found
        };
        if let Some(name) = name {
            if !name.is_empty() {
                let proto_nid = ctx.mkid(&[&ctx.stem.clone(), &name])?;
                // The label carries the ObjC protocol brackets.
                ctx.add_node(&proto_nid, &format!("<{name}>"), line);
                let f = ctx.file_nid.clone();
                ctx.add_edge(&f, &proto_nid, "contains", line, None);
                // `@protocol Derived <Base, Other>` nests under
                // `protocol_reference_list`, which is a DIFFERENT node from the
                // `parameterized_arguments` @interface adoption uses -- so these
                // were never emitted at all.
                for child in children(node) {
                    if child.kind() != "protocol_reference_list" {
                        continue;
                    }
                    for sub in children(child) {
                        if sub.kind() == "identifier" {
                            let bname = ctx.text(sub)?.to_string();
                            let base_nid = ctx.ensure_named_node(&bname)?;
                            if base_nid != proto_nid {
                                ctx.add_edge(&proto_nid, &base_nid, "implements", line, None);
                            }
                        }
                    }
                }
                for child in children(node) {
                    walk(ctx, child, Some(&proto_nid))?;
                }
            }
        }
        return Ok(());
    }

    if matches!(t, "method_declaration" | "method_definition") {
        let container = parent_nid.unwrap_or(&ctx.file_nid).to_string();
        // A class method starts with `+`, an instance method with `-`, emitted as
        // the first child. The selector is the concatenation of the DIRECT
        // `identifier` children: one for a simple selector (`-go`), several for a
        // compound one (`-tableView:numberOfRowsInSection:` ->
        // "tableViewnumberOfRowsInSection"). `method_parameter` holds argument
        // types and names, not selector keywords, so it is correctly skipped.
        let mut prefix = "-";
        for child in children(node) {
            if child.kind() == "+" {
                prefix = "+";
                break;
            }
            if child.kind() == "-" {
                prefix = "-";
                break;
            }
        }
        let mut parts = String::new();
        for c in children(node) {
            if c.kind() == "identifier" {
                parts.push_str(ctx.text(c)?);
            }
        }
        if !parts.is_empty() {
            let method_nid = ctx.mkid(&[&container, &parts])?;
            ctx.add_node(&method_nid, &format!("{prefix}{parts}"), line);
            ctx.add_edge(&container, &method_nid, "method", line, None);
            if t == "method_definition" {
                ctx.method_bodies.push((method_nid, node, container));
            }
        }
        return Ok(());
    }

    for child in children(node) {
        walk(ctx, child, parent_nid)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn walk_calls<'tree>(
    ctx: &mut Ctx<'_, 'tree>,
    n: Node<'tree>,
    caller_nid: &str,
    container_nid: &str,
    all_method_nids: &[String],
    sibling_nids: &HashSet<String>,
    method_bodies_keys: &[(String, String)],
    seen_calls: &mut HashSet<(String, String)>,
) -> R<()> {
    let t = n.kind();
    if t == "message_expression" {
        let meth = n.child_by_field_name("method");
        let recv = n.child_by_field_name("receiver");
        // `[[Foo alloc] init]`: link the allocating method to the allocated type.
        if let (Some(m), Some(r)) = (meth, recv) {
            if m.kind() == "identifier" && ctx.text(m)? == "alloc" && r.kind() == "identifier" {
                let tname = ctx.text(r)?.to_string();
                let ref_line = n.start_position().row + 1;
                let type_nid = ctx.ensure_named_node(&tname)?;
                if type_nid != caller_nid {
                    ctx.add_reference_edge(caller_nid, &type_nid, "type", ref_line);
                }
            }
        }
        // Reconstruct the selector from every child carrying the FIELD NAME
        // "method", so a `self`/`super`/ClassName receiver is never mistaken for
        // a selector and a compound send resolves too. The whole second pass used
        // to be dead code for ObjC because the grammar emits these as
        // `identifier`, not `selector`/`keyword_argument_list` (#1475).
        let mut method_name = String::new();
        for (field, child) in children_with_fields(n) {
            if field == Some("method") && child.kind() == "identifier" {
                method_name.push_str(ctx.text(child)?);
            }
        }
        if !method_name.is_empty() {
            let line = n.start_position().row + 1;
            // `_make_id("", method_name).lstrip("_")`.
            let needle = make_id_ascii(&["", &method_name])
                .ok_or("non_ascii_id")?
                .trim_start_matches('_')
                .to_string();
            for candidate in all_method_nids {
                if candidate.ends_with(&needle) {
                    let pair = (caller_nid.to_string(), candidate.clone());
                    if !seen_calls.contains(&pair) && caller_nid != candidate {
                        seen_calls.insert(pair);
                        ctx.add_edge(caller_nid, candidate, "calls", line, Some("call"));
                    }
                }
            }
            // Also emit a raw_call so the cross-file resolver can type the
            // receiver and link to a method in ANOTHER file.
            if let Some(r) = recv {
                if r.kind() == "identifier" {
                    let receiver = ctx.text(r)?.to_string();
                    ctx.raw_calls.push(vec![
                        ("caller_nid", Val::S(caller_nid.to_string())),
                        ("callee", Val::S(method_name.clone())),
                        ("is_member_call", Val::B(true)),
                        ("source_file", Val::S(ctx.str_path.to_string())),
                        ("source_location", Val::S(format!("L{line}"))),
                        ("receiver", Val::S(receiver)),
                        ("lang", Val::Static("objc")),
                    ]);
                } else if r.kind() == "field_expression" {
                    // `[self.bar doIt]`: capture ONLY the exact `self.<field>`
                    // shape. Anything else (`obj.prop`, a chain, `Foo.shared`)
                    // stays dropped -- passing the dotted text through would let a
                    // capitalized `Foo.shared` enter the explicit-class arm, where
                    // the key strips the dot and collides with a real class
                    // `FooShared`: a fabricated edge (#1556).
                    let kids = children(r);
                    if kids.len() == 3
                        && kids[0].kind() == "identifier"
                        && ctx.text(kids[0])? == "self"
                        && kids[1].kind() == "."
                        && kids[2].kind() == "field_identifier"
                    {
                        let receiver = ctx.text(kids[2])?.to_string();
                        ctx.raw_calls.push(vec![
                            ("caller_nid", Val::S(caller_nid.to_string())),
                            ("callee", Val::S(method_name.clone())),
                            ("is_member_call", Val::B(true)),
                            ("source_file", Val::S(ctx.str_path.to_string())),
                            ("source_location", Val::S(format!("L{line}"))),
                            ("receiver", Val::S(receiver)),
                            ("receiver_kind", Val::Static("self_field")),
                            ("lang", Val::Static("objc")),
                        ]);
                    }
                }
            }
        }
    } else if t == "field_expression" {
        // `self.name` -- dot-syntax sugar for `[self name]`. Resolved to a sibling
        // method of the SAME class by EXACT id match: a suffix match would
        // mis-resolve `self.name` to `-surname`, and would let a
        // substring-colliding sibling suppress the real `-name` edge (#1475).
        for child in children(n) {
            if child.kind() != "field_identifier" {
                continue;
            }
            let field_name = ctx.text(child)?.to_string();
            let target = ctx.mkid(&[container_nid, &field_name])?;
            if sibling_nids.contains(&target) && target != caller_nid {
                let pair = (caller_nid.to_string(), target.clone());
                if seen_calls.insert(pair) {
                    let line = n.start_position().row + 1;
                    ctx.add_edge(caller_nid, &target, "accesses", line, None);
                }
            }
        }
    } else if t == "selector_expression" {
        // `@selector(doSomething:withParam:)`. Emitted only when EXACTLY ONE
        // method matches, to avoid ambiguous fan-out, and matched exactly so
        // `-doThing` stays distinct from `-reallyDoThing` (#1475).
        let mut sel_name = String::new();
        for c in children(n) {
            if c.kind() == "identifier" {
                sel_name.push_str(ctx.text(c)?);
            }
        }
        if !sel_name.is_empty() {
            let mut matches: Vec<String> = Vec::new();
            for (m, cont) in method_bodies_keys {
                let expect = ctx.mkid(&[cont, &sel_name])?;
                if *m == expect && m != caller_nid && !matches.contains(m) {
                    matches.push(m.clone());
                }
            }
            matches.sort();
            if matches.len() == 1 {
                let pair = (caller_nid.to_string(), matches[0].clone());
                if seen_calls.insert(pair) {
                    let line = n.start_position().row + 1;
                    ctx.add_edge(caller_nid, &matches[0], "calls", line, Some("call"));
                }
            }
        }
    }
    for child in children(n) {
        walk_calls(
            ctx,
            child,
            caller_nid,
            container_nid,
            all_method_nids,
            sibling_nids,
            method_bodies_keys,
            seen_calls,
        )?;
    }
    Ok(())
}

pub fn walk_objc<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
    res: &crate::Resolvers<'py>,
) -> PyResult<Outcome<'py>> {
    // ObjC resolves quoted `#import "Foo.h"` through the same resolver the C
    // walker uses; without it the include target id cannot be computed.
    // `res.c` is the C include resolver: `extract_objc` calls the same
    // `_resolve_c_include_path` the C walker does.
    let resolver: Option<&dyn PathResolver> = Some(&res.c);
    match extract(py, path, source, resolver) {
        Ok(dict) => Ok(Outcome::Native(dict)),
        Err(reason) => Ok(Outcome::Defer(reason)),
    }
}

fn extract<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
    resolver: Option<&dyn PathResolver>,
) -> Result<Bound<'py, PyDict>, &'static str> {
    if std::str::from_utf8(source).is_err() {
        return Err("source_not_utf8");
    }
    // Blank the two macros, preserving LENGTH so byte offsets and line numbers
    // are unchanged. Done on an owned copy; `ctx.src` then points at it.
    let mut blanked = source.to_vec();
    for m in BLANK_MACROS {
        let pat = m.as_bytes();
        let mut from = 0usize;
        while let Some(pos) = blanked[from..]
            .windows(pat.len())
            .position(|w| w == pat)
            .map(|p| p + from)
        {
            for b in &mut blanked[pos..pos + pat.len()] {
                *b = b' ';
            }
            from = pos + pat.len();
        }
    }

    let stem = file_stem(path).ok_or("path_needs_pathlib")?;
    let file_nid = make_id_ascii(&[path]).ok_or("non_ascii_path")?;

    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_objc::LANGUAGE.into())
        .map_err(|_| "grammar_load_failed")?;
    let tree = parser.parse(&blanked, None).ok_or("parse_failed")?;
    let root = tree.root_node();
    if root.has_error() {
        return Err("parse_error");
    }

    let mut ctx = Ctx {
        src: &blanked,
        str_path: path,
        stem,
        file_nid: file_nid.clone(),
        nodes: Vec::new(),
        edges: Vec::new(),
        raw_calls: Vec::new(),
        seen_ids: HashSet::new(),
        method_bodies: Vec::new(),
        type_table: Vec::new(),
        type_table_seen: HashSet::new(),
        field_types: Vec::new(),
        resolver,
    };

    let file_label = path.rsplit('/').next().unwrap_or(path).to_string();
    ctx.add_node(&file_nid, &file_label, 1);

    walk(&mut ctx, root, None)?;

    // A LIST in `nodes` insertion order, not a set -- see the module doc.
    let all_method_nids: Vec<String> = ctx
        .nodes
        .iter()
        .filter(|n| n.id != file_nid)
        .map(|n| n.id.clone())
        .collect();
    let mut class_method_nids: HashMap<String, HashSet<String>> = HashMap::new();
    for (m_nid, _, container) in &ctx.method_bodies {
        class_method_nids
            .entry(container.clone())
            .or_default()
            .insert(m_nid.clone());
    }
    let method_bodies_keys: Vec<(String, String)> = ctx
        .method_bodies
        .iter()
        .map(|(m, _, c)| (m.clone(), c.clone()))
        .collect();

    // The type table is built from EVERY body first, before any call resolution.
    let bodies: Vec<(String, Node, String)> = ctx.method_bodies.clone();
    for (_, body, _) in &bodies {
        local_var_types(&mut ctx, *body)?;
    }

    let mut seen_calls: HashSet<(String, String)> = HashSet::new();
    for (caller_nid, body, container_nid) in &bodies {
        let empty = HashSet::new();
        let siblings = class_method_nids.get(container_nid).unwrap_or(&empty).clone();
        walk_calls(
            &mut ctx,
            *body,
            caller_nid,
            container_nid,
            &all_method_nids,
            &siblings,
            &method_bodies_keys,
            &mut seen_calls,
        )?;
    }

    let out = PyDict::new(py);
    let nodes = PyList::empty(py);
    for n in &ctx.nodes {
        nodes
            .append(emit::node_to_py(py, n, false, false).map_err(|_| "py_error")?)
            .map_err(|_| "py_error")?;
    }
    let edges = PyList::empty(py);
    for e in &ctx.edges {
        edges
            .append(emit::edge_to_py(py, e).map_err(|_| "py_error")?)
            .map_err(|_| "py_error")?;
    }
    let raw_calls = PyList::empty(py);
    for c in &ctx.raw_calls {
        raw_calls
            .append(emit::raw_call_to_py(py, c).map_err(|_| "py_error")?)
            .map_err(|_| "py_error")?;
    }
    out.set_item("nodes", nodes).map_err(|_| "py_error")?;
    out.set_item("edges", edges).map_err(|_| "py_error")?;
    out.set_item("raw_calls", raw_calls).map_err(|_| "py_error")?;
    out.set_item("input_tokens", 0i64).map_err(|_| "py_error")?;
    out.set_item("output_tokens", 0i64).map_err(|_| "py_error")?;

    if !ctx.type_table.is_empty() {
        let d = PyDict::new(py);
        d.set_item("path", path).map_err(|_| "py_error")?;
        let t = PyDict::new(py);
        for (k, v) in &ctx.type_table {
            t.set_item(k, v).map_err(|_| "py_error")?;
        }
        d.set_item("table", t).map_err(|_| "py_error")?;
        out.set_item("objc_type_table", d).map_err(|_| "py_error")?;
    }

    // Drop TOMBSTONED (conflicting) entries and then empty tables, in insertion
    // order -- the Python builds these with two comprehensions over dicts.
    let mut field_tables: Vec<(String, Vec<(String, String)>)> = Vec::new();
    for (cls, tbl) in &ctx.field_types {
        let kept: Vec<(String, String)> = tbl
            .iter()
            .filter_map(|(f, t)| t.as_ref().map(|t| (f.clone(), t.clone())))
            .collect();
        if !kept.is_empty() {
            field_tables.push((cls.clone(), kept));
        }
    }
    if !field_tables.is_empty() {
        let d = PyDict::new(py);
        d.set_item("path", path).map_err(|_| "py_error")?;
        let tables = PyDict::new(py);
        for (cls, tbl) in &field_tables {
            let inner = PyDict::new(py);
            for (f, t) in tbl {
                inner.set_item(f, t).map_err(|_| "py_error")?;
            }
            tables.set_item(cls, inner).map_err(|_| "py_error")?;
        }
        d.set_item("tables", tables).map_err(|_| "py_error")?;
        out.set_item("objc_field_types", d).map_err(|_| "py_error")?;
    }

    Ok(out)
}

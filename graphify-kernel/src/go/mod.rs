//! Go: a BESPOKE walker, like `bash/`.
//!
//! `extract_go` has its own walk, its own `pkg_scope` (the parent DIRECTORY's
//! name, so methods on one type across several files of a package share a
//! canonical type node), and a case-collision salt that nothing else on the
//! kernel has. None of that fits `EngineConfig`.
//!
//! Nothing here touches the filesystem, so unlike Bash there is no scope
//! deferral: every Go file the grammar parses is handled.

use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tree_sitter::{Node, Parser};

use crate::engine::R;
use crate::ids::{file_stem, make_id_ascii, parent_name};
use crate::js::ast::{children, text_checked};
use crate::js::emit::{self, EdgeRow, NodeRow, RawCall, Val};
use crate::py::helpers::BUILTIN_GLOBALS;
use crate::Outcome;

/// `_GO_PREDECLARED_TYPES`: never a user-defined type reference.
const PREDECLARED_TYPES: &[&str] = &[
    "bool", "byte", "complex64", "complex128", "error", "float32", "float64", "int", "int8",
    "int16", "int32", "int64", "rune", "string", "uint", "uint8", "uint16", "uint32", "uint64",
    "uintptr", "any", "comparable",
];

/// `_GO_PREDECLARED_FUNCS`, filtered ONLY when the callee is a bare identifier.
///
/// Deliberately Go-local rather than added to the shared `BUILTIN_GLOBALS`:
/// `new`, `close` and friends are ordinary method names in the eleven other
/// languages that consult the shared set, and listing them there would kill
/// every in-file Rust `Type::new()` edge. Bare-identifier-only for the same
/// reason within Go -- `h.append(v)` and `pkg.Delete(x)` are selector callees
/// and are genuine calls.
///
/// The full spec list, not a hand-picked subset: `len`, `max`, `min` and `print`
/// carry the same shadowing hazard as `append`, which on one 8.9k-node Go
/// codebase collected 330 phantom inbound `calls` edges onto a single unexported
/// `append` method.
const PREDECLARED_FUNCS: &[&str] = &[
    "append", "cap", "clear", "close", "complex", "copy", "delete", "imag", "len", "make", "max",
    "min", "new", "panic", "print", "println", "real", "recover",
];

const TYPE_WRAPPERS: &[&str] = &[
    "pointer_type",
    "slice_type",
    "array_type",
    "map_type",
    "channel_type",
    "parenthesized_type",
];

struct Ctx<'a> {
    src: &'a [u8],
    str_path: &'a str,
    stem: String,
    pkg_scope: String,
    file_nid: String,
    nodes: Vec<NodeRow>,
    edges: Vec<EdgeRow>,
    raw_calls: Vec<RawCall>,
    seen_ids: HashSet<String>,
    function_bodies: Vec<(String, usize, usize)>,
    imported_pkgs: Vec<(String, String)>,
    case_groups: HashMap<String, Vec<String>>,
    label_to_nid: HashMap<String, String>,
    seen_call_pairs: HashSet<(String, String)>,
}

impl<'a> Ctx<'a> {
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

    /// A SOURCELESS stub for a name not declared in this file -- typically a type
    /// defined in another file of the same package -- so the corpus-level rewire
    /// can collapse it onto the real definition (#1402).
    fn ensure_named_node(&mut self, name: &str, _line: usize) -> R<String> {
        let scoped = self.mkid(&[&self.pkg_scope.clone(), name])?;
        if self.seen_ids.contains(&scoped) {
            return Ok(scoped);
        }
        let bare = self.mkid(&[name])?;
        if !self.seen_ids.contains(&bare) {
            self.seen_ids.insert(bare.clone());
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
}

/// `_go_collect_type_refs`: `(name, is_generic_arg)` per referenced type.
fn collect_type_refs(
    ctx: &Ctx,
    node: Option<Node>,
    generic: bool,
    out: &mut Vec<(String, bool)>,
) -> R<()> {
    let node = match node {
        Some(n) => n,
        None => return Ok(()),
    };
    match node.kind() {
        "type_identifier" => {
            let text = ctx.text(node)?;
            if !text.is_empty() && !PREDECLARED_TYPES.contains(&text) {
                out.push((text.to_string(), generic));
            }
            return Ok(());
        }
        "qualified_type" => {
            // The package qualifier is KEPT, so the generic stub rewire cannot
            // attach `testing.T` to an unrelated local type or function named T.
            let text = ctx.text(node)?;
            if !text.is_empty() {
                out.push((text.to_string(), generic));
            }
            return Ok(());
        }
        "generic_type" => {
            if let Some(type_field) = node.child_by_field_name("type") {
                collect_type_refs(ctx, Some(type_field), generic, out)?;
            }
            for c in children(node) {
                if c.kind() == "type_arguments" {
                    for arg in children(c) {
                        if arg.is_named() {
                            collect_type_refs(ctx, Some(arg), true, out)?;
                        }
                    }
                }
            }
            return Ok(());
        }
        k if TYPE_WRAPPERS.contains(&k) => {
            for c in children(node) {
                if c.is_named() {
                    collect_type_refs(ctx, Some(c), generic, out)?;
                }
            }
            return Ok(());
        }
        _ => {}
    }
    if node.is_named() {
        for c in children(node) {
            if c.is_named() {
                collect_type_refs(ctx, Some(c), generic, out)?;
            }
        }
    }
    Ok(())
}

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
            ctx.add_edge(owner, &tgt, "references", line, Some(c));
        }
    }
    Ok(())
}

/// `emit_go_method_refs`: parameter and result types.
fn emit_method_refs(ctx: &mut Ctx, func_node: Node, func_nid: &str, line: usize) -> R<()> {
    if let Some(params) = func_node.child_by_field_name("parameters") {
        for p in children(params) {
            if p.kind() != "parameter_declaration" {
                continue;
            }
            emit_type_refs(
                ctx,
                func_nid,
                p.child_by_field_name("type"),
                "parameter_type",
                line,
            )?;
        }
    }
    let result = match func_node.child_by_field_name("result") {
        Some(r) => r,
        None => return Ok(()),
    };
    if result.kind() == "parameter_list" {
        for p in children(result) {
            if p.kind() != "parameter_declaration" {
                continue;
            }
            let mut type_node = p.child_by_field_name("type");
            if type_node.is_none() {
                type_node = children(p).into_iter().find(|c| c.is_named());
            }
            emit_type_refs(ctx, func_nid, type_node, "return_type", line)?;
        }
    } else {
        emit_type_refs(ctx, func_nid, Some(result), "return_type", line)?;
    }
    Ok(())
}

/// The receiver's TYPE name, with a leading `*` stripped.
fn receiver_type_of(ctx: &Ctx, node: Node) -> R<Option<String>> {
    let receiver = match node.child_by_field_name("receiver") {
        Some(r) => r,
        None => return Ok(None),
    };
    for param in children(receiver) {
        if param.kind() == "parameter_declaration" {
            if let Some(type_node) = param.child_by_field_name("type") {
                let t = ctx.text(type_node)?;
                return Ok(Some(t.trim_start_matches('*').trim().to_string()));
            }
            break;
        }
    }
    Ok(None)
}

fn plain_symbol_nid(ctx: &Ctx, node: Node) -> R<Option<(String, String)>> {
    let name_node = match node.child_by_field_name("name") {
        Some(n) => n,
        None => return Ok(None),
    };
    let name = ctx.text(name_node)?.to_string();
    let base = if node.kind() == "method_declaration" {
        match receiver_type_of(ctx, node)? {
            Some(rt) => ctx.mkid(&[&ctx.pkg_scope, &rt])?,
            None => ctx.stem.clone(),
        }
    } else {
        ctx.stem.clone()
    };
    Ok(Some((ctx.mkid(&[&base, &name])?, name)))
}

fn scan_declarations(ctx: &mut Ctx, node: Node) -> R<()> {
    if matches!(node.kind(), "function_declaration" | "method_declaration") {
        if let Some((nid, name)) = plain_symbol_nid(ctx, node)? {
            let group = ctx.case_groups.entry(nid).or_default();
            if !group.contains(&name) {
                group.push(name);
            }
        }
        return Ok(());
    }
    for child in children(node) {
        scan_declarations(ctx, child)?;
    }
    Ok(())
}

/// The #2779 case-collision salt.
///
/// Node ids are casefolded, so `Run` and `run` declared in one file produce the
/// same id and the second was silently dropped -- the unexported half vanished
/// and its call sites bound by bare name to a same-named function in another
/// package, which Go's visibility rules make impossible.
///
/// The EXPORTED member keeps the plain id: only exported symbols are reachable
/// across packages, so cross-package edges (and edges cached in graph.json from
/// files an incremental rebuild does not touch) target it, and keeping its id
/// stable means adding or removing an unexported sibling re-points nothing. When
/// the collision has no unique exported member (`Run`/`RUN`), EVERY member is
/// salted rather than picking one arbitrarily, so the result never depends on
/// declaration order.
fn symbol_nid(ctx: &Ctx, plain_nid: &str, name: &str) -> R<String> {
    let names = match ctx.case_groups.get(plain_nid) {
        Some(n) if n.len() >= 2 => n,
        _ => return Ok(plain_nid.to_string()),
    };
    if !name.is_ascii() {
        return Err("non_ascii_id");
    }
    let exported: Vec<&String> = names
        .iter()
        .filter(|n| n.chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
        .collect();
    if exported.len() == 1 && exported[0] == name {
        return Ok(plain_nid.to_string());
    }
    use sha1::{Digest, Sha1};
    let digest = Sha1::digest(name.as_bytes());
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    ctx.mkid(&[plain_nid, &hex[..6]])
}

fn walk<'tree>(ctx: &mut Ctx, node: Node<'tree>) -> R<()> {
    match node.kind() {
        "function_declaration" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let func_name = ctx.text(name_node)?.to_string();
                let line = node.start_position().row + 1;
                let plain = ctx.mkid(&[&ctx.stem.clone(), &func_name])?;
                let func_nid = symbol_nid(ctx, &plain, &func_name)?;
                ctx.add_node(&func_nid, &format!("{func_name}()"), line);
                let f = ctx.file_nid.clone();
                ctx.add_edge(&f, &func_nid, "contains", line, None);
                emit_method_refs(ctx, node, &func_nid, line)?;
                if let Some(body) = node.child_by_field_name("body") {
                    ctx.function_bodies
                        .push((func_nid, body.start_byte(), body.end_byte()));
                }
            }
            return Ok(());
        }
        "method_declaration" => {
            let receiver_type = receiver_type_of(ctx, node)?;
            let name_node = match node.child_by_field_name("name") {
                Some(n) => n,
                None => return Ok(()),
            };
            let method_name = ctx.text(name_node)?.to_string();
            let line = node.start_position().row + 1;
            let method_nid = match receiver_type {
                Some(rt) if !rt.is_empty() => {
                    let parent_nid = ctx.mkid(&[&ctx.pkg_scope.clone(), &rt])?;
                    ctx.add_node(&parent_nid, &rt, line);
                    let plain = ctx.mkid(&[&parent_nid, &method_name])?;
                    let nid = symbol_nid(ctx, &plain, &method_name)?;
                    ctx.add_node(&nid, &format!(".{method_name}()"), line);
                    ctx.add_edge(&parent_nid, &nid, "method", line, None);
                    nid
                }
                _ => {
                    let plain = ctx.mkid(&[&ctx.stem.clone(), &method_name])?;
                    let nid = symbol_nid(ctx, &plain, &method_name)?;
                    ctx.add_node(&nid, &format!("{method_name}()"), line);
                    let f = ctx.file_nid.clone();
                    ctx.add_edge(&f, &nid, "contains", line, None);
                    nid
                }
            };
            emit_method_refs(ctx, node, &method_nid, line)?;
            if let Some(body) = node.child_by_field_name("body") {
                ctx.function_bodies
                    .push((method_nid, body.start_byte(), body.end_byte()));
            }
            return Ok(());
        }
        "type_declaration" => {
            for child in children(node) {
                if child.kind() != "type_spec" {
                    continue;
                }
                let name_node = match child.child_by_field_name("name") {
                    Some(n) => n,
                    None => continue,
                };
                let type_name = ctx.text(name_node)?.to_string();
                let line = child.start_position().row + 1;
                let type_nid = ctx.mkid(&[&ctx.pkg_scope.clone(), &type_name])?;
                ctx.add_node(&type_nid, &type_name, line);
                let f = ctx.file_nid.clone();
                ctx.add_edge(&f, &type_nid, "contains", line, None);
                let type_body = children(child)
                    .into_iter()
                    .find(|tc| matches!(tc.kind(), "struct_type" | "interface_type"));
                let type_body = match type_body {
                    Some(b) => b,
                    None => continue,
                };
                if type_body.kind() == "struct_type" {
                    for fdl in children(type_body) {
                        if fdl.kind() != "field_declaration_list" {
                            continue;
                        }
                        for field in children(fdl) {
                            if field.kind() != "field_declaration" {
                                continue;
                            }
                            // An UNNAMED field is an embed, not a reference.
                            let has_name = children(field)
                                .into_iter()
                                .any(|fc| fc.kind() == "field_identifier");
                            let mut type_node = field.child_by_field_name("type");
                            if type_node.is_none() {
                                type_node = children(field)
                                    .into_iter()
                                    .find(|fc| fc.is_named() && fc.kind() != "field_identifier");
                            }
                            let mut refs: Vec<(String, bool)> = Vec::new();
                            collect_type_refs(ctx, type_node, false, &mut refs)?;
                            let fline = field.start_position().row + 1;
                            for (ref_name, generic) in refs {
                                let tgt = ctx.ensure_named_node(&ref_name, fline)?;
                                if tgt == type_nid {
                                    continue;
                                }
                                if !has_name && !generic {
                                    ctx.add_edge(&type_nid, &tgt, "embeds", fline, None);
                                } else {
                                    let c = if generic { "generic_arg" } else { "field" };
                                    ctx.add_edge(&type_nid, &tgt, "references", fline, Some(c));
                                }
                            }
                        }
                    }
                } else {
                    for elem in children(type_body) {
                        if elem.kind() != "type_elem" {
                            continue;
                        }
                        let mut refs: Vec<(String, bool)> = Vec::new();
                        for sub in children(elem) {
                            if sub.is_named() {
                                collect_type_refs(ctx, Some(sub), false, &mut refs)?;
                            }
                        }
                        let eline = elem.start_position().row + 1;
                        for (ref_name, generic) in refs {
                            let tgt = ctx.ensure_named_node(&ref_name, eline)?;
                            if tgt == type_nid {
                                continue;
                            }
                            if !generic {
                                ctx.add_edge(&type_nid, &tgt, "embeds", eline, None);
                            } else {
                                ctx.add_edge(
                                    &type_nid,
                                    &tgt,
                                    "references",
                                    eline,
                                    Some("generic_arg"),
                                );
                            }
                        }
                    }
                }
            }
            return Ok(());
        }
        "import_declaration" => {
            for child in children(node) {
                if child.kind() == "import_spec_list" {
                    for spec in children(child) {
                        if spec.kind() == "import_spec" {
                            import_spec(ctx, spec)?;
                        }
                    }
                } else if child.kind() == "import_spec" {
                    import_spec(ctx, child)?;
                }
            }
            return Ok(());
        }
        _ => {}
    }
    for child in children(node) {
        walk(ctx, child)?;
    }
    Ok(())
}

fn import_spec(ctx: &mut Ctx, spec: Node) -> R<()> {
    let path_node = match spec.child_by_field_name("path") {
        Some(p) => p,
        None => return Ok(()),
    };
    let raw = ctx.text(path_node)?.trim_matches('"').to_string();
    // `go_pkg_` prefixed so a stdlib name like `context` cannot collide with a
    // local file of the same basename.
    let tgt = ctx.mkid(&["go", "pkg", &raw])?;
    let f = ctx.file_nid.clone();
    ctx.add_edge(
        &f,
        &tgt,
        "imports_from",
        spec.start_position().row + 1,
        Some("import"),
    );
    let alias = spec.child_by_field_name("name");
    let local_name = match alias {
        Some(a) => ctx.text(a)?.to_string(),
        None => raw.rsplit('/').next().unwrap_or(&raw).to_string(),
    };
    if !local_name.is_empty() && local_name != "_" && local_name != "." {
        // Last write wins, as a dict assignment does.
        ctx.imported_pkgs.retain(|(k, _)| k != &local_name);
        ctx.imported_pkgs.push((local_name, raw));
    }
    Ok(())
}

fn walk_calls<'tree>(ctx: &mut Ctx, node: Node<'tree>, caller_nid: &str) -> R<()> {
    if matches!(node.kind(), "function_declaration" | "method_declaration") {
        return Ok(());
    }
    if node.kind() == "call_expression" {
        let func_node = node.child_by_field_name("function");
        let mut callee_name: Option<String> = None;
        let mut is_member_call = false;
        let mut is_bare_identifier = false;
        let mut package_receiver: Option<String> = None;
        let mut import_path: Option<String> = None;
        if let Some(func_node) = func_node {
            if func_node.kind() == "identifier" {
                is_bare_identifier = true;
                callee_name = Some(ctx.text(func_node)?.to_string());
            } else if func_node.kind() == "selector_expression" {
                let field = func_node.child_by_field_name("field");
                let operand = func_node.child_by_field_name("operand");
                let receiver_name = match operand {
                    Some(o) => ctx.text(o)?.to_string(),
                    None => String::new(),
                };
                // A package-qualified call (`fmt.Println`) can resolve
                // cross-file; a receiver method call (`s.logger.Log`) has no
                // import evidence and is left member-flagged.
                let imported = ctx
                    .imported_pkgs
                    .iter()
                    .find(|(k, _)| *k == receiver_name)
                    .map(|(_, v)| v.clone());
                is_member_call = imported.is_none();
                if let Some(path) = imported {
                    package_receiver = Some(receiver_name);
                    import_path = Some(path);
                }
                if let Some(field) = field {
                    callee_name = Some(ctx.text(field)?.to_string());
                }
            }
        }
        if is_bare_identifier {
            if let Some(name) = callee_name.as_deref() {
                if PREDECLARED_FUNCS.contains(&name) {
                    // A bare `append(s, x)` is the builtin, never a same-named
                    // method a sibling file declares. Dropped BEFORE both
                    // branches, so it neither emits an in-file phantom edge nor
                    // reaches raw_calls for the cross-file pass to bind.
                    callee_name = None;
                }
            }
        }
        if let Some(callee) = callee_name {
            if !BUILTIN_GLOBALS.contains(&callee.as_str()) {
                // An imported selector must never resolve through a bare local.
                let tgt_nid = if import_path.is_some() {
                    None
                } else {
                    ctx.label_to_nid.get(&callee).cloned()
                };
                let line = node.start_position().row + 1;
                match tgt_nid {
                    Some(tgt) if tgt != caller_nid => {
                        let pair = (caller_nid.to_string(), tgt.clone());
                        if ctx.seen_call_pairs.insert(pair) {
                            let sf = ctx.str_path.to_string();
                            ctx.edges.push(EdgeRow {
                                source: caller_nid.to_string(),
                                target: tgt,
                                relation: "calls",
                                fields: vec![
                                    ("context", Val::Static("call")),
                                    ("confidence", Val::Static("EXTRACTED")),
                                    ("source_file", Val::S(sf)),
                                    ("source_location", Val::S(format!("L{line}"))),
                                    ("weight", Val::F(1.0)),
                                ],
                            });
                        }
                    }
                    // `elif callee_name:` -- reached when the lookup missed AND
                    // when it hit the caller itself, unlike the other walkers.
                    _ => {
                        ctx.raw_calls.push(vec![
                            ("caller_nid", Val::S(caller_nid.to_string())),
                            ("callee", Val::S(callee)),
                            ("is_member_call", Val::B(is_member_call)),
                            ("language", Val::Static("go")),
                            (
                                "receiver",
                                match &package_receiver {
                                    Some(r) => Val::S(r.clone()),
                                    None => Val::None,
                                },
                            ),
                            (
                                "import_path",
                                match &import_path {
                                    Some(p) => Val::S(p.clone()),
                                    None => Val::None,
                                },
                            ),
                            ("source_file", Val::S(ctx.str_path.to_string())),
                            ("source_location", Val::S(format!("L{line}"))),
                        ]);
                    }
                }
            }
        }
    }
    for child in children(node) {
        walk_calls(ctx, child, caller_nid)?;
    }
    Ok(())
}

pub fn walk_go<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
    _res: &crate::Resolvers<'py>,
) -> PyResult<Outcome<'py>> {
    match extract(py, path, source) {
        Ok(dict) => Ok(Outcome::Native(dict)),
        Err(reason) => Ok(Outcome::Defer(reason)),
    }
}

fn find_node<'tree>(root: Node<'tree>, start: usize, end: usize) -> Option<Node<'tree>> {
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        if n.start_byte() == start && n.end_byte() == end && n.kind() == "block" {
            return Some(n);
        }
        if n.start_byte() <= start && n.end_byte() >= end {
            stack.extend(children(n));
        }
    }
    None
}

fn extract<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
) -> Result<Bound<'py, PyDict>, &'static str> {
    if std::str::from_utf8(source).is_err() {
        return Err("source_not_utf8");
    }
    let stem = file_stem(path).ok_or("path_needs_pathlib")?;
    // `path.parent.name or stem`: the package directory, so methods on one type
    // across several files of a package share a canonical type node.
    let parent = parent_name(path).ok_or("path_needs_pathlib")?;
    let pkg_scope = if parent.is_empty() { stem.clone() } else { parent };
    let file_nid = make_id_ascii(&[path]).ok_or("non_ascii_path")?;

    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_go::LANGUAGE.into())
        .map_err(|_| "grammar_load_failed")?;
    let tree = parser.parse(source, None).ok_or("parse_failed")?;
    let root = tree.root_node();
    if root.has_error() {
        return Err("parse_error");
    }

    let mut ctx = Ctx {
        src: source,
        str_path: path,
        stem,
        pkg_scope,
        file_nid: file_nid.clone(),
        nodes: Vec::new(),
        edges: Vec::new(),
        raw_calls: Vec::new(),
        seen_ids: HashSet::new(),
        function_bodies: Vec::new(),
        imported_pkgs: Vec::new(),
        case_groups: HashMap::new(),
        label_to_nid: HashMap::new(),
        seen_call_pairs: HashSet::new(),
    };

    let file_label = path.rsplit('/').next().unwrap_or(path).to_string();
    ctx.add_node(&file_nid, &file_label, 1);

    scan_declarations(&mut ctx, root)?;
    walk(&mut ctx, root)?;

    for n in &ctx.nodes {
        let mut label = String::new();
        for (k, v) in &n.fields {
            if *k == "label" {
                if let Val::S(s) = v {
                    label = s.clone();
                }
            }
        }
        let normalised = label.trim_matches(|c| c == '(' || c == ')').trim_start_matches('.');
        ctx.label_to_nid.insert(normalised.to_string(), n.id.clone());
    }

    let bodies = ctx.function_bodies.clone();
    for (caller_nid, start, end) in bodies {
        let body = find_node(root, start, end).ok_or("body_not_found")?;
        walk_calls(&mut ctx, body, &caller_nid)?;
    }

    let mut clean: Vec<&EdgeRow> = Vec::new();
    for e in &ctx.edges {
        let target_ok =
            ctx.seen_ids.contains(&e.target) || matches!(e.relation, "imports" | "imports_from");
        if ctx.seen_ids.contains(&e.source) && target_ok {
            clean.push(e);
        }
    }

    let out = PyDict::new(py);
    let nodes = PyList::empty(py);
    for n in &ctx.nodes {
        nodes
            .append(emit::node_to_py(py, n, false, false).map_err(|_| "py_error")?)
            .map_err(|_| "py_error")?;
    }
    let edges = PyList::empty(py);
    for e in clean {
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
    let imports = PyDict::new(py);
    for (k, v) in &ctx.imported_pkgs {
        imports.set_item(k, v).map_err(|_| "py_error")?;
    }
    out.set_item("nodes", nodes).map_err(|_| "py_error")?;
    out.set_item("edges", edges).map_err(|_| "py_error")?;
    out.set_item("raw_calls", raw_calls).map_err(|_| "py_error")?;
    out.set_item("go_imports", imports).map_err(|_| "py_error")?;
    Ok(out)
}

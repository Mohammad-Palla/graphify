//! PowerShell: a BESPOKE walker.
//!
//! # Why the declaration walk descends a function body TWICE
//!
//! `function_statement` pushes its body to `function_bodies` for the call pass
//! AND walks it immediately, because `Import-Module` and dot-sourcing inside a
//! function must still emit FILE-level `imports_from` edges (#1331). Removing
//! either descent loses something: the first loses call edges, the second loses
//! imports declared inside functions.
//!
//! # Parse ceiling
//!
//! 62.8% over 548 files -- the lowest of any language routed here except
//! Fortran's fixed-form dialect. PowerShell/PowerShell itself is 64.1%; the two
//! small corpora are worse (PSReadLine 33.3% of 9 files, posh-git 50.0% of 32).
//! Still well clear of the rate floor: at 63% native, routing pays.

use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tree_sitter::{Node, Parser};

use crate::engine::R;
use crate::ids::{file_stem, make_id_ascii};
use crate::js::ast::{children, text_checked};
use crate::js::emit::{self, EdgeRow, NodeRow, RawCall, Val};
use crate::Outcome;

/// `_PS_SKIP`: a keyword or an import command, never a function call. Compared
/// LOWERCASED.
const PS_SKIP: &[&str] = &[
    "using", "return", "if", "else", "elseif", "foreach", "for", "while", "do", "switch", "try",
    "catch", "finally", "throw", "break", "continue", "exit", "param", "begin", "process", "end",
    // Handled as an import edge, not a call.
    "import-module",
];

/// `re.sub(r'\.[^.]+$', '', s)`: drop a trailing extension.
///
/// The pattern is anchored at `$` and `[^.]+` demands at least one non-dot
/// character, so `"a.."` is left ALONE while `"a.b.c"` becomes `"a.b"`.
fn strip_extension(s: &str) -> &str {
    match s.rfind('.') {
        Some(i) if i + 1 < s.len() => &s[..i],
        _ => s,
    }
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
    function_bodies: Vec<(String, Node<'tree>)>,
    seen_call_pairs: HashSet<(String, String)>,
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
}

/// `_find_script_block_body`: the `script_block_body` under a `script_block`, or
/// the `script_block` itself when it has none.
fn find_script_block_body<'tree>(node: Node<'tree>) -> Option<Node<'tree>> {
    for child in children(node) {
        if child.kind() == "script_block" {
            for sc in children(child) {
                if sc.kind() == "script_block_body" {
                    return Some(sc);
                }
            }
            return Some(child);
        }
    }
    None
}

/// `_ps_type_name`: `type_literal > type_spec > type_name > type_identifier`.
fn ps_type_name<'a>(ctx: &Ctx<'a, '_>, type_literal: Option<Node>) -> R<Option<&'a str>> {
    let tl = match type_literal {
        Some(t) => t,
        None => return Ok(None),
    };
    for spec in children(tl) {
        if spec.kind() != "type_spec" {
            continue;
        }
        for tname in children(spec) {
            if tname.kind() != "type_name" {
                continue;
            }
            for tid in children(tname) {
                if tid.kind() == "type_identifier" {
                    return Ok(Some(ctx.text(tid)?));
                }
            }
        }
    }
    Ok(None)
}

fn walk<'tree>(
    ctx: &mut Ctx<'_, 'tree>,
    node: Node<'tree>,
    parent_class_nid: Option<&str>,
) -> R<()> {
    let t = node.kind();

    if t == "function_statement" {
        let name_node = children(node)
            .into_iter()
            .find(|c| c.kind() == "function_name");
        if let Some(nn) = name_node {
            let func_name = ctx.text(nn)?.to_string();
            let line = node.start_position().row + 1;
            let func_nid = ctx.mkid(&[&ctx.stem.clone(), &func_name])?;
            ctx.add_node(&func_nid, &format!("{func_name}()"), line);
            let f = ctx.file_nid.clone();
            ctx.add_edge(&f, &func_nid, "contains", line, None);
            if let Some(body) = find_script_block_body(node) {
                ctx.function_bodies.push((func_nid, body));
                // The SECOND descent -- see the module doc.
                walk(ctx, body, parent_class_nid)?;
            }
        }
        return Ok(());
    }

    if t == "enum_statement" {
        // A PowerShell enum is a first-class type, referenced via a `[Color]`
        // type literal. Without this branch those references resolved to a
        // sourceless phantom stub instead of the real in-file definition, and the
        // enum's members were lost entirely.
        let name_node = children(node).into_iter().find(|c| c.kind() == "simple_name");
        if let Some(nn) = name_node {
            let enum_name = ctx.text(nn)?.to_string();
            let line = node.start_position().row + 1;
            let enum_nid = ctx.mkid(&[&ctx.stem.clone(), &enum_name])?;
            ctx.add_node(&enum_nid, &enum_name, line);
            let f = ctx.file_nid.clone();
            ctx.add_edge(&f, &enum_nid, "contains", line, None);
            for member in children(node) {
                if member.kind() != "enum_member" {
                    continue;
                }
                let mn = children(member).into_iter().find(|c| c.kind() == "simple_name");
                let mn = match mn {
                    Some(m) => m,
                    None => continue,
                };
                let member_name = ctx.text(mn)?.to_string();
                let m_line = member.start_position().row + 1;
                let member_nid = ctx.mkid(&[&enum_nid, &member_name])?;
                ctx.add_node(&member_nid, &member_name, m_line);
                ctx.add_edge(&enum_nid, &member_nid, "contains", m_line, None);
            }
        }
        return Ok(());
    }

    if t == "class_statement" {
        let name_node = children(node).into_iter().find(|c| c.kind() == "simple_name");
        if let Some(nn) = name_node {
            let class_name = ctx.text(nn)?.to_string();
            let line = node.start_position().row + 1;
            let class_nid = ctx.mkid(&[&ctx.stem.clone(), &class_name])?;
            ctx.add_node(&class_nid, &class_name, line);
            let f = ctx.file_nid.clone();
            ctx.add_edge(&f, &class_nid, "contains", line, None);
            // PowerShell has no syntactic base-vs-interface split, so (matching
            // the C# convention) the FIRST base after `:` is the superclass and
            // every later one is an interface. Unlike the ObjC superclass scan,
            // `colon_seen` is NOT cleared -- every simple_name after the colon
            // counts, which is what makes `base_index` meaningful.
            let mut colon_seen = false;
            let mut base_index = 0usize;
            for child in children(node) {
                if child.kind() == ":" {
                    colon_seen = true;
                } else if colon_seen && child.kind() == "simple_name" {
                    let bname = ctx.text(child)?.to_string();
                    let base_nid = ctx.ensure_named_node(&bname)?;
                    if base_nid != class_nid {
                        let rel = if base_index == 0 {
                            "inherits"
                        } else {
                            "implements"
                        };
                        ctx.add_edge(&class_nid, &base_nid, rel, line, None);
                    }
                    base_index += 1;
                }
            }
            for child in children(node) {
                walk(ctx, child, Some(&class_nid))?;
            }
        }
        return Ok(());
    }

    if t == "class_property_definition" {
        if let Some(parent) = parent_class_nid {
            let type_literal = children(node).into_iter().find(|c| c.kind() == "type_literal");
            let type_name = ps_type_name(ctx, type_literal)?.map(|s| s.to_string());
            if let Some(type_name) = type_name {
                if !type_name.is_empty() {
                    let line = node.start_position().row + 1;
                    let target = ctx.ensure_named_node(&type_name)?;
                    if target != parent {
                        ctx.add_edge(parent, &target, "references", line, Some("field"));
                    }
                }
            }
            return Ok(());
        }
        // With no enclosing class the branch is not taken and the node falls
        // through to the generic recursion.
    }

    if t == "class_method_definition" {
        let name_node = children(node).into_iter().find(|c| c.kind() == "simple_name");
        if let Some(nn) = name_node {
            let method_name = ctx.text(nn)?.to_string();
            let line = node.start_position().row + 1;
            let method_nid = match parent_class_nid {
                Some(p) => {
                    let nid = ctx.mkid(&[p, &method_name])?;
                    ctx.add_node(&nid, &format!(".{method_name}()"), line);
                    ctx.add_edge(p, &nid, "method", line, None);
                    nid
                }
                None => {
                    let nid = ctx.mkid(&[&ctx.stem.clone(), &method_name])?;
                    ctx.add_node(&nid, &format!("{method_name}()"), line);
                    let f = ctx.file_nid.clone();
                    ctx.add_edge(&f, &nid, "contains", line, None);
                    nid
                }
            };
            // The return type is the `type_literal` SIBLING of simple_name -- the
            // FIRST one, which is why the parameter types come from the
            // parameter list rather than from a later sibling.
            let rt = children(node).into_iter().find(|c| c.kind() == "type_literal");
            let return_type_name = ps_type_name(ctx, rt)?.map(|s| s.to_string());
            if let Some(rtn) = return_type_name {
                if !rtn.is_empty() {
                    let target = ctx.ensure_named_node(&rtn)?;
                    if target != method_nid {
                        ctx.add_edge(&method_nid, &target, "references", line, Some("return_type"));
                    }
                }
            }
            let param_list = children(node)
                .into_iter()
                .find(|c| c.kind() == "class_method_parameter_list");
            if let Some(pl) = param_list {
                for p in children(pl) {
                    if p.kind() != "class_method_parameter" {
                        continue;
                    }
                    let ptl = children(p).into_iter().find(|c| c.kind() == "type_literal");
                    let ptype_name = ps_type_name(ctx, ptl)?.map(|s| s.to_string());
                    let ptype_name = match ptype_name {
                        Some(n) if !n.is_empty() => n,
                        _ => continue,
                    };
                    let p_line = p.start_position().row + 1;
                    let target = ctx.ensure_named_node(&ptype_name)?;
                    if target != method_nid {
                        ctx.add_edge(
                            &method_nid,
                            &target,
                            "references",
                            p_line,
                            Some("parameter_type"),
                        );
                    }
                }
            }
            if let Some(body) = find_script_block_body(node) {
                ctx.function_bodies.push((method_nid, body));
            }
        }
        return Ok(());
    }

    if t == "command" {
        let line = node.start_position().row + 1;
        // Dot-sourcing: `. ./Shared.psm1`. Spelled with a
        // `command_invokation_operator` and a `command_name_expr`, NOT a
        // `command_name`, so it has to be checked before the command-name arm.
        let invoke_op = children(node)
            .into_iter()
            .find(|c| c.kind() == "command_invokation_operator");
        if let Some(op) = invoke_op {
            if ctx.text(op)?.trim() == "." {
                let name_expr = children(node)
                    .into_iter()
                    .find(|c| c.kind() == "command_name_expr");
                if let Some(ne) = name_expr {
                    let name_node = children(ne)
                        .into_iter()
                        .find(|c| c.kind() == "command_name");
                    if let Some(nn) = name_node {
                        let raw_path = ctx.text(nn)?;
                        // `^[./\\]+` then the extension, then backslashes to
                        // slashes, then the last path segment.
                        let trimmed =
                            raw_path.trim_start_matches(|c| c == '.' || c == '/' || c == '\\');
                        let no_ext = strip_extension(trimmed).replace('\\', "/");
                        let module_name =
                            no_ext.rsplit('/').next().unwrap_or("").to_string();
                        if !module_name.is_empty() {
                            let tgt = ctx.mkid(&[&module_name])?;
                            let f = ctx.file_nid.clone();
                            ctx.add_edge(&f, &tgt, "imports_from", line, None);
                        }
                    }
                }
                // Returns whether or not anything was emitted.
                return Ok(());
            }
        }

        let cmd_name_node = children(node).into_iter().find(|c| c.kind() == "command_name");
        if let Some(cn) = cmd_name_node {
            let cmd_text = ctx.text(cn)?.to_lowercase();
            if cmd_text == "using" {
                let mut tokens: Vec<String> = Vec::new();
                for child in children(node) {
                    if child.kind() != "command_elements" {
                        continue;
                    }
                    for el in children(child) {
                        if el.kind() == "generic_token" {
                            tokens.push(ctx.text(el)?.to_string());
                        }
                    }
                }
                let module_tokens: Vec<&String> = tokens
                    .iter()
                    .filter(|t| {
                        !matches!(
                            t.to_lowercase().as_str(),
                            "namespace" | "module" | "assembly"
                        )
                    })
                    .collect();
                if let Some(last) = module_tokens.last() {
                    let module_name = last.rsplit('.').next().unwrap_or("").to_string();
                    let tgt = ctx.mkid(&[&module_name])?;
                    let f = ctx.file_nid.clone();
                    ctx.add_edge(&f, &tgt, "imports_from", line, None);
                }
            } else if cmd_text == "import-module" {
                // The module name is the FIRST `generic_token`, or the one right
                // after a `-Name` / `-N` flag, which overrides it.
                let mut module_name: Option<String> = None;
                let mut expect_name = false;
                for child in children(node) {
                    if child.kind() != "command_elements" {
                        continue;
                    }
                    for el in children(child) {
                        if el.kind() == "command_parameter" {
                            let param_text =
                                ctx.text(el)?.trim_start_matches('-').to_lowercase();
                            expect_name = param_text == "name" || param_text == "n";
                        } else if el.kind() == "generic_token" {
                            let token = ctx.text(el)?.to_string();
                            if module_name.is_none() || expect_name {
                                module_name = Some(token);
                                expect_name = false;
                            }
                        }
                    }
                }
                if let Some(mn) = module_name {
                    let bare = strip_extension(&mn)
                        .rsplit('/')
                        .next()
                        .unwrap_or("")
                        .rsplit('\\')
                        .next()
                        .unwrap_or("")
                        .to_string();
                    if !bare.is_empty() {
                        let tgt = ctx.mkid(&[&bare])?;
                        let f = ctx.file_nid.clone();
                        ctx.add_edge(&f, &tgt, "imports_from", line, None);
                    }
                }
            }
        }
        return Ok(());
    }

    for child in children(node) {
        walk(ctx, child, parent_class_nid)?;
    }
    Ok(())
}

fn walk_calls<'tree>(
    ctx: &mut Ctx<'_, 'tree>,
    node: Node<'tree>,
    caller_nid: &str,
    label_to_nid: &HashMap<String, String>,
) -> R<()> {
    if matches!(node.kind(), "function_statement" | "class_statement") {
        return Ok(());
    }
    if node.kind() == "command" {
        let cmd_name_node = children(node).into_iter().find(|c| c.kind() == "command_name");
        if let Some(cn) = cmd_name_node {
            // The RAW text is kept for the raw_call callee; only the lookup and
            // the skip test are lowercased.
            let cmd_text = ctx.text(cn)?.to_string();
            let lowered = cmd_text.to_lowercase();
            if !PS_SKIP.contains(&lowered.as_str()) {
                let line = node.start_position().row + 1;
                let tgt = label_to_nid.get(&lowered).cloned();
                // The `elif` hangs off `if tgt_nid and tgt_nid != caller_nid`, so
                // a command resolving to the CALLER ITSELF still produces a
                // raw_call -- the Zig/Elixir shape again.
                match tgt {
                    Some(t) if t != caller_nid => {
                        let pair = (caller_nid.to_string(), t.clone());
                        if ctx.seen_call_pairs.insert(pair) {
                            ctx.add_edge(caller_nid, &t, "calls", line, None);
                        }
                    }
                    _ => {
                        if !cmd_text.is_empty() {
                            ctx.raw_calls.push(vec![
                                ("caller_nid", Val::S(caller_nid.to_string())),
                                ("callee", Val::S(cmd_text)),
                                ("is_member_call", Val::B(false)),
                                ("source_file", Val::S(ctx.str_path.to_string())),
                                ("source_location", Val::S(format!("L{line}"))),
                            ]);
                        }
                    }
                }
            }
        }
    }
    for child in children(node) {
        walk_calls(ctx, child, caller_nid, label_to_nid)?;
    }
    Ok(())
}

pub fn walk_powershell<'py>(
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

fn extract<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
) -> Result<Bound<'py, PyDict>, &'static str> {
    if std::str::from_utf8(source).is_err() {
        return Err("source_not_utf8");
    }
    let stem = file_stem(path).ok_or("path_needs_pathlib")?;
    let file_nid = make_id_ascii(&[path]).ok_or("non_ascii_path")?;

    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_powershell::LANGUAGE.into())
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
        file_nid: file_nid.clone(),
        nodes: Vec::new(),
        edges: Vec::new(),
        raw_calls: Vec::new(),
        seen_ids: HashSet::new(),
        function_bodies: Vec::new(),
        seen_call_pairs: HashSet::new(),
    };

    let file_label = path.rsplit('/').next().unwrap_or(path).to_string();
    ctx.add_node(&file_nid, &file_label, 1);

    walk(&mut ctx, root, None)?;

    // `{label.strip("()").lstrip(".").lower(): id}` -- a dict comprehension, so
    // a later node with the same normalised label OVERWRITES an earlier one.
    let mut label_to_nid: HashMap<String, String> = HashMap::new();
    for n in &ctx.nodes {
        if let Some((_, Val::S(label))) = n.fields.first() {
            let key = label
                .trim_matches(|c| c == '(' || c == ')')
                .trim_start_matches('.')
                .to_lowercase();
            label_to_nid.insert(key, n.id.clone());
        }
    }

    let bodies: Vec<(String, Node)> = ctx.function_bodies.clone();
    for (caller_nid, body) in bodies {
        walk_calls(&mut ctx, body, &caller_nid, &label_to_nid)?;
    }

    // BOTH `imports_from` and `imports` survive a missing target here.
    let clean: Vec<&EdgeRow> = ctx
        .edges
        .iter()
        .filter(|e| {
            ctx.seen_ids.contains(&e.source)
                && (ctx.seen_ids.contains(&e.target)
                    || matches!(e.relation, "imports_from" | "imports"))
        })
        .collect();

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
    out.set_item("nodes", nodes).map_err(|_| "py_error")?;
    out.set_item("edges", edges).map_err(|_| "py_error")?;
    out.set_item("raw_calls", raw_calls).map_err(|_| "py_error")?;
    Ok(out)
}

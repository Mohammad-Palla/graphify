//! OCaml: a BESPOKE walker, and the first here driven by TWO grammars.
//!
//! `.ml` implementation files use `language_ocaml`; `.mli` interface files use
//! `language_ocaml_interface`. They are different grammars with different node
//! kinds (`value_definition` vs `value_specification`), so the walker handles
//! both and the SEAM decides which by suffix -- `extract_ocaml` passes a
//! different `BespokeGrammar`, and the language key it routes to (`ocaml` or
//! `ocaml_interface`) carries the choice. Sniffing the suffix here would put the
//! decision in two places.
//!
//! # Two shapes this walker does NOT have
//!
//! * **No `raw_calls`.** The result is `{"nodes", "edges"}` and nothing else.
//!   Calls are resolved in a SECOND pass over `call_sites` at the end of the
//!   file, because `let rec ... and ...` means a call can precede its
//!   definition, and a one-pass walk would miss every forward reference.
//! * **No edge cleaning.** Every other bespoke walker filters edges whose target
//!   is not in `seen_ids`; `extract_ocaml` does not, because `ref_stub` has
//!   already put a node behind every target it emits.
//!
//! # Parse ceiling
//!
//! 99.0% over 4,403 files (ocaml, dune, mirage) -- the largest corpus gated here
//! and the second-highest ceiling after Elixir's 100%.

use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tree_sitter::{Node, Parser};

use crate::engine::R;
use crate::ids::{file_stem, make_id_ascii};
use crate::js::ast::{children, text_checked};
use crate::js::emit::{self, EdgeRow, NodeRow, Val};
use crate::Outcome;

/// The child kinds `last_name` accepts, in the Python's membership order.
const NAME_KINDS: &[&str] = &["value_name", "module_name", "constructor_name"];

struct Ctx<'a> {
    src: &'a [u8],
    str_path: &'a str,
    stem: String,
    file_nid: String,
    nodes: Vec<NodeRow>,
    edges: Vec<EdgeRow>,
    seen_ids: HashSet<String>,
    /// Same-file definitions, for call resolution. A name defined more than once
    /// is moved to `ambiguous` and never resolved locally again.
    local_defs: HashMap<String, String>,
    ambiguous: HashSet<String>,
    /// Modules DEFINED in this file. `M.f` may bind to a local `f` only when `M`
    /// is one of these -- otherwise `M` is an external library and binding would
    /// be a false edge.
    local_modules: HashSet<String>,
    /// `(caller_nid, callee, qualifier_root, full_path_text, line)`, resolved
    /// after the walk so forward references work.
    call_sites: Vec<(String, String, Option<String>, String, usize)>,
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

    /// No `context` slot: `extract_ocaml`'s `add_edge` has none. `confidence`
    /// varies, though -- `INFERRED` for anything resolved through a stub.
    fn add_edge(
        &mut self,
        src: &str,
        tgt: &str,
        relation: &'static str,
        line: usize,
        confidence: &'static str,
    ) {
        self.edges.push(EdgeRow {
            source: src.to_string(),
            target: tgt.to_string(),
            relation,
            fields: vec![
                ("confidence", Val::Static(confidence)),
                ("source_file", Val::S(self.str_path.to_string())),
                ("source_location", Val::S(format!("L{line}"))),
                ("weight", Val::F(1.0)),
            ],
        });
    }

    /// A SOURCELESS stub for a cross-file target -- an `open`ed module, or a call
    /// to a name not defined here.
    ///
    /// The corpus rewire collapses it onto the unique real definition; external
    /// names (Stdlib, Core) dangle and are pruned. A *sourced* stub would bake
    /// this file's path into the id and block the rewire, which is the #1402
    /// phantom-duplicate bug.
    fn ref_stub(&mut self, name: &str) -> R<String> {
        let nid = self.mkid(&[name])?;
        if self.seen_ids.insert(nid.clone()) {
            self.nodes.push(NodeRow {
                id: nid.clone(),
                fields: vec![
                    ("label", Val::S(name.to_string())),
                    ("file_type", Val::Static("code")),
                    ("source_file", Val::Static("")),
                    ("source_location", Val::Static("")),
                    ("origin_file", Val::S(self.str_path.to_string())),
                ],
            });
        }
        Ok(nid)
    }

    fn register_def(&mut self, name: &str, nid: &str) {
        if self.ambiguous.contains(name) {
            return;
        }
        if let Some(existing) = self.local_defs.get(name) {
            if existing != nid {
                self.ambiguous.insert(name.to_string());
                self.local_defs.remove(name);
                return;
            }
        }
        self.local_defs.insert(name.to_string(), nid.to_string());
    }
}

fn line_of(node: Node) -> usize {
    node.start_position().row + 1
}

fn named_child_text<'a>(ctx: &Ctx<'a>, node: Node, child_kind: &str) -> R<Option<&'a str>> {
    for child in children(node) {
        if child.kind() == child_kind {
            return Ok(Some(ctx.text(child)?));
        }
    }
    Ok(None)
}

/// The final component of a `*_path` node: `Geo.area` -> `area`.
///
/// DIRECT children only, and the LAST match wins. The qualifier is nested under
/// a child `module_path`, so a deep search would wrongly return the module
/// qualifier instead of the name.
fn last_name<'a>(ctx: &Ctx<'a>, path_node: Node) -> R<Option<&'a str>> {
    let mut found: Option<&'a str> = None;
    for n in children(path_node) {
        if NAME_KINDS.contains(&n.kind()) {
            found = Some(ctx.text(n)?);
        }
    }
    Ok(found)
}

/// The leftmost (outermost) module segment of a `*_path`: `Stdlib.List.map` ->
/// `Stdlib`. `None` for an unqualified path, which has no child `module_path`.
fn path_root_module<'a>(ctx: &Ctx<'a>, path_node: Node) -> R<Option<&'a str>> {
    let mp = children(path_node)
        .into_iter()
        .find(|c| c.kind() == "module_path");
    let mut node = match mp {
        Some(m) => m,
        None => return Ok(None),
    };
    loop {
        let inner = children(node).into_iter().find(|c| c.kind() == "module_path");
        match inner {
            Some(i) => node = i,
            None => break,
        }
    }
    let mn = children(node).into_iter().find(|c| c.kind() == "module_name");
    match mn {
        Some(m) => Ok(Some(ctx.text(m)?)),
        None => Ok(None),
    }
}

/// `defines` from the file node, `contains` from anything else.
fn containment(container_nid: &str, file_nid: &str) -> &'static str {
    if container_nid == file_nid {
        "defines"
    } else {
        "contains"
    }
}

fn emit_type(ctx: &mut Ctx, binding: Node, container_nid: &str) -> R<()> {
    let type_name = match named_child_text(ctx, binding, "type_constructor")? {
        Some(t) if !t.is_empty() => t.to_string(),
        _ => return Ok(()),
    };
    let line = line_of(binding);
    let nid = ctx.mkid(&[&ctx.stem.clone(), &type_name])?;
    ctx.add_node(&nid, &type_name, line);
    let rel = containment(container_nid, &ctx.file_nid);
    ctx.add_edge(container_nid, &nid, rel, line, "EXTRACTED");
    ctx.register_def(&type_name, &nid);

    // Variant constructors: variant_declaration -> constructor_declaration ->
    // constructor_name. Note the constructor node's LINE, not the binding's.
    for vd in children(binding) {
        if vd.kind() != "variant_declaration" {
            continue;
        }
        for cd in children(vd) {
            if cd.kind() != "constructor_declaration" {
                continue;
            }
            let cname = match named_child_text(ctx, cd, "constructor_name")? {
                Some(c) if !c.is_empty() => c.to_string(),
                _ => continue,
            };
            let cnid = ctx.mkid(&[&ctx.stem.clone(), &type_name, &cname])?;
            ctx.add_node(&cnid, &cname, line_of(cd));
            ctx.add_edge(&nid, &cnid, "contains", line_of(cd), "EXTRACTED");
        }
    }
    Ok(())
}

fn walk<'tree>(
    ctx: &mut Ctx<'_>,
    node: Node<'tree>,
    container_nid: &str,
    enclosing_value: &str,
) -> R<()> {
    let t = node.kind();

    if t == "open_module" {
        let mp = children(node).into_iter().find(|c| c.kind() == "module_path");
        let name = match mp {
            Some(m) => last_name(ctx, m)?.map(|s| s.to_string()),
            None => None,
        };
        if let Some(name) = name {
            if !name.is_empty() {
                let stub = ctx.ref_stub(&name)?;
                ctx.add_edge(container_nid, &stub, "imports_from", line_of(node), "INFERRED");
            }
        }
        return Ok(());
    }

    if t == "module_definition" {
        let binding = children(node)
            .into_iter()
            .find(|c| c.kind() == "module_binding");
        if let Some(binding) = binding {
            let mname = named_child_text(ctx, binding, "module_name")?.map(|s| s.to_string());
            if let Some(mname) = mname {
                if !mname.is_empty() {
                    let line = line_of(node);
                    let mnid = ctx.mkid(&[&ctx.stem.clone(), &mname])?;
                    ctx.add_node(&mnid, &mname, line);
                    let rel = containment(container_nid, &ctx.file_nid);
                    ctx.add_edge(container_nid, &mnid, rel, line, "EXTRACTED");
                    ctx.register_def(&mname, &mnid);
                    ctx.local_modules.insert(mname);
                    for child in children(binding) {
                        walk(ctx, child, &mnid, enclosing_value)?;
                    }
                    return Ok(());
                }
            }
        }
        // No binding, or an unnamed one: FALLS THROUGH to the generic recursion
        // at the bottom rather than returning.
    }

    if t == "module_type_definition" {
        // `.mli` only.
        let mname = named_child_text(ctx, node, "module_type_name")?.map(|s| s.to_string());
        if let Some(mname) = mname {
            if !mname.is_empty() {
                let line = line_of(node);
                let mnid = ctx.mkid(&[&ctx.stem.clone(), &mname])?;
                ctx.add_node(&mnid, &mname, line);
                let rel = containment(container_nid, &ctx.file_nid);
                ctx.add_edge(container_nid, &mnid, rel, line, "EXTRACTED");
                ctx.register_def(&mname, &mnid);
                ctx.local_modules.insert(mname);
                for child in children(node) {
                    walk(ctx, child, &mnid, enclosing_value)?;
                }
                return Ok(());
            }
        }
        // Falls through, as above.
    }

    if t == "value_definition" {
        // Only a structure/top-level binding is a real definition. `let x = e in
        // body` is ALSO a `value_definition`, nested under a `let_expression` --
        // it must not mint a node, and must not steal call attribution from the
        // enclosing named function.
        let is_toplevel = node
            .parent()
            .is_some_and(|p| matches!(p.kind(), "compilation_unit" | "structure"));
        for lb in children(node) {
            if lb.kind() != "let_binding" {
                continue;
            }
            let vname = named_child_text(ctx, lb, "value_name")?.map(|s| s.to_string());
            let mut new_scope = enclosing_value.to_string();
            if let Some(vname) = &vname {
                if !vname.is_empty() && is_toplevel {
                    let line = line_of(lb);
                    let nid = ctx.mkid(&[&ctx.stem.clone(), vname])?;
                    ctx.add_node(&nid, vname, line);
                    let rel = containment(container_nid, &ctx.file_nid);
                    ctx.add_edge(container_nid, &nid, rel, line, "EXTRACTED");
                    ctx.register_def(vname, &nid);
                    new_scope = nid;
                }
            }
            // Descend into the body. A unit/pattern/local binding keeps the
            // OUTER scope, so its calls stay attributed to the enclosing
            // function rather than to nothing.
            for child in children(lb) {
                walk(ctx, child, container_nid, &new_scope)?;
            }
        }
        return Ok(());
    }

    if t == "value_specification" {
        // `.mli` only: a declaration with no body, so no scope and no calls.
        let vname = named_child_text(ctx, node, "value_name")?.map(|s| s.to_string());
        if let Some(vname) = vname {
            if !vname.is_empty() {
                let line = line_of(node);
                let nid = ctx.mkid(&[&ctx.stem.clone(), &vname])?;
                ctx.add_node(&nid, &vname, line);
                let rel = containment(container_nid, &ctx.file_nid);
                ctx.add_edge(container_nid, &nid, rel, line, "EXTRACTED");
                ctx.register_def(&vname, &nid);
            }
        }
        return Ok(());
    }

    if t == "type_definition" {
        for binding in children(node) {
            if binding.kind() == "type_binding" {
                emit_type(ctx, binding, container_nid)?;
            }
        }
        return Ok(());
    }

    if t == "application_expression" {
        // `node.named_children[0]` -- the first NAMED child, which is the
        // function position.
        let fn_node = children(node).into_iter().find(|c| c.is_named());
        if let Some(fn_node) = fn_node {
            if fn_node.kind() == "value_path" {
                let callee = last_name(ctx, fn_node)?.map(|s| s.to_string());
                if let Some(callee) = callee {
                    if !callee.is_empty() {
                        let caller = if enclosing_value.is_empty() {
                            ctx.file_nid.clone()
                        } else {
                            enclosing_value.to_string()
                        };
                        let root = path_root_module(ctx, fn_node)?.map(|s| s.to_string());
                        let full = ctx.text(fn_node)?.to_string();
                        ctx.call_sites
                            .push((caller, callee, root, full, line_of(node)));
                    }
                }
            }
        }
        // Falls through deliberately: the ARGUMENTS may hold further
        // applications and definitions.
    }

    for child in children(node) {
        walk(ctx, child, container_nid, enclosing_value)?;
    }
    Ok(())
}

pub fn walk_ocaml<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
    _res: &crate::Resolvers<'py>,
) -> PyResult<Outcome<'py>> {
    match extract(py, path, source, false) {
        Ok(dict) => Ok(Outcome::Native(dict)),
        Err(reason) => Ok(Outcome::Defer(reason)),
    }
}

pub fn walk_ocaml_interface<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
    _res: &crate::Resolvers<'py>,
) -> PyResult<Outcome<'py>> {
    match extract(py, path, source, true) {
        Ok(dict) => Ok(Outcome::Native(dict)),
        Err(reason) => Ok(Outcome::Defer(reason)),
    }
}

fn extract<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
    interface: bool,
) -> Result<Bound<'py, PyDict>, &'static str> {
    if std::str::from_utf8(source).is_err() {
        return Err("source_not_utf8");
    }
    let stem = file_stem(path).ok_or("path_needs_pathlib")?;
    let file_nid = make_id_ascii(&[path]).ok_or("non_ascii_path")?;

    let grammar: tree_sitter::Language = if interface {
        tree_sitter_ocaml::LANGUAGE_OCAML_INTERFACE.into()
    } else {
        tree_sitter_ocaml::LANGUAGE_OCAML.into()
    };
    let mut parser = Parser::new();
    parser
        .set_language(&grammar)
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
        seen_ids: HashSet::new(),
        local_defs: HashMap::new(),
        ambiguous: HashSet::new(),
        local_modules: HashSet::new(),
        call_sites: Vec::new(),
    };

    let file_label = path.rsplit('/').next().unwrap_or(path).to_string();
    ctx.add_node(&file_nid, &file_label, 1);

    walk(&mut ctx, root, &file_nid.clone(), "")?;

    // Pass 2. A qualified call `M.f` where `M` is NOT defined in this file is an
    // external-library call, so it must NOT bind to a same-named local `f` --
    // that would be a false edge, and a self-loop when the caller IS that local
    // `f`. Keying the stub by the FULL qualified name keeps it from collapsing
    // onto the local one in the corpus rewire, while an unqualified call still
    // gets a bare-name stub so `Geo.area` can collapse onto another file's
    // `area`.
    let sites = std::mem::take(&mut ctx.call_sites);
    for (caller, callee, qualifier, full_path, line) in sites {
        let external = match &qualifier {
            Some(q) => !ctx.local_modules.contains(q),
            None => false,
        };
        if external {
            let key = if ctx.local_defs.contains_key(&callee) {
                full_path
            } else {
                callee
            };
            let stub = ctx.ref_stub(&key)?;
            ctx.add_edge(&caller, &stub, "calls", line, "INFERRED");
        } else if let Some(tgt) = ctx.local_defs.get(&callee).cloned() {
            ctx.add_edge(&caller, &tgt, "calls", line, "EXTRACTED");
        } else {
            let stub = ctx.ref_stub(&callee)?;
            ctx.add_edge(&caller, &stub, "calls", line, "INFERRED");
        }
    }

    // NO edge cleaning: `extract_ocaml` returns `edges` unfiltered.
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
    out.set_item("nodes", nodes).map_err(|_| "py_error")?;
    out.set_item("edges", edges).map_err(|_| "py_error")?;
    Ok(out)
}

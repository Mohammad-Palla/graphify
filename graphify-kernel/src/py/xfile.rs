//! Cross-file import material for `_resolve_cross_file_imports`, from the
//! phase-2 parse.
//!
//! That pass is 7.1s of django's SERIAL phase 3, and with the native walker live
//! it was still doing 2,016 `tree_sitter.parse` calls and a 4,274,926-call Python
//! walk in the PARENT -- re-deriving from scratch what this walker had in hand a
//! moment earlier. django's Python files were being parsed three times per build.
//!
//! # What crosses the boundary, and what does not
//!
//! The pass has two halves. Pass 1 builds a CORPUS-WIDE index (`stem_to_entities`,
//! `bare_to_qualified`) and pass 2 resolves each file's imports against it. Only
//! the per-file half is portable -- a worker sees one file and cannot know the
//! corpus -- and that is exactly the expensive half, because it is the one that
//! parses and walks.
//!
//! So this emits the two per-file intermediates and nothing else:
//!
//! * `imports` -- for each `from X import A as B`, the bare module name, whether
//!   the import was relative, and the `(imported, local)` pairs. Resolving that to
//!   a target stem needs `path.parent`, `_file_stem` and the global index, all of
//!   which stay in Python.
//! * `refs` -- `identifier text -> [(owning top-level symbol NAME, first line)]`.
//!
//! # Why `refs` carries a NAME and not a node id
//!
//! `crossfile_py_java_cs_bash` runs LATE in phase 3 -- after `id_remap` and
//! `disambiguate_ids` -- so by the time it reads `file_result["nodes"]` the ids
//! have been rewritten. A payload built in phase 2 carries the pre-remap ids, and
//! emitting those produced 16 wrong edges on django: the `uses` edge pointed at a
//! stale id, lost the dedup tie-break to a `calls` edge on the same pair, and
//! shifted the clustering from 2,117 communities to 2,074.
//!
//! Nothing caught that except a full-graph comparison. A differential at this
//! pass's own boundary reported IDENTICAL, because calling it directly on fresh
//! per-file results means no remap has happened yet and both sides agree.
//!
//! Labels survive the remap where ids do not, so the walker emits the definition
//! NAME -- exactly the key the Python looks up in `name_to_nid` -- and the parent
//! resolves it against its own post-remap map. Membership is identical either
//! way, since the remap changes ids and not labels.
//!
//! # Why `refs` is filtered, and why that is exact
//!
//! The emitting loop is `for name, tgt in import_targets: for src, line in
//! ref_sources.get(name, {})`. `ref_sources` is therefore only ever READ at keys
//! that are local names of this file's own from-imports -- every other entry is
//! built and thrown away. Restricting the payload to that set is not an
//! approximation; the discarded entries are unreachable. On django it turns a
//! ~300k-entry payload into a few dozen names per file.
//!
//! # Order is part of the output
//!
//! The edges come out in `import_targets` order, then `ref_sources[name]` order,
//! and both are Python dict insertion order. So imports are emitted in walk order
//! with their pairs in child order, and each name's nids in first-encounter order
//! -- a list, not a map, because a map would lose it.

use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tree_sitter::Node;

use crate::js::ast::children;
use super::{Ctx, R};

pub struct Import {
    /// A `relative_import` child was reached, which stops the scan and wins.
    pub relative: bool,
    /// Candidate bare module names, in the order Python tries them.
    ///
    /// NOT a single name. `resolve_import`'s loop is
    /// `if child.type == "dotted_name" and target_fq is None`, so it keeps trying
    /// LATER `dotted_name` children until one resolves against the corpus index --
    /// and the imported names are themselves direct `dotted_name` children of the
    /// statement. So `from unknownmod import Foo`, with `unknownmod` absent from
    /// the index, goes on to try `Foo` as a module name. That is surprising, but
    /// it is the behaviour, and this walker reproduces behaviour rather than
    /// correcting it. The worker cannot know which candidate resolves, so it emits
    /// all of them in order and the parent stops at the first hit.
    ///
    /// For a relative import this holds at most one entry: the first
    /// `dotted_name` inside the `relative_import`.
    pub bares: Vec<String>,
    /// `(imported_name, local_name)`, in child order.
    pub pairs: Vec<(String, String)>,
}

pub struct XFile {
    pub imports: Vec<Import>,
    /// `identifier -> [(owner symbol name, first line)]`, both in
    /// first-encounter order.
    pub refs: Vec<(String, Vec<(String, usize)>)>,
}

/// The set of local symbol names, in first-writer order.
///
/// The Python builds `name -> nid`; this needs only the KEYS, because the payload
/// carries names and the parent owns the mapping (see the module docstring). Built
/// from the walk's own nodes, which is what lets it run inside the worker. The
/// filters mirror the Python exactly:
/// the file node ends in `.py`; a function/method label ends in `()` and is
/// stripped to its name (a METHOD keeps its leading dot, so `.bar` can never
/// match a module-scope definition, which is the intent); `ensure_named_node`
/// stubs are sourceless and excluded by the `source_file != str_path` test there.
/// Rationale nodes cannot appear -- they are appended after the kernel returns.
/// First writer wins on a collision -- which for a name SET is simply membership.
fn local_symbol_names(ctx: &Ctx) -> HashSet<String> {
    let mut out: HashSet<String> = HashSet::new();
    for n in &ctx.nodes {
        if n.id == ctx.file_nid {
            continue;
        }
        let mut label: Option<&str> = None;
        let mut sourced = false;
        for (k, v) in &n.fields {
            match (*k, v) {
                ("label", crate::js::emit::Val::S(s)) => label = Some(s.as_str()),
                ("source_file", crate::js::emit::Val::S(s)) => sourced = s == ctx.str_path,
                _ => {}
            }
        }
        let Some(label) = label else { continue };
        if !sourced || label.is_empty() || label.ends_with(".py") {
            continue;
        }
        let sym = label.strip_suffix("()").unwrap_or(label);
        if sym.is_empty() {
            continue;
        }
        out.insert(sym.to_string());
    }
    out
}

/// The local names this file's `from ... import ...` statements bind -- the only
/// keys `ref_sources` is ever read at. See the module docstring.
fn imported_local_names(imports: &[Import]) -> HashSet<String> {
    let mut out = HashSet::new();
    for imp in imports {
        for (_, local) in &imp.pairs {
            out.insert(local.clone());
        }
    }
    out
}

/// `resolve_import`'s parsing half: the bare module name and the imported pairs.
/// Resolution against the corpus index stays in Python.
fn read_import(ctx: &Ctx, node: Node) -> R<Import> {
    let mut relative = false;
    let mut bares: Vec<String> = Vec::new();
    for child in children(node) {
        if child.kind() == "relative_import" {
            // Wins outright and stops the scan: the Python `break`s the outer loop
            // after setting `target_fq` from the relative path, discarding any
            // candidate an earlier `dotted_name` had produced.
            relative = true;
            bares.clear();
            for sub in children(child) {
                if sub.kind() == "dotted_name" {
                    let t = ctx.text(sub)?;
                    bares.push(t.rsplit('.').next().unwrap_or(t).to_string());
                    break;
                }
            }
            break;
        }
        if child.kind() == "dotted_name" {
            let t = ctx.text(child)?;
            bares.push(t.rsplit('.').next().unwrap_or(t).to_string());
        }
    }

    // Imported names come AFTER the `import` keyword token.
    let mut pairs = Vec::new();
    let mut past_import_kw = false;
    for child in children(node) {
        if child.kind() == "import" {
            past_import_kw = true;
            continue;
        }
        if !past_import_kw {
            continue;
        }
        let (imported, local) = match child.kind() {
            "dotted_name" => {
                let t = ctx.text(child)?.to_string();
                (t.clone(), t)
            }
            "aliased_import" => {
                let Some(name_node) = child.child_by_field_name("name") else {
                    continue;
                };
                let imported = ctx.text(name_node)?.to_string();
                let local = match child.child_by_field_name("alias") {
                    Some(a) => ctx.text(a)?.to_string(),
                    None => imported.clone(),
                };
                (imported, local)
            }
            _ => continue,
        };
        if imported.is_empty() || local.is_empty() {
            continue;
        }
        pairs.push((imported, local));
    }
    Ok(Import {
        relative,
        bares,
        pairs,
    })
}

/// Pass A: the import statements, in walk order.
fn collect_imports(ctx: &Ctx, node: Node, out: &mut Vec<Import>) -> R<()> {
    if node.kind() == "import_from_statement" {
        out.push(read_import(ctx, node)?);
        // Python returns without descending: identifiers inside an import are the
        // import itself, not a use.
        return Ok(());
    }
    for child in children(node) {
        collect_imports(ctx, child, out)?;
    }
    Ok(())
}

/// Pass B: `visit`, restricted to the names that can ever be read.
fn visit(
    ctx: &Ctx,
    node: Node,
    current_owner: Option<&str>,
    names: &HashSet<String>,
    wanted: &HashSet<String>,
    order: &mut Vec<String>,
    refs: &mut HashMap<String, (Vec<(String, usize)>, HashSet<String>)>,
) -> R<()> {
    if node.kind() == "import_from_statement" {
        return Ok(());
    }
    // Only set at module scope, so a nested def never overrides its container.
    let mut current = current_owner.map(|s| s.to_string());
    if current.is_none()
        && matches!(node.kind(), "class_definition" | "function_definition")
    {
        if let Some(name_node) = node.child_by_field_name("name") {
            let name = ctx.text(name_node)?;
            if names.contains(name) {
                current = Some(name.to_string());
            }
        }
    }
    if node.kind() == "identifier" {
        if let Some(cur) = current.as_deref() {
            let text = ctx.text(node)?;
            if wanted.contains(text) {
                let slot = refs.entry(text.to_string()).or_insert_with(|| {
                    order.push(text.to_string());
                    (Vec::new(), HashSet::new())
                });
                // `slot.setdefault(current_nid, line)`: first line wins. Keying
                // by owner NAME rather than nid is equivalent -- the Python map is
                // injective, since each node contributes exactly one name.
                if slot.1.insert(cur.to_string()) {
                    slot.0.push((cur.to_string(), node.start_position().row + 1));
                }
            }
        }
    }
    for child in children(node) {
        visit(ctx, child, current.as_deref(), names, wanted, order, refs)?;
    }
    Ok(())
}

pub fn collect(ctx: &Ctx, root: Node) -> R<XFile> {
    let mut imports = Vec::new();
    collect_imports(ctx, root, &mut imports)?;
    let names = local_symbol_names(ctx);
    // Python skips the whole file -- parse included -- when it has no local
    // symbols, so no refs can exist. Imports are still emitted; with no refs they
    // produce no edges, exactly as the skip does.
    let wanted = if names.is_empty() {
        HashSet::new()
    } else {
        imported_local_names(&imports)
    };
    let mut order: Vec<String> = Vec::new();
    let mut refs: HashMap<String, (Vec<(String, usize)>, HashSet<String>)> = HashMap::new();
    if !wanted.is_empty() {
        visit(ctx, root, None, &names, &wanted, &mut order, &mut refs)?;
    }
    let refs = order
        .into_iter()
        .map(|k| {
            let v = refs.remove(&k).expect("order and refs are built together");
            (k, v.0)
        })
        .collect();
    Ok(XFile { imports, refs })
}

pub fn to_py<'py>(py: Python<'py>, x: &XFile) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let imports = PyList::empty(py);
    for imp in &x.imports {
        let pairs = PyList::empty(py);
        for (a, b) in &imp.pairs {
            pairs.append((a, b))?;
        }
        let bares = PyList::empty(py);
        for b in &imp.bares {
            bares.append(b)?;
        }
        imports.append((imp.relative, bares, pairs))?;
    }
    out.set_item("imports", imports)?;
    let refs = PyList::empty(py);
    for (name, entries) in &x.refs {
        let lst = PyList::empty(py);
        for (nid, line) in entries {
            lst.append((nid, *line))?;
        }
        refs.append((name, lst))?;
    }
    out.set_item("refs", refs)?;
    Ok(out)
}

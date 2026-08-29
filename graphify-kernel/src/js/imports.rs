//! Import handling: the hybrid boundary with Python's resolver.
//!
//! `_resolve_js_import_target` probes the filesystem -- extension candidates,
//! index files, `is_file()` tests, tsconfig `paths` aliases, workspace package
//! manifests. Reproducing it in Rust would be a large, heavily special-cased
//! surface for an I/O-bound job that measured ~5% of phase 2 against ~88% for the
//! walk, so it stays in Python. What crosses the boundary is the *answer*.
//!
//! # Why a callback and not a pre-resolved map
//!
//! The obvious shape is: scan the file for specifiers, resolve them all in Python,
//! then walk with a lookup table. That needs a parse to do the scanning and
//! another to do the walk -- and parse is precisely the part of phase 2 that is C
//! on both sides and cannot be made faster, so doubling it spends the win before
//! the walk starts. It also over-resolves, since a file that ends up deferring for
//! an unrelated reason has already paid for every specifier in it.
//!
//! So [`Resolver`] wraps a Python callable and is consulted lazily, memoized per
//! file: one parse, and a specifier costs a resolution only if the walk actually
//! reaches it. Roughly 5-10 calls per file at ~1us of FFI each, against filesystem
//! work that dominates them by orders of magnitude.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use tree_sitter::Node;

use super::ast::{children, line_of, strip_chars};
use super::emit::{EdgeRow, Val};
use super::{Ctx, R};

/// What Python's `_resolve_js_import_target` said about one specifier, plus the
/// two follow-up questions its callers ask about the result (`is_file()` and
/// whether the path lands in `node_modules`). Answering those here rather than
/// in Rust keeps every filesystem decision on the Python side of the boundary.
#[derive(Clone)]
pub struct Resolution {
    pub tgt_nid: String,
    /// `str(resolved_path)`, or `None` when resolution fell through to the
    /// `ref`-namespaced external id.
    pub path: Option<String>,
    pub is_file: bool,
    pub in_node_modules: bool,
    /// `_file_stem(resolved_path)`.
    pub target_stem: Option<String>,
}

/// A lazily-consulted view of Python's resolver, memoized for one file.
///
/// `None` from the callback means Python's `_resolve_js_import_target` returned
/// `None` (an empty specifier). A missing callback, or one that raises, is a
/// *deferral* -- never an assumption that the specifier is external, because
/// guessing there would silently drop or invent an `imports_from` edge.
pub struct Resolver<'py> {
    callback: Option<Bound<'py, PyAny>>,
    cache: RefCell<HashMap<String, Option<Resolution>>>,
    /// `_resolve_js_module_path(raw, file.parent)` then `.resolve()`, as a string.
    ///
    /// A SECOND resolver, not a reuse of the first: `_resolve_js_import_target`
    /// and `_resolve_js_module_path` are different functions with different
    /// fallbacks -- the former mints a `ref`-namespaced id for an unresolvable
    /// specifier, the latter simply returns None -- and the symbol-fact collector
    /// calls the latter. Sharing one would silently change which specifiers
    /// produce facts.
    module_callback: Option<Bound<'py, PyAny>>,
    module_cache: RefCell<HashMap<String, Option<String>>>,
}

impl<'py> Resolver<'py> {
    pub fn new(
        callback: Option<Bound<'py, PyAny>>,
        module_callback: Option<Bound<'py, PyAny>>,
    ) -> Self {
        Resolver {
            callback,
            cache: RefCell::new(HashMap::new()),
            module_callback,
            module_cache: RefCell::new(HashMap::new()),
        }
    }

    /// `_resolve_js_module_path(raw, dir).resolve()` as a string, memoized per
    /// file. `Err` when no resolver was supplied: the facts must defer rather
    /// than silently drop every import fact in the file.
    pub fn resolve_module(&self, raw: &str) -> R<Option<String>> {
        if let Some(v) = self.module_cache.borrow().get(raw) {
            return Ok(v.clone());
        }
        let cb = self.module_callback.as_ref().ok_or("no_module_resolver")?;
        let ret = cb.call1((raw,)).map_err(|_| "module_resolver_raised")?;
        let parsed: Option<String> = ret.extract().map_err(|_| "module_resolver_bad_shape")?;
        self.module_cache
            .borrow_mut()
            .insert(raw.to_string(), parsed.clone());
        Ok(parsed)
    }

    fn get(&self, raw: &str) -> R<Option<Resolution>> {
        if let Some(v) = self.cache.borrow().get(raw) {
            return Ok(v.clone());
        }
        let cb = self.callback.as_ref().ok_or("no_resolver")?;
        let ret = cb.call1((raw,)).map_err(|_| "resolver_raised")?;
        let parsed = if ret.is_none() {
            None
        } else {
            let (tgt_nid, path, is_file, in_node_modules, target_stem): (
                String,
                Option<String>,
                bool,
                bool,
                Option<String>,
            ) = ret.extract().map_err(|_| "resolver_bad_shape")?;
            Some(Resolution {
                tgt_nid,
                path,
                is_file,
                in_node_modules,
                target_stem,
            })
        };
        self.cache
            .borrow_mut()
            .insert(raw.to_string(), parsed.clone());
        Ok(parsed)
    }
}

// ── strip sets, one per Python call site ────────────────────────────────────
// These differ, and the differences are load-bearing: a specifier written with a
// space inside the quotes strips differently in `_import_js` (which includes the
// space) than in `_js_external_import_names` (which does not), so the two sites
// can legitimately look up different keys for the same string node.
const STRIP_IMPORT: &str = "'\"` ";
const STRIP_EXTERNAL: &str = "\"'`";
const STRIP_DYN_STRING: &str = "'\" ";
const STRIP_DYN_TEMPLATE: &str = "`";

/// `_js_import_binds_external`.
fn binds_external(ctx: &Ctx, raw: &str) -> R<bool> {
    match ctx.res.get(raw)? {
        None => Ok(false), // empty specifier -- binds nothing
        Some(r) => match r.path {
            None => Ok(true), // unresolved after relative / alias / workspace lookup
            Some(_) => Ok(r.in_node_modules),
        },
    }
}

/// `_js_external_import_names`.
pub fn external_import_names(ctx: &Ctx, root: Node) -> R<HashSet<String>> {
    let mut bound = HashSet::new();
    walk_ext(ctx, root, &mut bound)?;
    Ok(bound)
}

fn walk_ext(ctx: &Ctx, n: Node, bound: &mut HashSet<String>) -> R<()> {
    for c in children(n) {
        if c.kind() == "import_statement" {
            if let Some(src_node) = c.child_by_field_name("source") {
                let raw = strip_chars(ctx.text(src_node)?, STRIP_EXTERNAL);
                if binds_external(ctx, &raw)? {
                    for child in children(c) {
                        if child.kind() == "import_clause" {
                            clause_names(ctx, child, bound)?;
                        }
                    }
                }
            }
            continue; // never recurse into an import_statement
        }
        walk_ext(ctx, c, bound)?;
    }
    Ok(())
}

fn clause_names(ctx: &Ctx, clause: Node, bound: &mut HashSet<String>) -> R<()> {
    for c in children(clause) {
        match c.kind() {
            // import Default from "pkg"
            "identifier" => {
                bound.insert(ctx.text(c)?.to_string());
            }
            // import * as NS from "pkg"
            "namespace_import" => {
                for ident in children(c) {
                    if ident.kind() == "identifier" {
                        bound.insert(ctx.text(ident)?.to_string());
                    }
                }
            }
            // import { A, B as C } from "pkg"
            "named_imports" => {
                for spec in children(c) {
                    if spec.kind() != "import_specifier" {
                        continue;
                    }
                    let idents: Vec<Node> = children(spec)
                        .into_iter()
                        .filter(|g| g.kind() == "identifier")
                        .collect();
                    // `B as C` exposes both names; only the LAST is bound here.
                    if let Some(last) = idents.last() {
                        bound.insert(ctx.text(*last)?.to_string());
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// The key order Python's import-edge literals use: `context` third, then
/// `confidence`, and `target_file` last when the target resolved to a real file.
fn import_edge(
    ctx: &Ctx,
    source: &str,
    target: &str,
    relation: &'static str,
    context: &'static str,
    line: usize,
    target_file: Option<&str>,
) -> EdgeRow {
    let mut fields = vec![
        ("context", Val::Static(context)),
        ("confidence", Val::Static("EXTRACTED")),
        ("source_file", Val::S(ctx.str_path.to_string())),
        ("source_location", Val::S(format!("L{line}"))),
        ("weight", Val::F(1.0)),
    ];
    if let Some(tf) = target_file {
        fields.push(("target_file", Val::S(tf.to_string())));
    }
    EdgeRow {
        source: source.to_string(),
        target: target.to_string(),
        relation,
        fields,
    }
}

/// `_import_js`.
pub fn import_js(ctx: &mut Ctx, node: Node) -> R<()> {
    let is_reexport = node.kind() == "export_statement";
    if is_reexport {
        let mut has_from = false;
        for child in children(node) {
            if !matches!(child.kind(), "from" | "identifier") {
                continue;
            }
            if child.kind() == "from" || ctx.text(child)? == "from" {
                has_from = true;
                break;
            }
        }
        if !has_from {
            has_from = children(node).iter().any(|c| c.kind() == "string");
            if !has_from {
                return Ok(());
            }
        }
    }

    let mut module_string: Option<Node> = None;
    for child in children(node) {
        if child.kind() == "string" {
            module_string = Some(child);
            break;
        }
        if child.kind() == "import_require_clause" {
            // `import x = require("./m")`: the string sits inside the clause, so
            // the direct-child scan above never sees it.
            module_string = children(child).into_iter().find(|s| s.kind() == "string");
            break;
        }
    }

    let mut resolved_path: Option<String> = None;
    let mut target_stem: Option<String> = None;
    let line = line_of(node);
    if let Some(ms) = module_string {
        let raw = strip_chars(ctx.text(ms)?, STRIP_IMPORT);
        if let Some(r) = ctx.res.get(&raw)? {
            let mut tgt_nid = r.tgt_nid;
            resolved_path = r.path;
            target_stem = r.target_stem;
            // `_resolve_js_import_path` returns the ATTEMPTED path when no local
            // file exists. A static ES import must treat that as unresolved
            // rather than minting a checkout-specific target id (#2457).
            if resolved_path.is_some() && !r.is_file {
                tgt_nid = ctx.mkid(&["ref", &raw])?;
                resolved_path = None;
                target_stem = None;
            }
            let context = if is_reexport { "re-export" } else { "import" };
            let edge = import_edge(
                ctx,
                &ctx.file_nid.clone(),
                &tgt_nid,
                "imports_from",
                context,
                line,
                resolved_path.as_deref(),
            );
            ctx.edges.push(edge);
        }
    }

    let Some(rpath) = resolved_path else {
        return Ok(());
    };
    let target_stem = target_stem.ok_or("missing_target_stem")?;

    if is_reexport {
        // export { foo, bar } from './module'
        for child in children(node) {
            if child.kind() != "export_clause" {
                continue;
            }
            for spec in children(child) {
                if spec.kind() != "export_specifier" {
                    continue;
                }
                let Some(name_node) = spec.child_by_field_name("name") else {
                    continue;
                };
                let sym = ctx.text(name_node)?.to_string();
                if sym == "default" {
                    continue; // default re-exports do not match a symbol id
                }
                let target = ctx.mkid(&[&target_stem, &sym])?;
                let edge = import_edge(
                    ctx,
                    &ctx.file_nid.clone(),
                    &target,
                    "re_exports",
                    "re-export",
                    line,
                    Some(&rpath),
                );
                ctx.edges.push(edge);
            }
        }
    } else {
        // import { Foo, type Bar } from './bar'
        for child in children(node) {
            if child.kind() != "import_clause" {
                continue;
            }
            for sub in children(child) {
                if sub.kind() != "named_imports" {
                    continue;
                }
                for spec in children(sub) {
                    if spec.kind() != "import_specifier" {
                        continue;
                    }
                    let Some(name_node) = spec.child_by_field_name("name") else {
                        continue;
                    };
                    let sym = ctx.text(name_node)?.to_string();
                    let target = ctx.mkid(&[&target_stem, &sym])?;
                    let edge = import_edge(
                        ctx,
                        &ctx.file_nid.clone(),
                        &target,
                        "imports",
                        "import",
                        line,
                        Some(&rpath),
                    );
                    ctx.edges.push(edge);
                }
            }
        }
    }
    Ok(())
}

/// `_find_require_call`.
fn find_require_call<'tree>(value: Option<Node<'tree>>) -> Option<Node<'tree>> {
    let value = value?;
    if value.kind() == "call_expression" {
        if let Some(fnode) = value.child_by_field_name("function") {
            if fnode.kind() == "identifier" {
                return Some(value);
            }
        }
    }
    if value.kind() == "member_expression" {
        return find_require_call(value.child_by_field_name("object"));
    }
    None
}

/// `_require_imports_js`. Returns whether any require import was found.
pub fn require_imports_js(ctx: &mut Ctx, node: Node, importer_nid: &str) -> R<bool> {
    if !matches!(node.kind(), "lexical_declaration" | "variable_declaration") {
        return Ok(false);
    }
    let mut found = false;
    for child in children(node) {
        if child.kind() != "variable_declarator" {
            continue;
        }
        let value = child.child_by_field_name("value");
        let Some(call) = find_require_call(value) else {
            continue;
        };
        let Some(fnode) = call.child_by_field_name("function") else {
            continue;
        };
        if ctx.text(fnode)? != "require" {
            continue;
        }
        let Some(args) = call.child_by_field_name("arguments") else {
            continue;
        };
        let mut raw: Option<String> = None;
        for arg in children(args) {
            if arg.kind() == "string" {
                raw = Some(strip_chars(ctx.text(arg)?, STRIP_IMPORT));
                break;
            }
        }
        let Some(raw) = raw.filter(|r| !r.is_empty()) else {
            continue;
        };
        let Some(r) = ctx.res.get(&raw)? else { continue };
        let tgt_nid = r.tgt_nid;
        let resolved_path = r.path;
        let target_stem = r.target_stem;
        let line = line_of(node);
        let edge = import_edge(
            ctx,
            importer_nid,
            &tgt_nid,
            "imports_from",
            "import",
            line,
            resolved_path.as_deref(),
        );
        ctx.edges.push(edge);
        found = true;

        // Symbol-level edges for destructured / accessor binders. Python's
        // `if name_node is object_pattern ... elif value is member_expression`,
        // kept as one chain so the two branches cannot both run.
        let mut sym_names: Vec<String> = Vec::new();
        let name_node = child.child_by_field_name("name");
        if name_node.map_or(false, |n| n.kind() == "object_pattern") {
            // `const { a, b: alias } = require('./m')` -- one edge per property key
            for prop in children(name_node.expect("checked above")) {
                if prop.kind() == "shorthand_property_identifier_pattern" {
                    sym_names.push(ctx.text(prop)?.to_string());
                } else if prop.kind() == "pair_pattern" {
                    if let Some(key) = prop.child_by_field_name("key") {
                        sym_names.push(ctx.text(key)?.to_string());
                    }
                }
            }
        } else if let Some(v) = value {
            // `const x = require('./m').y` -- the symbol is the property accessed
            if v.kind() == "member_expression" {
                if let Some(prop) = v.child_by_field_name("property") {
                    sym_names.push(ctx.text(prop)?.to_string());
                }
            }
        }

        if let Some(ts) = target_stem {
            for sym in sym_names {
                let target = ctx.mkid(&[&ts, &sym])?;
                let edge = import_edge(
                    ctx,
                    importer_nid,
                    &target,
                    "imports",
                    "import",
                    line,
                    None,
                );
                ctx.edges.push(edge);
            }
        }
    }
    Ok(found)
}

/// `_dynamic_import_js`. Returns whether the node WAS a dynamic import, in which
/// case the caller skips normal call handling.
pub fn dynamic_import_js(ctx: &mut Ctx, node: Node, caller_nid: &str) -> R<bool> {
    let mut func_node = node.child_by_field_name("function");
    if func_node.is_none() {
        let kids = children(node);
        match kids.first() {
            Some(first) if ctx.text(*first)? == "import" => func_node = Some(*first),
            _ => return Ok(false),
        }
    }
    let func_node = func_node.ok_or("dyn_import_no_func")?;
    if ctx.text(func_node)? != "import" {
        return Ok(false);
    }
    let Some(args) = node.child_by_field_name("arguments") else {
        return Ok(true); // an import() with no args -- nothing to emit
    };
    for arg in children(args) {
        let raw = if arg.kind() == "template_string" {
            // A template with a substitution cannot be resolved statically.
            if children(arg).iter().any(|c| c.kind() == "template_substitution") {
                break;
            }
            strip_chars(ctx.text(arg)?, STRIP_DYN_TEMPLATE)
        } else if arg.kind() == "string" {
            strip_chars(ctx.text(arg)?, STRIP_DYN_STRING)
        } else {
            continue;
        };
        if raw.is_empty() {
            break;
        }
        let Some(r) = ctx.res.get(&raw)? else { break };
        let tgt_nid = r.tgt_nid;
        let resolved_path = r.path;
        let pair = (caller_nid.to_string(), tgt_nid.clone());
        if ctx.seen_dyn_import_pairs.insert(pair) {
            let line = line_of(node);
            let mut fields = vec![
                ("context", Val::Static("import")),
                // Marked deferred so find_import_cycles does not read it as a
                // static import and report a phantom file cycle (#1241).
                ("deferred", Val::B(true)),
                ("confidence", Val::Static("EXTRACTED")),
                ("source_file", Val::S(ctx.str_path.to_string())),
                ("source_location", Val::S(format!("L{line}"))),
                ("weight", Val::F(1.0)),
            ];
            if let Some(rp) = &resolved_path {
                fields.push(("target_file", Val::S(rp.clone())));
            }
            ctx.edges.push(EdgeRow {
                source: caller_nid.to_string(),
                target: tgt_nid,
                relation: "imports_from",
                fields,
            });
        }
        break;
    }
    Ok(true)
}

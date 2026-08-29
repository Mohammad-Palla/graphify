//! `_import_python`, and the Python-callback boundary its relative form needs.
//!
//! An absolute `import pkg.mod` / `from pkg.mod import x` needs no filesystem at
//! all: the target id is `_make_id(module_name)`, pure string work. A RELATIVE
//! `from ..pkg import x` does -- it walks `Path.parent` `dots-1` times, joins the
//! dotted remainder, and probes disk through `_probe_python_module_candidate`
//! (`is_dir()` + `__init__.py`, then `is_file()`, then a `.py` suffix), falling
//! back to a speculative path when nothing exists, and finally asks `is_file()`
//! again to decide whether to stamp `target_file`.
//!
//! All of that stays in Python behind [`Resolver`]. Reproducing pathlib's
//! normalization and four filesystem probes in Rust would be a large surface for
//! an I/O-bound job, and any disagreement would silently retarget an
//! `imports_from` edge -- the failure mode this design exists to prevent.

use std::cell::RefCell;
use std::collections::HashMap;

use pyo3::prelude::*;
use tree_sitter::Node;

use crate::js::ast::{children, line_of};
use crate::js::emit::{EdgeRow, Val};
use super::{Ctx, R};

/// What Python said about one RELATIVE import specifier: the target path it
/// resolved to (always a path -- `_import_python` never leaves `target_path`
/// unset on the relative branch) and whether that path exists.
#[derive(Clone)]
pub struct Relative {
    pub target_path: String,
    pub is_file: bool,
}

/// A lazily-consulted view of Python's relative-import resolution, memoized per
/// file.
///
/// A missing callback, or one that raises, is a *deferral* -- never an assumption
/// about where the import points, because guessing there would silently retarget
/// or drop an `imports_from` edge.
pub struct Resolver<'py> {
    callback: Option<Bound<'py, PyAny>>,
    cache: RefCell<HashMap<String, Relative>>,
}

impl<'py> Resolver<'py> {
    pub fn new(callback: Option<Bound<'py, PyAny>>) -> Self {
        Resolver {
            callback,
            cache: RefCell::new(HashMap::new()),
        }
    }

    pub fn resolve_relative(&self, raw: &str) -> R<Relative> {
        if let Some(v) = self.cache.borrow().get(raw) {
            return Ok(v.clone());
        }
        let cb = self.callback.as_ref().ok_or("no_py_import_resolver")?;
        let ret = cb.call1((raw,)).map_err(|_| "py_import_resolver_raised")?;
        let (target_path, is_file): (String, bool) = ret
            .extract()
            .map_err(|_| "py_import_resolver_bad_shape")?;
        let out = Relative {
            target_path,
            is_file,
        };
        self.cache.borrow_mut().insert(raw.to_string(), out.clone());
        Ok(out)
    }
}

/// Python's `str.strip()` with no argument, for ASCII.
///
/// NOT `str::trim`. Python strips every character for which `str.isspace()` is
/// true, and in the ASCII range that includes the four file/group/record/unit
/// separators `\x1c`-`\x1f`, which Rust's `char::is_whitespace` (Unicode
/// White_Space) does not. A non-ASCII whitespace character is left in place here
/// rather than guessed at -- it then reaches `make_id_ascii`, which defers the
/// file, so the disagreement costs a deferral and never a different id.
fn py_strip(s: &str) -> &str {
    let is_py_space = |c: char| matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c')
        || matches!(c, '\x1c' | '\x1d' | '\x1e' | '\x1f');
    s.trim_matches(is_py_space)
}

/// `_import_python`. Emits the `imports` / `imports_from` edges for one import
/// node; returns nothing to the walk (Python's handler returns `None`, so the
/// `imported_modules` module-node branch above it is dead for Python -- only
/// Swift's handler returns pairs).
pub fn import_python(ctx: &mut Ctx, node: Node) -> R<()> {
    let line = line_of(node);
    match node.kind() {
        "import_statement" => {
            for child in children(node) {
                if !matches!(child.kind(), "dotted_name" | "aliased_import") {
                    continue;
                }
                let raw = ctx.text(child)?;
                // `raw.partition(" as ")`: everything before the first " as ",
                // and everything after it (empty when absent).
                let (raw_module, raw_alias) = match raw.find(" as ") {
                    Some(i) => (&raw[..i], &raw[i + 4..]),
                    None => (raw, ""),
                };
                let module_name = py_strip(raw_module).trim_start_matches('.');
                let tgt_nid = ctx.mkid(&[module_name])?;
                let mut fields = vec![
                    ("context", Val::Static("import")),
                    ("confidence", Val::Static("EXTRACTED")),
                    ("source_file", Val::S(ctx.str_path.to_string())),
                    ("source_location", Val::S(format!("L{line}"))),
                    ("weight", Val::F(1.0)),
                ];
                if !raw_alias.is_empty() {
                    // `import pkg.mod as alias` binds `alias`, not `mod`'s stem
                    // -- stashed so the cross-file member-call resolver can match
                    // `alias.func()` against this edge instead of dropping it
                    // (#2082).
                    fields.push(("local_alias", Val::S(py_strip(raw_alias).to_string())));
                }
                let source = ctx.file_nid.clone();
                ctx.edges.push(EdgeRow {
                    source,
                    target: tgt_nid,
                    relation: "imports",
                    fields,
                });
            }
        }
        "import_from_statement" => {
            let Some(module_node) = node.child_by_field_name("module_name") else {
                return Ok(());
            };
            let raw = ctx.text(module_node)?;
            let mut target_path: Option<Relative> = None;
            let tgt_nid = if raw.starts_with('.') {
                let rel = ctx.res.resolve_relative(raw)?;
                let nid = ctx.mkid(&[&rel.target_path])?;
                target_path = Some(rel);
                nid
            } else {
                ctx.mkid(&[raw])?
            };
            let mut fields = vec![
                ("context", Val::Static("import")),
                ("confidence", Val::Static("EXTRACTED")),
                ("source_file", Val::S(ctx.str_path.to_string())),
                ("source_location", Val::S(format!("L{line}"))),
                ("weight", Val::F(1.0)),
            ];
            // Existence-gated, mirroring `_import_js` (#1814): a speculative
            // import of a nonexistent sibling must stay dangling. The stamp is
            // transient and popped before graph.json ships (#2213).
            if let Some(rel) = target_path {
                if rel.is_file {
                    fields.push(("target_file", Val::S(rel.target_path)));
                }
            }
            let source = ctx.file_nid.clone();
            ctx.edges.push(EdgeRow {
                source,
                target: tgt_nid,
                relation: "imports_from",
                fields,
            });
        }
        _ => {}
    }
    Ok(())
}

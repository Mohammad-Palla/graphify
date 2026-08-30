//! `_import_c`: `#include`.

use std::cell::RefCell;
use std::collections::HashMap;

use pyo3::prelude::*;
use tree_sitter::Node;

use crate::engine::{Ctx, PathResolver, R};
use crate::js::ast::children;
use crate::js::emit::{EdgeRow, Val};

/// `_resolve_c_include_path`, answered by Python.
///
/// Memoized per file: a header included from several places costs one call.
pub struct Resolver<'py> {
    callback: Option<Bound<'py, PyAny>>,
    cache: RefCell<HashMap<String, Option<String>>>,
}

impl<'py> Resolver<'py> {
    pub fn new(callback: Option<Bound<'py, PyAny>>) -> Self {
        Resolver {
            callback,
            cache: RefCell::new(HashMap::new()),
        }
    }
}

impl<'py> PathResolver for Resolver<'py> {
    fn resolve(&self, raw: &str) -> R<Option<String>> {
        if let Some(v) = self.cache.borrow().get(raw) {
            return Ok(v.clone());
        }
        let cb = self.callback.as_ref().ok_or("no_c_include_resolver")?;
        let ret = cb.call1((raw,)).map_err(|_| "c_include_resolver_raised")?;
        let out: Option<String> = ret.extract().map_err(|_| "c_include_resolver_bad_shape")?;
        self.cache.borrow_mut().insert(raw.to_string(), out.clone());
        Ok(out)
    }
}

/// `str.strip('"<> ')`: every leading and trailing character in that set.
fn strip_delims(s: &str) -> &str {
    s.trim_matches(|c| c == '"' || c == '<' || c == '>' || c == ' ')
}

pub fn import_c(ctx: &mut Ctx, node: Node) -> R<()> {
    for child in children(node) {
        if !matches!(
            child.kind(),
            "string_literal" | "system_lib_string" | "string"
        ) {
            continue;
        }
        let raw = strip_delims(ctx.text(child)?).to_string();
        let line = node.start_position().row + 1;
        // A quoted include is resolved to a real file so the edge lands on the
        // node `_extract_generic` mints for THAT file; an angle-bracket include
        // is a system header and is never probed.
        if child.kind() != "system_lib_string" && !raw.is_empty() {
            let resolver = ctx.path_resolver.ok_or("no_c_include_resolver")?;
            if let Some(resolved) = resolver.resolve(&raw)? {
                let target = ctx.mkid(&[&resolved])?;
                // Built as a dict literal, so `context` is third and
                // `target_file` is LAST -- the resolved-target stamp without
                // which an include whose header lives outside this batch keeps
                // an absolute-path id no later pass learns to relativize (#2243).
                ctx.edges.push(EdgeRow {
                    source: ctx.file_nid.clone(),
                    target,
                    relation: "imports",
                    fields: vec![
                        ("context", Val::Static("import")),
                        ("confidence", Val::Static("EXTRACTED")),
                        ("source_file", Val::S(ctx.str_path.to_string())),
                        ("source_location", Val::S(format!("L{line}"))),
                        ("weight", Val::F(1.0)),
                        ("target_file", Val::S(resolved)),
                    ],
                });
                return Ok(());
            }
        }
        // `raw.split("/")[-1].split(".")[0]` -- the bare header stem.
        let module_name = raw.rsplit('/').next().unwrap_or("");
        let module_name = module_name.split('.').next().unwrap_or("");
        if !module_name.is_empty() {
            let target = ctx.mkid(&[module_name])?;
            ctx.add_import_edge(&target, line);
        }
        return Ok(());
    }
    Ok(())
}

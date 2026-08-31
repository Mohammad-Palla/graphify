//! Lua, on the shared engine.
//!
//! The cheapest language ported so far, and worth stating why so the next one is
//! estimated the same way: `engine.py` contains **zero** `_is_lua` guards and
//! zero `tree_sitter_lua` guards. Everything Lua-specific lives in
//! `_LUA_CONFIG`'s data plus one `import_handler`, so this module adds no engine
//! hook positions at all -- the first language since Java to cost nothing
//! structurally.
//!
//! `class_types` is EMPTY: Lua has no class syntax the extractor recognises, so
//! the class branch of `walk` never fires and every table-based "class" idiom is
//! simply recursed through. That is Python's behaviour, not a gap being papered
//! over here.

use pyo3::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use tree_sitter::Node;

use crate::engine::{Ctx, EngineConfig, LangHooks, PathResolver, R};
use crate::js::emit::{EdgeRow, Val};
use crate::Outcome;

/// `_resolve_lua_import_target`, answered by Python.
///
/// It walks up to six directories looking for `{mod}.lua`, `{mod}.luau`,
/// `{mod}/init.lua` or `{mod}/init.luau` and falls back to `_make_id` of the
/// dotted name. That is four `is_file()` probes per level against an
/// attacker-controllable tree -- exactly the class of question the walker asks
/// Python rather than reimplementing, for the same reason `c/imports.rs` does.
///
/// Memoized per file: a module required from several places costs one call.
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
        let cb = self.callback.as_ref().ok_or("no_lua_import_resolver")?;
        let ret = cb.call1((raw,)).map_err(|_| "lua_import_resolver_raised")?;
        let out: Option<String> = ret
            .extract()
            .map_err(|_| "lua_import_resolver_bad_shape")?;
        self.cache.borrow_mut().insert(raw.to_string(), out.clone());
        Ok(out)
    }
}

/// Python's `re` `\s` in Unicode mode.
///
/// NOT `char::is_whitespace()` alone. CPython's `Py_UNICODE_ISSPACE` is the
/// Unicode White_Space property **plus** `\x1c`-`\x1f` (the file / group /
/// record / unit separators), which are not White_Space and which Rust's
/// `is_whitespace()` therefore rejects. The difference only shows up on a file
/// containing a separator control character next to a `require`, which is why it
/// would never have been caught by a corpus -- and why it is spelled out rather
/// than assumed.
fn py_re_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `re.search(r"require\s*[\('\"]\s*['\"]?([^'\")\s]+)", text)`, by hand.
///
/// Hand-written rather than pulling in the `regex` crate: the kernel has no
/// regex dependency and this is the only pattern any walker needs, so adding one
/// would be a large dependency for a fifteen-line scan.
///
/// The two subtleties, both verified against the Python rather than reasoned
/// about:
/// * `[\('"]` is REQUIRED and matches exactly one character. A bare
///   `require x` does not match at all, so no edge is emitted.
/// * `['"]?` is greedy but backtracking cannot change the answer: if consuming
///   the optional quote leaves the `+` with nothing, the un-consumed branch
///   starts on that same quote, which `[^'")\s]` also rejects. Both fail, so a
///   single forward pass is equivalent.
///
/// `re.search` scans left to right, so the FIRST `require` that matches wins and
/// a later one is never reached.
fn find_require(text: &str) -> Option<&str> {
    let b: Vec<(usize, char)> = text.char_indices().collect();
    let n = b.len();
    let mut i = 0usize;
    while i < n {
        if !text[b[i].0..].starts_with("require") {
            i += 1;
            continue;
        }
        // Position just past the literal, in char units.
        let mut j = i + "require".chars().count();
        while j < n && py_re_space(b[j].1) {
            j += 1;
        }
        // `[\('"]` -- required, exactly one.
        if j >= n || !matches!(b[j].1, '(' | '\'' | '"') {
            i += 1;
            continue;
        }
        j += 1;
        while j < n && py_re_space(b[j].1) {
            j += 1;
        }
        // `['"]?`
        if j < n && matches!(b[j].1, '\'' | '"') {
            j += 1;
        }
        // `([^'")\s]+)`
        let start = j;
        while j < n && !matches!(b[j].1, '\'' | '"' | ')') && !py_re_space(b[j].1) {
            j += 1;
        }
        if j > start {
            let lo = b[start].0;
            let hi = if j < n { b[j].0 } else { text.len() };
            return Some(&text[lo..hi]);
        }
        i += 1;
    }
    None
}

struct Lua;

impl LangHooks for Lua {
    /// `_import_lua`.
    ///
    /// The edge literal is spelled out in the Python's key order --
    /// source, target, relation, context, confidence, confidence_score,
    /// source_file, source_location, weight -- because CPython dicts preserve
    /// insertion order and the export does not sort keys, so key order is part
    /// of "byte-identical". The per-file parity harness canonicalises with
    /// `sort_keys=True` and would NOT catch a reordering here; the end-to-end
    /// graph comparison would.
    fn import_handler<'tree>(&self, ctx: &mut Ctx<'_, 'tree>, node: Node<'tree>) -> R<()> {
        let text = ctx.text(node)?;
        let raw_module = match find_require(text) {
            Some(m) => m.to_string(),
            None => return Ok(()),
        };
        // `if raw_module:` -- the capture is `+` so it cannot be empty, but the
        // Python tests it and so does this, rather than relying on that.
        if raw_module.is_empty() {
            return Ok(());
        }
        let resolver = ctx.path_resolver.ok_or("no_lua_import_resolver")?;
        let tgt_nid = match resolver.resolve(&raw_module)? {
            // `if tgt_nid:` -- the Python drops the edge on an empty string.
            Some(s) if !s.is_empty() => s,
            _ => return Ok(()),
        };
        let file_nid = ctx.file_nid.clone();
        let line = node.start_position().row + 1;
        let src_file = ctx.str_path.to_string();
        ctx.edges.push(EdgeRow {
            source: file_nid,
            target: tgt_nid,
            relation: "imports",
            // `context` comes immediately after `relation` here, NOT last as in
            // `add_edge_ctx`. `_import_lua` builds its dict literal in this
            // order and CPython preserves it, so this is not cosmetic.
            fields: vec![
                ("context", Val::Static("import")),
                ("confidence", Val::Static("EXTRACTED")),
                ("confidence_score", Val::F(1.0)),
                ("source_file", Val::S(src_file)),
                // `str(node.start_point[0] + 1)` -- a bare number, not `L{n}`.
                ("source_location", Val::S(line.to_string())),
                ("weight", Val::F(1.0)),
            ],
        });
        Ok(())
    }
}

static HOOKS: Lua = Lua;

pub static CONFIG: EngineConfig = EngineConfig {
    language: "lua",
    grammar: || tree_sitter_lua::LANGUAGE.into(),
    class_types: &[],
    function_types: &["function_declaration"],
    import_types: &["variable_declaration"],
    call_types: &["function_call"],
    function_boundary_types: &["function_declaration"],
    static_prop_types: &[],
    helper_fn_names: &[],
    container_bind_methods: &[],
    event_listener_properties: &[],
    name_field: "name",
    name_fallback_child_types: &["identifier", "method_index_expression"],
    body_field: "body",
    body_fallback_child_types: &["block"],
    call_function_field: "name",
    call_accessor_node_types: &["method_index_expression"],
    call_accessor_field: "name",
    call_accessor_object_field: "",
    function_label_parens: true,
    resolve_function_name: None,
    sanitize_symbol_name: None,
    type_table_key: None,
    hooks: &HOOKS,
};

pub fn walk_lua<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
    res: &crate::Resolvers<'py>,
) -> PyResult<Outcome<'py>> {
    crate::engine::run(
        py,
        &CONFIG,
        path,
        source,
        crate::engine::no_receiver_types,
        Some(&res.lua),
    )
}

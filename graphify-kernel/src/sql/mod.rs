//! SQL: a BESPOKE walker, and the only one here that WALKS AN ERRORED TREE.
//!
//! # Why the `has_error` defer is lifted, and why that is safe
//!
//! Every other walker in this kernel bails on `root.has_error()`. SQL cannot:
//! tree-sitter-sql leaves an ERROR node in **81.5%** of real files (PL/pgSQL
//! bodies with `OUT`/`INOUT` params, tagged dollar quotes, `PERFORM`, `:=`;
//! Firebird `COMPUTED BY`; assorted dialect syntax), and `extract_sql` is BUILT
//! around that -- it has an `ERROR` branch, a `fb_proc_or_trigger` branch, and
//! two whole-file regex fallbacks gated on `root.has_error`. Deferring on
//! `has_error` here would defer on four files in five and route almost nothing.
//!
//! Walking an errored tree is only safe if the kernel's tree is EXACTLY Python's
//! tree, error recovery included. That was measured rather than assumed:
//!
//! > PyPI `tree-sitter-sql` 0.3.11 is ABI 15; the crate `tree-sitter-sequel`
//! > 0.3.11 is ABI 14. All 729 node-kind names and all 54 field names are
//! > identical by id. Parsing all 3,442 files of postgres + sqlfluff with both
//! > and comparing a preorder digest of EVERY node -- kind, byte range, MISSING
//! > and ERROR flags -- gives **3,442 identical trees out of 3,442, including
//! > all 2,797 files that contain ERROR nodes.**
//!
//! That is the evidence the lifted defer rests on. It is a per-language
//! decision, not a global relaxation: no other walker gets it without the same
//! measurement.
//!
//! # Shape notes
//!
//! * The FILE node's `source_location` is `None`, not `"L1"`. Every other walker
//!   here writes a line; this one writes a JSON null.
//! * `_add_node` emits the node AND its `contains` edge together, so a
//!   deduplicated node contributes neither.
//! * `_ref_stub` deliberately emits NO `contains` edge: a sourced/contained stub
//!   would get the referencing file's path baked into its id by disambiguation,
//!   blocking the corpus-level rewire (#2324).

use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use regex::Regex;
use tree_sitter::{Node, Parser};

use crate::engine::R;
use crate::ids::{file_stem, make_id_ascii};
use crate::js::ast::{children, text_checked};
use crate::js::emit::{self, EdgeRow, NodeRow, Val};
use crate::Outcome;

/// The nine recovery patterns, compiled once per process.
///
/// Each is the Python literal, transliterated. `(?i)` is `re.IGNORECASE`; `\w`
/// is Unicode in both engines; the `regex` crate's `find_iter` is leftmost-first
/// and non-overlapping, matching `re.finditer`.
struct Patterns {
    /// `CREATE [OR REPLACE] FUNCTION|PROCEDURE [IF NOT EXISTS] <qualified name>`,
    /// where each part is bare or double-quoted -- a bare `[\w$.]+` stopped dead
    /// at the leading quote and silently dropped every quoted PL/pgSQL routine
    /// (#2180).
    create_routine: Regex,
    /// Firebird `CREATE [OR REPLACE|ALTER] PROCEDURE|TRIGGER|FUNCTION <name>`,
    /// ANCHORED (`re.match`, hence the leading `^`).
    fb_header: Regex,
    fb_for: Regex,
    /// A `WITH <name> [(cols)] AS (` binding, which is statement-local and must
    /// not become a table stub.
    cte: Regex,
    from_join_into: Regex,
    update_tbl: Regex,
    references: Regex,
    create_table_open: Regex,
    /// The end of a `CREATE TABLE` block, for the whole-file REFERENCES scan.
    block_end: Regex,
}

impl Patterns {
    fn new() -> Self {
        // `expect` is safe: these are compile-time-constant literals, and a bad
        // one is a build-breaking programming error, not a runtime condition.
        Self {
            create_routine: Regex::new(
                r"(?i)CREATE\s+(?:OR\s+REPLACE\s+)?(?:FUNCTION|PROCEDURE)\s+(?:IF\s+NOT\s+EXISTS\s+)?((?:"[^"\n]+"|[\w$]+)(?:\s*\.\s*(?:"[^"\n]+"|[\w$]+))*)",
            )
            .expect("create_routine"),
            fb_header: Regex::new(
                r"(?i)^CREATE\s+(?:OR\s+(?:REPLACE|ALTER)\s+)?(PROCEDURE|TRIGGER|FUNCTION)\s+([\w$]+)",
            )
            .expect("fb_header"),
            fb_for: Regex::new(r"(?i)\bFOR\s+([\w$]+)").expect("fb_for"),
            cte: Regex::new(
                r"(?i)(?:\bWITH\s+(?:RECURSIVE\s+)?|,\s*)([\w$]+)\s*(?:\([^()]*\))?\s+AS\s*\(",
            )
            .expect("cte"),
            from_join_into: Regex::new(r"(?i)\b(?:FROM|JOIN|INTO)\s+([\w$]+)")
                .expect("from_join_into"),
            update_tbl: Regex::new(r"(?i)\bUPDATE\s+([\w$]+)").expect("update_tbl"),
            references: Regex::new(r"(?i)\bREFERENCES\s+([\w$]+)").expect("references"),
            create_table_open: Regex::new(r"(?i)CREATE\s+TABLE\s+([\w$]+)\s*\(")
                .expect("create_table_open"),
            block_end: Regex::new(r"(?i)(?:^|\n)(?:CREATE|SET\s+TERM|ALTER)\s")
                .expect("block_end"),
        }
    }
}

fn patterns() -> &'static Patterns {
    use std::sync::OnceLock;
    static P: OnceLock<Patterns> = OnceLock::new();
    P.get_or_init(Patterns::new)
}

/// `_norm_ident`: split on `.`, strip ONE pair of delimiters per part, lowercase.
///
/// Postgres/ANSI `"x"`, MySQL `` `x` `` and T-SQL `[x]` all normalise the same
/// way, so `"public"."users"`, `public.users` and `PUBLIC.USERS` are one key.
/// Used ONLY for `table_nids` keys and lookups -- node ids and labels keep the
/// original text.
fn norm_ident(name: &str) -> String {
    name.split('.')
        .map(|part| {
            let p = part.trim();
            let b = p.as_bytes();
            let stripped = if b.len() >= 2 {
                let (f, l) = (b[0], b[b.len() - 1]);
                if (f == l && (f == b'"' || f == b'`')) || (f == b'[' && l == b']') {
                    &p[1..p.len() - 1]
                } else {
                    p
                }
            } else {
                p
            };
            stripped.to_lowercase()
        })
        .collect::<Vec<String>>()
        .join(".")
}

/// The 1-based line of a byte offset inside `text`, added to a base line.
fn line_offset(text: &str, upto: usize) -> usize {
    text[..upto].matches('\n').count()
}

struct Ctx<'a> {
    src: &'a [u8],
    src_text: &'a str,
    str_path: &'a str,
    stem: String,
    file_nid: String,
    nodes: Vec<NodeRow>,
    edges: Vec<EdgeRow>,
    seen_ids: HashSet<String>,
    /// `norm_ident(name) -> nid`, insertion-ordered because the bare-name alias
    /// pass iterates it and the Python's dict is ordered.
    table_nids: Vec<(String, String)>,
    table_index: HashMap<String, usize>,
}

impl<'a> Ctx<'a> {
    fn text(&self, node: Node) -> R<&'a str> {
        text_checked(node, self.src).ok_or("invalid_utf8_text")
    }

    fn mkid(&self, parts: &[&str]) -> R<String> {
        make_id_ascii(parts).ok_or("non_ascii_id")
    }

    fn table_get(&self, key: &str) -> Option<&str> {
        self.table_index
            .get(key)
            .map(|i| self.table_nids[*i].1.as_str())
    }

    fn table_set(&mut self, key: String, nid: String) {
        match self.table_index.get(&key) {
            Some(i) => self.table_nids[*i].1 = nid,
            None => {
                self.table_index.insert(key.clone(), self.table_nids.len());
                self.table_nids.push((key, nid));
            }
        }
    }

    /// The node AND its `contains` edge, together -- a deduplicated node adds
    /// neither.
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
        let f = self.file_nid.clone();
        self.edges.push(EdgeRow {
            source: f,
            target: nid.to_string(),
            relation: "contains",
            fields: vec![
                ("confidence", Val::Static("EXTRACTED")),
                ("source_file", Val::S(self.str_path.to_string())),
                ("source_location", Val::S(format!("L{line}"))),
                ("weight", Val::F(1.0)),
            ],
        });
    }

    fn add_edge(&mut self, src: &str, tgt: &str, relation: &'static str, line: usize) {
        self.edges.push(EdgeRow {
            source: src.to_string(),
            target: tgt.to_string(),
            relation,
            fields: vec![
                ("confidence", Val::Static("EXTRACTED")),
                ("source_file", Val::S(self.str_path.to_string())),
                ("source_location", Val::S(format!("L{line}"))),
                ("weight", Val::F(1.0)),
            ],
        });
    }

    /// A SOURCELESS bare-name stub for a table referenced but not defined here.
    ///
    /// SQL references are NAME-based, so a table created in another migration can
    /// only resolve at corpus level. NO `contains` edge, deliberately -- see the
    /// module doc.
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

    /// `table_nids.get(norm) or _ref_stub(name)`.
    fn resolve_or_stub(&mut self, name: &str) -> R<String> {
        match self.table_get(&norm_ident(name)) {
            Some(nid) => Ok(nid.to_string()),
            None => self.ref_stub(name),
        }
    }
}

/// The first `object_reference` child's text.
fn obj_name<'a>(ctx: &Ctx<'a>, n: Node) -> R<Option<&'a str>> {
    for c in children(n) {
        if c.kind() == "object_reference" {
            return Ok(Some(ctx.text(c)?));
        }
    }
    Ok(None)
}

/// The `object_reference` following a `keyword_references` sibling.
fn referenced_name<'a>(ctx: &Ctx<'a>, parent: Node) -> R<Option<&'a str>> {
    let mut found_ref = false;
    for cc in children(parent) {
        if cc.kind() == "keyword_references" {
            found_ref = true;
        } else if found_ref && cc.kind() == "object_reference" {
            return Ok(Some(ctx.text(cc)?));
        }
    }
    Ok(None)
}

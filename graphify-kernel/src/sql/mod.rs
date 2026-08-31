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
/// One name part: bare, or delimited by `""`, ``` `` ```, `''` or `[]`. A
/// delimited part may hold anything but its own delimiter -- a Databricks role
/// is routinely `finance-team`, whose hyphen no `\w` class will match.
const NAME_PART: &str = r#"(?:"[^"\n]+"|`[^`\n]+`|'[^'\n]+'|\[[^\]\n]+\]|[\w$]+)"#;

/// Object types that ARE graph entities. Postgres defaults to TABLE when the
/// keyword is omitted, so the keyword is optional at the call site.
const GRANTABLE_OBJECT_TYPES: &[&str] = &[
    "materialized view", "foreign table", "procedure", "function",
    "sequence", "routine", "table", "view",
];

/// Object types that are NOT graph entities. A grant on one of these is real,
/// correctly parsed SQL -- there is simply no node to hang it on, so nothing is
/// emitted rather than a stub being invented for a database or a schema.
const UNGRAPHED_OBJECT_TYPES: &[&str] = &[
    "all tables in schema", "all sequences in schema", "all functions in schema",
    "all procedures in schema", "all routines in schema",
    "foreign data wrapper", "foreign server", "large object", "tablespace",
    "database", "language", "parameter", "schema", "domain", "type",
    "property graph",
    "external location", "storage credential", "metastore", "connection",
    "catalog", "share", "provider", "recipient",
];

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
    /// A GRANT/REVOKE statement: line-anchored, running to the first `;`.
    grant_stmt: Regex,
    /// `\bON\b`, `\bTO\b`, `\bFROM\b` -- the clause separators.
    kw_on: Regex,
    kw_to: Regex,
    kw_from: Regex,
    grant_tail: Regex,
    revoke_option_for: Regex,
    role_qualifier: Regex,
    tsql_securable_class: Regex,
    object_name: Regex,
    role_name: Regex,
    /// A routine's trailing argument list, dropped so the node keys on the name.
    arg_list: Regex,
    /// `CREATE POLICY <name> ON <table> ...;`
    policy_stmt: Regex,
    /// The clauses that end a policy's role/command header.
    policy_body: Regex,
    policy_for: Regex,
}

impl Patterns {
    fn new() -> Self {
        // `expect` is safe: these are compile-time-constant literals, and a bad
        // one is a build-breaking programming error, not a runtime condition.
        Self {
            // `r#""#` because the pattern itself contains double quotes: a SQL
            // name part may be delimited (`"public"."fn"`), and a bare
            // `[\w$.]+` stopped dead at the leading quote.
            create_routine: Regex::new(
                r#"(?i)CREATE\s+(?:OR\s+REPLACE\s+)?(?:FUNCTION|PROCEDURE)\s+(?:IF\s+NOT\s+EXISTS\s+)?((?:"[^"\n]+"|[\w$]+)(?:\s*\.\s*(?:"[^"\n]+"|[\w$]+))*)"#,
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
            // `(?ims)`: line-anchored `^`, case-insensitive, and `.` spans
            // newlines because a grant on a routine routinely wraps.
            grant_stmt: Regex::new(r"(?ims)^[ \t]*(GRANT|REVOKE)\b(.*?);")
                .expect("grant_stmt"),
            kw_on: Regex::new(r"(?i)\bON\b").expect("kw_on"),
            kw_to: Regex::new(r"(?i)\bTO\b").expect("kw_to"),
            kw_from: Regex::new(r"(?i)\bFROM\b").expect("kw_from"),
            // `format!`, not `concat!`: NAME_PART is a const, and `concat!`
            // takes only literals.
            grant_tail: Regex::new(&format!(
                r"(?i)\s+(?:WITH\s+(?:GRANT|ADMIN|INHERIT|SET)\s+OPTION|GRANTED\s+BY\s+{np}|AS\s+{np}|CASCADE|RESTRICT)\s*$",
                np = NAME_PART))
                .expect("grant_tail"),
            revoke_option_for: Regex::new(r"(?i)^\s*(?:GRANT|ADMIN)\s+OPTION\s+FOR\b")
                .expect("revoke_option_for"),
            role_qualifier: Regex::new(r"(?i)^(?:GROUP|ROLE|USER)\s+").expect("role_qualifier"),
            tsql_securable_class: Regex::new(r"(?i)^([A-Za-z_]+)\s*::\s*")
                .expect("tsql_securable_class"),
            object_name: Regex::new(&format!(
                r"^{np}(?:\s*\.\s*{np})*$", np = NAME_PART))
                .expect("object_name"),
            role_name: Regex::new(&format!(
                r"^({np})(?:\s*@\s*{np})?$", np = NAME_PART))
                .expect("role_name"),
            arg_list: Regex::new(r"(?s)\s*\(.*\)\s*$").expect("arg_list"),
            policy_stmt: Regex::new(&format!(
                r"(?ims)^[ \t]*CREATE\s+POLICY\s+(?P<name>{np})\s+ON\s+(?P<table>{np}(?:\s*\.\s*{np})*)(?P<rest>.*?);",
                np = NAME_PART))
                .expect("policy_stmt"),
            policy_body: Regex::new(r"(?i)\b(?:USING|WITH\s+CHECK)\b").expect("policy_body"),
            policy_for: Regex::new(r"(?i)\bFOR\s+(ALL|SELECT|INSERT|UPDATE|DELETE)\b")
                .expect("policy_for"),
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

/// `_split_top_level`: split on commas that are NOT inside parentheses.
///
/// A routine's argument list is comma-separated too, so a naive split tears
/// `FUNCTION f(uuid, text), g(int)` into four objects instead of two.
fn split_top_level(text: &str) -> Vec<String> {
    let mut parts: Vec<String> = Vec::new();
    let mut depth: i32 = 0;
    let mut current = String::new();
    for ch in text.chars() {
        match ch {
            '(' => depth += 1,
            ')' => depth = (depth - 1).max(0),
            _ => {}
        }
        if ch == ',' && depth == 0 {
            parts.push(std::mem::take(&mut current));
        } else {
            current.push(ch);
        }
    }
    parts.push(current);
    parts
        .into_iter()
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect()
}

/// `_split_keyword`: split at the first TOP-LEVEL whole-word occurrence.
///
/// Top-level because a routine signature can contain the word inside its
/// parentheses; first because the trailing clauses (`WITH GRANT OPTION`) come
/// after the role list, never before it.
fn split_keyword<'t>(text: &'t str, kw: &Regex) -> Option<(&'t str, &'t str)> {
    for m in kw.find_iter(text) {
        let before = &text[..m.start()];
        let depth = before.matches('(').count() as i64 - before.matches(')').count() as i64;
        if depth <= 0 {
            return Some((before, &text[m.end()..]));
        }
    }
    None
}

/// Python's `" ".join(s.split())`.
fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<&str>>().join(" ")
}

/// `_parse_grant`: split a GRANT/REVOKE body into (privileges, objects, roles).
///
/// `None` means "well-formed SQL this graph has no node for", and the caller
/// then emits NOTHING. That is the point: the failure being replaced is a
/// guessed edge shipped as EXTRACTED at confidence 1.0, and a grant on a
/// DATABASE has no object node to point at.
fn parse_grant(p: &Patterns, verb: &str, body: &str) -> Option<(String, Vec<String>, Vec<String>)> {
    let body = p.revoke_option_for.replace(body, "");
    // No ON => role membership (`GRANT admin TO alice`); both sides are roles,
    // so there is no object to carry a privilege edge.
    let (privileges_part, rest) = split_keyword(&body, &p.kw_on)?;

    // Postgres spells it `REVOKE ... FROM role`; T-SQL spells it
    // `REVOKE ... TO role`. Try the dialect-native keyword, then the other --
    // 25 corpus statements are T-SQL and were dropped by FROM alone.
    let primary = if verb == "GRANT" { &p.kw_to } else { &p.kw_from };
    let (objects_part, roles_part) = match split_keyword(rest, primary) {
        Some(v) => v,
        None if verb == "REVOKE" => split_keyword(rest, &p.kw_to)?,
        None => return None,
    };

    let mut objects_part = objects_part.trim().to_string();
    if let Some(cm) = p.tsql_securable_class.captures(&objects_part) {
        // T-SQL scopes by class: `OBJECT::t`, `TYPE::t`, `SCHEMA::s`. Only
        // OBJECT names something this graph holds.
        if cm[1].to_lowercase() != "object" {
            return None;
        }
        let end = cm.get(0).unwrap().end();
        objects_part = objects_part[end..].trim().to_string();
    }
    let lowered = objects_part.to_lowercase();
    if UNGRAPHED_OBJECT_TYPES.iter().any(|t| lowered.starts_with(t)) {
        return None;
    }
    for object_type in GRANTABLE_OBJECT_TYPES {
        if lowered.starts_with(&format!("{object_type} "))
            || lowered.starts_with(&format!("{object_type}\n"))
        {
            objects_part = objects_part[object_type.len()..].trim().to_string();
            break;
        }
    }

    let mut objects: Vec<String> = Vec::new();
    for raw in split_top_level(&objects_part) {
        let name = p.arg_list.replace(&raw, "").trim().to_string();
        // `GRANT ... ON db_name.* TO x` (MySQL) names a whole database, not an
        // object, so there is nothing to point an edge at.
        if !name.is_empty() && !name.ends_with(".*") && p.object_name.is_match(&name) {
            objects.push(name);
        }
    }

    // Applied until stable: `... CASCADE` and `... GRANTED BY x` can both trail.
    let mut roles_part = roles_part.to_string();
    loop {
        let trimmed = p.grant_tail.replace(&roles_part, "").to_string();
        if trimmed == roles_part {
            break;
        }
        roles_part = trimmed;
    }
    let mut roles: Vec<String> = Vec::new();
    for raw in split_top_level(&roles_part) {
        let name = p.role_qualifier.replace(&raw, "").trim().to_string();
        if name.is_empty() {
            continue;
        }
        if let Some(rm) = p.role_name.captures(&name) {
            roles.push(rm[1].to_string());
        }
    }

    if objects.is_empty() || roles.is_empty() {
        return None;
    }
    Some((collapse_ws(privileges_part).to_uppercase(), objects, roles))
}

/// `_parse_policy`: split a CREATE POLICY tail into (command, roles).
///
/// `roles` is empty when there is no TO clause, the majority case (127 of 163
/// measured). Postgres then applies the policy to PUBLIC, and the caller
/// materialises that as an INFERRED edge -- the role is a language default, not
/// something written in the source.
///
/// Everything from USING / WITH CHECK onward is dropped: those are arbitrary SQL
/// expressions that can contain the words FOR and TO, and can name tables the
/// policy does not "apply to" in any sense worth an edge.
fn parse_policy(p: &Patterns, rest: &str) -> (Option<String>, Vec<String>) {
    let head = match p.policy_body.find(rest) {
        Some(m) => &rest[..m.start()],
        None => rest,
    };
    let command = p
        .policy_for
        .captures(head)
        .map(|c| c[1].to_uppercase());

    let mut roles: Vec<String> = Vec::new();
    if let Some((_, roles_part)) = split_keyword(head, &p.kw_to) {
        for raw in split_top_level(roles_part) {
            let name = p.role_qualifier.replace(&raw, "").trim().to_string();
            if let Some(rm) = p.role_name.captures(&name) {
                roles.push(rm[1].to_string());
            }
        }
    }
    (command, roles)
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
    ///
    /// And NO `origin_file`, which every other stub-emitting walker here does
    /// set. That is not an omission: `_disambiguate_colliding_node_ids` falls
    /// back to `origin_file` for a node with no `source_file`, so stamping it
    /// salted this stub with the REFERENCING file's path and split one table
    /// across every migration that mentioned it -- 28 nodes for one table
    /// created once, measured. Withholding it is how a walker states "file is
    /// not part of this name's identity"; a SQL table name is resolved by the
    /// database globally. Keep this in step with `_ref_stub` in
    /// `graphify/extractors/sql.py`: the key set here IS the parity contract.
    ///
    /// The id is NAMESPACED `sql_table_*`, mirroring Go's `go_type_*`. A bare id
    /// shares an id space with FILE nodes -- Postgres has a table `pg_class` AND
    /// a `pg_class.h` whose file node also reduces to `pg_class` -- and the
    /// sourceless stub then absorbed the `#include` edges meant for the header:
    /// 454 C imports silently retargeted onto a SQL table, measured on postgres.
    fn ref_stub(&mut self, name: &str) -> R<String> {
        let nid = self.mkid(&["sql", "table", name])?;
        if self.seen_ids.insert(nid.clone()) {
            self.nodes.push(NodeRow {
                id: nid.clone(),
                fields: vec![
                    ("label", Val::S(name.to_string())),
                    ("file_type", Val::Static("code")),
                    ("source_file", Val::Static("")),
                    ("source_location", Val::Static("")),
                ],
            });
        }
        Ok(nid)
    }

    /// A SOURCELESS shared node for a database ROLE.
    ///
    /// Namespaced `sql_role_*` for the same reason tables are `sql_table_*`: a
    /// role lives in the database, not in a file, so it must be ONE node across
    /// the corpus and must stay out of the file-node id space.
    ///
    /// The label is `role <name>`, not the bare name, and that is load-bearing.
    /// `_rewire_unique_stub_nodes` matches a sourceless stub onto a unique real
    /// definition by LABEL KEY alone, so a role `anon` would be absorbed by a
    /// table named `anon` -- turning `f --grants_to--> anon` into a claim about
    /// a table, which is precisely the role/table confusion this change exists
    /// to remove. `roleanon` and `anon` are different key spaces.
    fn role_stub(&mut self, name: &str) -> R<String> {
        // A quoted `'svc'` is one role with `svc`; `norm_ident` strips `"`,
        // backticks and `[]` but not `'`.
        let bytes = name.as_bytes();
        let bare = if bytes.len() >= 2 && bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\'' {
            &name[1..name.len() - 1]
        } else {
            name
        };
        let role = norm_ident(bare);
        let nid = self.mkid(&["sql", "role", &role])?;
        if self.seen_ids.insert(nid.clone()) {
            self.nodes.push(NodeRow {
                id: nid.clone(),
                fields: vec![
                    ("label", Val::S(format!("role {role}"))),
                    ("file_type", Val::Static("code")),
                    ("source_file", Val::Static("")),
                    ("source_location", Val::Static("")),
                ],
            });
        }
        Ok(nid)
    }

    /// `add_edge` plus a trailing `privileges` key.
    ///
    /// Which privileges, verbatim: the whole point of a grant edge is WHAT was
    /// granted, since "never grant execute to anon" cannot be answered from a
    /// bare object->role link. `privileges` is LAST, matching the Python, which
    /// sets it after `weight`.
    fn add_grant_edge(&mut self, src: &str, tgt: &str, relation: &'static str,
                      line: usize, privileges: &str) {
        self.edges.push(EdgeRow {
            source: src.to_string(),
            target: tgt.to_string(),
            relation,
            fields: vec![
                ("confidence", Val::Static("EXTRACTED")),
                ("source_file", Val::S(self.str_path.to_string())),
                ("source_location", Val::S(format!("L{line}"))),
                ("weight", Val::F(1.0)),
                ("privileges", Val::S(privileges.to_string())),
            ],
        });
    }

    /// `add_edge` plus an optional trailing `command`, and a settable
    /// confidence.
    ///
    /// `command` is set ONLY when the source says `FOR <cmd>`: an absent FOR
    /// means ALL, and writing that value into an EXTRACTED edge would be the
    /// invented-fact-at-confidence-1.0 problem in miniature.
    fn add_policy_edge(&mut self, src: &str, tgt: &str, relation: &'static str,
                       line: usize, command: Option<&str>,
                       confidence: &'static str) {
        let mut fields = vec![
            ("confidence", Val::Static(confidence)),
            ("source_file", Val::S(self.str_path.to_string())),
            ("source_location", Val::S(format!("L{line}"))),
            ("weight", Val::F(1.0)),
        ];
        if let Some(cmd) = command {
            fields.push(("command", Val::S(cmd.to_string())));
        }
        self.edges.push(EdgeRow {
            source: src.to_string(),
            target: tgt.to_string(),
            relation,
            fields,
        });
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

/// `_walk_from_refs`: FROM/JOIN table references, skipping CTE names.
///
/// **Scoping is the whole point.** A CTE is visible only inside the query that
/// declares it, so the active set is extended PER SUBTREE: each node's directly
/// owned `cte` children join the set passed into THAT node's recursion only. A
/// single statement-wide pre-collect would also suppress an OUTER reference to a
/// real table that merely shares a subquery-CTE's name
/// (`... FROM t2 JOIN (WITH t2 AS (...) SELECT ...) sub`), dropping the real
/// `-> t2` edge (#2577).
fn walk_from_refs(
    ctx: &mut Ctx,
    node: Node,
    caller_nid: &str,
    line: usize,
    cte_names: &HashSet<String>,
) -> R<()> {
    let mut own: HashSet<String> = HashSet::new();
    for c in children(node) {
        if c.kind() != "cte" {
            continue;
        }
        // The FIRST identifier is the CTE's name; later ones are its column list
        // (`WITH levels(a, b) AS (...)`), which must NOT be skipped as tables.
        for cc in children(c) {
            if matches!(cc.kind(), "identifier" | "object_reference") {
                own.insert(norm_ident(ctx.text(cc)?));
                break;
            }
        }
    }
    // A fresh set only when this node owns CTEs, mirroring `frozenset(a | b)`.
    let active: HashSet<String> = if own.is_empty() {
        cte_names.clone()
    } else {
        cte_names.union(&own).cloned().collect()
    };

    if matches!(node.kind(), "from" | "join") {
        for c in children(node) {
            if c.kind() != "relation" {
                continue;
            }
            for cc in children(c) {
                if cc.kind() != "object_reference" {
                    continue;
                }
                let tbl = ctx.text(cc)?.to_string();
                if active.contains(&norm_ident(&tbl)) {
                    continue;
                }
                let tbl_nid = ctx.resolve_or_stub(&tbl)?;
                // The RELATION's line, not the statement's.
                let rel_line = c.start_position().row + 1;
                ctx.add_edge(caller_nid, &tbl_nid, "reads_from", rel_line);
            }
        }
    }
    for child in children(node) {
        walk_from_refs(ctx, child, caller_nid, line, &active)?;
    }
    Ok(())
}

fn walk(ctx: &mut Ctx, node: Node) -> R<()> {
    let t = node.kind();
    let line = node.start_position().row + 1;
    let p = patterns();

    if t == "create_table" {
        let name = obj_name(ctx, node)?.map(|s| s.to_string());
        if let Some(name) = name {
            if !name.is_empty() {
                let nid = ctx.mkid(&[&ctx.stem.clone(), &name])?;
                ctx.add_node(&nid, &name, line);
                ctx.table_set(norm_ident(&name), nid.clone());
                for col in children(node) {
                    if col.kind() != "column_definitions" {
                        continue;
                    }
                    let has_error = children(col).into_iter().any(|cd| cd.kind() == "ERROR");
                    let mut seen_refs: HashSet<String> = HashSet::new();
                    for cd in children(col) {
                        if cd.kind() == "column_definition" {
                            // Inline column-level REFERENCES.
                            if let Some(ref_name) = referenced_name(ctx, cd)?.map(|s| s.to_string())
                            {
                                let ref_nid = ctx.resolve_or_stub(&ref_name)?;
                                ctx.add_edge(&nid, &ref_nid, "references", line);
                                seen_refs.insert(norm_ident(&ref_name));
                            }
                        } else if cd.kind() == "constraints" {
                            // Table-level FOREIGN KEY ... REFERENCES constraints.
                            for constraint in children(cd) {
                                if constraint.kind() != "constraint" {
                                    continue;
                                }
                                if let Some(ref_name) =
                                    referenced_name(ctx, constraint)?.map(|s| s.to_string())
                                {
                                    let ref_nid = ctx.resolve_or_stub(&ref_name)?;
                                    ctx.add_edge(&nid, &ref_nid, "references", line);
                                    seen_refs.insert(norm_ident(&ref_name));
                                }
                            }
                        }
                    }
                    if has_error {
                        // Dialect syntax (Firebird COMPUTED BY) makes the parser
                        // drop the trailing constraints block entirely. Regex the
                        // raw column_definitions text as a fallback.
                        let col_text = ctx.text(col)?.to_string();
                        for cap in p.references.captures_iter(&col_text) {
                            let ref_name = cap[1].to_string();
                            if seen_refs.insert(norm_ident(&ref_name)) {
                                let ref_nid = ctx.resolve_or_stub(&ref_name)?;
                                ctx.add_edge(&nid, &ref_nid, "references", line);
                            }
                        }
                    }
                }
            }
        }
    } else if t == "create_view" {
        let name = obj_name(ctx, node)?.map(|s| s.to_string());
        if let Some(name) = name {
            if !name.is_empty() {
                let nid = ctx.mkid(&[&ctx.stem.clone(), &name])?;
                ctx.add_node(&nid, &name, line);
                ctx.table_set(norm_ident(&name), nid.clone());
                walk_from_refs(ctx, node, &nid, line, &HashSet::new())?;
            }
        }
    } else if t == "create_function" || t == "create_procedure" {
        let name = obj_name(ctx, node)?.map(|s| s.to_string());
        if let Some(name) = name {
            if !name.is_empty() {
                let nid = ctx.mkid(&[&ctx.stem.clone(), &name])?;
                ctx.add_node(&nid, &format!("{name}()"), line);
                // A routine is NOT registered in `table_nids`.
                walk_from_refs(ctx, node, &nid, line, &HashSet::new())?;
            }
        }
    } else if t == "alter_table" {
        let name = obj_name(ctx, node)?.map(|s| s.to_string());
        if let Some(name) = name {
            if !name.is_empty() {
                let src_nid = match ctx.table_get(&norm_ident(&name)) {
                    Some(n) => n.to_string(),
                    None => {
                        // Subject table not defined here: a sourceless stub, not a
                        // sourced wrong-stem node (#2324).
                        let stub = ctx.ref_stub(&name)?;
                        ctx.table_set(norm_ident(&name), stub.clone());
                        stub
                    }
                };
                for child in children(node) {
                    if child.kind() != "add_constraint" {
                        continue;
                    }
                    for cc in children(child) {
                        if cc.kind() != "constraint" {
                            continue;
                        }
                        if let Some(ref_name) = referenced_name(ctx, cc)?.map(|s| s.to_string()) {
                            let ref_nid = ctx.resolve_or_stub(&ref_name)?;
                            ctx.add_edge(&src_nid, &ref_nid, "references", line);
                        }
                    }
                }
            }
        }
    } else if t == "create_trigger" {
        // The trigger name is the first `object_reference` after
        // `keyword_trigger`; the table is the first one after `keyword_for`.
        let mut trig_name: Option<String> = None;
        let mut tbl_name: Option<String> = None;
        let mut after_trigger = false;
        let mut after_for = false;
        for c in children(node) {
            let k = c.kind();
            if k == "keyword_trigger" {
                after_trigger = true;
            } else if after_trigger && trig_name.is_none() && k == "object_reference" {
                trig_name = Some(ctx.text(c)?.to_string());
            } else if k == "keyword_for" {
                after_for = true;
            } else if after_for && tbl_name.is_none() && k == "object_reference" {
                tbl_name = Some(ctx.text(c)?.to_string());
            }
        }
        if let Some(trig) = trig_name {
            if !trig.is_empty() {
                let trig_nid = ctx.mkid(&[&ctx.stem.clone(), &trig])?;
                ctx.add_node(&trig_nid, &trig, line);
                if let Some(tbl) = tbl_name {
                    let tbl_nid = ctx.resolve_or_stub(&tbl)?;
                    ctx.add_edge(&trig_nid, &tbl_nid, "triggers", line);
                }
            }
        }
    } else if t == "ERROR" {
        // tree-sitter-sql cannot parse a PL/pgSQL CREATE FUNCTION/PROCEDURE body
        // and emits an ERROR node instead, dropping the object. One ERROR blob can
        // swallow several statements, so scan for EVERY create in it.
        //
        // Deliberately NOT scanning the body for FROM/JOIN references: PL/pgSQL
        // loop variables and locals would produce junk `reads_from` targets.
        let text = ctx.text(node)?.to_string();
        for cap in p.create_routine.captures_iter(&text) {
            let m = cap.get(1).unwrap();
            let name = m.as_str().to_string();
            let m_line = line + line_offset(&text, cap.get(0).unwrap().start());
            let nid = ctx.mkid(&[&ctx.stem.clone(), &name])?;
            ctx.add_node(&nid, &format!("{name}()"), m_line);
        }
    } else if t == "fb_proc_or_trigger" {
        let text = ctx.text(node)?.to_string();
        if let Some(m) = p.fb_header.captures(&text) {
            let obj_type = m[1].to_uppercase();
            let obj_name_s = m[2].to_string();
            let obj_nid = ctx.mkid(&[&ctx.stem.clone(), &obj_name_s])?;
            let label = if obj_type == "TRIGGER" {
                obj_name_s.clone()
            } else {
                format!("{obj_name_s}()")
            };
            ctx.add_node(&obj_nid, &label, line);
            if obj_type == "TRIGGER" {
                if let Some(fm) = p.fb_for.captures(&text) {
                    let tbl = fm[1].to_string();
                    let tbl_nid = ctx.resolve_or_stub(&tbl)?;
                    ctx.add_edge(&obj_nid, &tbl_nid, "triggers", line);
                }
            }
            let mut non_tables: HashSet<String> = [
                "select", "where", "set", "dual", "null", "true", "false", "first", "skip",
                "rows", "next", "only", "lateral",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            // Same CTE-blindness as the AST path (#2577). The regex has no scope
            // tree, so the skip is BODY-WIDE -- the right trade for a recovery path.
            for cm in p.cte.captures_iter(&text) {
                non_tables.insert(norm_ident(&cm[1]));
            }
            let mut seen_tbls: HashSet<String> = HashSet::new();
            for rm in p.from_join_into.captures_iter(&text) {
                let tbl = rm[1].to_string();
                let n = norm_ident(&tbl);
                if !non_tables.contains(&n) && seen_tbls.insert(n) {
                    let tbl_nid = ctx.resolve_or_stub(&tbl)?;
                    ctx.add_edge(&obj_nid, &tbl_nid, "reads_from", line);
                }
            }
            for rm in p.update_tbl.captures_iter(&text) {
                let tbl = rm[1].to_string();
                let n = norm_ident(&tbl);
                if !non_tables.contains(&n) && seen_tbls.insert(n) {
                    let tbl_nid = ctx.resolve_or_stub(&tbl)?;
                    ctx.add_edge(&obj_nid, &tbl_nid, "reads_from", line);
                }
            }
        }
    }

    for child in children(node) {
        walk(ctx, child)?;
    }
    Ok(())
}

/// Pre-pass: register every table/view DEFINED here before walking, so a forward
/// reference (an FK to a table created later in the same file) resolves to the
/// real sourced node instead of falling back to a stub.
fn collect_defined_names(ctx: &mut Ctx, node: Node) -> R<()> {
    if matches!(node.kind(), "create_table" | "create_view") {
        if let Some(name) = obj_name(ctx, node)?.map(|s| s.to_string()) {
            if !name.is_empty() {
                let nid = ctx.mkid(&[&ctx.stem.clone(), &name])?;
                ctx.table_set(norm_ident(&name), nid);
            }
        }
    }
    for child in children(node) {
        collect_defined_names(ctx, child)?;
    }
    Ok(())
}

pub fn walk_sql<'py>(
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
    // `source.decode("utf-8", errors="replace")` would let this proceed on
    // invalid UTF-8, but the replacement changes byte offsets against the tree,
    // so deferring is the only faithful answer.
    let src_text = std::str::from_utf8(source).map_err(|_| "source_not_utf8")?;
    let stem = file_stem(path).ok_or("path_needs_pathlib")?;
    let file_nid = make_id_ascii(&[path]).ok_or("non_ascii_path")?;

    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_sequel::LANGUAGE.into())
        .map_err(|_| "grammar_load_failed")?;
    let tree = parser.parse(source, None).ok_or("parse_failed")?;
    let root = tree.root_node();
    // NO `has_error` defer. See the module doc for the measurement that justifies
    // it -- this is the one walker here that walks an errored tree.

    let mut ctx = Ctx {
        src: source,
        src_text,
        str_path: path,
        stem,
        file_nid: file_nid.clone(),
        // The FILE node is seeded directly, because its `source_location` is a
        // JSON null rather than a line and it carries no `contains` edge.
        nodes: vec![NodeRow {
            id: file_nid.clone(),
            fields: vec![
                (
                    "label",
                    Val::S(path.rsplit('/').next().unwrap_or(path).to_string()),
                ),
                ("file_type", Val::Static("code")),
                ("source_file", Val::S(path.to_string())),
                ("source_location", Val::None),
            ],
        }],
        edges: Vec::new(),
        seen_ids: HashSet::from([file_nid.clone()]),
        table_nids: Vec::new(),
        table_index: HashMap::new(),
    };

    collect_defined_names(&mut ctx, root)?;

    // Secondary bare-name aliases: `REFERENCES users` should resolve to a
    // schema-qualified `public.users` when that is UNAMBIGUOUS. Never shadow an
    // explicit definition, and skip a bare name defined under more than one
    // schema (the `None` tombstone).
    let mut bare_candidates: Vec<(String, Option<String>)> = Vec::new();
    let mut bare_index: HashMap<String, usize> = HashMap::new();
    for (key, alias_nid) in ctx.table_nids.clone() {
        if !key.contains('.') {
            continue;
        }
        let bare = key.rsplit('.').next().unwrap_or("").to_string();
        match bare_index.get(&bare) {
            Some(i) => {
                // `bare_candidates.get(bare, alias_nid) == alias_nid`: a second,
                // DIFFERENT definition tombstones the alias.
                if bare_candidates[*i].1.as_deref() != Some(alias_nid.as_str()) {
                    bare_candidates[*i].1 = None;
                }
            }
            None => {
                bare_index.insert(bare.clone(), bare_candidates.len());
                bare_candidates.push((bare, Some(alias_nid)));
            }
        }
    }
    for (bare, alias_nid) in bare_candidates {
        if let Some(nid) = alias_nid {
            if ctx.table_get(&bare).is_none() {
                ctx.table_set(bare, nid);
            }
        }
    }

    // Only these top-level shapes are entered, and `statement` is descended one
    // level rather than walked -- a `transaction` wraps `statement` children, so
    // it IS walked whole (#2953).
    for stmt in children(root) {
        match stmt.kind() {
            "statement" => {
                for child in children(stmt) {
                    walk(&mut ctx, child)?;
                }
            }
            "transaction" => walk(&mut ctx, stmt)?,
            "fb_proc_or_trigger" | "set_term" | "declare_external_function" | "ERROR" => {
                walk(&mut ctx, stmt)?
            }
            _ => {}
        }
    }

    let p = patterns();

    // Whole-file fallback 1: REFERENCES missed because ERROR nodes pushed the
    // constraints out of the tree. Snapshotted AFTER the walk so edges already
    // captured are not re-emitted.
    let mut emitted: HashSet<(String, String)> = ctx
        .edges
        .iter()
        .filter(|e| e.relation == "references")
        .map(|e| (e.source.clone(), e.target.clone()))
        .collect();
    for m in p.create_table_open.captures_iter(src_text) {
        let tbl_name = m[1].to_string();
        let tbl_nid = match ctx.table_get(&norm_ident(&tbl_name)) {
            Some(n) => n.to_string(),
            None => continue,
        };
        let start = m.get(0).unwrap().start();
        let tbl_line = line_offset(src_text, start) + 1;
        let tail = &src_text[start..];
        // The Python searches `tail[1:]` and then slices `tail[:end.start()+1]`,
        // so the offset is relative to the ONE-character-shifted view.
        let block = match p.block_end.find(&tail[1..]) {
            Some(e) => &tail[..e.start() + 1],
            None => tail,
        };
        for rm in p.references.captures_iter(block) {
            let ref_name = rm[1].to_string();
            let ref_nid = ctx.resolve_or_stub(&ref_name)?;
            if emitted.insert((tbl_nid.clone(), ref_nid.clone())) {
                ctx.add_edge(&tbl_nid, &ref_nid, "references", tbl_line);
            }
        }
    }

    // Whole-file fallback 2 (#2180): a PL/pgSQL body can break the parse in shapes
    // the ERROR branch never sees -- the statement shredded into loose top-level
    // tokens with only the offending line inside an ERROR node, or a quoted name.
    // Gated on a FAILED parse so a clean file cannot have routines fabricated from
    // commented-out DDL, from DDL inside an EXECUTE '...' body, or from MySQL
    // `CREATE FUNCTION IF NOT EXISTS` (which would capture `IF`). `add_node`
    // dedupes by id, so routines already recovered are not emitted twice.
    if root.has_error() {
        for m in p.create_routine.captures_iter(src_text) {
            let fn_name = m[1].to_string();
            let fn_line = line_offset(src_text, m.get(0).unwrap().start()) + 1;
            let nid = ctx.mkid(&[&ctx.stem.clone(), &fn_name])?;
            ctx.add_node(&nid, &format!("{fn_name}()"), fn_line);
        }
    }

    // ── GRANT / REVOKE ───────────────────────────────────────────────────────
    // tree-sitter-sql has NO rule for these: the kind list holds `create_role`,
    // `alter_role`, `drop_role` and nothing else. Every GRANT/REVOKE lands in an
    // ERROR node (1,393 of 1,584 measured on postgres + sqlfluff), and recovery
    // actively MANGLES them -- `GRANT SELECT ON t TO r` recovers as a real
    // `select` statement with `ON t` as a term. So this scans text and ignores
    // the tree, like the REFERENCES and routine fallbacks above.
    //
    // Line-anchored, which is what keeps commented-out DDL out: `-- GRANT ...`
    // cannot match. Measured across both corpora, 0 of 1,421 line-start
    // GRANT/REVOKEs sit inside a comment or a quoted string. One inside a
    // multi-line `EXECUTE '...'` body would still be picked up; that is the
    // known residual, recorded rather than guessed at.
    for m in p.grant_stmt.captures_iter(src_text) {
        let whole = m.get(0).unwrap();
        let verb = m[1].to_uppercase();
        let line = line_offset(src_text, whole.start()) + 1;
        let Some((privileges, objects, roles)) = parse_grant(p, &verb, &m[2]) else {
            continue;
        };
        let relation = if verb == "GRANT" { "grants_to" } else { "revokes_from" };
        for obj in &objects {
            let by_table = ctx.table_get(&norm_ident(obj)).map(str::to_string);
            let obj_nid = match by_table {
                Some(nid) => nid,
                None => {
                    let stem = ctx.stem.clone();
                    let candidate = ctx.mkid(&[&stem, obj])?;
                    if ctx.seen_ids.contains(&candidate) {
                        candidate
                    } else {
                        ctx.ref_stub(obj)?
                    }
                }
            };
            for role in &roles {
                let role_nid = ctx.role_stub(role)?;
                ctx.add_grant_edge(&obj_nid, &role_nid, relation, line, &privileges);
            }
        }
    }

    // ── CREATE POLICY ────────────────────────────────────────────────────────
    // Same story as GRANT/REVOKE: no grammar rule, so a whole-file text scan.
    // 163 CREATE POLICY statements across the corpora produced nothing at all --
    // no node, no edge -- so row-level security was entirely invisible.
    //
    // ALTER POLICY and DROP POLICY (70 more) are deliberately NOT handled. The
    // graph is a static snapshot with no notion of statement order, so folding
    // an ALTER's role list into the CREATE's would claim the policy applies to
    // the union of every role it ever had, and `ALTER POLICY p ON t RENAME TO x`
    // would read `x` as a role. Silence beats either.
    for m in p.policy_stmt.captures_iter(src_text) {
        let whole = m.get(0).unwrap();
        let line = line_offset(src_text, whole.start()) + 1;
        let pol_name = m.name("name").unwrap().as_str().to_string();
        let tbl_name = m.name("table").unwrap().as_str().to_string();
        let rest = m.name("rest").unwrap().as_str().to_string();

        // Keyed on (file, TABLE, policy name): that is how SQL namespaces a
        // policy. `CREATE POLICY p1 ON t1` and `p1 ON t2` are two policies, and
        // a policy `foo` is unrelated to a TABLE `foo`. Keying on (file, name)
        // alone collided a policy `foo` with the table `foo` in the same
        // postgres file; `add_node` deduped them onto one node and the graph's
        // same-endpoint collapse then dropped that table's `references` edge.
        let stem = ctx.stem.clone();
        let pol_nid = ctx.mkid(&[&stem, &tbl_name, &pol_name])?;
        // Labelled `policy <name>`, like roles are `role <name>`:
        // `_rewire_unique_stub_nodes` picks rewire targets by label key, and a
        // policy is not a table, so an unresolved table reference must never
        // bind to one. Measured on postgres: policies named `p`/`p1`/`p2`
        // entered that label space and changed a cross-file `reads_from`.
        ctx.add_node(&pol_nid, &format!("policy {pol_name}"), line);

        let by_table = ctx.table_get(&norm_ident(&tbl_name)).map(str::to_string);
        let tbl_nid = match by_table {
            Some(nid) => nid,
            None => ctx.ref_stub(&tbl_name)?,
        };
        ctx.add_policy_edge(&pol_nid, &tbl_nid, "secures", line, None, "EXTRACTED");

        let (command, roles) = parse_policy(p, &rest);
        let cmd = command.as_deref();
        for role in &roles {
            let role_nid = ctx.role_stub(role)?;
            ctx.add_policy_edge(&pol_nid, &role_nid, "applies_to", line, cmd, "EXTRACTED");
        }
        if roles.is_empty() {
            // "If no role is specified, the policy applies to PUBLIC" -- the
            // documented Postgres default. Materialised, because a policy that
            // silently applies to everyone is the case an audit most needs to
            // see, and tagged INFERRED because the role is not in the text.
            // That distinction is what the confidence vocabulary is for.
            let role_nid = ctx.role_stub("public")?;
            ctx.add_policy_edge(&pol_nid, &role_nid, "applies_to", line, cmd, "INFERRED");
        }
    }

    let _ = ctx.src_text;
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

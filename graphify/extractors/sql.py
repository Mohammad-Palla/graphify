"""Sql extractor. Moved verbatim from graphify/extract.py."""
from __future__ import annotations

import re

from pathlib import Path
from graphify.extractors import kernel as _kernel
from graphify.extractors.base import _file_stem, _make_id


# A GRANT/REVOKE statement, anchored at line start and running to the first `;`.
# DOTALL because the statement routinely wraps across lines (a function signature
# with several parameter types is the common case).
_GRANT_STMT = re.compile(
    r"^[ \t]*(GRANT|REVOKE)\b(?P<body>.*?);",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)

# Object types that ARE graph entities, and so can carry a grant edge. Postgres
# defaults to TABLE when the keyword is omitted, which is why the keyword is
# optional at the call site.
_GRANTABLE_OBJECT_TYPES = (
    "materialized view", "foreign table", "procedure", "function",
    "sequence", "routine", "table", "view",
)

# Object types that are NOT graph entities. A grant on one of these is real SQL
# and correctly parsed -- there is simply no node to hang it on, so nothing is
# emitted rather than a stub being invented for a database or a schema.
_UNGRAPHED_OBJECT_TYPES = (
    "all tables in schema", "all sequences in schema", "all functions in schema",
    "all procedures in schema", "all routines in schema",
    "foreign data wrapper", "foreign server", "large object", "tablespace",
    "database", "language", "parameter", "schema", "domain", "type",
    # SQL/PGQ (Postgres 18). A real object, but not one this graph models.
    "property graph",
    # Databricks / Unity Catalog.
    "external location", "storage credential", "metastore", "connection",
    "catalog", "share", "provider", "recipient",
)

# Trailing clauses that are not part of the role list.
# Applied repeatedly: `... CASCADE` and `... GRANTED BY x` can both be present.
_GRANT_TAIL = re.compile(
    r"\s+(?:WITH\s+(?:GRANT|ADMIN|INHERIT|SET)\s+OPTION"
    r"|GRANTED\s+BY\s+[\w$\"`\[\]]+"   # Postgres
    r"|AS\s+[\w$\"`\[\]]+"              # T-SQL grantor
    r"|CASCADE|RESTRICT)\s*$",
    re.IGNORECASE,
)

# A name part: bare, or delimited by "", ``, '' or []. Delimited parts may
# contain anything but their delimiter -- a Databricks role is routinely
# `finance-team`, whose hyphen no `\w` pattern will match.
_PART = r"""(?:"[^"\n]+"|`[^`\n]+`|'[^'\n]+'|\[[^\]\n]+\]|[\w$]+)"""
_OBJECT_NAME = re.compile(r"^" + _PART + r"(?:\s*\.\s*" + _PART + r")*$")
# MySQL names an account `user@host` (`'svc'@'%'`). Only the user part is kept:
# the id would collapse to the same slug either way (`_make_id` drops `@` and
# `%`), so keeping the host would leave two labels fighting over one id.
# Consequence, recorded rather than hidden: `svc@localhost` and `svc@%` are one
# node, though MySQL treats them as different accounts.
_ROLE_NAME = re.compile(r"^(" + _PART + r")(?:\s*@\s*" + _PART + r")?$")

# T-SQL scopes the object by class: `OBJECT::t`, `TYPE::t`, `SCHEMA::s`. Only
# OBJECT names something this graph holds.
_TSQL_SECURABLE_CLASS = re.compile(r"^([A-Za-z_]+)\s*::\s*", re.IGNORECASE)
# `REVOKE GRANT OPTION FOR <privs> ON ...`, and role-qualifier noise in a role list.
_REVOKE_OPTION_FOR = re.compile(r"^\s*(?:GRANT|ADMIN)\s+OPTION\s+FOR\b", re.IGNORECASE)
_ROLE_QUALIFIER = re.compile(r"^(?:GROUP|ROLE|USER)\s+", re.IGNORECASE)


def _split_top_level(text: str) -> list[str]:
    """Split on commas that are not inside parentheses.

    A routine's argument list is comma-separated too, so a naive `split(",")`
    tears `FUNCTION f(uuid, text), g(int)` into four objects instead of two.
    """
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def _split_keyword(text: str, keyword: str) -> tuple[str, str] | None:
    """Split at the first top-level occurrence of `keyword` as a whole word.

    Top-level because a routine signature can contain the word inside its
    parentheses, and the FIRST occurrence because the trailing clauses
    (`WITH GRANT OPTION`) come after the role list, never before it.
    """
    depth = 0
    for match in re.finditer(r"\b" + keyword + r"\b", text, re.IGNORECASE):
        depth = (text.count("(", 0, match.start())
                 - text.count(")", 0, match.start()))
        if depth <= 0:
            return text[: match.start()], text[match.end():]
    return None


# `CREATE POLICY <name> ON <table> ...;` -- line-anchored and running to the
# first `;`, exactly like the grant scan and for the same reason: the grammar
# has no POLICY rule either.
_POLICY_STMT = re.compile(
    r"^[ \t]*CREATE\s+POLICY\s+(?P<name>" + _PART + r")"
    r"\s+ON\s+(?P<table>" + _PART + r"(?:\s*\.\s*" + _PART + r")*)"
    r"(?P<rest>.*?);",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
# The clauses that end the role/command header and begin the expressions.
_POLICY_BODY = re.compile(r"\b(?:USING|WITH\s+CHECK)\b", re.IGNORECASE)
_POLICY_FOR = re.compile(r"\bFOR\s+(ALL|SELECT|INSERT|UPDATE|DELETE)\b", re.IGNORECASE)


def _parse_policy(rest: str) -> tuple[str | None, list[str]]:
    """Split a CREATE POLICY tail into (command, roles).

    `roles` is empty when the statement has no TO clause, which is the majority
    case (127 of 163 measured). Postgres then applies the policy to PUBLIC; the
    caller materialises that as an INFERRED edge rather than an EXTRACTED one,
    because the role is a language default and is not written in the source.

    Everything from USING / WITH CHECK onward is dropped. Those are arbitrary
    SQL expressions that can contain the words FOR and TO, and can name tables
    this policy does not "apply to" in any sense worth an edge.
    """
    body = _POLICY_BODY.search(rest)
    head = rest[: body.start()] if body else rest

    command_match = _POLICY_FOR.search(head)
    command = command_match.group(1).upper() if command_match else None

    roles: list[str] = []
    split_to = _split_keyword(head, "TO")
    if split_to is not None:
        for raw in _split_top_level(split_to[1]):
            role_match = _ROLE_NAME.match(_ROLE_QUALIFIER.sub("", raw).strip())
            if role_match:
                roles.append(role_match.group(1))
    return command, roles


def _parse_grant(verb: str, body: str) -> tuple[str, list[str], list[str]] | None:
    """Split a GRANT/REVOKE body into (privileges, objects, roles).

    Returns None when the statement is well-formed SQL that this graph has no
    node for, so the caller emits nothing. That is deliberate: the failure this
    replaces is a guessed edge presented as EXTRACTED at confidence 1.0, and a
    grant on a DATABASE has no object node to point at.
    """
    body = _REVOKE_OPTION_FOR.sub("", body)
    # No ON => role membership (`GRANT admin TO alice`). Both sides are roles,
    # so there is no object to hang a privilege edge on.
    split_on = _split_keyword(body, "ON")
    if split_on is None:
        return None
    privileges_part, rest = split_on

    # Postgres spells it `REVOKE ... FROM role`; T-SQL spells it
    # `REVOKE ... TO role`. Try the dialect-native keyword first, then the other
    # -- 25 of the corpus statements are T-SQL and were dropped by FROM alone.
    split_roles = _split_keyword(rest, "TO" if verb == "GRANT" else "FROM")
    if split_roles is None and verb == "REVOKE":
        split_roles = _split_keyword(rest, "TO")
    if split_roles is None:
        return None
    objects_part, roles_part = split_roles

    objects_part = objects_part.strip()
    class_match = _TSQL_SECURABLE_CLASS.match(objects_part)
    if class_match:
        if class_match.group(1).lower() != "object":
            return None  # TYPE::, SCHEMA::, ASSEMBLY:: -- not graph entities
        objects_part = objects_part[class_match.end():].strip()
    lowered = objects_part.lower()
    if lowered.startswith(_UNGRAPHED_OBJECT_TYPES):
        return None
    for object_type in _GRANTABLE_OBJECT_TYPES:
        if lowered.startswith(object_type + " ") or lowered.startswith(object_type + "\n"):
            objects_part = objects_part[len(object_type):].strip()
            break

    objects = []
    for raw in _split_top_level(objects_part):
        # Drop a routine's argument list: the node is keyed on the name.
        name = re.sub(r"\s*\(.*\)\s*$", "", raw, flags=re.DOTALL).strip()
        # `GRANT ... ON db_name.* TO x` (MySQL) names a whole database, not an
        # object, so there is nothing to point an edge at.
        if name and not name.endswith(".*") and _OBJECT_NAME.match(name):
            objects.append(name)

    roles = []
    while True:
        trimmed = _GRANT_TAIL.sub("", roles_part)
        if trimmed == roles_part:
            break
        roles_part = trimmed
    for raw in _split_top_level(roles_part):
        name = _ROLE_QUALIFIER.sub("", raw).strip()
        role_match = _ROLE_NAME.match(name) if name else None
        if role_match:
            roles.append(role_match.group(1))

    if not objects or not roles:
        return None
    privileges = " ".join(privileges_part.split()).upper()
    return privileges, objects, roles


def _norm_ident(name: str) -> str:
    """Normalize a SQL identifier for name-based reference resolution.

    Splits on `.`, strips one pair of surrounding delimiters from each part
    (double quotes for Postgres/ANSI, backticks for MySQL, brackets for
    T-SQL), lowercases, and rejoins. So `"public"."users"`, `public.users`,
    and `PUBLIC.USERS` all normalize to `public.users`. Used ONLY for
    `table_nids` keys and lookups — node ids and display labels keep the
    original text.
    """
    parts = []
    for part in name.split("."):
        p = part.strip()
        if len(p) >= 2 and ((p[0] == p[-1] and p[0] in ('"', "`"))
                            or (p[0] == "[" and p[-1] == "]")):
            p = p[1:-1]
        parts.append(p.lower())
    return ".".join(parts)


_KERNEL_GRAMMAR = _kernel.BespokeGrammar("tree_sitter_sql")


def extract_sql(path: Path, content: str | bytes | None = None) -> dict:
    """Extract tables, views, functions, and relationships from .sql files via tree-sitter."""
    # `content` is supplied by the --postgres introspection path, which
    # reconstructs DDL in memory rather than reading a file; pass it through as
    # `source_override` so the kernel walks the same bytes.
    native = _kernel.try_extract(
        path, _KERNEL_GRAMMAR,
        source_override=(content.encode("utf-8") if isinstance(content, str)
                         else content),
    )
    if native is not None:
        return native
    try:
        import tree_sitter_sql as tssql
        from tree_sitter import Language, Parser
    except ImportError as e:
        import importlib.util
        # An installed-but-broken grammar (e.g. a C extension built for a
        # different Python ABI, #2602) raises ImportError here too. Reporting
        # that as "not installed" sends the user to a no-op `pip install`, so
        # distinguish a genuinely-absent module from one that failed to load
        # and surface the real exception in the latter case.
        if importlib.util.find_spec("tree_sitter_sql") is None:
            return {"nodes": [], "edges": [],
                    "error": "tree_sitter_sql not installed. Run: pip install tree-sitter-sql"}
        return {"nodes": [], "edges": [],
                "error": f"tree_sitter_sql is installed but failed to load: {e}"}

    try:
        language = Language(tssql.language())
        parser = Parser(language)
        source = (
            content.encode("utf-8") if isinstance(content, str)
            else content if content is not None
            else path.read_bytes()
        )
        tree = parser.parse(source)
        root = tree.root_node
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}


    stem = _file_stem(path)
    str_path = str(path)
    file_nid = _make_id(str_path)
    nodes: list[dict] = [{"id": file_nid, "label": path.name, "file_type": "code",
                           "source_file": str_path, "source_location": None}]
    edges: list[dict] = []
    seen_ids: set[str] = {file_nid}
    table_nids: dict[str, str] = {}  # name → nid for reference resolution

    def _read(n) -> str:
        return source[n.start_byte:n.end_byte].decode("utf-8", errors="replace")

    def _obj_name(n) -> str | None:
        for c in n.children:
            if c.type == "object_reference":
                return _read(c)
        return None

    def _add_node(nid: str, label: str, line: int) -> None:
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({"id": nid, "label": label, "file_type": "code",
                           "source_file": str_path, "source_location": f"L{line}"})
            edges.append({"source": file_nid, "target": nid, "relation": "contains",
                           "confidence": "EXTRACTED", "source_file": str_path,
                           "source_location": f"L{line}", "weight": 1.0})

    def _add_edge(src: str, tgt: str, relation: str, line: int,
                  privileges: str | None = None, command: str | None = None,
                  confidence: str = "EXTRACTED") -> None:
        edge = {"source": src, "target": tgt, "relation": relation,
                "confidence": confidence, "source_file": str_path,
                "source_location": f"L{line}", "weight": 1.0}
        if command is not None:
            # Which statements the policy covers (`FOR SELECT`). Only ever set
            # when the source says so -- an absent FOR means ALL, and inventing
            # that value inside an EXTRACTED edge is the thing being avoided.
            edge["command"] = command
        if privileges is not None:
            # Which privileges, verbatim from the statement. The whole point of
            # a grant edge is WHAT was granted: "never grant execute to anon" is
            # unanswerable from a bare role->object link.
            edge["privileges"] = privileges
        edges.append(edge)

    def _role_stub(name: str) -> str:
        """Sourceless shared node for a database ROLE.

        Namespaced `sql_role_*` for the same reason table stubs are namespaced
        `sql_table_*`: a role lives in the database, not in a file, so it must be
        one node across the corpus, and it must not land in the id space of file
        nodes.

        The label is `role <name>`, not the bare name, and that is load-bearing
        rather than cosmetic. `_rewire_unique_stub_nodes` matches a sourceless
        stub onto a unique real definition purely by label key, so a role `anon`
        would be ABSORBED by a table named `anon` -- turning
        `f --grants_to--> anon` into a statement about a table. That is exactly
        the role/table confusion this whole change exists to remove, so the
        labels are kept in different key spaces (`roleanon` vs `anon`).
        """
        # Normalised like a table key: `"public"`, `PUBLIC` and `public` are one
        # role. Strictly, a quoted role name is case-sensitive and `"Public"` is
        # a different role from `public` -- but folding them keeps one node for
        # the overwhelmingly common case, and splitting on quoting style would
        # scatter the very node an audit query goes looking for.
        if len(name) >= 2 and name[0] == name[-1] == "'":
            name = name[1:-1]
        role = _norm_ident(name)
        nid = _make_id("sql", "role", role)
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({"id": nid, "label": f"role {role}", "file_type": "code",
                           "source_file": "", "source_location": ""})
        return nid

    def _ref_stub(name: str) -> str:
        """Sourceless bare-name stub for a table referenced but not defined here.

        SQL references are NAME-based, so a table defined in another file (e.g.
        prisma migration m2 referencing a table created in m1) can only resolve
        at the corpus level. Minting `_make_id(stem, name)` under THIS file's
        stem fabricated a node-less compound id — an absolute-path slug when the
        input path was absolute — that could never match the real definition
        (#2324). Instead emit a SOURCELESS stub, mirroring the Go extractor's
        cross-file pattern (#1402): `_rewire_unique_stub_nodes` collapses it
        onto the unique real table definition, and an unresolvable name survives
        as a portable name-only node instead of dangling. No contains edge: a
        sourced/contained stub would get the referencing file's path baked into
        its id by disambiguation, blocking the rewire.

        No ``origin_file`` either, for the same reason and by the same route.
        That hint is what ``_disambiguate_colliding_node_ids`` falls back to for
        a SOURCELESS node, so stamping it re-introduced exactly the compound id
        this stub exists to avoid. Disambiguation runs BEFORE
        ``_rewire_unique_stub_nodes``, so the rewire was left with nothing to
        collapse: measured on 20 migrations referencing one table created once,
        ``public.account`` became 28 nodes.

        Omitting it is the deliberate #1462 opt-out, not an oversight. #1462
        keeps sourceless stubs per-file distinct only where Graphify cannot tell
        whether two same-named references mean the same external entity (two
        files importing ``pathlib.Path``). SQL is the case where it can: a table
        name is resolved by the database globally, never per-file, so two
        references to ``public.account`` are the same table by definition. Go
        states the same fact with a canonical ``go_type_*`` id; SQL has no such
        pass, and withholding the hint is the equivalent statement.

        Sharing a stub is not binding it: ``_rewire_unique_stub_nodes`` still
        absorbs one only when EXACTLY ONE real definition carries the name, so an
        ambiguous name survives as a single shared name-only node rather than
        mis-binding to an arbitrary definition.

        The id is NAMESPACED (`sql_table_*`), exactly as Go namespaces its shared
        external types `go_type_*`, and for a reason found by measurement rather
        than taste. A bare `_make_id(name)` puts the stub in the same id space as
        FILE nodes: Postgres has both a table `pg_class` and a header
        `src/include/catalog/pg_class.h`, whose file node also reduces to
        `pg_class`. Disambiguation salts the colliding FILE apart but skips a
        node with no source_key, so the stub kept the bare `pg_class` and then
        absorbed the `#include "catalog/pg_class.h"` edges aimed at the header:
        **454 C imports silently retargeted onto a SQL table** on postgres alone.
        Namespacing removes the collision instead of arbitrating it.
        """
        nid = _make_id("sql", "table", name)
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({"id": nid, "label": name, "file_type": "code",
                           "source_file": "", "source_location": ""})
        return nid

    def walk(node) -> None:
        t = node.type
        line = node.start_point[0] + 1

        if t == "create_table":
            name = _obj_name(node)
            if name:
                nid = _make_id(stem, name)
                _add_node(nid, name, line)
                table_nids[_norm_ident(name)] = nid
                # Foreign key REFERENCES
                for col in node.children:
                    if col.type == "column_definitions":
                        has_error = any(cd.type == "ERROR" for cd in col.children)
                        seen_refs: set[str] = set()
                        for cd in col.children:
                            if cd.type == "column_definition":
                                # Inline column-level REFERENCES
                                ref_name: str | None = None
                                found_ref = False
                                for cc in cd.children:
                                    if cc.type == "keyword_references":
                                        found_ref = True
                                    elif found_ref and cc.type == "object_reference":
                                        ref_name = _read(cc)
                                        break
                                if ref_name:
                                    ref_nid = table_nids.get(_norm_ident(ref_name)) or _ref_stub(ref_name)
                                    _add_edge(nid, ref_nid, "references", line)
                                    seen_refs.add(_norm_ident(ref_name))
                            elif cd.type == "constraints":
                                # Table-level FOREIGN KEY ... REFERENCES ... constraints
                                for constraint in cd.children:
                                    if constraint.type != "constraint":
                                        continue
                                    ref_name = None
                                    found_ref = False
                                    for cc in constraint.children:
                                        if cc.type == "keyword_references":
                                            found_ref = True
                                        elif found_ref and cc.type == "object_reference":
                                            ref_name = _read(cc)
                                            break
                                    if ref_name:
                                        ref_nid = table_nids.get(_norm_ident(ref_name)) or _ref_stub(ref_name)
                                        _add_edge(nid, ref_nid, "references", line)
                                        seen_refs.add(_norm_ident(ref_name))
                        if has_error:
                            # Dialect-specific syntax (e.g. Firebird COMPUTED BY) causes ERROR
                            # nodes that make the parser drop the trailing constraints block.
                            # Regex-scan the raw column_definitions text as fallback.
                            col_text = _read(col)
                            for rm in re.finditer(r"\bREFERENCES\s+([\w$]+)", col_text, re.IGNORECASE):
                                ref_name = rm.group(1)
                                if _norm_ident(ref_name) not in seen_refs:
                                    ref_nid = table_nids.get(_norm_ident(ref_name)) or _ref_stub(ref_name)
                                    _add_edge(nid, ref_nid, "references", line)
                                    seen_refs.add(_norm_ident(ref_name))

        elif t == "create_view":
            name = _obj_name(node)
            if name:
                nid = _make_id(stem, name)
                _add_node(nid, name, line)
                table_nids[_norm_ident(name)] = nid
                # FROM/JOIN table references inside view body
                _walk_from_refs(node, nid, line)

        elif t == "create_function":
            name = _obj_name(node)
            if name:
                nid = _make_id(stem, name)
                _add_node(nid, f"{name}()", line)
                _walk_from_refs(node, nid, line)

        elif t == "create_procedure":
            name = _obj_name(node)
            if name:
                nid = _make_id(stem, name)
                _add_node(nid, f"{name}()", line)
                _walk_from_refs(node, nid, line)

        elif t == "alter_table":
            name = _obj_name(node)
            if name:
                src_nid = table_nids.get(_norm_ident(name))
                if not src_nid:
                    # Subject table not defined in this file: sourceless stub,
                    # not a sourced wrong-stem node (#2324).
                    src_nid = _ref_stub(name)
                    table_nids[_norm_ident(name)] = src_nid
                for child in node.children:
                    if child.type == "add_constraint":
                        for cc in child.children:
                            if cc.type != "constraint":
                                continue
                            found_ref = False
                            ref_name: str | None = None
                            for ccc in cc.children:
                                if ccc.type == "keyword_references":
                                    found_ref = True
                                elif found_ref and ccc.type == "object_reference":
                                    ref_name = _read(ccc)
                                    break
                            if ref_name:
                                ref_nid = (table_nids.get(_norm_ident(ref_name))
                                           or _ref_stub(ref_name))
                                _add_edge(src_nid, ref_nid, "references", line)

        elif t == "create_trigger":
            trig_name: str | None = None
            tbl_name: str | None = None
            after_trigger = False
            after_for = False
            for c in node.children:
                if c.type == "keyword_trigger":
                    after_trigger = True
                elif after_trigger and not trig_name and c.type == "object_reference":
                    trig_name = _read(c)
                elif c.type == "keyword_for":
                    after_for = True
                elif after_for and not tbl_name and c.type == "object_reference":
                    tbl_name = _read(c)
            if trig_name:
                trig_nid = _make_id(stem, trig_name)
                _add_node(trig_nid, trig_name, line)
                if tbl_name:
                    tbl_nid = table_nids.get(_norm_ident(tbl_name)) or _ref_stub(tbl_name)
                    _add_edge(trig_nid, tbl_nid, "triggers", line)

        elif t == "ERROR":
            # tree-sitter-sql cannot parse PL/pgSQL CREATE FUNCTION/PROCEDURE
            # bodies (OUT/INOUT params, tagged dollar quotes, PERFORM, :=) and
            # emits an ERROR node instead, silently dropping the object.
            # Regex-scan the raw text as fallback, mirroring the
            # fb_proc_or_trigger recovery below. One ERROR blob can swallow
            # several statements, so scan for every CREATE in it. We deliberately
            # do not scan the body for FROM/JOIN references: PL/pgSQL loop
            # variables and locals would produce junk reads_from targets.
            #
            # Each name part is either a bare identifier or a double-quoted
            # (delimited) one, so schema-qualified generated DDL such as
            # CREATE OR REPLACE FUNCTION "public"."fn"(...) is recovered too.
            # A bare [\w$.]+ stops dead at the leading quote, which silently
            # dropped every quoted PL/pgSQL routine (#2180).
            text = _read(node)
            for m in re.finditer(
                r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:FUNCTION|PROCEDURE)\s+"
                r"(?:IF\s+NOT\s+EXISTS\s+)?"
                r"((?:\"[^\"\n]+\"|[\w$]+)(?:\s*\.\s*(?:\"[^\"\n]+\"|[\w$]+))*)",
                text, re.IGNORECASE,
            ):
                name = m.group(1)
                m_line = line + text[: m.start()].count("\n")
                nid = _make_id(stem, name)
                _add_node(nid, f"{name}()", m_line)

        elif t == "fb_proc_or_trigger":
            text = _read(node)
            m = re.match(
                r"CREATE\s+(?:OR\s+(?:REPLACE|ALTER)\s+)?"
                r"(PROCEDURE|TRIGGER|FUNCTION)\s+([\w$]+)",
                text, re.IGNORECASE,
            )
            if m:
                obj_type = m.group(1).upper()
                obj_name = m.group(2)
                obj_nid = _make_id(stem, obj_name)
                label = obj_name if obj_type == "TRIGGER" else f"{obj_name}()"
                _add_node(obj_nid, label, line)
                if obj_type == "TRIGGER":
                    fm = re.search(r"\bFOR\s+([\w$]+)", text, re.IGNORECASE)
                    if fm:
                        tbl = fm.group(1)
                        tbl_nid = table_nids.get(_norm_ident(tbl)) or _ref_stub(tbl)
                        _add_edge(obj_nid, tbl_nid, "triggers", line)
                _NON_TABLES = {
                    "select", "where", "set", "dual", "null", "true", "false",
                    "first", "skip", "rows", "next", "only", "lateral",
                }
                # Same CTE-blindness as the AST path (#2577): a `WITH <name> AS (`
                # binding is statement-local, not a table, so its name must not
                # become a reads_from stub. The regex has no scope tree, so the
                # skip is body-wide — the right trade for a recovery path.
                for cm in re.finditer(
                    r"(?:\bWITH\s+(?:RECURSIVE\s+)?|,\s*)([\w$]+)\s*(?:\([^()]*\))?\s+AS\s*\(",
                    text, re.IGNORECASE,
                ):
                    _NON_TABLES.add(_norm_ident(cm.group(1)))
                seen_tbls: set[str] = set()
                for rm in re.finditer(r"\b(?:FROM|JOIN|INTO)\s+([\w$]+)", text, re.IGNORECASE):
                    tbl = rm.group(1)
                    if _norm_ident(tbl) not in _NON_TABLES and _norm_ident(tbl) not in seen_tbls:
                        seen_tbls.add(_norm_ident(tbl))
                        tbl_nid = table_nids.get(_norm_ident(tbl)) or _ref_stub(tbl)
                        _add_edge(obj_nid, tbl_nid, "reads_from", line)
                for rm in re.finditer(r"\bUPDATE\s+([\w$]+)", text, re.IGNORECASE):
                    tbl = rm.group(1)
                    if _norm_ident(tbl) not in _NON_TABLES and _norm_ident(tbl) not in seen_tbls:
                        seen_tbls.add(_norm_ident(tbl))
                        tbl_nid = table_nids.get(_norm_ident(tbl)) or _ref_stub(tbl)
                        _add_edge(obj_nid, tbl_nid, "reads_from", line)

        for child in node.children:
            walk(child)

    def _walk_from_refs(node, caller_nid: str, line: int,
                        cte_names: frozenset[str] = frozenset()) -> None:
        """Recursively find FROM/JOIN table references inside a node, skipping CTEs.

        A name bound by `WITH <name> AS (...)` is not a table: emitting it as a
        `reads_from` target minted a bare `_ref_stub`, and because that stub is
        intentionally sourceless (see `_ref_stub`) it carried no schema, file, or
        language namespace, so a CTE named `levels` or `slug` collided with any
        same-named node from another language during the build (#2577).

        Scoping matters: a CTE is visible only inside the query that declares it,
        and a `WITH` inside a subquery is scoped to that subquery alone. So the
        active set is extended PER SUBTREE — each node's directly-owned `cte`
        children (`create_query` for a statement-level WITH, `subquery` for a
        nested one) join the set passed down into that node's recursion only. A
        single statement-wide pre-collect would also suppress an OUTER reference
        to a real table that merely shares a subquery-CTE's name
        (`... FROM t2 JOIN (WITH t2 AS (...) SELECT ...) sub`), dropping the
        real `-> t2` edge.
        """
        own: set[str] = set()
        for c in node.children:
            if c.type != "cte":
                continue
            # First identifier is the CTE's name; later ones are its column
            # list (`WITH levels(a, b) AS (...)`), which must not be skipped.
            for cc in c.children:
                if cc.type in ("identifier", "object_reference"):
                    own.add(_norm_ident(_read(cc)))
                    break
        if own:
            cte_names = frozenset(cte_names | own)
        if node.type in ("from", "join"):
            for c in node.children:
                if c.type == "relation":
                    for cc in c.children:
                        if cc.type == "object_reference":
                            tbl = _read(cc)
                            if _norm_ident(tbl) in cte_names:
                                continue
                            tbl_nid = table_nids.get(_norm_ident(tbl)) or _ref_stub(tbl)
                            _add_edge(caller_nid, tbl_nid, "reads_from",
                                      c.start_point[0] + 1)
        for child in node.children:
            _walk_from_refs(child, caller_nid, line, cte_names)

    # Pre-pass: register every table/view DEFINED in this file before walking,
    # so forward references (a FK to a table created later in the same file)
    # still resolve to the real sourced node instead of falling back to a stub.
    def _collect_defined_names(node) -> None:
        if node.type in ("create_table", "create_view"):
            name = _obj_name(node)
            if name:
                table_nids[_norm_ident(name)] = _make_id(stem, name)
        for child in node.children:
            _collect_defined_names(child)

    _collect_defined_names(root)

    # Secondary bare-name aliases: a reference written without a schema
    # (`REFERENCES users`) should resolve to a schema-qualified definition
    # (`public.users`) when that is unambiguous. Never shadow an explicit
    # definition, and skip bare names defined under more than one schema.
    bare_candidates: dict[str, str | None] = {}
    for key, alias_nid in table_nids.items():
        if "." in key:
            bare = key.rsplit(".", 1)[1]
            bare_candidates[bare] = (
                alias_nid if bare_candidates.get(bare, alias_nid) == alias_nid else None
            )
    for bare, alias_nid in bare_candidates.items():
        if alias_nid is not None and bare not in table_nids:
            table_nids[bare] = alias_nid

    for stmt in root.children:
        if stmt.type == "statement":
            for child in stmt.children:
                walk(child)
        elif stmt.type == "transaction":
            # BEGIN; ... COMMIT; wraps DDL in a transaction node whose children
            # are statement nodes, not direct create_table nodes (#2953).
            walk(stmt)
        elif stmt.type in ("fb_proc_or_trigger", "set_term", "declare_external_function", "ERROR"):
            walk(stmt)

    # Global regex fallback: catch any REFERENCES missed due to ERROR nodes in the parse tree
    # (e.g. Firebird COMPUTED BY columns push constraints out of the tree entirely).
    # Snapshot after tree walk so we don't re-emit edges already captured above.
    emitted = {(e["source"], e["target"]) for e in edges if e["relation"] == "references"}
    src_text = source.decode("utf-8", errors="replace")
    for m in re.finditer(r"CREATE\s+TABLE\s+([\w$]+)\s*\(", src_text, re.IGNORECASE):
        tbl_name = m.group(1)
        tbl_nid = table_nids.get(_norm_ident(tbl_name))
        if tbl_nid is None:
            continue
        tbl_line = src_text[: m.start()].count("\n") + 1
        tail = src_text[m.start():]
        end = re.search(r"(?:^|\n)(?:CREATE|SET\s+TERM|ALTER)\s", tail[1:], re.IGNORECASE)
        block = tail[: end.start() + 1] if end else tail
        for rm in re.finditer(r"\bREFERENCES\s+([\w$]+)", block, re.IGNORECASE):
            ref_name = rm.group(1)
            ref_nid = table_nids.get(_norm_ident(ref_name)) or _ref_stub(ref_name)
            if (tbl_nid, ref_nid) not in emitted:
                _add_edge(tbl_nid, ref_nid, "references", tbl_line)
                emitted.add((tbl_nid, ref_nid))

    # Global regex fallback for routines (#2180). PL/pgSQL bodies break the parse
    # in more than one shape, and only the first was recovered before:
    #   1. the whole CREATE lands in one ERROR node          -> handled in walk()
    #   2. the statement is shredded into loose top-level tokens
    #      (keyword_create/keyword_function/object_reference/... ) and the ERROR
    #      node holds only the offending body line, e.g. `PERFORM x();` or
    #      `x := 1;` -- so no CREATE text is inside any ERROR node at all
    #   3. the name is a quoted identifier ("public"."fn"), which a bare
    #      [\w$.]+ pattern cannot match
    # Shapes 2 and 3 silently dropped the routine: no node, no warning, exit 0.
    # Scanning the raw source catches all three, and _add_node dedupes by id so
    # routines already recovered from the tree are not emitted twice.
    #
    # Gate on a failed parse: a cleanly-parsing file must NOT have routines
    # fabricated from commented-out DDL, DDL inside EXECUTE '...' string bodies,
    # or MySQL `CREATE FUNCTION IF NOT EXISTS` (which would capture `IF`). Every
    # observed drop shape leaves an ERROR node in the tree, so has_error loses
    # nothing while protecting clean corpora (#2180 follow-up).
    if root.has_error:
        for m in re.finditer(
            r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:FUNCTION|PROCEDURE)\s+"
            r"(?:IF\s+NOT\s+EXISTS\s+)?"
            r"((?:\"[^\"\n]+\"|[\w$]+)(?:\s*\.\s*(?:\"[^\"\n]+\"|[\w$]+))*)",
            src_text, re.IGNORECASE,
        ):
            fn_name = m.group(1)
            fn_line = src_text[: m.start()].count("\n") + 1
            _add_node(_make_id(stem, fn_name), f"{fn_name}()", fn_line)

    # ── GRANT / REVOKE ────────────────────────────────────────────────────────
    # tree-sitter-sql has NO grammar rule for these. The kind list contains
    # `create_role`/`alter_role`/`drop_role` and nothing else: every GRANT and
    # REVOKE lands in an ERROR node (1,393 of 1,584 measured on postgres +
    # sqlfluff), and error recovery actively MANGLES them --
    # `GRANT SELECT ON t TO r` recovers as a real `select` statement with `ON t`
    # as a term. So this is a whole-file text scan, like the REFERENCES and
    # routine fallbacks above, and it deliberately ignores the tree.
    #
    # Before this, 1,421 GRANT/REVOKE statements across both corpora produced
    # ZERO edges -- a silent gap in the area where a wrong answer costs most
    # ("never grant execute to anon" is a rule people enforce in CI).
    #
    # Anchored at line start, which is what keeps commented-out and
    # string-embedded DDL out: `-- GRANT ...` cannot match. Measured across both
    # corpora, 0 of 1,421 line-start GRANT/REVOKEs sit inside a comment or a
    # quoted string. A grant inside a multi-line `EXECUTE '...'` body would
    # still be picked up; that is the known residual exposure, and it is
    # recorded rather than guessed at.
    for m in _GRANT_STMT.finditer(src_text):
        verb = m.group(1).upper()
        line = src_text[: m.start()].count("\n") + 1
        parsed = _parse_grant(verb, m.group("body"))
        if parsed is None:
            # Not a privilege grant on a nameable object: role membership
            # (`GRANT admin TO alice`, no ON), or an object type that is not a
            # graph entity (SCHEMA, DATABASE, LANGUAGE, ALL TABLES IN SCHEMA).
            # Emitting nothing is the point -- a guessed target here is worse
            # than no edge.
            continue
        privileges, objects, roles = parsed
        relation = "grants_to" if verb == "GRANT" else "revokes_from"
        for obj in objects:
            obj_nid = (table_nids.get(_norm_ident(obj))
                       or (_make_id(stem, obj) if _make_id(stem, obj) in seen_ids
                           else None)
                       or _ref_stub(obj))
            for role in roles:
                _add_edge(obj_nid, _role_stub(role), relation, line,
                          privileges=privileges)

    # ── CREATE POLICY ─────────────────────────────────────────────────────────
    # Same story as GRANT/REVOKE: no grammar rule, so a whole-file text scan.
    # 163 CREATE POLICY statements across the corpora produced nothing at all --
    # no node, no edge -- so row-level security was entirely invisible.
    #
    # ALTER POLICY and DROP POLICY (70 more) are deliberately NOT handled. The
    # graph is a static snapshot with no notion of statement order, so merging an
    # ALTER's role list into the CREATE's would claim the policy applies to the
    # union of every role it ever had, and `ALTER POLICY p ON t RENAME TO x`
    # would read `x` as a role. Silence beats either.
    for m in _POLICY_STMT.finditer(src_text):
        line = src_text[: m.start()].count("\n") + 1
        pol_name = m.group("name")
        tbl_name = m.group("table")
        # Keyed on (file, TABLE, policy name), because that is how SQL itself
        # namespaces a policy: `CREATE POLICY p1 ON t1` and `CREATE POLICY p1 ON
        # t2` are two different policies, and a policy `foo` is unrelated to a
        # TABLE `foo`. Keying on (file, name) alone did both kinds of damage --
        # measured on postgres, a policy `foo` collided with the table `foo` in
        # the same file, `_add_node` deduped them onto one node, and the graph's
        # same-endpoint edge collapse then DROPPED that table's `references`
        # edge. A new node kind must not enter an existing id space.
        pol_nid = _make_id(stem, tbl_name, pol_name)
        # Labelled `policy <name>`, not the bare name, for the same reason roles
        # are labelled `role <name>`: `_rewire_unique_stub_nodes` picks its
        # rewire targets by label key, and a policy is not a table, so an
        # unresolved table reference must never bind to one. Measured on
        # postgres: policies named `p`, `p1`, `p2` entered that label space and
        # silently changed a cross-file `reads_from` in an unrelated file.
        _add_node(pol_nid, f"policy {pol_name}", line)

        tbl_nid = table_nids.get(_norm_ident(tbl_name)) or _ref_stub(tbl_name)
        _add_edge(pol_nid, tbl_nid, "secures", line)

        command, roles = _parse_policy(m.group("rest"))
        for role in roles:
            _add_edge(pol_nid, _role_stub(role), "applies_to", line,
                      command=command)
        if not roles:
            # "If no role is specified, the policy applies to PUBLIC" -- the
            # documented Postgres default. Materialised, because a policy that
            # silently applies to everyone is the case an audit most needs to
            # see, and tagged INFERRED because the role is not in the text. That
            # distinction is what the confidence vocabulary is for.
            _add_edge(pol_nid, _role_stub("public"), "applies_to", line,
                      command=command, confidence="INFERRED")

    return {"nodes": nodes, "edges": edges}

"""C constructs the corpora cannot reach, pinned against `_extract_generic`.

`harness/kernel_walker_parity.py` compares the native C walker against
`_extract_generic` over 2,137 real files (curl 1,014, redis 756, libuv 367) at
DIVERGENT 0, which is the primary evidence.

C is the first language on the shared engine with an EMPTY `class_types` and the
first with a `resolve_function_name_fn`, so the two things worth pinning directly
are the declarator unwrapping (a corpus contains whatever declarator shapes it
happens to contain) and the `#include` seam, whose resolved branch depends on a
file actually existing on disk.

Each test compares the two implementations on one source.
"""
from __future__ import annotations

import contextlib
import json

import pytest

from graphify.extract import _C_CONFIG
from graphify.extractors import kernel as kseam
from graphify.extractors.engine import _extract_generic

kernel = pytest.importorskip("graphify_kernel", reason="native kernel not built")


def _canon(o) -> str:
    return json.dumps(o, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)


@contextlib.contextmanager
def _seam_disabled():
    original = kseam.try_extract
    kseam.try_extract = lambda *a, **kw: None
    try:
        yield
    finally:
        kseam.try_extract = original


def _both(tmp_path, source: str, name: str = "sample.c"):
    p = tmp_path / name
    p.write_text(source, encoding="utf-8")
    src = p.read_bytes()
    native, reason = kernel.extract_file(
        str(p), src, "c", None, None, None, kseam._c_include_resolver(str(p)))
    with _seam_disabled():
        expected = _extract_generic(p, _C_CONFIG)
    return native, reason, expected


def _assert_match(tmp_path, source: str, name: str = "sample.c"):
    native, reason, expected = _both(tmp_path, source, name)
    assert native is not None, f"kernel deferred: {reason}"
    assert _canon(native) == _canon(expected)
    return native


def _labels(result):
    return {n.get("label") for n in result["nodes"]}


def _edges(result, relation):
    return [e for e in result["edges"] if e["relation"] == relation]


# ── declarator unwrapping ────────────────────────────────────────────────────

def test_plain_function(tmp_path):
    r = _assert_match(tmp_path, "int add(int a, int b) { return a + b; }")
    assert "add()" in _labels(r)


def test_pointer_returning_function_unwraps_to_its_name(tmp_path):
    """`char *dup(...)` wraps the function_declarator in a pointer_declarator, so
    the name is not a `name` field anywhere on the definition."""
    r = _assert_match(tmp_path, "char *dup(const char *s) { return 0; }")
    assert "dup()" in _labels(r)


def test_double_pointer_function(tmp_path):
    r = _assert_match(tmp_path, "char **split(char *s) { return 0; }")
    assert "split()" in _labels(r)


def test_function_returning_a_function_pointer_is_dropped_by_both(tmp_path):
    """`void (*sig(int))(int)` produces NO function node, on either side.

    `_get_c_func_name` recurses on the `declarator` FIELD first and only scans
    children for a bare `identifier` as a fallback. The outer declarator's field
    is a `parenthesized_declarator`, which has no `declarator` field and no
    direct `identifier` child -- so the walk returns None and the definition is
    skipped. Pinned as a deliberate limitation rather than fixed: the native
    walker must reproduce `_extract_generic`, including where it gives up."""
    r = _assert_match(tmp_path, "void (*sig(int n))(int) { return 0; }")
    assert not any(lbl and lbl.startswith("sig") for lbl in _labels(r))


def test_a_declaration_with_no_declarator_is_skipped(tmp_path):
    """`resolve_function_name_fn` is an elif, so it REPLACES the name-field
    lookup: with no declarator the function is dropped rather than falling back
    to a `name` field that C never has."""
    r = _assert_match(tmp_path, "int x = 1;\nint f(void) { return x; }")
    assert "f()" in _labels(r)


# ── type references ──────────────────────────────────────────────────────────

def test_return_type_reference(tmp_path):
    r = _assert_match(tmp_path, "typedef struct s widget_t;\nwidget_t make(void) { }")
    assert any(e.get("context") == "return_type" for e in _edges(r, "references"))


def test_parameter_type_reference(tmp_path):
    r = _assert_match(tmp_path, "void take(widget_t w) { }")
    assert any(e.get("context") == "parameter_type" for e in _edges(r, "references"))


def test_pointer_parameter_type_is_unwrapped(tmp_path):
    r = _assert_match(tmp_path, "void take(widget_t *w) { }")
    assert any(e.get("context") == "parameter_type" for e in _edges(r, "references"))


def test_primitive_types_produce_no_reference(tmp_path):
    """`primitive_type` and `sized_type_specifier` are skipped whole."""
    r = _assert_match(tmp_path, "unsigned long count(int n) { return n; }")
    assert not _edges(r, "references")


def test_a_macro_wrapped_return_type_becomes_a_reference(tmp_path):
    """`UNUSED static int f(void)` is why libuv's headers are what they are: when
    the macro DOES parse, it reads as the return type and mints a stub node."""
    r = _assert_match(tmp_path, "typedef int UNUSED;\nUNUSED f(void) { return 0; }")
    assert any(e.get("context") == "return_type" for e in _edges(r, "references"))


# ── includes ─────────────────────────────────────────────────────────────────

def test_system_include_uses_the_bare_stem(tmp_path):
    """An angle-bracket include is a system header and is never probed on disk."""
    r = _assert_match(tmp_path, "#include <stdio.h>\nint f(void) { return 0; }")
    e = next(e for e in r["edges"] if e["relation"] == "imports")
    assert "target_file" not in e
    assert e["context"] == "import"


def test_quoted_include_that_does_not_exist_falls_back_to_the_stem(tmp_path):
    r = _assert_match(tmp_path, '#include "nowhere/absent.h"\nint f(void) { return 0; }')
    e = next(e for e in r["edges"] if e["relation"] == "imports")
    assert "target_file" not in e


def test_quoted_include_that_exists_is_resolved_to_the_file(tmp_path):
    """The resolved branch stamps `target_file` LAST, and it only fires when the
    header is really on disk -- which is why it needs a fixture, not a string."""
    (tmp_path / "helper.h").write_text("int helper(void);\n")
    r = _assert_match(tmp_path, '#include "helper.h"\nint f(void) { return 0; }')
    e = next(e for e in r["edges"] if e["relation"] == "imports")
    assert e["target_file"].endswith("helper.h")
    assert list(e)[-1] == "target_file"


def test_include_in_a_subdirectory_resolves_relative_to_the_including_file(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "dep.h").write_text("int dep(void);\n")
    r = _assert_match(tmp_path, '#include "sub/dep.h"\nint f(void) { return 0; }')
    e = next(e for e in r["edges"] if e["relation"] == "imports")
    assert e["target_file"].endswith("dep.h")


def test_missing_include_resolver_defers_rather_than_guessing(tmp_path):
    """Resolution has no safe default: without the callback the file defers, the
    same rule the JS and Python resolvers follow."""
    p = tmp_path / "sample.c"
    p.write_text('#include "helper.h"\nint f(void) { return 0; }')
    native, reason = kernel.extract_file(str(p), p.read_bytes(), "c",
                                         None, None, None, None)
    assert native is None
    assert reason == "no_c_include_resolver"


# ── calls ────────────────────────────────────────────────────────────────────

def test_direct_call_binds_to_a_same_file_definition(tmp_path):
    r = _assert_match(tmp_path, """
int helper(void) { return 1; }
int caller(void) { return helper(); }
""")
    assert _edges(r, "calls")


def test_member_call_through_a_field_expression(tmp_path):
    r = _assert_match(tmp_path, """
int caller(struct s *p) { return p->fn(); }
""")
    assert any(c["is_member_call"] for c in r["raw_calls"])


# ── deferral, not divergence ─────────────────────────────────────────────────

def test_a_function_like_macro_in_declaration_position_defers(tmp_path):
    """tree-sitter has no preprocessor, so an unknown macro wrapping a
    declaration derails the parse. This is the single biggest source of C
    deferrals -- 31.8% of real files -- and it is a parser limit both sides
    share, not a walker gap."""
    native, reason, _ = _both(
        tmp_path, "HEAP_EXPORT(void heap_init(struct heap* h));\nint f(void) { return 0; }")
    assert native is None
    assert reason == "parse_error"

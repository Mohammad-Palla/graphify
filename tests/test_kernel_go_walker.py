"""Go constructs the corpora cannot reach, pinned against `extract_go`.

`harness/kernel_walker_parity.py` compares the native Go walker against
`extract_go` over 18,694 real files (kubernetes 17,865, prometheus 730, gin 99)
at DIVERGENT 0.

Two things a corpus can never promise it contains, and both are silent when
wrong: the #2779 CASE-COLLISION salt (a file declaring both `Run` and `run`),
and the predeclared-function filter that keeps a method named `append` from
absorbing every builtin call in the corpus. Both are covered directly.

Each test compares the two implementations on one source.
"""
from __future__ import annotations

import contextlib
import json

import pytest

from graphify.extractors import kernel as kseam

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


def _both(tmp_path, source: str, name: str = "sample.go"):
    from graphify.extract import extract_go

    d = tmp_path / "pkgdir"
    d.mkdir(exist_ok=True)
    p = d / name
    p.write_text(source, encoding="utf-8")
    native, reason = kernel.extract_file(str(p), p.read_bytes(), "go",
                                         None, None, None, None)
    with _seam_disabled():
        expected = extract_go(p)
    return native, reason, expected


def _assert_match(tmp_path, source: str, name: str = "sample.go"):
    native, reason, expected = _both(tmp_path, source, name)
    assert native is not None, f"kernel deferred: {reason}"
    assert _canon(native) == _canon(expected)
    return native


def _labels(r):
    return {n.get("label") for n in r["nodes"]}


def _rel(r, relation):
    return [e for e in r["edges"] if e["relation"] == relation]


# ── declarations ─────────────────────────────────────────────────────────────

def test_function_declaration(tmp_path):
    r = _assert_match(tmp_path, "package p\n\nfunc Run() {}\n")
    assert "Run()" in _labels(r)


def test_method_is_owned_by_its_receiver_type(tmp_path):
    """The receiver type node is scoped to the package DIRECTORY, so methods on
    one type across several files of a package share a canonical node."""
    r = _assert_match(tmp_path, """package p

type Server struct{}

func (s *Server) Start() {}
""")
    assert "Server" in _labels(r)
    assert ".Start()" in _labels(r)
    assert _rel(r, "method")


def test_pointer_receiver_star_is_stripped(tmp_path):
    r = _assert_match(tmp_path, """package p

type S struct{}

func (s *S) A() {}
func (s S) B() {}
""")
    assert len(_rel(r, "method")) == 2


def test_struct_field_types_are_referenced(tmp_path):
    r = _assert_match(tmp_path, """package p

type Server struct {
    logger Logger
}
""")
    assert any(e.get("context") == "field" for e in _rel(r, "references"))


def test_an_unnamed_struct_field_is_an_embed(tmp_path):
    """An embed has no `field_identifier`, which is the only thing separating it
    from an ordinary field."""
    r = _assert_match(tmp_path, """package p

type Server struct {
    Logger
}
""")
    assert _rel(r, "embeds")


def test_interface_element_is_an_embed(tmp_path):
    r = _assert_match(tmp_path, """package p

type ReadWriter interface {
    Reader
}
""")
    assert _rel(r, "embeds")


def test_predeclared_types_are_not_references(tmp_path):
    r = _assert_match(tmp_path, """package p

type S struct { n int; s string; ok bool }
""")
    assert not _rel(r, "references")


def test_qualified_type_keeps_its_package_qualifier(tmp_path):
    """`testing.T` stays qualified so the generic stub rewire cannot attach it to
    an unrelated local type or function named `T`."""
    r = _assert_match(tmp_path, """package p

func Helper(t *testing.T) {}
""")
    assert any(n.get("label") == "testing.T" for n in r["nodes"])


def test_generic_type_yields_base_and_arguments(tmp_path):
    r = _assert_match(tmp_path, """package p

type Box struct { items List[Widget] }
""")
    ctxs = {e.get("context") for e in _rel(r, "references")}
    assert {"field", "generic_arg"} <= ctxs


# ── the case-collision salt (#2779) ──────────────────────────────────────────

def test_case_only_collision_keeps_both_symbols(tmp_path):
    """Node ids are casefolded, so `Run` and `run` produce the same id and the
    second was silently dropped -- the unexported half vanished and its call
    sites bound by bare name to a same-named function in ANOTHER package, which
    Go's visibility rules make impossible."""
    r = _assert_match(tmp_path, """package p

func Run() {}
func run() {}
""")
    ids = {n["id"] for n in r["nodes"]}
    assert len(ids) == 3  # file + both functions


def test_the_exported_member_keeps_the_plain_id(tmp_path):
    """Only exported symbols are reachable across packages, so cross-package
    edges target it; keeping its id stable means adding or removing an
    unexported sibling re-points nothing."""
    r = _assert_match(tmp_path, """package p

func Run() {}
func run() {}
""")
    exported = next(n for n in r["nodes"] if n["label"] == "Run()")
    assert exported["id"].endswith("_run")


def test_a_collision_with_no_unique_exported_member_salts_every_member(tmp_path):
    """`Run`/`RUN` has two exported members, so picking one would depend on
    declaration order. Both are salted instead."""
    r = _assert_match(tmp_path, """package p

func Run() {}
func RUN() {}
""")
    ids = {n["id"] for n in r["nodes"]}
    assert len(ids) == 3


def test_methods_on_the_same_type_collide_independently(tmp_path):
    r = _assert_match(tmp_path, """package p

type S struct{}

func (s S) Get() {}
func (s S) get() {}
""")
    assert len(_rel(r, "method")) == 2


# ── calls ────────────────────────────────────────────────────────────────────

def test_same_file_call_binds(tmp_path):
    r = _assert_match(tmp_path, """package p

func helper() {}
func Run() { helper() }
""")
    assert _rel(r, "calls")


def test_a_bare_predeclared_function_is_never_a_call(tmp_path):
    """A method named `append` would otherwise absorb every builtin call in the
    corpus -- 330 phantom inbound edges on one 8.9k-node codebase. Dropped
    before BOTH branches, so it does not reach raw_calls either."""
    r = _assert_match(tmp_path, """package p

func (h *history) append(v int) {}
func Run(s []int) { s = append(s, 1) }
""")
    assert not _rel(r, "calls")
    assert not [c for c in r["raw_calls"] if c["callee"] == "append"]


def test_a_selector_named_like_a_builtin_is_still_a_call(tmp_path):
    """The filter is bare-identifier-only: `h.append(v)` is a genuine call."""
    r = _assert_match(tmp_path, """package p

type h struct{}
func (x *h) append(v int) {}
func Run(a *h) { a.append(1) }
""")
    assert _rel(r, "calls") or r["raw_calls"]


def test_package_qualified_call_records_the_import_path(tmp_path):
    r = _assert_match(tmp_path, """package p

import "fmt"

func Run() { fmt.Println("hi") }
""")
    rc = next(c for c in r["raw_calls"] if c["callee"] == "Println")
    assert rc["import_path"] == "fmt"
    assert rc["receiver"] == "fmt"
    assert rc["is_member_call"] is False


def test_a_receiver_method_call_has_no_import_evidence(tmp_path):
    r = _assert_match(tmp_path, """package p

func Run(s *Server) { s.logger.Log("x") }
""")
    rc = next(c for c in r["raw_calls"] if c["callee"] == "Log")
    assert rc["is_member_call"] is True
    assert rc["import_path"] is None


def test_an_imported_selector_never_resolves_through_a_bare_local(tmp_path):
    """A local `Println` must not capture `fmt.Println`."""
    r = _assert_match(tmp_path, """package p

import "fmt"

func Println() {}
func Run() { fmt.Println("hi") }
""")
    assert not _rel(r, "calls")


# ── imports ──────────────────────────────────────────────────────────────────

def test_import_is_prefixed_so_stdlib_cannot_collide_with_a_local_file(tmp_path):
    r = _assert_match(tmp_path, 'package p\n\nimport "context"\n')
    e = next(e for e in r["edges"] if e["relation"] == "imports_from")
    assert e["target"].startswith("go_pkg_")


def test_aliased_import_tracks_the_alias(tmp_path):
    r = _assert_match(tmp_path, """package p

import f "fmt"

func Run() { f.Println("hi") }
""")
    assert r["go_imports"]["f"] == "fmt"
    rc = next(c for c in r["raw_calls"] if c["callee"] == "Println")
    assert rc["import_path"] == "fmt"


def test_blank_and_dot_imports_are_not_tracked(tmp_path):
    r = _assert_match(tmp_path, """package p

import (
    _ "embed"
    . "math"
)
""")
    assert "_" not in r["go_imports"] and "." not in r["go_imports"]


def test_grouped_and_single_import_forms_both_work(tmp_path):
    r = _assert_match(tmp_path, """package p

import "os"

import (
    "io"
)
""")
    assert set(r["go_imports"]) == {"os", "io"}

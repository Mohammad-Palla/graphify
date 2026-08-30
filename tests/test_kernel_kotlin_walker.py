"""Kotlin constructs the corpora cannot reach, pinned against `_extract_generic`.

`harness/kernel_walker_parity.py` compares the native Kotlin walker against
`_extract_generic` over 4,226 real files (ktor 2,527, kotlinx.coroutines 1,082,
okhttp 617) at DIVERGENT 0.

Kotlin uses more of the engine than any other language -- eleven of the sixteen
hook positions -- and three of them exist only because it reached them:
`on_function_body` (anonymous objects inside a function), `result_extra` (the
declared package), and `Ctx::initializer_nodes` (a call in a property
initializer, which lives in no function at all).

Each test compares the two implementations on one source.
"""
from __future__ import annotations

import contextlib
import json

import pytest

from graphify.extract import _KOTLIN_CONFIG
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


def _both(tmp_path, source: str, name: str = "Sample.kt"):
    p = tmp_path / name
    p.write_text(source, encoding="utf-8")
    native, reason = kernel.extract_file(str(p), p.read_bytes(), "kotlin",
                                         None, None, None, None)
    with _seam_disabled():
        expected = _extract_generic(p, _KOTLIN_CONFIG)
    return native, reason, expected


def _assert_match(tmp_path, source: str, name: str = "Sample.kt"):
    native, reason, expected = _both(tmp_path, source, name)
    assert native is not None, f"kernel deferred: {reason}"
    assert _canon(native) == _canon(expected)
    return native


def _labels(r):
    return {n.get("label") for n in r["nodes"]}


def _rel(r, relation):
    return [e for e in r["edges"] if e["relation"] == relation]


# ── declarations ─────────────────────────────────────────────────────────────

def test_class_and_object_are_both_class_types(tmp_path):
    r = _assert_match(tmp_path, """
class Server
object Registry
""")
    assert {"Server", "Registry"} <= _labels(r)


def test_constructor_invocation_is_inherits_and_a_bare_type_is_implements(tmp_path):
    r = _assert_match(tmp_path, """
class A : Base(), Runner
""")
    assert _rel(r, "inherits")
    assert _rel(r, "implements")


def test_delegation_by_still_emits_implements(tmp_path):
    """`class Foo : Bar by baz` wraps `Bar` in an `explicit_delegation`."""
    r = _assert_match(tmp_path, """
class A(b: Bar) : Bar by b
""")
    assert _rel(r, "implements")


def test_generic_supertype_arguments_are_referenced(tmp_path):
    r = _assert_match(tmp_path, "class A : Holder<Widget>()\n")
    assert any(e.get("context") == "generic_arg" for e in _rel(r, "references"))


def test_enum_entries_become_case_of_nodes(tmp_path):
    r = _assert_match(tmp_path, """
enum class Level { DEBUG, INFO }
""")
    assert len(_rel(r, "case_of")) == 2


def test_a_companion_object_is_transparent(tmp_path):
    """Its members belong to the ENCLOSING class. Recursing into the class_body's
    children directly matters: a bare class_body would default-recurse and drop
    the parent link, leaving companion `fun`s file-level."""
    r = _assert_match(tmp_path, """
class A {
    companion object {
        fun create() {}
    }
}
""")
    assert _rel(r, "method")
    assert ".create()" in _labels(r)


# ── types ────────────────────────────────────────────────────────────────────

def test_property_type_is_referenced_only_inside_a_class(tmp_path):
    """Top-level properties keep their pre-#2565 no-references behaviour."""
    r = _assert_match(tmp_path, """
class A {
    val repo: Repo? = null
}
""")
    assert any(e.get("context") == "field" for e in _rel(r, "references"))


def test_a_top_level_property_emits_no_field_reference(tmp_path):
    r = _assert_match(tmp_path, "val repo: Repo? = null\n")
    assert not [e for e in _rel(r, "references") if e.get("context") == "field"]


def test_builtin_types_are_never_referenced(tmp_path):
    """The Kotlin set is combined with Java's 180 names -- Kotlin compiles to the
    JVM and references `java.*` types freely."""
    r = _assert_match(tmp_path, """
class A {
    val n: Int = 0
    val s: String = ""
    val m: HashMap<String, Int>? = null
}
""")
    assert not _rel(r, "references")


def test_parameter_and_return_types(tmp_path):
    r = _assert_match(tmp_path, """
class A {
    fun build(g: Gadget): Widget? = null
}
""")
    ctxs = {e.get("context") for e in _rel(r, "references")}
    assert {"parameter_type", "return_type"} <= ctxs


def test_a_bare_identifier_inside_a_function_type_is_a_reference(tmp_path):
    """`(value: T) -> R?` -- the walk emits the bare `identifier` it reaches
    FIRST rather than descending past it. Missing that arm made the walker yield
    `T` where the Python yields `value`, on 26 ktor and coroutines files."""
    r = _assert_match(tmp_path, """
class A {
    fun <T, R : Any> go(transform: (value: T) -> R?) {}
}
""")
    assert any(n.get("label") == "value" for n in r["nodes"])


# ── initializers (#2565) and anonymous objects (#2347) ───────────────────────

def test_a_call_in_a_property_initializer_is_collected(tmp_path):
    """`val repo = createRepo()` is a call that lives in no function, so without
    seeding the initializer it produced no edge at all."""
    r = _assert_match(tmp_path, """
fun createRepo() {}

class A {
    val repo = createRepo()
}
""")
    assert _rel(r, "calls")


def test_a_nested_argument_call_in_an_initializer_is_reached(tmp_path):
    """The WHOLE expression is seeded, not just `call_types`, so
    `HttpClient(base())` reaches the inner call too."""
    r = _assert_match(tmp_path, """
fun base() {}

class A {
    val c = HttpClient(base())
}
""")
    assert _rel(r, "calls")


def test_a_delegated_property_initializer_is_collected(tmp_path):
    r = _assert_match(tmp_path, """
fun createRepo() {}

class A {
    val repo by lazy { createRepo() }
}
""")
    assert _rel(r, "calls")


def test_a_literal_initializer_yields_nothing(tmp_path):
    r = _assert_match(tmp_path, "class A {\n    val plain = 5\n}\n")
    assert not _rel(r, "calls")


def test_an_anonymous_object_in_a_function_becomes_a_node(tmp_path):
    """The function branch never recurses into a body and `object_literal` is not
    a class type, so the literal's members -- and every call inside them -- got
    no nodes at all (#2347)."""
    r = _assert_match(tmp_path, """
fun run() {
    val l = object : Listener {
        fun onEvent() {}
    }
}
""")
    assert ".onEvent()" in _labels(r)
    assert _rel(r, "contains")


def test_an_anonymous_object_with_no_supertype_is_labelled_by_line(tmp_path):
    r = _assert_match(tmp_path, """
fun run() {
    val l = object {
        fun onEvent() {}
    }
}
""")
    assert any(lbl and lbl.startswith("object@L") for lbl in _labels(r))


def test_a_local_function_s_literals_are_dropped_by_both(tmp_path):
    """The scan stops at a nested `function_declaration` boundary -- a local
    fun's literals are not the enclosing function's.

    And nothing else picks them up: the function branch never recurses into a
    body, so `inner` is never emitted as a declaration either, and its object
    literal disappears entirely. Pinned as a deliberate limitation rather than
    fixed -- the native walker must reproduce `_extract_generic`, including
    where it gives up."""
    r = _assert_match(tmp_path, """
fun outer() {
    fun inner() {
        val l = object : Listener {
            fun onEvent() {}
        }
    }
}
""")
    assert _labels(r) == {"Sample.kt", "outer()"}


# ── calls ────────────────────────────────────────────────────────────────────

def test_a_bare_call_binds_in_file(tmp_path):
    r = _assert_match(tmp_path, """
fun helper() {}
fun run() { helper() }
""")
    assert _rel(r, "calls")


def test_a_dotted_fqn_call_stamps_a_qualified_prefix(tmp_path):
    """#2550: `com.example.Foo.bar()` is a NESTED navigation chain whose last
    identifier alone rarely matches in-file, so the call was dropped. Three or
    more plain-identifier segments means a real FQN, not `recv.method()`."""
    r = _assert_match(tmp_path, """
fun run() { com.example.Foo.bar() }
""")
    rc = next(c for c in r["raw_calls"] if c["callee"] == "bar")
    assert rc["lang"] == "kotlin"
    assert rc["qualified_prefix"] == "com.example.Foo"


def test_a_two_segment_receiver_call_is_not_a_qualified_fqn(tmp_path):
    r = _assert_match(tmp_path, "fun run(recv: Thing) { recv.method() }\n")
    rc = next(c for c in r["raw_calls"] if c["callee"] == "method")
    assert "qualified_prefix" not in rc


def test_a_non_identifier_chain_segment_bails(tmp_path):
    """A receiver that is a call or an expression must never read as a qualified
    name."""
    r = _assert_match(tmp_path, "fun run() { make().a.b() }\n")
    assert not [c for c in r["raw_calls"] if c.get("qualified_prefix")]


# ── the package, and imports ─────────────────────────────────────────────────

def test_the_declared_package_rides_on_the_result(tmp_path):
    """It qualifies every node in the file; the import-target and qualified-call
    resolvers key their per-package symbol indexes off it (#2526/#2550)."""
    r = _assert_match(tmp_path, "package com.example.app\n\nclass A\n")
    assert r["kotlin_package"] == "com.example.app"


def test_a_file_with_no_package_has_no_key(tmp_path):
    r = _assert_match(tmp_path, "class A\n")
    assert "kotlin_package" not in r


def test_import_stamps_the_target_fqn(tmp_path):
    r = _assert_match(tmp_path, "import a.b.C\n\nclass X\n")
    e = next(e for e in r["edges"] if e["relation"] == "imports")
    assert e["metadata"]["target_fqn"] == "a.b.C"


def test_an_aliased_import_records_the_alias(tmp_path):
    r = _assert_match(tmp_path, "import a.b.C as D\n\nclass X\n")
    e = next(e for e in r["edges"] if e["relation"] == "imports")
    assert e["metadata"]["alias"] == "D"
    assert list(e["metadata"]) == ["target_fqn", "alias"]


def test_a_wildcard_import_emits_nothing(tmp_path):
    """It imports a whole PACKAGE: the last segment is a package name, so a
    symbol-level edge would dangle on or collide with an unrelated node."""
    r = _assert_match(tmp_path, "import a.b.*\n\nclass X\n")
    assert not [e for e in r["edges"] if e["relation"] == "imports"]

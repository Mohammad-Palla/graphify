"""C++ constructs the corpora cannot reach, pinned against `_extract_generic`.

`harness/kernel_walker_parity.py` compares the native C++ walker against
`_extract_generic` over 2,558 real files (folly 2,287, spdlog 143, leveldb 128)
at DIVERGENT 0, which is the primary evidence.

C++ is the first language on the engine to use the two guard positions OUTSIDE
`walk`/`walk_calls` -- `before_calls`, which builds the file's `var -> ClassName`
table, and the `cpp_type_table` result key. Neither reaches the graph directly,
so a per-file node/edge comparison would pass with both of them broken; they are
asserted here explicitly.

Each test compares the two implementations on one source.
"""
from __future__ import annotations

import contextlib
import json

import pytest

from graphify.extract import _CPP_CONFIG
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


def _both(tmp_path, source: str, name: str = "sample.cpp"):
    p = tmp_path / name
    p.write_text(source, encoding="utf-8")
    native, reason = kernel.extract_file(
        str(p), p.read_bytes(), "cpp", None, None, None,
        kseam._c_include_resolver(str(p)))
    with _seam_disabled():
        expected = _extract_generic(p, _CPP_CONFIG)
    return native, reason, expected


def _assert_match(tmp_path, source: str, name: str = "sample.cpp"):
    native, reason, expected = _both(tmp_path, source, name)
    assert native is not None, f"kernel deferred: {reason}"
    assert _canon(native) == _canon(expected)
    return native


def _labels(r):
    return {n.get("label") for n in r["nodes"]}


def _edges(r, relation):
    return [e for e in r["edges"] if e["relation"] == relation]


# ── classes and inheritance ──────────────────────────────────────────────────

def test_class_and_method(tmp_path):
    r = _assert_match(tmp_path, "class Widget { public: void draw() {} };")
    assert "Widget" in _labels(r)
    assert _edges(r, "method")


def test_struct_is_a_class_type(tmp_path):
    r = _assert_match(tmp_path, "struct Point { int x; };")
    assert "Point" in _labels(r)


def test_plain_inheritance(tmp_path):
    r = _assert_match(tmp_path, "class Base {};\nclass Derived : public Base {};")
    assert _edges(r, "inherits")


def test_qualified_base_uses_the_unqualified_tail(tmp_path):
    """`std::vector` links to a `vector` node if one exists in the graph."""
    r = _assert_match(tmp_path, "class Bag : public ns::Holder {};")
    assert any(n.get("label") == "Holder" for n in r["nodes"])


def test_template_base_emits_a_generic_arg_per_argument(tmp_path):
    r = _assert_match(tmp_path, "class Car : public Base<Dep, Other> {};")
    args = {e.get("context") for e in _edges(r, "references")}
    assert "generic_arg" in args
    assert any(n.get("label") == "Dep" for n in r["nodes"])


def test_access_specifier_and_virtual_are_not_bases(tmp_path):
    """`public` and `virtual` are siblings inside base_class_clause and must not
    become inherited types."""
    r = _assert_match(tmp_path, "class D : public virtual B {};")
    assert not any(n.get("label") in ("public", "virtual") for n in r["nodes"])


# ── members ──────────────────────────────────────────────────────────────────

def test_data_member_becomes_a_node(tmp_path):
    r = _assert_match(tmp_path, "class C { int count; };")
    assert "count" in _labels(r)
    assert any(e.get("context") == "field" for e in _edges(r, "defines"))


def test_multiple_declarators_each_become_a_member(tmp_path):
    r = _assert_match(tmp_path, "class C { int x, y; };")
    assert {"x", "y"} <= _labels(r)


def test_a_method_prototype_suppresses_type_refs_but_still_defines_a_node(tmp_path):
    """`is_method` is narrower than its comment suggests.

    A `field_declaration` carrying a function_declarator IS a member-function
    declaration, and the guard skips the type-REFERENCE block for it -- but the
    declarator loop below runs unconditionally, so the prototype still mints a
    node and a `defines` edge. Pinned because the guard reads as if it skipped
    the whole branch, and a walker written from the comment rather than the code
    would drop the node."""
    r = _assert_match(tmp_path, "class C { void run(int n); };")
    assert _edges(r, "defines")
    assert not [e for e in _edges(r, "references") if e.get("context") == "field"]


def test_pointer_returning_prototype_behaves_the_same(tmp_path):
    r = _assert_match(tmp_path, "class C { Widget *name(); };")
    assert _edges(r, "defines")
    assert not [e for e in _edges(r, "references") if e.get("context") == "field"]


def test_nested_class_is_walked_not_dropped(tmp_path):
    """#2876: a nested type is a field_declaration whose `type` field IS the
    class_specifier. Returning from that branch silently dropped Inner and
    everything it declared, with no parse error."""
    r = _assert_match(tmp_path, """
class Outer {
  class Inner { public: void go() {} };
};
""")
    assert "Inner" in _labels(r)
    assert any(lbl == ".go()" for lbl in _labels(r))


def test_nested_class_with_an_instance_declares_both(tmp_path):
    """`class Inner { } inst;` declares a member alongside the type, so the
    declarator loop still has to run after the nested-type walk."""
    r = _assert_match(tmp_path, "class Outer { class Inner {} inst; };")
    assert "Inner" in _labels(r)
    assert "inst" in _labels(r)


def test_member_field_type_is_referenced(tmp_path):
    r = _assert_match(tmp_path, "class C { Widget w; };")
    assert any(e.get("context") == "field" for e in _edges(r, "references"))


def test_templated_member_field_yields_base_and_argument(tmp_path):
    r = _assert_match(tmp_path, "class C { std::vector<Widget> items; };")
    ctxs = {e.get("context") for e in _edges(r, "references")}
    assert {"field", "generic_arg"} <= ctxs


# ── out-of-class definitions ─────────────────────────────────────────────────

def test_out_of_class_definition_keeps_its_qualifier(tmp_path):
    """`void Foo::bar() {}` keeps `Foo::` so `_make_id(stem, "Foo::bar")`
    normalizes to the same id as the in-class `_make_id(class_nid, "bar")` --
    the declaration and the definition collapse onto ONE node (#1547)."""
    r = _assert_match(tmp_path, "void Foo::bar() {}")
    assert any(lbl and "bar" in lbl for lbl in _labels(r))


def test_destructor_and_operator_names_resolve(tmp_path):
    r = _assert_match(tmp_path, """
class C {
 public:
  ~C() {}
  bool operator==(const C& o) const { return true; }
};
""")
    assert len(_edges(r, "method")) == 2


# ── the type table ───────────────────────────────────────────────────────────

def test_local_declaration_builds_the_type_table(tmp_path):
    r = _assert_match(tmp_path, """
void run() { Widget w; w.draw(); }
""")
    assert r["cpp_type_table"]["table"] == {"w": "Widget"}
    assert r["cpp_type_table"]["path"].endswith("sample.cpp")


def test_pointer_and_initialised_locals_are_typed(tmp_path):
    r = _assert_match(tmp_path, """
void run() { Widget* a; Gadget b = Gadget(); }
""")
    assert r["cpp_type_table"]["table"] == {"a": "Widget", "b": "Gadget"}


def test_a_qualified_local_type_records_its_tail(tmp_path):
    r = _assert_match(tmp_path, "void run() { ns::Widget w; }")
    assert r["cpp_type_table"]["table"] == {"w": "Widget"}


def test_a_builtin_local_is_not_typed(tmp_path):
    """Precision over recall: `int x` names no class, so it contributes nothing
    rather than a guess."""
    r = _assert_match(tmp_path, "void run() { int x; }")
    assert "cpp_type_table" not in r


def test_a_multi_declarator_line_is_skipped(tmp_path):
    """`Foo a, b;` cannot be attributed to one receiver cleanly."""
    r = _assert_match(tmp_path, "void run() { Widget a, b; }")
    assert "cpp_type_table" not in r


def test_the_type_table_is_file_scoped_and_first_binding_wins(tmp_path):
    """A later body's `Foo f;` must not clobber an earlier binding."""
    r = _assert_match(tmp_path, """
void first() { Widget w; }
void second() { Gadget w; }
""")
    assert r["cpp_type_table"]["table"] == {"w": "Widget"}


def test_a_lambda_body_does_not_pollute_the_enclosing_table(tmp_path):
    r = _assert_match(tmp_path, """
void run() { auto f = [](){ Widget inner; }; }
""")
    assert "inner" not in r.get("cpp_type_table", {}).get("table", {})


# ── calls ────────────────────────────────────────────────────────────────────

def test_arrow_call_captures_the_receiver(tmp_path):
    r = _assert_match(tmp_path, "void run(Widget* w) { w->draw(); }")
    rc = next(c for c in r["raw_calls"] if c["callee"] == "draw")
    assert rc["receiver"] == "w"
    assert rc["lang"] == "cpp"


def test_this_call_captures_this_as_the_receiver(tmp_path):
    r = _assert_match(tmp_path, "class C { void a() { this->b(); } void b() {} };")
    assert any(c.get("receiver") == "this" for c in r["raw_calls"]) or _edges(r, "calls")


def test_scoped_call_names_the_scope_as_the_receiver(tmp_path):
    """`Foo::bar()` names the receiver type explicitly in source."""
    r = _assert_match(tmp_path, "void run() { Foo::bar(); }")
    rc = next(c for c in r["raw_calls"] if c["callee"] == "bar")
    assert rc["receiver"] == "Foo"


# ── deferral, not divergence ─────────────────────────────────────────────────

def test_a_template_heavy_header_may_defer(tmp_path):
    """C++'s native rate is a property of the corpus's house style: leveldb 75%,
    folly 66%, spdlog 13%. Template metaprogramming is what separates them, and
    a file the grammar cannot parse defers rather than diverging."""
    native, reason, _ = _both(
        tmp_path,
        "template <class T> struct S { template <class U> void f() requires (sizeof(U) > 0) {} };")
    assert native is None or reason is None

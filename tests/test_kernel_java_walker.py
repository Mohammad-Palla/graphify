"""Java constructs the corpora cannot reach, pinned against `_extract_generic`.

`harness/kernel_walker_parity.py` compares the native Java walker against
`_extract_generic` over 3,589 real files (guava 3,275, gson 264, the java corpus
50) at DIVERGENT 0, which is the primary evidence. But a corpus only exercises
what it happens to contain, and the parity run on gson initially passed 200 of
264 files while the walker had a real bug -- it tested `BUILTIN_GLOBALS` with a
binary search over an array that is grouped by language rather than sorted, so
most of the set read as absent. The 64 files that caught it all happened to call
`parseInt`, `set` or `next`; a corpus without those names would have reported a
clean pass on a broken walker.

Each test compares the two implementations on one source.
"""
from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path

import pytest

from graphify.extract import _JAVA_CONFIG
from graphify.extractors import kernel as kseam
from graphify.extractors.engine import _extract_generic

kernel = pytest.importorskip("graphify_kernel", reason="native kernel not built")


def _canon(o) -> str:
    return json.dumps(o, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)


@contextlib.contextmanager
def _seam_disabled():
    """Make `_extract_generic` take the pure-Python path for the control arm.

    Without this the control IS the kernel: `_extract_generic` calls
    `kernel.try_extract` at its top, so now that `java` is in
    `supported_languages()` both arms would be native and every test here would
    pass vacuously.
    """
    original = kseam.try_extract
    kseam.try_extract = lambda *a, **kw: None
    try:
        yield
    finally:
        kseam.try_extract = original


def _both(tmp_path, source: str, name: str = "Sample.java"):
    """Run the native walker and `_extract_generic` on the same file."""
    p = tmp_path / name
    p.write_text(source, encoding="utf-8")
    src = p.read_bytes()
    native, reason = kernel.extract_file(str(p), src, "java", None, None, None)
    with _seam_disabled():
        expected = _extract_generic(p, _JAVA_CONFIG)
    return native, reason, expected


def _assert_match(tmp_path, source: str, name: str = "Sample.java"):
    native, reason, expected = _both(tmp_path, source, name)
    assert native is not None, f"kernel deferred: {reason}"
    assert _canon(native) == _canon(expected)
    return native


# ── the bug the corpus nearly missed ─────────────────────────────────────────

@pytest.mark.parametrize("callee", ["parseInt", "set", "next", "String", "Number"])
def test_builtin_global_callees_are_filtered(tmp_path, callee):
    """`_LANGUAGE_BUILTIN_GLOBALS` is a UNION across languages, so Java code
    calling a name that is builtin in some other language is filtered too.

    The set is not sorted, so a binary search over it silently misses most
    entries -- which is exactly the bug that reported 64 gson files DIVERGENT.
    """
    _assert_match(tmp_path, f"""
class Sample {{
  void run() {{ {callee}("1"); }}
}}
""")


def test_builtin_globals_constant_is_not_sorted():
    """Pins the PROPERTY the linear scan exists for.

    If `BUILTIN_GLOBALS` ever became sorted, a future reader would be right that
    a binary search is fine -- and wrong the moment someone appended to it. This
    test documents that the order is the Python source's grouping, so the
    membership test must not assume otherwise.
    """
    rs = Path(kernel.__file__).parent
    # The constant lives in the Rust source, which is not shipped in the wheel;
    # read it from the repo when present, else skip rather than fail a wheel-only
    # install.
    src = Path("/home/mrx/graphify/graphify-src/graphify-kernel/src/py/helpers.rs")
    if not src.exists():
        pytest.skip("kernel sources not present")
    body = re.search(r"pub const BUILTIN_GLOBALS: &\[&str\] = &\[(.*?)\];",
                     src.read_text(), re.S).group(1)
    vals = re.findall(r'"([^"]*)"', body)
    assert vals != sorted(vals), "BUILTIN_GLOBALS is sorted; a binary search would now be valid"


# ── declaration shapes ───────────────────────────────────────────────────────

def test_record_components_emit_field_references(tmp_path):
    """`record_declaration` reads its components from `parameters`, and the
    reference line is the COMPONENT's line, not the record's."""
    _assert_match(tmp_path, """
record Point(
  Widget x,
  Gadget y
) {}
""")


def test_spread_parameter_record_component(tmp_path):
    """A varargs record component has no `type` field; the type is the first
    named child that is neither `modifiers` nor `variable_declarator`."""
    _assert_match(tmp_path, "record Args(String name, Widget... rest) {}\n")


def test_enum_constant_with_anonymous_body(tmp_path):
    """`MONDAY { void greet(){} }` -- the constant's body members attach to the
    CONSTANT, not the enum, so `_java_extra_walk` re-walks them with const_nid."""
    _assert_match(tmp_path, """
enum Day {
  MONDAY { void greet() { helper(); } },
  TUESDAY;
  void helper() {}
}
""")


def test_annotation_type_element_preserves_qualified_names(tmp_path):
    """`annotation_type_element_declaration` is the ONE Java site that passes
    `preserve_qualified=True`, so a dotted return type keeps its qualifier."""
    _assert_match(tmp_path, """
@interface Marker {
  com.example.Widget value();
  String name() default "x";
}
""")


def test_inline_qualified_annotation_keeps_dotted_name(tmp_path):
    """#2504: `@org.pkg.Foo` keeps its full dotted name so a bare same-named
    local class cannot absorb it."""
    _assert_match(tmp_path, """
@org.pkg.Foo
@Bar
class Sample {
  @com.example.Inject void run() {}
}
class Foo {}
""")


def test_annotation_class_literal_refs(tmp_path):
    """Type names used as class literals inside annotation arguments."""
    _assert_match(tmp_path, """
@Uses({Widget.class, com.example.Gadget.class})
class Sample {}
""")


def test_interface_extends_and_implements(tmp_path):
    _assert_match(tmp_path, """
interface A extends B, C {}
class D extends Base implements A, java.io.Serializable {}
""")


def test_generic_bases_emit_generic_arg_references(tmp_path):
    """`_emit_java_parent_type` links the first `type` role and turns every
    `generic_arg` into a `references` edge."""
    _assert_match(tmp_path, """
class Sample extends Base<Widget, Gadget> implements Handler<Event> {}
""")


def test_type_parameters_are_not_emitted_as_references(tmp_path):
    """`T` is a type parameter in scope, not a type reference."""
    _assert_match(tmp_path, """
class Box<T> {
  T value;
  <U> U convert(T input, U other) { return other; }
}
""")


def test_builtin_types_are_not_emitted(tmp_path):
    _assert_match(tmp_path, """
class Sample {
  String name;
  Integer count;
  Widget custom;
  void run(Object o, Exception e, Widget w) {}
}
""")


def test_nested_class_is_contained_by_its_enclosing_type(tmp_path):
    """#2040: a nested type is contained by the enclosing type, not the file."""
    _assert_match(tmp_path, """
class Outer {
  static class Inner {
    void go() {}
  }
}
""")


def test_field_declaration_without_type_field_falls_through(tmp_path):
    """The Java field branch returns only when the declaration HAS a type; the
    Python code returns from inside `if type_node is not None`."""
    _assert_match(tmp_path, """
class Sample {
  Widget a, b;
  int primitive;
}
""")


# ── call pass ────────────────────────────────────────────────────────────────

def test_every_member_call_defers_to_receiver_resolution(tmp_path):
    """`_java_defer` is unconditional: any `a.b()` becomes a raw_call, never a
    `calls` edge, regardless of the receiver's case."""
    _assert_match(tmp_path, """
class Sample {
  Widget field;
  void run(Gadget param) {
    field.doThing();
    param.doOther();
    this.local();
    this.field.deep();
    Helper.staticCall();
    local();
  }
  void local() {}
}
""")


def test_receiver_type_is_stamped_from_fields_and_params(tmp_path):
    _assert_match(tmp_path, """
class Sample {
  Widget field;
  void run(Gadget param) {
    Thing localVar = null;
    field.a();
    param.b();
    localVar.c();
  }
}
""")


def test_conflicting_local_declaration_makes_a_receiver_ambiguous(tmp_path):
    """A local whose type conflicts with the field of the same name is dropped
    entirely -- raw call facts carry no lexical scope, so a wrong type would
    bind the call to the wrong method."""
    _assert_match(tmp_path, """
class Sample {
  Widget shared;
  void run() {
    Gadget shared = null;
    shared.doThing();
  }
}
""")


def test_lambda_parameter_shadowing_is_ambiguous(tmp_path):
    _assert_match(tmp_path, """
class Sample {
  Widget item;
  void run(java.util.List<Gadget> xs) {
    xs.forEach(item -> item.use());
  }
}
""")


def test_object_creation_reads_the_type_field(tmp_path):
    """#1373: `new Foo(...)` puts the constructed type in `type`, not `name`,
    and a generic/qualified form is reduced to its simple name."""
    _assert_match(tmp_path, """
class Sample {
  void run() {
    new Widget();
    new java.util.ArrayList<String>();
    new Outer.Inner();
  }
}
class Widget {}
""")


def test_anonymous_inner_class_methods_are_their_own_callers(tmp_path):
    """A `method_declaration` inside `new Runnable() { ... }` is a call boundary,
    so its calls attribute to it and not to the enclosing method."""
    _assert_match(tmp_path, """
class Sample {
  void run() {
    Runnable r = new Runnable() {
      public void run() { inner(); }
    };
  }
  void inner() {}
}
""")


def test_constructor_declaration_has_no_return_type(tmp_path):
    """A constructor has no `type` field, so the return-type branch is a no-op
    -- and must not read the class name as a return type."""
    _assert_match(tmp_path, """
class Sample {
  Widget dep;
  Sample(Gadget g) { g.use(); }
}
""")


# ── imports ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("stmt", [
    "import java.util.List;",
    "import java.util.*;",
    "import static org.junit.Assert.assertEquals;",
    "import static java.util.Arrays.*;",
    "import Bare;",
])
def test_import_forms(tmp_path, stmt):
    _assert_match(tmp_path, f"{stmt}\nclass Sample {{}}\n")


def test_static_import_emits_exactly_one_edge(tmp_path):
    """A static import names a member, and only the enclosing path is imported.

    NOT a test of the handler's `break`: tree-sitter-java gives an
    `import_declaration` at most one identifier child, so `break` and `continue`
    are indistinguishable here -- an injection sweep confirmed replacing one with
    the other changes nothing. This pins the edge COUNT, which is what reaches
    the graph.
    """
    native = _assert_match(tmp_path, "import static java.util.Arrays.asList;\nclass S {}\n")
    imports = [e for e in native["edges"] if e["relation"] == "imports"]
    assert len(imports) == 1


# ── deferral ─────────────────────────────────────────────────────────────────

def test_parse_error_defers(tmp_path):
    native, reason, _ = _both(tmp_path, "class Sample { void run( { } }\n")
    assert native is None
    assert reason == "parse_error"


def test_non_ascii_identifier_defers_rather_than_guessing(tmp_path):
    """The id recipe's Unicode fixpoint is not reproduced in Rust, so a
    non-ASCII name must defer rather than mint a possibly-different id."""
    native, reason, _ = _both(tmp_path, "class Ünicode {\n  void gruß() {}\n}\n")
    if native is not None:
        # ASCII-only ids happened to be derivable; then it must still match.
        _assert_match(tmp_path, "class Ünicode {\n  void gruß() {}\n}\n")
    else:
        assert reason == "non_ascii_id"

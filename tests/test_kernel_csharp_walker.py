"""C# constructs the corpora cannot reach, pinned against `_extract_generic`.

`harness/kernel_walker_parity.py` compares the native C# walker against
`_extract_generic` over 7,177 real files (EF Core 5,762, Newtonsoft.Json 945,
eShopOnWeb 254, Serilog 216) at DIVERGENT 0, which is the primary evidence. This
file covers what a corpus sample cannot promise it contains.

C# is the first language on the shared engine to emit a `metadata` block at all,
and metadata is the failure mode a corpus run is WORST at catching: the parity
harness canonicalizes with `sort_keys=True`, so a wrong key ORDER inside a
metadata dict survives every per-file comparison and only appears much later as a
whole-graph byte diff. Several tests below therefore assert the emitted key order
directly rather than only comparing the two implementations.

Each test compares the two implementations on one source.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest

from graphify.extract import _CSHARP_CONFIG
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
    `kernel.try_extract` at its top, so now that `csharp` is in
    `supported_languages()` both arms would be native and every test here would
    pass vacuously.
    """
    original = kseam.try_extract
    kseam.try_extract = lambda *a, **kw: None
    try:
        yield
    finally:
        kseam.try_extract = original


def _both(tmp_path, source: str, name: str = "Sample.cs"):
    p = tmp_path / name
    p.write_text(source, encoding="utf-8")
    src = p.read_bytes()
    native, reason = kernel.extract_file(str(p), src, "csharp", None, None, None)
    with _seam_disabled():
        expected = _extract_generic(p, _CSHARP_CONFIG)
    return native, reason, expected


def _assert_match(tmp_path, source: str, name: str = "Sample.cs"):
    native, reason, expected = _both(tmp_path, source, name)
    assert native is not None, f"kernel deferred: {reason}"
    assert _canon(native) == _canon(expected)
    return native


def _node(result, label):
    return next(n for n in result["nodes"] if n.get("label") == label)


def _edges(result, relation):
    return [e for e in result["edges"] if e["relation"] == relation]


# ── namespaces ───────────────────────────────────────────────────────────────

def test_block_namespace(tmp_path):
    """A block namespace qualifies the ids of the types it nests."""
    r = _assert_match(tmp_path, """
namespace Acme.Widgets {
    public class Gadget {
        public void Run() { }
    }
}
""")
    ns = _node(r, "Acme.Widgets")
    assert ns["type"] == "namespace"
    assert ns["metadata"]["kind"] == "csharp_namespace"
    assert ns["id"].startswith("csharp_namespace:")
    # The namespace node itself carries `namespace` but NOT `scope_chain`.
    assert "scope_chain" not in ns["metadata"]
    assert _node(r, "Gadget")["metadata"]["namespace"] == "Acme.Widgets"


def test_file_scoped_namespace_applies_to_the_rest_of_the_file(tmp_path):
    """`namespace Foo;` has no body: the push is never popped, so every later
    sibling declaration is qualified by it."""
    r = _assert_match(tmp_path, """
namespace Acme.Core;

public class First { }
public class Second { }
""")
    assert _node(r, "First")["metadata"]["namespace"] == "Acme.Core"
    assert _node(r, "Second")["metadata"]["namespace"] == "Acme.Core"


def test_nested_block_namespaces_join_with_a_dot(tmp_path):
    r = _assert_match(tmp_path, """
namespace Outer {
    namespace Inner {
        public class Deep { }
    }
}
""")
    assert _node(r, "Deep")["metadata"]["namespace"] == "Outer.Inner"


def test_namespace_scope_is_popped_after_its_body(tmp_path):
    """A type after a CLOSED block namespace must not inherit it -- the pop is
    the whole reason the block and file-scoped forms are separate branches."""
    r = _assert_match(tmp_path, """
namespace Boxed { public class Inside { } }
public class Outside { }
""")
    assert _node(r, "Inside")["metadata"]["namespace"] == "Boxed"
    assert "metadata" not in _node(r, "Outside")


def test_scope_chain_is_stamped_on_members_inside_a_namespace(tmp_path):
    r = _assert_match(tmp_path, """
namespace Acme {
    public class Thing { public void Go() { } }
}
""")
    chain = _node(r, "Thing")["metadata"]["scope_chain"]
    assert isinstance(chain, list) and len(chain) == 1 and chain[0].startswith("s")


# ── class metadata ───────────────────────────────────────────────────────────

def test_partial_class_is_stamped(tmp_path):
    r = _assert_match(tmp_path, "public partial class Split { }")
    assert _node(r, "Split")["metadata"]["is_partial"] is True


def test_nested_type_is_stamped_and_contained_by_its_outer_type(tmp_path):
    r = _assert_match(tmp_path, """
public class Outer {
    public class Inner { }
}
""")
    inner = _node(r, "Inner")
    assert inner["metadata"]["is_nested_type"] is True
    outer_id = _node(r, "Outer")["id"]
    assert any(e["source"] == outer_id and e["target"] == inner["id"]
               for e in _edges(r, "contains"))


def test_nested_partial_class_orders_is_nested_type_before_is_partial(tmp_path):
    """Key ORDER inside metadata reaches the exported JSON; `sort_keys=True` in
    the parity harness would not see it reversed."""
    r = _assert_match(tmp_path, """
public class Outer {
    public partial class Inner { }
}
""")
    assert list(_node(r, "Inner")["metadata"]) == [
        "is_nested_type", "is_partial",
    ]


# ── base types ───────────────────────────────────────────────────────────────

def test_i_prefix_convention_classifies_a_base_as_implements(tmp_path):
    r = _assert_match(tmp_path, "public class Impl : IThing { }")
    assert [e["relation"] for e in r["edges"] if e["relation"] in ("implements", "inherits")] == ["implements"]


def test_declared_interface_classifies_as_implements_without_the_i_prefix(tmp_path):
    """The pre-scan is the point: `Runnable` has no `I`, but it is declared as an
    interface in THIS file, so the base is `implements`."""
    r = _assert_match(tmp_path, """
public interface Runnable { }
public class Job : Runnable { }
""")
    assert any(e["relation"] == "implements" for e in r["edges"])


def test_plain_base_class_is_inherits(tmp_path):
    r = _assert_match(tmp_path, """
public class Animal { }
public class Dog : Animal { }
""")
    assert any(e["relation"] == "inherits" for e in r["edges"])


def test_generic_base_emits_a_generic_arg_reference_per_type_argument(tmp_path):
    r = _assert_match(tmp_path, "public class Repo : BaseRepo<Widget, Gadget> { }")
    args = {e["metadata"]["ref_token"] for e in _edges(r, "references")
            if e.get("context") == "generic_arg"}
    assert {"Widget", "Gadget"} <= args


def test_a_bare_type_parameter_is_never_a_base_reference(tmp_path):
    """`class Box<T> : Holder<T>` must not mint a node for `T`."""
    r = _assert_match(tmp_path, "public class Box<T> : Holder<T> { }")
    assert not any(n.get("label") == "T" for n in r["nodes"])


def test_qualified_base_records_its_qualifier(tmp_path):
    r = _assert_match(tmp_path, "public class Impl : Acme.Core.IThing { }")
    e = next(e for e in r["edges"] if e["relation"] == "implements")
    assert e["metadata"]["qualified"] is True
    assert e["metadata"]["ref_qualifier"] == "Acme.Core"
    # Order is `ref_token`, `qualified`, `ref_qualifier`.
    assert list(e["metadata"]) == ["ref_token", "qualified", "ref_qualifier"]


# ── members ──────────────────────────────────────────────────────────────────

def test_property_becomes_a_node_and_a_field_is_not(tmp_path):
    """The id recipe casefolds and strips leading underscores, so `_count` and
    `Count` collide; only the property is emitted (#3006)."""
    r = _assert_match(tmp_path, """
public class Counter {
    private Widget _count;
    public Widget Count { get; set; }
}
""")
    assert any(n.get("label") == "Count" for n in r["nodes"])
    assert [e["relation"] for e in r["edges"] if e["relation"] == "defines"] == ["defines"]


def test_field_generic_argument_is_referenced_not_just_the_outer_type(tmp_path):
    r = _assert_match(tmp_path, """
public class Holder {
    private Box<Widget> _box;
}
""")
    tokens = {(e["metadata"]["ref_token"], e.get("context"))
              for e in _edges(r, "references")}
    assert ("Box", "field") in tokens
    assert ("Widget", "generic_arg") in tokens


def test_enum_members_become_case_of_nodes(tmp_path):
    r = _assert_match(tmp_path, """
public enum Level { Debug, Info, Warning }
""")
    assert {e["relation"] for e in r["edges"] if e["relation"] == "case_of"} == {"case_of"}
    assert len(_edges(r, "case_of")) == 3


def test_case_colliding_enum_members_keep_only_the_first(tmp_path):
    """C# is case-sensitive but the id recipe casefolds: `Value` and `value`
    normalize to one id, and the FIRST declaration keeps it."""
    r = _assert_match(tmp_path, "public enum E { Value, value }")
    assert len(_edges(r, "case_of")) == 1


def test_record_and_struct_are_class_types(tmp_path):
    r = _assert_match(tmp_path, """
public record Point(int X, int Y);
public struct Size { }
""")
    labels = {n.get("label") for n in r["nodes"]}
    assert {"Point", "Size"} <= labels


def test_primary_constructor_parameters_reference_their_types(tmp_path):
    """C# 12 declares dependencies on the type declaration itself, where neither
    the field nor the property handler ever sees them."""
    r = _assert_match(tmp_path, "public class Service(IRepo repo, Widget w) { }")
    tokens = {e["metadata"]["ref_token"] for e in _edges(r, "references")}
    assert {"IRepo", "Widget"} <= tokens


def test_method_attribute_becomes_an_attribute_reference(tmp_path):
    r = _assert_match(tmp_path, """
public class Tests {
    [Fact]
    public void Works() { }
}
""")
    assert any(e.get("context") == "attribute"
               and e["metadata"]["ref_token"] == "Fact"
               for e in _edges(r, "references"))


def test_method_parameter_and_return_types_are_referenced(tmp_path):
    r = _assert_match(tmp_path, """
public class Svc {
    public Widget Build(Gadget g) { return null; }
}
""")
    ctxs = {(e["metadata"]["ref_token"], e.get("context")) for e in _edges(r, "references")}
    assert ("Gadget", "parameter_type") in ctxs
    assert ("Widget", "return_type") in ctxs


def test_preproc_wrapper_keeps_the_enclosing_class(tmp_path):
    """A method guarded by `#if` sits inside a `preproc_*` node. Dropping the
    parent there makes it look file-level (#2631)."""
    r = _assert_match(tmp_path, """
public class Guarded {
#if NET6_0
    public void Modern() { }
#endif
}
""")
    cls = _node(r, "Guarded")["id"]
    assert any(e["source"] == cls and e["relation"] == "method" for e in r["edges"])


# ── using directives ─────────────────────────────────────────────────────────

def test_plain_using_records_kind_and_target(tmp_path):
    r = _assert_match(tmp_path, "using System.Text;\npublic class C { }")
    e = next(e for e in r["edges"] if e["relation"] == "imports")
    assert e["metadata"]["using_kind"] == "namespace"
    assert e["metadata"]["target_fqn"] == "System.Text"
    assert e["metadata"]["scope_kind"] == "file"
    assert list(e["metadata"]) == ["using_kind", "target_fqn", "scope_kind"]


def test_static_using(tmp_path):
    r = _assert_match(tmp_path, "using static System.Math;\npublic class C { }")
    e = next(e for e in r["edges"] if e["relation"] == "imports")
    assert e["metadata"]["using_kind"] == "static"
    assert e["metadata"]["target_fqn"] == "System.Math"


def test_alias_using_keeps_the_alias_and_escapes_the_target(tmp_path):
    """`sanitize_metadata` HTML-escapes every metadata string, and a generic
    alias target is the one place C# routinely puts `<` and `>` in one."""
    r = _assert_match(tmp_path,
                      "using L = System.Collections.Generic.List<int>;\npublic class C { }")
    e = next(e for e in r["edges"] if e["relation"] == "imports")
    assert e["metadata"]["using_kind"] == "alias"
    assert e["metadata"]["alias"] == "L"
    assert e["metadata"]["target_fqn"] == "System.Collections.Generic.List&lt;int&gt;"
    assert list(e["metadata"]) == ["using_kind", "alias", "target_fqn", "scope_kind"]


def test_global_using_is_stripped_before_the_using_test(tmp_path):
    r = _assert_match(tmp_path, "global using System.Linq;\npublic class C { }")
    assert any(e["metadata"]["target_fqn"] == "System.Linq"
               for e in r["edges"] if e["relation"] == "imports")


def test_using_inside_a_namespace_records_the_enclosing_scope(tmp_path):
    r = _assert_match(tmp_path, """
namespace Acme {
    using System.Text;
    public class C { }
}
""")
    e = next(e for e in r["edges"] if e["relation"] == "imports")
    assert e["metadata"]["scope_kind"] == "namespace"
    assert e["metadata"]["scope_id"].startswith("s")


# ── calls and receiver typing ────────────────────────────────────────────────

def test_member_call_defers_with_the_receiver_type_from_a_field(tmp_path):
    r = _assert_match(tmp_path, """
public class Caller {
    private Store _store;
    public void Go() { _store.Save(); }
}
""")
    rc = next(c for c in r["raw_calls"] if c["callee"] == "Save")
    assert rc["lang"] == "csharp"
    assert rc["receiver"] == "_store"
    assert rc["receiver_type"] == "Store"


def test_receiver_type_is_resolved_at_the_call_offset_not_method_wide(tmp_path):
    """#2472: an untypable `out var` binding in one block must not wipe the
    typed binding that covers a call in a sibling block."""
    r = _assert_match(tmp_path, """
public class Caller {
    public void Go(bool flag) {
        if (flag) { Store s = new Store(); s.Save(); }
        else { Cache.TryGet(out var s); }
    }
}
""")
    rc = next(c for c in r["raw_calls"] if c["callee"] == "Save")
    assert rc["receiver_type"] == "Store"


def test_lambda_parameter_shadows_only_inside_the_lambda(tmp_path):
    r = _assert_match(tmp_path, """
public class Caller {
    private Store s;
    public void Go() {
        Run(x => x.Ping());
        s.Save();
    }
}
""")
    save = next(c for c in r["raw_calls"] if c["callee"] == "Save")
    assert save["receiver_type"] == "Store"
    ping = next(c for c in r["raw_calls"] if c["callee"] == "Ping")
    assert "receiver_type" not in ping


def test_a_local_disagreeing_with_a_field_drops_the_name_entirely(tmp_path):
    r = _assert_match(tmp_path, """
public class Caller {
    private Store thing;
    public void Go() { Cache thing = new Cache(); thing.Save(); }
}
""")
    rc = next(c for c in r["raw_calls"] if c["callee"] == "Save")
    assert "receiver_type" not in rc


def test_pattern_binding_types_the_receiver(tmp_path):
    """#2346: `is Leaf lf` is a declaration_pattern with `type` + `name`."""
    r = _assert_match(tmp_path, """
public class Caller {
    public void Go(object o) { if (o is Leaf lf) { lf.Ping(); } }
}
""")
    rc = next(c for c in r["raw_calls"] if c["callee"] == "Ping")
    assert rc["receiver_type"] == "Leaf"


def test_new_expression_links_the_constructed_type(tmp_path):
    r = _assert_match(tmp_path, """
public class Caller {
    public void Go() { var x = new Widget(); }
}
""")
    assert any(c["callee"] == "Widget" for c in r["raw_calls"])


def test_qualified_new_keeps_the_namespace_prefix(tmp_path):
    r = _assert_match(tmp_path, """
public class Caller {
    public void Go() { var x = new Acme.Core.Cache(); }
}
""")
    rc = next(c for c in r["raw_calls"] if c["callee"] == "Cache")
    assert rc["qualified_prefix"] == "Acme.Core"
    assert list(rc)[-2:] == ["lang", "qualified_prefix"]


def test_call_site_type_arguments_become_generic_arg_references(tmp_path):
    """#2911: the DI shape. Without this the type arguments never became nodes
    and the dependency edges were silently erased."""
    r = _assert_match(tmp_path, """
public class Startup {
    public void Configure(IServiceCollection services) {
        services.AddScoped<ISvc, Impl>();
    }
}
""")
    tokens = {e["metadata"]["ref_token"] for e in _edges(r, "references")
              if e.get("context") == "generic_arg"}
    assert {"ISvc", "Impl"} <= tokens


def test_static_generic_call_type_arguments_are_referenced(tmp_path):
    r = _assert_match(tmp_path, """
public class Caller {
    public void Go() { Make<Widget>(); }
}
""")
    assert any(e["metadata"]["ref_token"] == "Widget"
               for e in _edges(r, "references") if e.get("context") == "generic_arg")


def test_base_call_receiver_is_captured(tmp_path):
    r = _assert_match(tmp_path, """
public class Child : Parent {
    public void Go() { base.Setup(); }
}
""")
    rc = next(c for c in r["raw_calls"] if c["callee"] == "Setup")
    assert rc["receiver"] == "base"


def test_this_field_call_types_from_the_field_table(tmp_path):
    r = _assert_match(tmp_path, """
public class Caller {
    private Store store;
    public void Go() { this.store.Save(); }
}
""")
    rc = next(c for c in r["raw_calls"] if c["callee"] == "Save")
    assert rc["receiver"] == "store"
    assert rc["receiver_type"] == "Store"


# ── deferral, not divergence ─────────────────────────────────────────────────

def test_a_parse_error_defers_rather_than_guessing(tmp_path):
    """Python attaches a `parse_errors` block and keeps going; its recovery is
    authoritative, so the kernel hands the file back rather than reproducing it.
    `async` used as an ordinary identifier in EXPRESSION position is the single
    biggest source of this in real C# -- `await base.M(async)`, `M(async, x)`,
    `async ? a : b` -- because the grammar commits to an async-lambda parse. In
    declaration position (`void M(bool async)`) it parses fine, which is why the
    repro below is the call and not the parameter."""
    native, reason, _ = _both(
        tmp_path,
        "public class C { public async Task M(bool async) { await base.M(async); } }")
    assert native is None
    assert reason == "parse_error"


def test_a_non_ascii_identifier_defers(tmp_path):
    native, reason, _ = _both(tmp_path, "public class Café { }")
    assert native is None
    assert reason == "non_ascii_id"

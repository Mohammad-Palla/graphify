"""PHP constructs the corpora cannot reach, pinned against `_extract_generic`.

`harness/kernel_walker_parity.py` compares the native PHP walker against
`_extract_generic` over 14,495 real files (symfony 11,306, laravel 3,052,
guzzle 137) at DIVERGENT 0.

PHP is the first language whose `LanguageConfig` uses the four FRAMEWORK fields
-- `helper_fn_names`, `container_bind_methods`, `event_listener_properties`,
`static_prop_types` -- which encode Laravel conventions, not language syntax.
Those four blocks emit edges with a `confidence_score` key and no `context`, a
shape nothing else on the engine produces, and two of them resolve their target
through a CASE-INSENSITIVE label map. Guzzle and Symfony contain none of it, so
this file is where they are covered.

Each test compares the two implementations on one source.
"""
from __future__ import annotations

import contextlib
import json

import pytest

from graphify.extract import _PHP_CONFIG
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


def _both(tmp_path, source: str, name: str = "Sample.php"):
    p = tmp_path / name
    p.write_text(source, encoding="utf-8")
    native, reason = kernel.extract_file(str(p), p.read_bytes(), "php",
                                         None, None, None, None)
    with _seam_disabled():
        expected = _extract_generic(p, _PHP_CONFIG)
    return native, reason, expected


def _assert_match(tmp_path, source: str, name: str = "Sample.php"):
    native, reason, expected = _both(tmp_path, source, name)
    assert native is not None, f"kernel deferred: {reason}"
    assert _canon(native) == _canon(expected)
    return native


def _rel(r, relation):
    return [e for e in r["edges"] if e["relation"] == relation]


def _labels(r):
    return {n.get("label") for n in r["nodes"]}


# ── classes ──────────────────────────────────────────────────────────────────

def test_extends_implements_and_use(tmp_path):
    """Three different clause shapes, three different relations, and the edge
    line is the CLAUSE's -- not the class's, unlike every other language here."""
    r = _assert_match(tmp_path, """<?php
class Repo extends Base implements Contract {
    use Loggable;
}
""")
    assert _rel(r, "inherits")
    assert _rel(r, "implements")
    assert _rel(r, "mixes_in")


def test_namespaced_base_uses_the_unqualified_tail(tmp_path):
    """PHP namespaces separate with a BACKSLASH, so `\\App\\Models\\User`
    reduces to `User` -- not the forward-slash or dot every other language uses."""
    r = _assert_match(tmp_path, "<?php\nclass A extends \\App\\Models\\User {}\n")
    assert any(n.get("label") == "User" for n in r["nodes"])


def test_use_statement_imports_the_tail(tmp_path):
    r = _assert_match(tmp_path, "<?php\nuse App\\Models\\Order;\nclass A {}\n")
    e = next(e for e in r["edges"] if e["relation"] == "imports")
    assert e["context"] == "import"


# ── typed members and promotion ──────────────────────────────────────────────

def test_typed_property_references_its_type(tmp_path):
    r = _assert_match(tmp_path, "<?php\nclass A { private Widget $w; }\n")
    assert any(e.get("context") == "field" for e in _rel(r, "references"))


def test_union_property_type_references_every_member(tmp_path):
    r = _assert_match(tmp_path, "<?php\nclass A { private Widget|Gadget $w; }\n")
    tgts = {e["target"] for e in _rel(r, "references")}
    assert len(tgts) >= 2


def test_parameter_and_return_types(tmp_path):
    """PHP has no `return_type` field; the type node is found POSITIONALLY,
    after `formal_parameters`."""
    r = _assert_match(tmp_path, """<?php
class A { public function build(Gadget $g): Widget { } }
""")
    ctxs = {e.get("context") for e in _rel(r, "references")}
    assert {"parameter_type", "return_type"} <= ctxs


def test_promoted_constructor_parameter_is_both_a_parameter_and_a_field(tmp_path):
    """PHP 8 promotion declares a class field AND a parameter, and the two edges
    are emitted INTERLEAVED per referenced type -- not in two passes. Two passes
    yield the same edges in a different order, which `sort_keys=True` hides from
    the per-file harness and the exported JSON does not: 33 Symfony files
    diverged on exactly that."""
    r = _assert_match(tmp_path, """<?php
class A { public function __construct(private Repo $repo) {} }
""")
    ctxs = [e.get("context") for e in _rel(r, "references")]
    assert "parameter_type" in ctxs and "field" in ctxs
    assert ctxs.index("parameter_type") < ctxs.index("field")


# ── calls ────────────────────────────────────────────────────────────────────

def test_static_call_names_the_class_not_the_method(tmp_path):
    """`Helper::format()` resolves to `Helper`. The Python's choice, kept: the
    class is the thing the graph can resolve."""
    r = _assert_match(tmp_path, """<?php
class Helper { public static function format() {} }
class B { public function go() { Helper::format(); } }
""")
    assert _rel(r, "calls") or any(c["callee"] == "Helper" for c in r["raw_calls"])


def test_member_call_captures_the_method_name(tmp_path):
    r = _assert_match(tmp_path, """<?php
class B { public function go($obj) { $obj->save(); } }
""")
    assert any(c["callee"] == "save" and c["is_member_call"] for c in r["raw_calls"])


# ── the four framework blocks ────────────────────────────────────────────────

def test_config_helper_emits_a_uses_config_edge(tmp_path):
    """`config('app.debug')` links to the `app` config FILE, matched
    case-insensitively and with a `.php` suffix fallback."""
    r = _assert_match(tmp_path, """<?php
class App {}
class B { public function go() { config('app.debug'); } }
""")
    e = next(iter(_rel(r, "uses_config")), None)
    assert e is not None
    assert e["confidence_score"] == 1.0
    assert "context" not in e
    assert list(e) == ["source", "target", "relation", "confidence",
                       "confidence_score", "source_file", "source_location", "weight"]


def test_config_helper_with_a_non_string_argument_emits_nothing(tmp_path):
    r = _assert_match(tmp_path, """<?php
class App {}
class B { public function go($k) { config($k); } }
""")
    assert not _rel(r, "uses_config")


def test_container_binding_emits_bound_to(tmp_path):
    r = _assert_match(tmp_path, """<?php
class Contract {}
class Impl {}
class Provider {
    public function register() { $this->app->bind(Contract::class, Impl::class); }
}
""")
    assert _rel(r, "bound_to")


def test_container_binding_scans_past_a_non_class_first_child(tmp_path):
    """The Python breaks out of an argument's children only WHEN it finds a
    `::class`; breaking unconditionally dropped Laravel's
    ContextualAttributeBindingTest edge."""
    r = _assert_match(tmp_path, """<?php
class Contract {}
class Impl {}
class Provider {
    public function register() { $this->app->singleton(Contract::class, Impl::class); }
}
""")
    assert _rel(r, "bound_to")


def test_a_one_argument_binding_emits_nothing(tmp_path):
    """Exactly two class arguments, or no edge."""
    r = _assert_match(tmp_path, """<?php
class Impl {}
class Provider { public function register() { $this->app->bind(Impl::class); } }
""")
    assert not _rel(r, "bound_to")


def test_event_listener_array_emits_listened_by(tmp_path):
    """`$listen = [Event::class => [Listener::class]]` names classes that may be
    declared LATER, so the edges are harvested during the walk and resolved
    after the call pass."""
    r = _assert_match(tmp_path, """<?php
class Provider {
    protected $listen = [
        OrderPlaced::class => [SendEmail::class],
    ];
}
class OrderPlaced {}
class SendEmail {}
""")
    e = next(iter(_rel(r, "listened_by")), None)
    assert e is not None
    assert e["confidence_score"] == 1.0


def test_a_non_listener_property_falls_through_to_the_type_branch(tmp_path):
    """The listener harvest consumes the node only when it MATCHED; `$other`
    must still reach the property type-reference branch."""
    r = _assert_match(tmp_path, """<?php
class Provider { protected Widget $other; }
""")
    assert any(e.get("context") == "field" for e in _rel(r, "references"))


def test_static_property_access_emits_uses_static_prop(tmp_path):
    """`Foo::$bar` is not a call, so the edge is emitted at the body level of
    the call pass rather than in the call branch."""
    r = _assert_match(tmp_path, """<?php
class Config { public static $debug; }
class B { public function go() { return Config::$debug; } }
""")
    assert _rel(r, "uses_static_prop")


def test_class_constant_access_emits_references_constant(tmp_path):
    r = _assert_match(tmp_path, """<?php
class Limits { const MAX = 10; }
class B { public function go() { return Limits::MAX; } }
""")
    assert _rel(r, "references_constant")


def test_framework_edges_are_deduplicated(tmp_path):
    r = _assert_match(tmp_path, """<?php
class Limits { const MAX = 10; const MIN = 1; }
class B { public function go() { return Limits::MAX + Limits::MIN; } }
""")
    assert len(_rel(r, "references_constant")) == 1

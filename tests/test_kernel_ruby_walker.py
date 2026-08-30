"""Ruby constructs the corpora cannot reach, pinned against `_extract_generic`.

`harness/kernel_walker_parity.py` compares the native Ruby walker against
`_extract_generic` over 3,605 real files (rails 3,458, sinatra 147) at
DIVERGENT 0, 100% native.

Ruby is the only language on the engine that QUALIFIES a declaration's label
against its enclosing scope, the only one with a `sanitize_symbol_name_fn`, and
the only one whose receiver table is per-CALLER. All three are invisible in a
node count and silent when wrong.

Each test compares the two implementations on one source.
"""
from __future__ import annotations

import contextlib
import json

import pytest

from graphify.extract import _RUBY_CONFIG
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


def _both(tmp_path, source: str, name: str = "sample.rb"):
    p = tmp_path / name
    p.write_text(source, encoding="utf-8")
    native, reason = kernel.extract_file(str(p), p.read_bytes(), "ruby",
                                         None, None, None, None)
    with _seam_disabled():
        expected = _extract_generic(p, _RUBY_CONFIG)
    return native, reason, expected


def _assert_match(tmp_path, source: str, name: str = "sample.rb"):
    native, reason, expected = _both(tmp_path, source, name)
    assert native is not None, f"kernel deferred: {reason}"
    assert _canon(native) == _canon(expected)
    return native


def _labels(r):
    return {n.get("label") for n in r["nodes"]}


def _rel(r, relation):
    return [e for e in r["edges"] if e["relation"] == relation]


# ── scope qualification (#2302) ──────────────────────────────────────────────

def test_nested_class_label_is_qualified(tmp_path):
    r = _assert_match(tmp_path, """
module Billing
  class Invoice
  end
end
""")
    assert "Billing::Invoice" in _labels(r)


def test_compact_declaration_converges_on_the_same_label(tmp_path):
    """`class Billing::Invoice` splits into the same two segments, so both
    declaration styles produce ONE label."""
    r = _assert_match(tmp_path, "class Billing::Invoice\nend\n")
    assert "Billing::Invoice" in _labels(r)


def test_the_scope_is_popped_after_the_body(tmp_path):
    r = _assert_match(tmp_path, """
module A
  class Inside
  end
end
class Outside
end
""")
    assert "A::Inside" in _labels(r)
    assert "Outside" in _labels(r)


def test_a_module_gets_a_node_like_a_class(tmp_path):
    """Without `module` in class_types a utility module produced no node and its
    methods hung off the file with dot-less labels (#1640)."""
    r = _assert_match(tmp_path, """
module Util
  def self.run
  end
end
""")
    assert "Util" in _labels(r)


# ── method-name sanitization (#3077) ─────────────────────────────────────────

def test_bang_method_survives(tmp_path):
    """A trailing `!` would normalize away entirely, taking the method with it."""
    r = _assert_match(tmp_path, """
class C
  def save!
  end
end
""")
    assert ".save!()" in _labels(r)
    assert any(e["target"].endswith("_bang") for e in _rel(r, "method"))


def test_predicate_and_setter_methods_survive(tmp_path):
    r = _assert_match(tmp_path, """
class C
  def valid?
  end
  def name=(v)
  end
end
""")
    assert len(_rel(r, "method")) == 2
    tgts = " ".join(e["target"] for e in _rel(r, "method"))
    assert "_pred" in tgts and "_eq" in tgts


def test_the_label_keeps_the_raw_name(tmp_path):
    """The ID is sanitized; the LABEL is not."""
    r = _assert_match(tmp_path, "class C\n  def save!\n  end\nend\n")
    assert ".save!()" in _labels(r)


def test_bang_and_plain_siblings_do_not_collide(tmp_path):
    r = _assert_match(tmp_path, """
class C
  def save
  end
  def save!
  end
end
""")
    assert len(_rel(r, "method")) == 2


# ── inheritance and mixins ───────────────────────────────────────────────────

def test_superclass_emits_inherits(tmp_path):
    r = _assert_match(tmp_path, "class Dog < Animal\nend\n")
    assert _rel(r, "inherits")


def test_namespaced_superclass_uses_the_last_segment(tmp_path):
    r = _assert_match(tmp_path, "class Dog < Zoo::Animal\nend\n")
    assert _rel(r, "inherits")


def test_include_becomes_a_mixin_raw_call(tmp_path):
    """The module usually lives in another file, so resolution is deferred to
    the cross-file resolver rather than emitted as an edge (#1668)."""
    r = _assert_match(tmp_path, """
class C
  include Loggable
end
""")
    rc = next(c for c in r["raw_calls"] if c.get("is_mixin"))
    assert rc["callee"] == "Loggable"


def test_a_namespaced_mixin_keeps_its_full_path(tmp_path):
    """Truncating `ActiveSupport::Concern` to `Concern` fabricated edges to any
    local module of that name (#2302)."""
    r = _assert_match(tmp_path, """
class C
  include ActiveSupport::Concern
end
""")
    assert any(c["callee"] == "ActiveSupport::Concern" for c in r["raw_calls"])


def test_extend_self_is_not_a_mixin(tmp_path):
    """Only bare or namespaced CONSTANT arguments count."""
    r = _assert_match(tmp_path, """
module M
  extend self
end
""")
    assert not [c for c in r["raw_calls"] if c.get("is_mixin")]


def test_mixin_raw_calls_come_last(tmp_path):
    """Mixins are found in the class body, before the call pass has produced any
    raw_calls, and the Python appends them at the very end."""
    r = _assert_match(tmp_path, """
class C
  include Loggable
  def run
    helper_method
  end
end
""")
    assert r["raw_calls"][-1].get("is_mixin") is True


# ── class factories (#1640) ──────────────────────────────────────────────────

def test_struct_new_defines_a_class(tmp_path):
    r = _assert_match(tmp_path, "Point = Struct.new(:x, :y)\n")
    assert "Point" in _labels(r)


def test_data_define_defines_a_class(tmp_path):
    r = _assert_match(tmp_path, "Coord = Data.define(:lat)\n")
    assert "Coord" in _labels(r)


def test_class_new_with_a_superclass_emits_inherits(tmp_path):
    r = _assert_match(tmp_path, "Sub = Class.new(Base)\n")
    assert _rel(r, "inherits")


def test_factory_block_methods_attach_to_the_class(tmp_path):
    """Without descending into the block, the default recurse resets the parent
    and every method hangs off the file with a dot-less label."""
    r = _assert_match(tmp_path, """
Point = Struct.new(:x) do
  def norm
  end
end
""")
    assert _rel(r, "method")
    assert ".norm()" in _labels(r)


def test_a_factory_const_is_qualified_by_its_module(tmp_path):
    r = _assert_match(tmp_path, """
module Billing
  Invoice = Struct.new(:total)
end
""")
    assert "Billing::Invoice" in _labels(r)


def test_an_unrelated_constant_assignment_is_not_a_class(tmp_path):
    r = _assert_match(tmp_path, "LIMIT = 10\n")
    assert "LIMIT" not in _labels(r)


# ── calls and receiver typing ────────────────────────────────────────────────

def test_receiver_and_method_are_direct_fields(tmp_path):
    """Ruby's `call` node has no intermediate accessor node, so the generic
    accessor model does not apply."""
    r = _assert_match(tmp_path, """
class C
  def run(p)
    p.process
  end
end
""")
    rc = next(c for c in r["raw_calls"] if c["callee"] == "process")
    assert rc["receiver"] == "p"


def test_a_namespaced_receiver_keeps_its_whole_path(tmp_path):
    """Truncating bound `ActiveRecord::Base.transaction` to whatever single
    class named `Base` the corpus defined -- the god-node guard catches an
    ambiguous match, not a unique-but-wrong one (#3078)."""
    r = _assert_match(tmp_path, """
class C
  def run
    ActiveRecord::Base.transaction
  end
end
""")
    rc = next(c for c in r["raw_calls"] if c["callee"] == "transaction")
    assert rc["receiver"] == "ActiveRecord::Base"


def test_a_single_constant_new_binding_types_the_receiver(tmp_path):
    r = _assert_match(tmp_path, """
class C
  def run
    p = Processor.new
    p.process
  end
end
""")
    rc = next(c for c in r["raw_calls"] if c["callee"] == "process")
    assert rc["receiver_type"] == "Processor"


def test_a_reassigned_variable_is_ambiguous_and_carries_no_type(tmp_path):
    """A variable assigned to two different classes maps to None -- PRESENT with
    no type, which is different from being absent."""
    r = _assert_match(tmp_path, """
class C
  def run
    p = Processor.new
    p = Other.new
    p.process
  end
end
""")
    rc = next(c for c in r["raw_calls"] if c["callee"] == "process")
    assert rc["receiver_type"] is None


def test_a_variable_assigned_to_something_untypable_is_poisoned(tmp_path):
    r = _assert_match(tmp_path, """
class C
  def run
    p = Processor.new
    p = compute
    p.process
  end
end
""")
    rc = next(c for c in r["raw_calls"] if c["callee"] == "process")
    assert rc["receiver_type"] is None


def test_the_binding_table_is_per_caller(tmp_path):
    """Java's receiver table is per-method and C++'s per-file; Ruby's is
    per-CALLER, so a binding in one method must not type a receiver in another."""
    r = _assert_match(tmp_path, """
class C
  def a
    p = Processor.new
  end
  def b(p)
    p.process
  end
end
""")
    rc = next(c for c in r["raw_calls"] if c["callee"] == "process")
    assert rc["receiver_type"] is None

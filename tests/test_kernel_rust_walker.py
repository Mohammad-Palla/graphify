"""Rust constructs the corpora cannot reach, pinned against `extract_rust`.

`harness/kernel_walker_parity.py` compares the native Rust walker against
`extract_rust` over 3,901 real files (cargo 1,373, bun 1,527, tokio 793,
serde 208) at DIVERGENT 0.

Rust's shape is unlike anything else on the kernel: an `impl` block is not a
declaration, so methods hang off the type it implements FOR, and the same type
can be reopened in several blocks. Two behaviours are also easy to get subtly
wrong and impossible to see from a node count -- the FIRST collected type ref of
a trait bound is the supertrait and the rest are generic arguments, and a
`scoped_identifier` callee is allowed an in-file match but never a cross-file
one (#908).

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


def _both(tmp_path, source: str, name: str = "sample.rs"):
    from graphify.extract import extract_rust

    p = tmp_path / name
    p.write_text(source, encoding="utf-8")
    native, reason = kernel.extract_file(str(p), p.read_bytes(), "rust",
                                         None, None, None, None)
    with _seam_disabled():
        expected = extract_rust(p)
    return native, reason, expected


def _assert_match(tmp_path, source: str, name: str = "sample.rs"):
    native, reason, expected = _both(tmp_path, source, name)
    assert native is not None, f"kernel deferred: {reason}"
    assert _canon(native) == _canon(expected)
    return native


def _labels(r):
    return {n.get("label") for n in r["nodes"]}


def _rel(r, relation):
    return [e for e in r["edges"] if e["relation"] == relation]


# ── items ────────────────────────────────────────────────────────────────────

def test_free_function(tmp_path):
    r = _assert_match(tmp_path, "fn run() {}\n")
    assert "run()" in _labels(r)


def test_struct_enum_and_trait_are_all_items(tmp_path):
    r = _assert_match(tmp_path, """
struct S;
enum E { A }
trait T {}
""")
    assert {"S", "E", "T"} <= _labels(r)


def test_impl_methods_hang_off_the_type(tmp_path):
    """An `impl` block is not a declaration: the method belongs to the type it
    implements FOR, and the type can be reopened in several blocks."""
    r = _assert_match(tmp_path, """
struct Server;
impl Server { fn start(&self) {} }
impl Server { fn stop(&self) {} }
""")
    assert len(_rel(r, "method")) == 2
    assert ".start()" in _labels(r) and ".stop()" in _labels(r)


def test_trait_impl_emits_implements(tmp_path):
    r = _assert_match(tmp_path, """
struct S;
trait Runner {}
impl Runner for S {}
""")
    assert _rel(r, "implements")


def test_a_generic_trait_impl_references_its_arguments(tmp_path):
    """The FIRST collected ref is the trait; every later one is a generic arg."""
    r = _assert_match(tmp_path, """
struct S;
impl From<Widget> for S {}
""")
    assert _rel(r, "implements")
    assert any(e.get("context") == "generic_arg" for e in _rel(r, "references"))


def test_supertrait_bound_is_inherits(tmp_path):
    r = _assert_match(tmp_path, "trait Child: Parent {}\n")
    assert _rel(r, "inherits")


def test_struct_field_types_are_referenced(tmp_path):
    r = _assert_match(tmp_path, "struct S { logger: Logger }\n")
    assert any(e.get("context") == "field" for e in _rel(r, "references"))


def test_tuple_struct_positional_types_are_referenced(tmp_path):
    """A tuple struct nests its types directly under
    `ordered_field_declaration_list`, with no `field_declaration` wrapper."""
    r = _assert_match(tmp_path, "struct Wrapper(pub Logger, Config);\n")
    tgts = {e["target"] for e in _rel(r, "references")}
    assert len(tgts) >= 2


def test_tuple_enum_variant_payload_is_referenced(tmp_path):
    r = _assert_match(tmp_path, "enum Event { Click(Logger) }\n")
    assert _rel(r, "references")


def test_struct_enum_variant_payload_is_referenced(tmp_path):
    r = _assert_match(tmp_path, "enum Event { Resize { size: Dim } }\n")
    assert _rel(r, "references")


def test_primitive_types_produce_no_reference(tmp_path):
    r = _assert_match(tmp_path, "struct S { n: u32, ok: bool }\n")
    assert not _rel(r, "references")


def test_scoped_type_collapses_to_its_last_segment(tmp_path):
    """`std::io::Error` -> `Error`. The opposite of Go's `qualified_type`, which
    keeps the qualifier -- so the two walkers must not be assumed to agree."""
    r = _assert_match(tmp_path, "fn run() -> std::io::Error { todo!() }\n")
    assert any(n.get("label") == "Error" for n in r["nodes"])


def test_generic_type_yields_base_and_arguments(tmp_path):
    r = _assert_match(tmp_path, "struct S { items: Vec<Widget> }\n")
    ctxs = {e.get("context") for e in _rel(r, "references")}
    assert {"field", "generic_arg"} <= ctxs


def test_parameter_and_return_types(tmp_path):
    r = _assert_match(tmp_path, "fn build(g: Gadget) -> Widget { todo!() }\n")
    ctxs = {e.get("context") for e in _rel(r, "references")}
    assert {"parameter_type", "return_type"} <= ctxs


# ── calls ────────────────────────────────────────────────────────────────────

def test_same_file_call_binds(tmp_path):
    r = _assert_match(tmp_path, """
fn helper() {}
fn run() { helper(); }
""")
    assert _rel(r, "calls")


def test_a_scoped_call_binds_in_file_but_never_cross_file(tmp_path):
    """`Type::method()` gets an in-file EXTRACTED match, but no raw_call: a bare
    last-segment lookup ignores crate boundaries and invents INFERRED edges
    (#908)."""
    r = _assert_match(tmp_path, """
fn run() { Foo::helper(); }
""")
    assert not [c for c in r["raw_calls"] if c["callee"] == "helper"]


def test_a_scoped_call_to_a_local_name_still_binds(tmp_path):
    r = _assert_match(tmp_path, """
fn helper() {}
fn run() { Foo::helper(); }
""")
    assert _rel(r, "calls")


def test_blocklisted_trait_method_names_are_not_raw_calls(tmp_path):
    """`new`, `clone`, `unwrap` and friends are so common across traits that a
    bare cross-file lookup on them is noise. Rust-local, deliberately: putting
    them in the shared builtin set would break Go and eleven other languages."""
    r = _assert_match(tmp_path, """
fn run(x: Thing) { x.clone(); x.unwrap(); }
""")
    assert not r["raw_calls"]


def test_a_non_blocklisted_method_call_is_a_raw_call(tmp_path):
    r = _assert_match(tmp_path, "fn run(x: Thing) { x.frobnicate(); }\n")
    rc = next(c for c in r["raw_calls"] if c["callee"] == "frobnicate")
    assert rc["is_member_call"] is True


def test_the_blocklist_is_case_insensitive(tmp_path):
    r = _assert_match(tmp_path, "fn run(x: Thing) { x.Clone(); }\n")
    assert not r["raw_calls"]


# ── use declarations ─────────────────────────────────────────────────────────

def test_use_declaration_imports_the_last_segment(tmp_path):
    r = _assert_match(tmp_path, "use std::collections::HashMap;\n")
    e = next(e for e in r["edges"] if e["relation"] == "imports_from")
    assert e["context"] == "import"


def test_a_braced_use_takes_the_prefix_before_the_brace(tmp_path):
    r = _assert_match(tmp_path, "use std::io::{Read, Write};\n")
    assert [e for e in r["edges"] if e["relation"] == "imports_from"]


def test_a_glob_use_strips_the_star_and_the_separator(tmp_path):
    """`split("{")[0]` then rstrip `:`, `*`, `:` -- three strips in that order,
    which is not the same as stripping the union of those characters."""
    r = _assert_match(tmp_path, "use std::prelude::*;\n")
    assert [e for e in r["edges"] if e["relation"] == "imports_from"]

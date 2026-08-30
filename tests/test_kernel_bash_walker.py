"""Bash constructs the corpora cannot reach, pinned against `extract_bash`.

`harness/kernel_walker_parity.py` compares the native Bash walker against
`extract_bash` over 415 real files across seven corpora at DIVERGENT 0.

Bash is the first BESPOKE walker on the kernel: it does not go through
`_extract_generic`, it stamps `{"language": "bash", "kind": …}` on every node,
and it invents an `__entry` node that owns every top-level call. It is also the
first walker to defer by SCOPE rather than by parser limit -- `source` and `.sh`
invocation resolve through the filesystem -- so the deferral boundary itself is
behaviour worth pinning: a walker that quietly handled a `source` would emit a
graph missing every `bash_sources` entry, and no per-file node/edge comparison
would show it.

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


def _both(tmp_path, source: str, name: str = "sample.sh"):
    from graphify.extract import extract_bash

    p = tmp_path / name
    p.write_text(source, encoding="utf-8")
    native, reason = kernel.extract_file(str(p), p.read_bytes(), "bash",
                                         None, None, None, None)
    with _seam_disabled():
        expected = extract_bash(p)
    return native, reason, expected


def _assert_match(tmp_path, source: str, name: str = "sample.sh"):
    native, reason, expected = _both(tmp_path, source, name)
    assert native is not None, f"kernel deferred: {reason}"
    assert _canon(native) == _canon(expected)
    return native


def _labels(r):
    return {n.get("label") for n in r["nodes"]}


def _rel(r, relation):
    return [e for e in r["edges"] if e["relation"] == relation]


# ── the file and its entrypoint ──────────────────────────────────────────────

def test_file_and_entry_nodes(tmp_path):
    """Two nodes for an empty script: the file, and the synthetic entrypoint that
    owns top-level calls. The `__entry` suffix cannot collide with a function id
    because `file_nid` is path-derived and `_make_id(stem, name)` never is."""
    r = _assert_match(tmp_path, "echo hi\n")
    assert "sample.sh" in _labels(r)
    assert "sample.sh script" in _labels(r)
    assert _rel(r, "contains")


def test_every_node_carries_language_and_kind_metadata(tmp_path):
    r = _assert_match(tmp_path, "greet() { echo hi; }\n")
    kinds = {n["metadata"]["kind"] for n in r["nodes"]}
    assert {"file", "bash_entrypoint", "bash_function"} <= kinds
    assert all(n["metadata"]["language"] == "bash" for n in r["nodes"])


# ── functions and calls ──────────────────────────────────────────────────────

def test_function_definition_and_call(tmp_path):
    r = _assert_match(tmp_path, """
greet() { echo hi; }
greet
""")
    assert "greet()" in _labels(r)
    assert _rel(r, "defines")
    assert _rel(r, "calls")


def test_top_level_call_is_attributed_to_the_entry_node(tmp_path):
    r = _assert_match(tmp_path, """
greet() { echo hi; }
greet
""")
    entry = next(n["id"] for n in r["nodes"] if n["label"].endswith(" script"))
    assert any(e["source"] == entry for e in _rel(r, "calls"))


def test_function_calling_a_function(tmp_path):
    r = _assert_match(tmp_path, """
helper() { echo x; }
main() { helper; }
""")
    assert len(_rel(r, "calls")) == 1


def test_a_nested_definition_is_not_attributed_to_its_parent(tmp_path):
    """A nested function's body is walked separately, so its calls belong to it."""
    r = _assert_match(tmp_path, """
helper() { echo x; }
outer() {
  inner() { helper; }
  inner
}
""")
    assert "inner()" in _labels(r)
    srcs = {e["source"] for e in _rel(r, "calls")}
    inner = next(n["id"] for n in r["nodes"] if n["label"] == "inner()")
    assert inner in srcs


def test_an_undefined_callee_becomes_a_raw_call(tmp_path):
    """It may be a function from a sourced library. A genuine external command
    matches nothing in the cross-file resolver, so this cannot over-connect."""
    r = _assert_match(tmp_path, """
main() { some_library_fn; }
""")
    assert any(c["callee"] == "some_library_fn" for c in r["raw_calls"])
    assert all(c["language"] == "bash" for c in r["raw_calls"])


def test_source_and_runner_names_are_never_raw_calls(tmp_path):
    r = _assert_match(tmp_path, "main() { bash; }\n")
    assert not [c for c in r["raw_calls"] if c["callee"] == "bash"]


def test_raw_calls_are_deduplicated_per_caller(tmp_path):
    r = _assert_match(tmp_path, """
main() { some_fn; some_fn; }
""")
    assert len([c for c in r["raw_calls"] if c["callee"] == "some_fn"]) == 1


# ── the expansion rules ──────────────────────────────────────────────────────

def test_a_bare_command_substitution_is_not_a_call(tmp_path):
    """`$(build)` exposes `build` as a child command whose NAME token has no
    metacharacters -- only the parent does, so token filtering alone misses it."""
    r = _assert_match(tmp_path, """
build() { echo x; }
$(build)
""")
    assert not _rel(r, "calls")


def test_value_capture_from_a_substitution_is_a_call(tmp_path):
    """`x=$(fn)` is a real invocation (#2978), unlike a bare `$(fn)`."""
    r = _assert_match(tmp_path, """
build() { echo x; }
main() { local x=$(build); }
""")
    assert _rel(r, "calls")


def test_a_process_substitution_is_never_a_call(tmp_path):
    r = _assert_match(tmp_path, """
build() { echo x; }
main() { diff <(build) /dev/null; }
""")
    assert not _rel(r, "calls")


def test_a_name_with_a_metacharacter_is_rejected(tmp_path):
    r = _assert_match(tmp_path, 'main() { "$CMD"; }\n')
    assert not r["raw_calls"]


# ── declarations ─────────────────────────────────────────────────────────────

def test_top_level_export_defines_a_variable_node(tmp_path):
    r = _assert_match(tmp_path, "export ROOT=/tmp\n")
    assert "ROOT" in _labels(r)
    assert _rel(r, "defines")


def test_a_declaration_inside_a_function_is_not_a_file_variable(tmp_path):
    """The branch is gated on the parent being `program`."""
    r = _assert_match(tmp_path, "main() { export INNER=1; }\n")
    assert "INNER" not in _labels(r)


# ── the deferral boundary ────────────────────────────────────────────────────

def test_a_source_command_defers(tmp_path):
    """Resolution touches the filesystem and feeds `bash_sources`, which a
    cross-file resolver consumes. Handling it natively without reproducing the
    path policy would emit a graph silently missing those entries."""
    native, reason, _ = _both(tmp_path, 'source ./lib.sh\necho hi\n')
    assert native is None
    assert reason == "bash_source_command"


def test_a_dot_source_defers_too(tmp_path):
    native, reason, _ = _both(tmp_path, '. ./lib.sh\n')
    assert native is None
    assert reason == "bash_source_command"


def test_a_source_shadowed_by_a_local_function_does_not_defer(tmp_path):
    """A script may define its own `source`. The pre-pass collects every defined
    name first, so the shadow is known regardless of definition order."""
    r = _assert_match(tmp_path, """
source() { echo shadowed; }
source foo
""")
    assert "source()" in _labels(r)


def test_a_bare_source_with_no_argument_does_not_defer(tmp_path):
    """The Python only reaches the filesystem when there IS a path argument."""
    r = _assert_match(tmp_path, "source\n")
    assert r["nodes"]


def test_a_script_invocation_defers(tmp_path):
    native, reason, _ = _both(tmp_path, "bash ./deploy.sh\n")
    assert native is None
    assert reason == "bash_script_invocation"


def test_a_non_sh_command_does_not_defer(tmp_path):
    r = _assert_match(tmp_path, "make build\n")
    assert r["nodes"]


def test_bash_sources_is_always_present_and_empty_when_native(tmp_path):
    """A file that would populate it defers, so an empty list is the whole truth
    for every natively handled file."""
    r = _assert_match(tmp_path, "greet() { echo hi; }\n")
    assert r["bash_sources"] == []

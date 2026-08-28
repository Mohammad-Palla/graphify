"""The native-kernel routing seam must be invisible until a language is gated.

These tests pin the *failure* behaviour, not the happy path, because that is
where this project's pooling work actually went wrong: every one of the five
defects found in the earlier parallel-extraction code produced silent wrong
output rather than an exception. A kernel that half-loads, or that answers for a
language it was not gated for, must degrade to the Python path -- never guess.

None of these tests require the compiled kernel to be present; a fake module is
injected so they behave identically on a machine that never ran maturin.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from graphify.extractors import kernel


@pytest.fixture(autouse=True)
def _fresh_kernel():
    kernel.reset_for_test()
    yield
    kernel.reset_for_test()


def _fake_kernel(*, languages=(), tree_sitter_ok=True, extract=None, version="0.1.0"):
    mod = types.ModuleType("graphify_kernel")
    mod.version = lambda: version
    mod.supported_languages = lambda: list(languages)
    mod.selftest = lambda: {"version": version, "languages": list(languages),
                            "tree_sitter_ok": tree_sitter_ok}
    mod.extract_file = extract if extract is not None else (lambda p, s, l: None)
    return mod


def _install(monkeypatch, mod):
    monkeypatch.setitem(sys.modules, "graphify_kernel", mod)


class _Cfg:
    ts_module = "tree_sitter_typescript"
    ts_language_fn = "language_typescript"


# ── the kill switch ──────────────────────────────────────────────────────────

def test_env_zero_disables_without_importing(monkeypatch):
    """GRAPHIFY_KERNEL=0 must short-circuit before the import is even attempted."""
    exploding = types.ModuleType("graphify_kernel")

    def _boom():
        raise AssertionError("kernel must not be consulted when disabled")

    exploding.selftest = _boom
    _install(monkeypatch, exploding)
    monkeypatch.setenv("GRAPHIFY_KERNEL", "0")

    assert kernel.status() == "disabled_by_env"
    assert kernel.enabled_languages() == set()
    assert kernel.try_extract(Path("a.ts"), _Cfg()) is None


# ── fail-open on every load failure ──────────────────────────────────────────

def test_missing_kernel_is_not_an_error(monkeypatch):
    monkeypatch.delenv("GRAPHIFY_KERNEL", raising=False)
    monkeypatch.setitem(sys.modules, "graphify_kernel", None)  # import -> ImportError
    assert kernel.status().startswith("import_failed")
    assert kernel.try_extract(Path("a.ts"), _Cfg()) is None


def test_broken_grammar_disables_the_kernel(monkeypatch):
    """Imports fine, but tree-sitter is not actually linked: must NOT be used.

    This is the dangerous case -- a kernel that loads but is subtly wrong would
    emit plausible garbage, so the selftest gates it rather than the import.
    """
    monkeypatch.delenv("GRAPHIFY_KERNEL", raising=False)
    _install(monkeypatch, _fake_kernel(languages=("typescript",), tree_sitter_ok=False))
    assert kernel.status() == "selftest_grammar_failed"
    assert kernel.enabled_languages() == set()


def test_selftest_raising_disables_the_kernel(monkeypatch):
    monkeypatch.delenv("GRAPHIFY_KERNEL", raising=False)
    mod = _fake_kernel(languages=("typescript",))

    def _raise():
        raise RuntimeError("abi mismatch")

    mod.selftest = _raise
    _install(monkeypatch, mod)
    assert kernel.status() == "selftest_raised:RuntimeError"
    assert kernel.try_extract(Path("a.ts"), _Cfg()) is None


# ── routing ──────────────────────────────────────────────────────────────────

def test_grammar_pair_is_the_routing_key():
    """ts and tsx share a module and differ only by language fn -- both must resolve."""
    from graphify.extract import _JS_CONFIG, _PYTHON_CONFIG, _TS_CONFIG, _TSX_CONFIG
    assert kernel.language_for(_TS_CONFIG) == "typescript"
    assert kernel.language_for(_TSX_CONFIG) == "tsx"
    assert kernel.language_for(_JS_CONFIG) == "javascript"
    assert kernel.language_for(_PYTHON_CONFIG) == "python"


def test_ungated_language_defers_and_is_counted(monkeypatch):
    """A grammar the kernel knows but has NOT been parity-gated for must defer."""
    monkeypatch.delenv("GRAPHIFY_KERNEL", raising=False)
    _install(monkeypatch, _fake_kernel(languages=()))  # nothing gated
    assert kernel.try_extract(Path("a.ts"), _Cfg()) is None
    assert kernel.drain_counts() == {"defer:unsupported_language": 1}


def test_unknown_grammar_never_reaches_the_kernel(monkeypatch):
    """A language with no mapping must not even trigger the module load."""
    monkeypatch.delenv("GRAPHIFY_KERNEL", raising=False)

    class _Go:
        ts_module = "tree_sitter_go"
        ts_language_fn = "language"

    assert kernel.language_for(_Go()) is None
    assert kernel.try_extract(Path("a.go"), _Go()) is None
    assert kernel.drain_counts() == {}


def test_walker_exception_is_a_deferral_not_a_build_failure(monkeypatch, tmp_path):
    monkeypatch.delenv("GRAPHIFY_KERNEL", raising=False)

    def _explode(path, source, language):
        raise ValueError("walker bug")

    _install(monkeypatch, _fake_kernel(languages=("typescript",), extract=_explode))
    f = tmp_path / "a.ts"
    f.write_bytes(b"const x = 1;")
    assert kernel.try_extract(f, _Cfg()) is None
    assert kernel.drain_counts() == {"defer:kernel_raised:ValueError": 1}


def test_native_result_is_returned_and_counted(monkeypatch, tmp_path):
    monkeypatch.delenv("GRAPHIFY_KERNEL", raising=False)
    payload = {"nodes": [{"id": "n"}], "edges": []}
    _install(monkeypatch, _fake_kernel(languages=("typescript",),
                                       extract=lambda p, s, l: payload))
    f = tmp_path / "a.ts"
    f.write_bytes(b"const x = 1;")
    assert kernel.try_extract(f, _Cfg()) is payload
    assert kernel.drain_counts() == {"native:typescript": 1}


def test_source_override_is_honoured(monkeypatch, tmp_path):
    """Vue SFCs parse embedded <script> bytes; the kernel must not re-read the file."""
    monkeypatch.delenv("GRAPHIFY_KERNEL", raising=False)
    seen: dict = {}

    def _capture(path, source, language):
        seen["source"] = source
        return {"nodes": [], "edges": []}

    _install(monkeypatch, _fake_kernel(languages=("typescript",), extract=_capture))
    f = tmp_path / "a.vue"
    f.write_bytes(b"<template>WRAPPER</template>")
    kernel.try_extract(f, _Cfg(), source_override=b"const x = 1;")
    assert seen["source"] == b"const x = 1;"


# ── the invariant that matters most ──────────────────────────────────────────

def test_extraction_is_identical_with_and_without_the_kernel(tmp_path):
    """With no language gated, the real kernel must change nothing at all."""
    from graphify.extract import _get_extractor, _safe_extract

    src = tmp_path / "sample.ts"
    src.write_text(
        "import { helper } from './helper';\n"
        "export class Widget {\n"
        "  render(): number { return helper(1); }\n"
        "}\n"
        "export function make(): Widget { return new Widget(); }\n"
    )
    extractor = _get_extractor(src)

    kernel.reset_for_test()
    with_kernel = _safe_extract(extractor, src)

    import os
    os.environ["GRAPHIFY_KERNEL"] = "0"
    try:
        kernel.reset_for_test()
        without_kernel = _safe_extract(extractor, src)
    finally:
        os.environ.pop("GRAPHIFY_KERNEL", None)
        kernel.reset_for_test()

    assert with_kernel == without_kernel
    assert with_kernel["nodes"], "fixture should produce nodes"

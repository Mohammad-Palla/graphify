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


def _py_grammar_fingerprint(language):
    """The fingerprint the real Python-side grammar for `language` produces.

    A fake kernel has to report this, or the load-time grammar check will
    (correctly) drop the language -- which is the point of the check: the two
    sides must be parsing the same grammar. Tests that want the OPPOSITE are
    explicit about it (see `test_grammar_mismatch_drops_the_language`).
    """
    import importlib

    from tree_sitter import Language

    from graphify.extractors.kernel import _LANGUAGE_TO_GRAMMAR

    module_name, fn_name = _LANGUAGE_TO_GRAMMAR[language]
    lang = Language(getattr(importlib.import_module(module_name), fn_name)())
    return (lang.abi_version, lang.node_kind_count, lang.field_count)


def _fake_kernel(*, languages=(), tree_sitter_ok=True, extract=None, version="0.1.0",
                 grammars=None):
    mod = types.ModuleType("graphify_kernel")
    mod.version = lambda: version
    mod.supported_languages = lambda: list(languages)
    if grammars is None:
        grammars = {lang: _py_grammar_fingerprint(lang) for lang in languages}
    mod.selftest = lambda: {"version": version, "languages": list(languages),
                            "tree_sitter_ok": tree_sitter_ok,
                            "grammars": dict(grammars)}
    # The real signature: `(result, defer_reason)`, exactly one of them non-None,
    # plus the resolver callbacks the walkers call back into (JS import, JS
    # module, Python relative import, C include, Lua require). Pinned rather than
    # `*args`: a
    # stub that swallowed any arity would let the seam and the kernel drift apart
    # silently, and the seam converts every TypeError into a deferral -- so the
    # drift would show up as a quietly falling native rate, not a failure.
    # `test_seam_passes_every_resolver_the_kernel_takes` checks that against the
    # real module rather than against this list.
    mod.extract_file = (extract if extract is not None
                        else (lambda p, s, l, r=None, m=None, pr=None, ci=None, li=None:
                              (None, "fake")))
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
    """A language with no mapping must not even trigger the module load.

    The example grammar is chosen from the ones that are NOT routed, rather than
    named: this test previously hard-coded `tree_sitter_go`, which stopped being
    unrouted the day Go was ported and turned a real invariant into a false
    failure. Picking an unrouted one keeps the test about the invariant.
    """
    monkeypatch.delenv("GRAPHIFY_KERNEL", raising=False)
    from graphify.extractors.kernel import _GRAMMAR_TO_LANGUAGE

    unrouted = next(
        (m for m in ("tree_sitter_zig", "tree_sitter_dart", "tree_sitter_elixir",
                     "tree_sitter_ocaml", "tree_sitter_julia")
         if (m, "language") not in _GRAMMAR_TO_LANGUAGE),
        None,
    )
    assert unrouted is not None, "every candidate grammar is now routed"

    class _Unrouted:
        ts_module = unrouted
        ts_language_fn = "language"

    assert kernel.language_for(_Unrouted()) is None
    assert kernel.try_extract(Path("a.xx"), _Unrouted()) is None
    assert kernel.drain_counts() == {}


def test_walker_exception_is_a_deferral_not_a_build_failure(monkeypatch, tmp_path):
    monkeypatch.delenv("GRAPHIFY_KERNEL", raising=False)

    def _explode(path, source, language, resolve_import=None, resolve_module=None,
                resolve_py_import=None, resolve_c_include=None,
                 resolve_lua_import=None):
        raise ValueError("walker bug")

    _install(monkeypatch, _fake_kernel(languages=("typescript",), extract=_explode))
    f = tmp_path / "a.ts"
    f.write_bytes(b"const x = 1;")
    assert kernel.try_extract(f, _Cfg()) is None
    assert kernel.drain_counts() == {"defer:kernel_raised:ValueError": 1}


def test_native_result_is_returned_and_counted(monkeypatch, tmp_path):
    monkeypatch.delenv("GRAPHIFY_KERNEL", raising=False)
    payload = {"nodes": [{"id": "n"}], "edges": []}
    _install(monkeypatch, _fake_kernel(
        languages=("typescript",),
        extract=lambda p, s, l, r=None, m=None, pr=None, ci=None, li=None: (payload, None)))
    f = tmp_path / "a.ts"
    f.write_bytes(b"const x = 1;")
    assert kernel.try_extract(f, _Cfg()) is payload
    assert kernel.drain_counts() == {"native:typescript": 1}


def test_source_override_is_honoured(monkeypatch, tmp_path):
    """Vue SFCs parse embedded <script> bytes; the kernel must not re-read the file."""
    monkeypatch.delenv("GRAPHIFY_KERNEL", raising=False)
    seen: dict = {}

    def _capture(path, source, language, resolve_import=None, resolve_module=None,
                 resolve_py_import=None, resolve_c_include=None,
                 resolve_lua_import=None):
        seen["source"] = source
        return ({"nodes": [], "edges": []}, None)

    _install(monkeypatch, _fake_kernel(languages=("typescript",), extract=_capture))
    f = tmp_path / "a.vue"
    f.write_bytes(b"<template>WRAPPER</template>")
    kernel.try_extract(f, _Cfg(), source_override=b"const x = 1;")
    assert seen["source"] == b"const x = 1;"


def test_real_rust_panic_is_a_deferral(monkeypatch, tmp_path):
    """A panic in the native walker must degrade to the Python path.

    Uses the REAL compiled kernel's `debug_panic`, not a stand-in, because the
    thing being tested is a fact about PyO3: it raises
    `pyo3_runtime.PanicException`, which derives from BaseException rather than
    Exception. An `except Exception` in the seam would miss it, the panic would
    pass through `_safe_extract` (also `except Exception`) and escape the pool
    worker, and ProcessPoolExecutor would report a failure for every file that
    worker held. Faking the exception would test our own assumption about its base
    class instead of the assumption itself.
    """
    real = pytest.importorskip("graphify_kernel")
    if not hasattr(real, "debug_panic"):
        pytest.skip("kernel predates the debug_panic hook")
    monkeypatch.delenv("GRAPHIFY_KERNEL", raising=False)

    def _panic(path, source, language, resolve_import=None, resolve_module=None,
               resolve_py_import=None, resolve_c_include=None,
                 resolve_lua_import=None):
        real.debug_panic()

    _install(monkeypatch, _fake_kernel(languages=("typescript",), extract=_panic))
    f = tmp_path / "a.ts"
    f.write_bytes(b"const x = 1;")
    assert kernel.try_extract(f, _Cfg()) is None
    assert list(kernel.drain_counts()) == ["defer:kernel_raised:PanicException"]


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
def test_interrupts_are_not_swallowed(monkeypatch, tmp_path, exc):
    """Containment must not extend to Ctrl-C: that is how a long extraction stops
    responding to it (the defect fixed in d0edc4d)."""
    monkeypatch.delenv("GRAPHIFY_KERNEL", raising=False)

    def _interrupt(path, source, language, resolve_import=None, resolve_module=None,
                  resolve_py_import=None, resolve_c_include=None,
                 resolve_lua_import=None):
        raise exc()

    _install(monkeypatch, _fake_kernel(languages=("typescript",), extract=_interrupt))
    f = tmp_path / "a.ts"
    f.write_bytes(b"const x = 1;")
    with pytest.raises(exc):
        kernel.try_extract(f, _Cfg())


def test_deferral_is_counted_by_reason(monkeypatch, tmp_path):
    """A deferral without a reason is just a percentage: it says a gap exists but
    never which one, so the next construct to implement has to be guessed."""
    monkeypatch.delenv("GRAPHIFY_KERNEL", raising=False)
    _install(monkeypatch, _fake_kernel(
        languages=("typescript",),
        extract=lambda p, s, l, r=None, m=None, pr=None, ci=None, li=None: (None, "decorator")))
    f = tmp_path / "a.ts"
    f.write_bytes(b"const x = 1;")
    assert kernel.try_extract(f, _Cfg()) is None
    assert kernel.drain_counts() == {"defer:typescript:decorator": 1}


def test_grammar_mismatch_drops_the_language(monkeypatch, tmp_path):
    """A kernel built against a different grammar revision must not be trusted.

    This is not hypothetical: `tree-sitter-javascript` was 0.25.0 in the venv and
    0.23.1 in the crate, and the two parse `await f()` differently and disagree
    about which files contain an error node. The walker was faithfully walking a
    different tree than `_extract_generic` would, and nothing else in the build
    would have said so -- the crate pins a semver RANGE and `pip install
    --upgrade` moves the other side silently.
    """
    monkeypatch.delenv("GRAPHIFY_KERNEL", raising=False)
    called: list = []

    def _extract(path, source, language, resolve_import=None, resolve_module=None,
                 resolve_py_import=None, resolve_c_include=None,
                 resolve_lua_import=None):
        called.append(path)
        return ({"nodes": [], "edges": []}, None)

    _install(monkeypatch, _fake_kernel(
        languages=("typescript",), extract=_extract,
        grammars={"typescript": (14, 1, 1)},  # not what Python loads
    ))
    f = tmp_path / "a.ts"
    f.write_bytes(b"const x = 1;")
    assert kernel.enabled_languages() == set()
    assert kernel.try_extract(f, _Cfg()) is None
    assert not called, "a mismatched grammar must never be asked to walk a file"


def test_missing_grammar_fingerprint_drops_the_language(monkeypatch, tmp_path):
    """Fails closed: an older kernel that reports no fingerprints is not trusted."""
    monkeypatch.delenv("GRAPHIFY_KERNEL", raising=False)
    _install(monkeypatch, _fake_kernel(languages=("typescript",), grammars={}))
    assert kernel.enabled_languages() == set()


def test_one_language_config_per_routed_grammar():
    """The routing key `(ts_module, ts_language_fn)` must identify a config uniquely.

    The kernel routes on the grammar pair because that pair IS the grammar
    `_extract_generic` would have loaded, so the kernel cannot parse a file with a
    different grammar than the Python path. But the walker also hard-codes the
    DISPATCH SETS of the config it was written against (`class_types`,
    `function_types`, ...). If a second `LanguageConfig` ever shares a routed
    grammar pair with different sets, the kernel would silently apply the wrong
    ones -- extra or missing nodes, no error.

    So: a routed grammar pair must map to exactly one config. Adding a second one
    means the kernel needs a finer routing key, and this test is where that gets
    noticed.
    """
    import dataclasses

    from graphify.extractors.models import LanguageConfig
    from graphify import extract as extract_module
    from graphify.extractors.kernel import _GRAMMAR_TO_LANGUAGE

    by_key: dict[tuple[str, str], list[str]] = {}
    for name in dir(extract_module):
        cfg = getattr(extract_module, name)
        if isinstance(cfg, LanguageConfig):
            by_key.setdefault((cfg.ts_module, cfg.ts_language_fn), []).append(name)

    for key, names in by_key.items():
        if key not in _GRAMMAR_TO_LANGUAGE:
            continue  # not routed; the kernel never sees it
        assert len(names) == 1, (
            f"grammar {key} is routed to the kernel but maps to {names}; "
            "the kernel would apply one config's dispatch sets to all of them"
        )
        # Guard the fields the walker hard-codes, in case the single config is
        # later rebuilt from another one with a tweak.
        cfg = getattr(extract_module, names[0])
        if _GRAMMAR_TO_LANGUAGE[key] not in _engine_languages():
            # A HAND-WRITTEN walker (js/, py/) omits these branches entirely, so
            # a non-empty set here would silently drop its edges. An
            # engine-driven language READS all four from `EngineConfig`; they are
            # compared field for field by the test below instead.
            for field in ("static_prop_types", "helper_fn_names",
                          "container_bind_methods", "event_listener_properties"):
                assert not getattr(cfg, field), (
                    f"{names[0]}.{field} is now non-empty; the native walker omits "
                    f"that branch entirely (see src/js/walk.rs) and would drop its edges"
                )
        # These two are only meaningful for a HAND-WRITTEN walker (js/, py/),
        # which ignores them: it hard-codes the sets it was written against, so
        # a non-empty tuple here would be silently dropped. An engine-driven
        # language reads them from `EngineConfig` at run time and is checked
        # field for field by the test below instead.
        if _GRAMMAR_TO_LANGUAGE[key] not in _engine_languages():
            assert cfg.name_fallback_child_types == (), names[0]
            assert cfg.body_fallback_child_types == (), names[0]
            assert cfg.resolve_function_name_fn is None, names[0]
            assert cfg.sanitize_symbol_name_fn is None, names[0]
        assert dataclasses.is_dataclass(cfg)


def _engine_languages() -> set[str]:
    """Languages driven by the shared Rust engine, or empty if it is not built."""
    try:
        import graphify_kernel
    except ImportError:
        return set()
    if not hasattr(graphify_kernel, "engine_configs"):
        return set()
    return set(graphify_kernel.engine_configs())


def test_engine_configs_match_their_language_config():
    """Every dispatch field an engine-driven walker reads must equal the Python's.

    Routing on `(ts_module, ts_language_fn)` proves the two sides load the same
    GRAMMAR. It says nothing about whether they dispatch on the same NODE KINDS
    -- and the engine reads all of them from its own `EngineConfig`, a hand-typed
    copy of the `LanguageConfig`. A typo there (a missing `record_declaration`,
    an `attribute` where the Python says `name`) silently changes what becomes a
    node, on exactly the constructs a corpus may not contain.

    This is the check that scales: it gets stronger with each language added to
    the engine rather than needing a new exemption.
    """
    kernel_mod = pytest.importorskip("graphify_kernel",
                                     reason="native kernel not built")
    if not hasattr(kernel_mod, "engine_configs"):
        pytest.skip("kernel predates engine_configs()")

    from graphify import extract as extract_module
    from graphify.extractors.kernel import _LANGUAGE_TO_GRAMMAR, _GRAMMAR_TO_LANGUAGE
    from graphify.extractors.models import LanguageConfig

    by_grammar = {}
    for name in dir(extract_module):
        cfg = getattr(extract_module, name)
        if isinstance(cfg, LanguageConfig):
            by_grammar[(cfg.ts_module, cfg.ts_language_fn)] = (name, cfg)

    SET_FIELDS = ("class_types", "function_types", "import_types", "call_types",
                  "function_boundary_types", "call_accessor_node_types",
                  # The four Laravel-convention sets. On the config, not
                  # hard-coded inside `php/`, precisely so this test can see them.
                  "static_prop_types", "helper_fn_names",
                  "container_bind_methods", "event_listener_properties")
    TUPLE_FIELDS = ("name_fallback_child_types", "body_fallback_child_types")
    SCALAR_FIELDS = ("name_field", "body_field", "call_function_field",
                     "call_accessor_field", "call_accessor_object_field",
                     "function_label_parens")
    # Not a value comparison: the engine reports whether it has a resolver, and
    # the two must AGREE about that, because `None` selects a different branch of
    # the function-name lookup rather than meaning "do nothing".
    PRESENCE_FIELDS = ("resolve_function_name", "sanitize_symbol_name")

    engine = kernel_mod.engine_configs()
    assert engine, "the engine reported no languages"
    for language, native in engine.items():
        grammar = _LANGUAGE_TO_GRAMMAR[language]
        name, cfg = by_grammar[grammar]
        for field in SET_FIELDS:
            assert set(native[field]) == set(getattr(cfg, field)), f"{name}.{field}"
        for field in TUPLE_FIELDS:
            assert tuple(native[field]) == tuple(getattr(cfg, field)), f"{name}.{field}"
        for field in SCALAR_FIELDS:
            assert native[field] == getattr(cfg, field), f"{name}.{field}"
        for field in PRESENCE_FIELDS:
            assert native[field] == (getattr(cfg, f"{field}_fn") is not None), \
                f"{name}.{field}_fn"
        assert _GRAMMAR_TO_LANGUAGE[grammar] == language


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

    # `js_symbol_facts` is the ONE documented asymmetry: the native path also
    # emits phase 3's symbol-resolution facts from the parse it already did, so
    # that `collect_js` does not re-parse the file (a full build called
    # tree-sitter's `parse()` 2.07 times per JS/TS-family file before this). It is
    # a memo of work done later, not part of the graph, and its own equivalence is
    # gated by `harness/kernel_facts_parity.py` (DIVERGENT 0 over 11,868 files)
    # plus the cold graph gate. Everything that reaches the graph must still match
    # exactly.
    assert set(with_kernel) - set(without_kernel) <= {"js_symbol_facts"}
    with_kernel.pop("js_symbol_facts", None)
    assert with_kernel == without_kernel
    assert with_kernel["nodes"], "fixture should produce nodes"


def test_seam_calls_the_real_signature():
    """The stubs above pin `extract_file`'s arity; this pins them to REALITY.

    `try_extract` wraps the call in `except BaseException` and turns any failure
    into a deferral, so calling the real kernel with the wrong number of arguments
    does not raise -- it silently defers every file and the only symptom is a
    native rate that quietly drops to zero. The stub tests cannot catch that,
    because they stub the thing that would disagree. So probe the real module.
    """
    kernel_mod = pytest.importorskip("graphify_kernel",
                                     reason="native kernel not built")
    from graphify.extractors import kernel as seam
    src = b"const x = 1;\n"
    result, reason = kernel_mod.extract_file(
        "/tmp/probe.ts", src, "typescript",
        seam._import_resolver("/tmp/probe.ts"),
        seam._module_resolver("/tmp/probe.ts"),
        seam._py_import_resolver("/tmp/probe.ts"),
        seam._c_include_resolver("/tmp/probe.ts"),
    )
    assert (result is None) != (reason is None), (
        "extract_file must return exactly one of (result, reason)"
    )


def test_seam_passes_every_resolver_the_kernel_takes(tmp_path, monkeypatch):
    """The seam must fill EVERY resolver slot the kernel declares.

    Each resolver defaults to None on the Rust side, and a walker that finds its
    resolver missing defers -- correctly, but silently. So forgetting to pass a
    newly added one does not raise, does not fail a parity run (the harness
    passes its own), and shows up only as a native rate that is quietly lower
    than it should be. Adding the C include resolver as a fourth parameter is
    exactly that shape of change, which is why this is checked against the real
    signature instead of a hand-maintained list.
    """
    import inspect

    kernel_mod = pytest.importorskip("graphify_kernel",
                                     reason="native kernel not built")
    from graphify.extractors import kernel as seam

    expected = len(inspect.signature(kernel_mod.extract_file).parameters)
    seen: list[int] = []

    def _record(*args, **kwargs):
        seen.append(len(args) + len(kwargs))
        return (None, "recorded")

    monkeypatch.setattr(seam, "_kernel", kernel_mod, raising=False)
    monkeypatch.setattr(kernel_mod, "extract_file", _record, raising=False)
    src = tmp_path / "probe.ts"
    src.write_text("const x = 1;\n")
    seam.try_extract(src, _Cfg())

    assert seen, "the seam did not call extract_file at all"
    assert seen[0] == expected, (
        f"the seam passes {seen[0]} arguments but extract_file declares "
        f"{expected}; a resolver slot is being left to its None default"
    )

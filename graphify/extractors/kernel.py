"""Routing seam for the native extraction kernel (`graphify_kernel`).

The kernel is **optional and fail-open**. If the binary is missing, stale, built
against a different ABI, or broken in any way, this module reports that once and
every file takes the pure-Python path it always took. Graphify must never
require a compiled component to extract a repo.

Why the seam is this defensive
------------------------------
Earlier perf work in this project introduced five defects in ~100 lines of
process-pool code (`GRAPHIFY_BUILD_PERF.md`, sixth update). Every single one
produced **silent wrong output or a hang, never an exception** -- and neither the
4,700-test suite nor the cold equivalence gate caught any of them, because each
needed a non-default worker count, a git hook, or a pool failure to trigger.

A native walker is a far larger surface with the same failure mode: a missing
node-type rule does not crash, it silently drops an edge. So:

* the kernel decides *nothing* about which language it handles -- it answers only
  for languages Graphify already resolved to an extractor, via an explicit
  whitelist, so it can never disagree with `_get_extractor`'s filename special
  cases (``.blade.php``, MCP configs, ObjC-vs-MATLAB ``.m``, C-vs-C++ ``.h``);
* any file it does not fully understand is **deferred**, not guessed at;
* deferrals are counted by reason, because a rising deferral rate is the only
  visible signal that a walker has quietly stopped handling something;
* ``GRAPHIFY_KERNEL=0`` turns the whole thing off without a rebuild.

Cache interaction
-----------------
Extraction results are cached by file hash with no kernel marker, so a cache can
hold a mix of native and Python results, and disabling the kernel can serve a
natively-produced entry to a Python-path run. That is only sound because the
parity gate requires **byte-identical** output per file before a language is
routed. If we ever allow deliberate divergence, the kernel version must enter
the cache key -- there is no third option.
"""
from __future__ import annotations

import collections
import os
import sys
from pathlib import Path
from typing import Any, Callable

# (ts_module, ts_language_fn) -> the language key the kernel understands.
#
# This pair IS the grammar `_extract_generic` would have loaded, so the kernel is
# structurally incapable of parsing a file with a different grammar than the
# Python path. Keying on the file suffix instead would reintroduce exactly the
# ambiguities `_get_extractor` spends thirty lines resolving (`.h` C-vs-C++,
# `.m` ObjC-vs-MATLAB, `.blade.php`), and keying on the extractor function is not
# even possible: `.ts`, `.tsx`, `.js` and `.jsx` all dispatch to one `extract_js`
# which picks the config internally.
_GRAMMAR_TO_LANGUAGE: dict[tuple[str, str], str] = {
    ("tree_sitter_typescript", "language_typescript"): "typescript",
    ("tree_sitter_typescript", "language_tsx"): "tsx",
    ("tree_sitter_javascript", "language"): "javascript",
    ("tree_sitter_python", "language"): "python",
}

# The reverse map, for the grammar-version check below: kernel language key ->
# (module, language function) to load on the Python side.
_LANGUAGE_TO_GRAMMAR: dict[str, tuple[str, str]] = {
    v: k for k, v in _GRAMMAR_TO_LANGUAGE.items()
}

_ENV_VAR = "GRAPHIFY_KERNEL"

# Result of the one-time load attempt.
_kernel: Any | None = None
_status: str = "not_loaded"
_loaded = False

# Deferral tally by reason, plus a native-success count, so the parity harness
# and `--timing` runs can report a RATE rather than a bare "it worked".
_counts: collections.Counter = collections.Counter()


def _disable(reason: str) -> None:
    global _kernel, _status
    _kernel = None
    _status = reason


def _load() -> Any | None:
    """Import and self-check the kernel exactly once. Never raises."""
    global _kernel, _status, _loaded
    if _loaded:
        return _kernel
    _loaded = True

    if os.environ.get(_ENV_VAR) == "0":
        _disable("disabled_by_env")
        return None

    try:
        import graphify_kernel as mod  # type: ignore
    except Exception as exc:  # ImportError, but also a bad .so at load time
        _disable(f"import_failed:{type(exc).__name__}")
        return None

    # A kernel that imports but is subtly wrong -- ABI mismatch, a grammar built
    # against another tree-sitter -- is worse than one that fails to import,
    # because it would produce plausible-looking garbage. Prove it works first.
    try:
        report = mod.selftest()
        if not report.get("tree_sitter_ok"):
            _disable("selftest_grammar_failed")
            return None
        if not isinstance(mod.version(), str):
            _disable("selftest_bad_version")
            return None
    except Exception as exc:
        _disable(f"selftest_raised:{type(exc).__name__}")
        return None

    try:
        supported = set(mod.supported_languages())
    except Exception as exc:
        _disable(f"supported_languages_raised:{type(exc).__name__}")
        return None

    # Drop any language whose grammar differs from the one Python would load. A
    # skew here is silent and total: the kernel walks a different tree than
    # `_extract_generic` would, faithfully, and every difference reads as a walker
    # bug. It has already happened once -- tree-sitter-javascript was 0.25.0 in
    # the venv against 0.23.1 in the crate, which parses `await f()` differently
    # and disagrees about which files contain an error node.
    #
    # Fails CLOSED, per language: an unreadable or unloadable grammar drops that
    # language rather than trusting it. Runs once per process, so the cost is a
    # handful of grammar loads the extractor was going to do anyway.
    kernel_fps = report.get("grammars") or {}
    for language in sorted(supported):
        if not _grammar_matches(language, kernel_fps.get(language)):
            supported.discard(language)
            _counts[f"disabled:grammar_mismatch:{language}"] += 1

    _kernel = mod
    _status = "ok"
    _kernel._graphify_supported = supported  # type: ignore[attr-defined]
    return mod


def _grammar_matches(language: str, kernel_fp: tuple | None) -> bool:
    """Whether the kernel's grammar for `language` is the one Python loads.

    Compared on ``(abi_version, node_kind_count, field_count)``. ABI version alone
    is far too coarse -- many grammar revisions share ABI 15 -- while the kind and
    field counts move with essentially any revision, which is what makes the triple
    a usable stand-in for "same grammar".
    """
    if kernel_fp is None:
        return False
    spec = _LANGUAGE_TO_GRAMMAR.get(language)
    if spec is None:
        return False
    module_name, fn_name = spec
    try:
        import importlib

        from tree_sitter import Language

        mod = importlib.import_module(module_name)
        lang = Language(getattr(mod, fn_name)())
        py_fp = (lang.abi_version, lang.node_kind_count, lang.field_count)
    except Exception:
        return False
    return tuple(kernel_fp) == py_fp


def status() -> str:
    """Human-readable load state, for diagnostics. Triggers the load attempt."""
    _load()
    return _status


def enabled_languages() -> set[str]:
    mod = _load()
    if mod is None:
        return set()
    return getattr(mod, "_graphify_supported", set())


def language_for(config: Any) -> str | None:
    """The kernel's language key for a LanguageConfig, or None if not routable.

    Deliberately cheap: this runs once per file, and the overwhelmingly common
    answer (until every language is ported) is None, which must cost a tuple
    build and a dict lookup and nothing else -- in particular it must not touch
    the filesystem.
    """
    try:
        return _GRAMMAR_TO_LANGUAGE.get((config.ts_module, config.ts_language_fn))
    except AttributeError:
        return None


# Lazily-bound Python helpers the native walker calls back into. Imported on
# first use, not at module import: `engine.py` imports THIS module at import
# time, and `resolution` / `base` sit on the other side of that edge -- binding
# them eagerly here would close an import cycle.
_resolve_js_import_target: Callable | None = None
_py_file_stem: Callable | None = None


def _bind_resolver_helpers() -> None:
    global _resolve_js_import_target, _py_file_stem
    if _resolve_js_import_target is not None:
        return
    from graphify.extractors.base import _file_stem
    from graphify.extractors.resolution import _resolve_js_import_target as _r
    _resolve_js_import_target = _r
    _py_file_stem = _file_stem


def _import_resolver(str_path: str) -> Callable[[str], tuple | None]:
    """A `(specifier) -> resolution` callable for one file's native walk.

    The native walker never touches the filesystem. Instead it calls this for each
    module specifier it reaches, and Graphify's own ``_resolve_js_import_target``
    answers -- so extension candidates, index files, tsconfig ``paths`` aliases and
    workspace manifests are resolved by exactly the code the Python path uses, and
    cannot drift from it. The two follow-up questions the Python callers ask about
    a resolved path (``is_file()``, and whether it lands in ``node_modules``) are
    answered here too, keeping every filesystem decision on this side.

    The kernel memoizes per file, so a specifier imported twice costs one call.
    """
    _bind_resolver_helpers()
    resolve = _resolve_js_import_target
    file_stem = _py_file_stem
    assert resolve is not None and file_stem is not None

    def _resolve(raw: str) -> tuple | None:
        got = resolve(raw, str_path)
        if got is None:
            return None
        tgt_nid, resolved_path = got
        if resolved_path is None:
            return (tgt_nid, None, False, False, None)
        return (
            tgt_nid,
            str(resolved_path),
            resolved_path.is_file(),
            "node_modules" in resolved_path.parts,
            file_stem(resolved_path),
        )

    return _resolve


def try_extract(path: Path, config: Any,
                source_override: bytes | None = None) -> dict | None:
    """Extract `path` natively, or return None to mean "use the Python path".

    Mirrors `_extract_generic`'s signature, including `source_override`: container
    formats (Vue SFCs) parse embedded `<script>` bytes while still keying
    nodes/edges off `path`, and the kernel has to honour that or it would silently
    extract the whole wrapper.

    Returns None -- never raises -- for every reason a caller might care about:
    the kernel is absent or disabled, the language is not routed, the walker
    deferred, or the walker blew up. A native failure is a deferral, not a build
    failure.
    """
    language = language_for(config)
    if language is None:
        return None
    mod = _load()
    if mod is None:
        return None
    if language not in getattr(mod, "_graphify_supported", ()):  # not parity-gated yet
        _counts["defer:unsupported_language"] += 1
        return None

    # Only now is reading the file worth it: on the deferral paths above the
    # Python extractor will read it anyway, and a second read per file would be
    # a real cost across 15k files.
    if source_override is not None:
        source = source_override
    else:
        try:
            source = path.read_bytes()
        except OSError:
            _counts["defer:read_failed"] += 1
            return None

    str_path = str(path)
    try:
        result, reason = mod.extract_file(
            str_path, source, language, _import_resolver(str_path)
        )
    except BaseException as exc:
        # BaseException, not Exception, and deliberately so.
        #
        # PyO3 turns a Rust panic into `pyo3_runtime.PanicException`, which derives
        # straight from BaseException -- verified, not assumed (see
        # `graphify_kernel.debug_panic` and the test that calls it). An
        # `except Exception` here would miss it, and the panic would then also pass
        # through `_safe_extract` (also `except Exception`) and escape the pool
        # worker, where ProcessPoolExecutor reports it as a failure for every file
        # that worker was holding. One malformed file would take out a batch.
        #
        # KeyboardInterrupt and SystemExit are re-raised: swallowing them is how a
        # long extraction stops responding to Ctrl-C (the defect fixed in d0edc4d).
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        _counts[f"defer:kernel_raised:{type(exc).__name__}"] += 1
        if os.environ.get("GRAPHIFY_DEBUG"):
            print(f"  kernel: {path} raised {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
        return None

    if result is None:
        # The reason is the construct the walker has no rule for, so the tally
        # ranks the gaps. Without it the only signal is a percentage, which says
        # a gap exists but never which one -- and the next construct to implement
        # then has to be guessed.
        _counts[f"defer:{language}:{reason or 'unknown'}"] += 1
        return None
    _counts[f"native:{language}"] += 1
    return result


def drain_counts() -> dict[str, int]:
    """Return and reset the tally.

    Extraction runs in `ProcessPoolExecutor` workers, so these counters live in
    the worker and are discarded when it exits. `_extract_single_file` therefore
    drains them per file and ships them back to the parent alongside its result,
    where `extract()` sums and prints them -- see `_KERNEL_TALLY` there.

    Draining per file rather than per worker is deliberate: there is no
    worker-exit hook, so any coarser scheme loses whichever worker happens to
    die, and an under-reported deferral rate is worse than none. A walker that is
    correct but quietly defers 60% of files shows DIVERGENT=0 while delivering
    none of the speedup, so the rate has to be visible or the counting is
    pointless.
    """
    if not _counts:
        return {}
    out = dict(_counts)
    _counts.clear()
    return out


def reset_for_test() -> None:
    """Forget the load attempt so a test can re-evaluate the environment."""
    global _kernel, _status, _loaded
    _kernel = None
    _status = "not_loaded"
    _loaded = False
    _counts.clear()

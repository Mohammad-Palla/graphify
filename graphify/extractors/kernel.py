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

    _kernel = mod
    _status = "ok"
    _kernel._graphify_supported = supported  # type: ignore[attr-defined]
    return mod


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

    try:
        result = mod.extract_file(str(path), source, language)
    except Exception as exc:
        # A panic or a PyO3 error surfaces here. Count it loudly by reason and
        # fall back; one bad file must not take out the build.
        _counts[f"defer:kernel_raised:{type(exc).__name__}"] += 1
        if os.environ.get("GRAPHIFY_DEBUG"):
            print(f"  kernel: {path} raised {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
        return None

    if result is None:
        _counts[f"defer:{language}"] += 1
        return None
    _counts[f"native:{language}"] += 1
    return result


def drain_counts() -> dict[str, int]:
    """Return and reset the tally.

    NOTE for whoever routes the first language: extraction runs in
    `ProcessPoolExecutor` workers, so these counters live in the worker and are
    discarded when it exits. They are accurate for in-process callers (the parity
    harness, tests) but a pooled build currently reports nothing. Ferrying them
    back -- alongside the per-file timings `_extract_single_file` already
    returns under GRAPHIFY_EXTRACT_PROFILE -- is a prerequisite for enabling any
    language, because an unreported deferral rate defeats the whole point of
    counting.
    """
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

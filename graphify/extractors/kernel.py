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
    ("tree_sitter_java", "language"): "java",
    ("tree_sitter_c_sharp", "language"): "csharp",
    ("tree_sitter_c", "language"): "c",
    ("tree_sitter_cpp", "language"): "cpp",
    ("tree_sitter_php", "language_php"): "php",
    ("tree_sitter_bash", "language"): "bash",
    ("tree_sitter_go", "language"): "go",
    ("tree_sitter_rust", "language"): "rust",
    ("tree_sitter_ruby", "language"): "ruby",
    ("tree_sitter_kotlin", "language"): "kotlin",
    ("tree_sitter_lua", "language"): "lua",
    ("tree_sitter_groovy", "language"): "groovy",
    ("tree_sitter_scala", "language"): "scala",
    ("tree_sitter_swift", "language"): "swift",
    ("tree_sitter_ocaml", "language_ocaml"): "ocaml",
    ("tree_sitter_ocaml", "language_ocaml_interface"): "ocaml_interface",
    # BESPOKE walkers still need an entry here. They reach the kernel through
    # `BespokeGrammar`, which supplies exactly this `(ts_module, ts_language_fn)`
    # pair for `language_for` to look up -- so a walker registered in
    # `languages.rs` but MISSING here is silently unreachable: `language_for`
    # returns None, `try_extract` defers before it even loads the kernel, and
    # every gate still passes because Python is then compared against Python.
    # `tests/test_kernel_seam.py::test_every_supported_language_is_reachable`
    # exists because that happened to six languages at once.
    ("tree_sitter_zig", "language"): "zig",
    ("tree_sitter_elixir", "language"): "elixir",
    ("tree_sitter_julia", "language"): "julia",
    ("tree_sitter_fortran", "language"): "fortran",
    ("tree_sitter_objc", "language"): "objc",
    ("tree_sitter_powershell", "language"): "powershell",
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


def _names_digest(names) -> str:
    """SHA-256 over grammar names, ``\0``-joined in id order.

    Must stay byte-for-byte identical to `names_digest` in `languages.rs`: join
    with ``\0``, hash the UTF-8 bytes, lowercase hex. A missing name is the empty
    string.
    """
    import hashlib

    joined = "\0".join(n or "" for n in names)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _grammar_matches(language: str, kernel_fp: tuple | None) -> bool:
    """Whether the kernel's grammar for `language` is the one Python loads.

    Compared on the DIGESTS of the full symbol tables -- every node-kind name by
    id and every field name by id -- not on their counts.

    That is strictly stronger than the old
    ``(abi_version, node_kind_count, field_count)``: a grammar revision that adds
    one kind and removes another keeps both counts and so passed silently, while
    parsing differently. The digests pin the whole table, name for name.

    ``abi_version`` is deliberately NOT compared, and the reason is measured.
    ABI is a property of the tree-sitter CLI that generated the parser, not of
    the grammar. PyPI ``tree-sitter-sql`` 0.3.11 is ABI 15 while the crate
    ``tree-sitter-sequel`` 0.3.11 is ABI 14, with all 729 kind names and 54 field
    names identical by id; parsing all 3,442 postgres + sqlfluff files with both
    and comparing a preorder digest of every node (kind, byte range, MISSING and
    ERROR flags) gave 3,442 identical trees out of 3,442, including all 2,797
    files containing ERROR nodes. Gating on ABI would reject that while adding no
    protection the digests do not already provide. The real safety net is
    per-language and unchanged: a DIVERGENT-0 parity run over real corpora.
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
        kinds = _names_digest(
            lang.node_kind_for_id(i) for i in range(lang.node_kind_count)
        )
        fields = _names_digest(
            lang.field_name_for_id(i) for i in range(lang.field_count + 1)
        )
    except Exception:
        return False
    # kernel_fp is (abi, kinds_digest, fields_digest); abi is diagnostic only.
    if len(kernel_fp) != 3:
        return False
    return (kernel_fp[1], kernel_fp[2]) == (kinds, fields)


def status() -> str:
    """Human-readable load state, for diagnostics. Triggers the load attempt."""
    _load()
    return _status


def enabled_languages() -> set[str]:
    mod = _load()
    if mod is None:
        return set()
    return getattr(mod, "_graphify_supported", set())


class BespokeGrammar:
    """A stand-in `LanguageConfig` for an extractor that has none.

    `extract_bash`, `extract_go` and `extract_rust` are hand-written walkers, not
    `_extract_generic` under a config -- so they have no `ts_module` /
    `ts_language_fn` for `language_for` to route on, and without this they could
    never reach the kernel at all.

    Two attributes, and deliberately not a `LanguageConfig`: the routing key stays
    ONE table keyed on the grammar pair, and nothing here can be mistaken for a
    config the engine could be driven by.
    """

    __slots__ = ("ts_module", "ts_language_fn")

    def __init__(self, ts_module: str, ts_language_fn: str = "language") -> None:
        self.ts_module = ts_module
        self.ts_language_fn = ts_language_fn


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
_resolve_js_module_path: Callable | None = None
_py_file_stem: Callable | None = None
_probe_python_module: Callable | None = None


def _bind_resolver_helpers() -> None:
    global _resolve_js_import_target, _resolve_js_module_path, _py_file_stem
    global _probe_python_module
    if _resolve_js_import_target is not None:
        return
    from graphify.extract import _probe_python_module_candidate
    from graphify.extractors.base import _file_stem
    from graphify.extractors.resolution import (
        _resolve_js_import_target as _r,
        _resolve_js_module_path as _m,
    )
    _resolve_js_import_target = _r
    _resolve_js_module_path = _m
    _py_file_stem = _file_stem
    _probe_python_module = _probe_python_module_candidate


def _c_include_resolver(str_path: str) -> Callable[[str], str | None]:
    """A `(raw) -> resolved-path-string | None` callable for one C file.

    `_resolve_c_include_path` is `Path(str_path).parent / raw` through
    `resolve_cached` plus an `is_file()` probe -- symlink resolution, `..`
    normalization and a disk hit, none of which the native walker may do itself.
    It calls Graphify's own function so the resolution cannot drift from the
    Python path, the same arrangement `_import_resolver` uses for JS.
    """
    from graphify.extractors.resolution import _resolve_c_include_path

    def _resolve(raw: str) -> str | None:
        target = _resolve_c_include_path(raw, str_path)
        return None if target is None else str(target)

    return _resolve


def _lua_import_resolver(str_path: str) -> Callable[[str], str | None]:
    """A `(raw_module) -> node-id | None` callable for one Lua file.

    `_resolve_lua_import_target` turns `require("pkg.b")` into a node id by
    walking up to six directories probing `pkg/b.lua`, `pkg/b.luau`,
    `pkg/b/init.lua` and `pkg/b/init.luau` -- up to 24 `is_file()` hits against a
    tree the walker must not touch itself. It calls Graphify's own function so
    the resolution cannot drift from the Python path.

    Note it returns an id, not a path: on a miss it falls back to
    `_make_id(raw_module)` rather than dropping the edge (#1075), so the only
    None here is the empty-module case the Python also drops.
    """
    from graphify.extractors.resolution import _resolve_lua_import_target

    def _resolve(raw_module: str) -> str | None:
        nid = _resolve_lua_import_target(raw_module, str_path)
        return nid or None

    return _resolve


def _module_resolver(str_path: str) -> Callable[[str], str | None]:
    """A `(specifier) -> resolved-path-string | None` callable for the fact pass.

    Deliberately a SECOND resolver rather than a reuse of `_import_resolver`.
    `_resolve_js_import_target` and `_resolve_js_module_path` are different
    functions with different fallbacks -- the first mints a `ref`-namespaced id
    for a specifier it cannot resolve, the second returns None -- and
    `_collect_js_facts_one_file` calls the second. Sharing one would silently
    change which specifiers produce facts.

    `.resolve()` is applied here because the Python collector applies it, and the
    resolved string is what lands in the fact tuples.
    """
    _bind_resolver_helpers()
    resolve = _resolve_js_module_path
    assert resolve is not None
    parent = Path(str_path).parent

    def _resolve(raw: str) -> str | None:
        target = resolve(raw, parent)
        return None if target is None else str(target.resolve())

    return _resolve


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


def _py_import_resolver(str_path: str) -> Callable[[str], tuple]:
    """A `(raw) -> (target_path_str, is_file)` callable for one Python file.

    This is `_import_python`'s RELATIVE branch, lifted verbatim. The native walker
    does the pure string work (an absolute `from pkg.mod import x` needs no
    filesystem at all, so it never calls in here), and everything that touches
    pathlib or disk happens on this side: the `.parent` walk for the leading dots,
    the dotted-to-slash join, `_probe_python_module_candidate`'s
    directory/`__init__.py`/file probes, the speculative fallback, and the final
    `is_file()` that gates the `target_file` stamp.

    Kept as a copy of the Python rather than a call into it because
    `_import_python` emits edges as its output -- there is no seam in it that
    returns just the resolution. The copy is byte-compared against the original by
    `harness/kernel_walker_parity.py` on every `.py` file in the corpora, which is
    what keeps it from drifting.
    """
    _bind_resolver_helpers()
    probe = _probe_python_module
    assert probe is not None
    parent = Path(str_path).parent

    def _resolve(raw: str) -> tuple:
        dots = len(raw) - len(raw.lstrip("."))
        module_name = raw.lstrip(".")
        base = parent
        for _ in range(dots - 1):
            base = base.parent
        candidate = base / module_name.replace(".", "/") if module_name else base
        resolved = probe(candidate)
        if resolved is not None:
            target_path = resolved
        else:
            rel = (module_name.replace(".", "/") + ".py") if module_name else "__init__.py"
            target_path = base / rel
        try:
            is_file = target_path.is_file()
        except OSError:
            is_file = False
        return (str(target_path), is_file)

    return _resolve


def js_facts_from_native(path: Path, payload: Any) -> tuple | None:
    """Turn the kernel's flat fact tuples into `_SymbolResolutionFacts` objects.

    The dataclasses stay defined in ONE place (`extractors.models`) and are
    constructed here rather than in Rust: the native side ships plain tuples, so
    a field added to any fact type is a Python-side change only, and the two
    implementations cannot drift into different record shapes.

    Called by the PARENT, not the worker, which is what keeps
    `js_symbol_facts` JSON-serializable while it rides in the extraction result.
    That matters because the result can still reach `save_cached`: the
    cache-bypass list is matched on the exact lowercase suffix, so a file named
    `A.TS` is cached like any other, and a dataclass in the payload made
    `json.dump` raise `TypeError: Object of type _SymbolResolutionFacts is not
    JSON serializable`. Tuples round-trip through the cache as lists, which
    unpack identically here.
    """
    from graphify.extractors.models import (
        _NamespaceExportFact,
        _StarExportFact,
        _SymbolAliasFact,
        _SymbolDeclarationFact,
        _SymbolExportFact,
        _SymbolImportFact,
        _SymbolResolutionFacts,
        _SymbolUseFact,
    )

    def build(d: dict):
        f = _SymbolResolutionFacts()
        f.declarations.extend(
            _SymbolDeclarationFact(path, name, line) for name, line in d["declarations"]
        )
        f.imports.extend(
            _SymbolImportFact(path, local, Path(target), imported, line)
            for local, target, imported, line in d["imports"]
        )
        f.aliases.extend(
            _SymbolAliasFact(path, alias, target, line)
            for alias, target, line in d["aliases"]
        )
        f.exports.extend(
            _SymbolExportFact(
                path, exported, line, local,
                None if target is None else Path(target), target_name,
            )
            for exported, line, local, target, target_name in d["exports"]
        )
        f.star_exports.extend(
            _StarExportFact(path, Path(target), line) for target, line in d["star_exports"]
        )
        f.namespace_exports.extend(
            _NamespaceExportFact(path, exported, Path(target), line)
            for exported, target, line in d["namespace_exports"]
        )
        f.uses.extend(
            _SymbolUseFact(path, source_id, local, relation, context, line)
            for source_id, local, relation, context, line in d["uses"]
        )
        return f

    try:
        return build(payload[0]), build(payload[1])
    except Exception:
        # A shape the converter does not understand must cost a deferral, not a
        # half-built fact set: phase 3 re-collects this file in Python.
        return None


def py_facts_from_native(path: Path, root: Path, payload: Any) -> tuple | None:
    """Turn the kernel's Python fact payload into `_SymbolResolutionFacts`.

    The native side ships the PARSED material -- each `from ... import ...` as
    `(level, module_name, line, [(imported, local)])` and each top-level
    function's calls as `(source_id, [(callee, line)])` -- and every filesystem
    decision is made here: `_resolve_python_module_path`, and the `is_file()`
    probes that redirect `from pkg import submod` to the submodule file (#1146).

    That split was measured, not assumed. Over django's 2,929 files the pass costs
    6.39s serially, of which parse is 2.26s and the walk 3.84s -- both free once
    phase 2 has done them -- against 0.28s of module resolution over 10,176 calls,
    which is what moves from the pool into this serial parent. 95.6% of the work
    moves to where it is already paid for. Had the ratio gone the other way the
    fusion would have been a loss, because unlike the cross-file pass this one was
    already parallel.

    Returns `(facts, deferred)` to match `collect_one`'s contract. Python has a
    single `uses` producer, so `deferred` is always empty.
    """
    from graphify.extractors.models import (
        _SymbolExportFact,
        _SymbolImportFact,
        _SymbolResolutionFacts,
        _SymbolUseFact,
    )
    from graphify.extractors.base import _file_stem, _make_id
    from graphify.extractors.resolution import _resolve_python_module_path

    try:
        facts = _SymbolResolutionFacts()
        deferred = _SymbolResolutionFacts()
        is_init = path.name == "__init__.py"
        for level, module_name, line, names in payload["imports"]:
            target_path = _resolve_python_module_path(module_name, path, root, level)
            if target_path is None:
                continue
            pkg_dir = target_path.parent if target_path.name == "__init__.py" else None
            for imported_name, local_name in names:
                if pkg_dir is not None:
                    sub_py = pkg_dir / f"{imported_name}.py"
                    sub_pkg = pkg_dir / imported_name / "__init__.py"
                    submodule = (sub_py if sub_py.is_file()
                                 else (sub_pkg if sub_pkg.is_file() else None))
                    if submodule is not None:
                        facts.module_imports.append((path, submodule, line, local_name))
                        continue
                facts.imports.append(
                    _SymbolImportFact(path, local_name, target_path, imported_name, line)
                )
                if is_init:
                    facts.exports.append(
                        _SymbolExportFact(path, local_name, line,
                                          target_path=target_path,
                                          target_name=imported_name)
                    )
        # The payload carries the function NAME; the id is minted here, from the
        # path this run is looking at. Baking `_make_id(_file_stem(path), name)`
        # into the payload embedded the scan root's slug in the portable AST cache
        # (#2257) -- caught by test_warm_cache_from_another_root_does_not_leak.
        stem = _file_stem(path)
        for source_name, calls in payload["uses"]:
            source_id = _make_id(stem, source_name)
            for callee, line in calls:
                facts.uses.append(
                    _SymbolUseFact(path, source_id, callee, "calls", "call", line)
                )
        return facts, deferred
    except Exception:
        # A shape the converter does not understand must cost a deferral, not a
        # half-built fact set: phase 3 re-collects this file in Python.
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

    str_path = str(path)
    try:
        result, reason = mod.extract_file(
            str_path, source, language,
            _import_resolver(str_path), _module_resolver(str_path),
            _py_import_resolver(str_path), _c_include_resolver(str_path),
            _lua_import_resolver(str_path),
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
    if "js_symbol_facts" in result:
        _counts[f"native_facts:{language}"] += 1
    if "py_rationale" in result:
        _counts[f"native_rationale:{language}"] += 1
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

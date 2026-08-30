"""Drift guard: the resolution tail must resolve paths through the memo.

`Path.resolve()` lstat-walks every component, so it is the one path helper that
touches the filesystem. The resolution tail asks for the same few thousand paths
over and over -- on superset, 460,113 calls carrying 10,998 distinct paths
(41.8x), 4.5 million `lstat` syscalls -- because the work is per FACT while the
paths are per FILE. `resolve_cached` already existed, with a per-extract()
lifecycle, and `extract.py`'s id-remap already used it; `resolution.py`'s two
hottest consumers (`_apply_symbol_resolution_facts` and `resolve_exported_origin`)
did not, and they are 86% of the calls.

Routing them through it took `phase3.augment_symbol_res` from 17.5s to 3.6s and
superset's whole cold build from 74.8s to 58.6s.

Nothing in the type system stops a future edit from writing `.resolve()` again --
it is the obvious spelling and it would be silently correct, just slow. Hence a
source-level guard, in the same spirit as the node-ID contract test.
"""
from __future__ import annotations

import re
from pathlib import Path

import graphify.extractors.resolution as resolution


def _code_lines(module) -> list[tuple[int, str]]:
    """Source lines with comments and docstring-only lines removed.

    Crude but sufficient: the guard is about CALLS, and the file's prose
    mentions `.resolve()` several times when explaining why the memo exists.
    """
    src = Path(module.__file__).read_text(encoding="utf-8").splitlines()
    out = []
    in_doc = False
    for i, raw in enumerate(src, 1):
        line = raw.strip()
        # Toggle on a line that opens or closes a triple-quoted block.
        ticks = line.count('"""') + line.count("'''")
        if in_doc:
            if ticks:
                in_doc = False
            continue
        if ticks == 1:
            in_doc = True
            continue
        if line.startswith("#"):
            continue
        out.append((i, raw.split("#", 1)[0]))
    return out


def test_resolution_never_calls_path_resolve_directly():
    offenders = [
        (n, l.strip()) for n, l in _code_lines(resolution)
        if re.search(r"\.resolve\(\)", l)
    ]
    assert not offenders, (
        "resolution.py must resolve through `resolve_cached`, not `.resolve()`:\n"
        + "\n".join(f"  line {n}: {t}" for n, t in offenders)
    )


def test_resolve_cached_is_imported_and_used():
    """Guards the other direction: the calls could be removed rather than routed."""
    src = Path(resolution.__file__).read_text(encoding="utf-8")
    assert "from graphify.paths import resolve_cached" in src
    assert src.count("resolve_cached(") >= 15


def test_resolve_cached_answers_identically_to_path_resolve(tmp_path):
    """The memo must be a drop-in: same answer, including through a symlink."""
    from graphify.paths import clear_resolve_cache, resolve_cached
    real = tmp_path / "real"
    real.mkdir()
    (real / "f.txt").write_text("x")
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):  # Windows without privilege
        link = real
    clear_resolve_cache()
    for p in (real / "f.txt", link / "f.txt", tmp_path, Path("."),
              tmp_path / "does_not_exist"):
        assert resolve_cached(p) == Path(p).resolve()
        assert resolve_cached(p) == Path(p).resolve()  # again, now cached


def test_resolve_cache_is_dropped_between_runs(tmp_path):
    """The lifecycle is the whole reason this memo is not process-lifetime: watch
    and the MCP server call extract() repeatedly in ONE process and must observe
    a symlink that has been repointed."""
    from graphify.paths import clear_resolve_cache, resolve_cached
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(a, target_is_directory=True)
    except (OSError, NotImplementedError):
        import pytest
        pytest.skip("symlinks unavailable")
    clear_resolve_cache()
    assert resolve_cached(link) == a.resolve()
    link.unlink()
    link.symlink_to(b, target_is_directory=True)
    assert resolve_cached(link) == a.resolve(), "should still be cached within a run"
    clear_resolve_cache()
    assert resolve_cached(link) == b.resolve(), "must re-read after the run boundary"

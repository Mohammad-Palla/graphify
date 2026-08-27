"""Regression tests for the per-file symbol-fact collection driver.

The driver optionally runs collection across a process pool. Every defect it
has had produced WRONG OUTPUT rather than an error, so these tests pin the
behaviours that fail silently:

  * facts must be identical whatever the worker count resolves to — a
    `return None` sentinel that stopped working once the mapper became a
    generator silently merged ZERO facts when the count resolved to 1, which
    is what `GRAPHIFY_MAX_WORKERS=1` (exported by the Windows post-commit hook)
    produces.
  * `parallel=False` must actually skip the pool; it is the documented escape
    hatch when process pools are unusable.
  * a rebuild watchdog's TimeoutError must propagate. `graphify hook install`
    arms a ONE-SHOT SIGALRM; swallowing it spends the alarm and then re-runs
    the whole collection with nothing armed.
"""
from __future__ import annotations

import pytest

from graphify.extractors import resolution as R
from graphify.extractors.models import _SymbolResolutionFacts


def _corpus(tmp_path, n):
    """n TypeScript files — enough to cross _FACTS_PARALLEL_THRESHOLD."""
    paths = []
    for i in range(n):
        p = tmp_path / f"mod{i}.ts"
        # Shapes chosen to populate SEVERAL fact lists, including both `uses`
        # producers: a top-level (non-exported) function body, and a class
        # member. `export function` does NOT work here — it wraps the
        # declaration in an export_statement, and _js_top_level_function_bodies
        # only scans direct children of the root.
        p.write_text(
            f"import {{ helper{i} }} from './mod{(i + 1) % n}';\n"
            f"function run{i}() {{ return helper{i}(); }}\n"
            f"export class Widget{i} {{\n"
            f"  render() {{ return run{i}(); }}\n"
            f"}}\n"
            f"export function helper{i}() {{ return 1; }}\n"
            f"export {{ run{i} as alias{i} }};\n",
            encoding="utf-8",
        )
        paths.append(p)
    return paths


def _collect(paths, **kwargs):
    facts = _SymbolResolutionFacts()
    R._collect_js_symbol_resolution_facts(paths, facts, **kwargs)
    return facts


def _signature(facts):
    return {
        f.name: len(getattr(facts, f.name))
        for f in __import__("dataclasses").fields(_SymbolResolutionFacts)
    }


@pytest.mark.parametrize("workers", ["1", "2", "4", "99"])
def test_worker_count_never_changes_the_facts(tmp_path, monkeypatch, workers):
    """`GRAPHIFY_MAX_WORKERS=1` used to yield zero facts, silently."""
    paths = _corpus(tmp_path, R._FACTS_PARALLEL_THRESHOLD + 20)
    monkeypatch.delenv("GRAPHIFY_MAX_WORKERS", raising=False)
    serial = _signature(_collect(paths, parallel=False))
    assert serial["uses"] > 0, "fixture produced no facts; test would be vacuous"

    monkeypatch.setenv("GRAPHIFY_MAX_WORKERS", workers)
    assert _signature(_collect(paths)) == serial


def test_explicit_max_workers_is_honoured(tmp_path, monkeypatch):
    paths = _corpus(tmp_path, R._FACTS_PARALLEL_THRESHOLD + 20)
    monkeypatch.delenv("GRAPHIFY_MAX_WORKERS", raising=False)
    seen = []
    real = R._map_facts_parallel
    monkeypatch.setattr(
        R, "_map_facts_parallel",
        lambda fn, sel, workers: (seen.append(workers), real(fn, sel, workers))[1],
    )
    _collect(paths, max_workers=2)
    assert seen == [2]


def test_parallel_false_skips_the_pool(tmp_path, monkeypatch):
    """The escape hatch extract()'s own BrokenProcessPool message advertises."""
    paths = _corpus(tmp_path, R._FACTS_PARALLEL_THRESHOLD + 20)
    monkeypatch.setattr(
        R, "_map_facts_parallel",
        lambda *a, **k: pytest.fail("pool used despite parallel=False"),
    )
    assert _signature(_collect(paths, parallel=False))["uses"] > 0


def test_below_threshold_stays_serial(tmp_path, monkeypatch):
    """Most repos are small; they must not pay process spawn."""
    paths = _corpus(tmp_path, 5)
    monkeypatch.setattr(
        R, "_map_facts_parallel",
        lambda *a, **k: pytest.fail("pool used below the threshold"),
    )
    assert _signature(_collect(paths))["uses"] > 0


def test_rebuild_watchdog_timeout_propagates(tmp_path, monkeypatch):
    """`graphify hook install` arms a ONE-SHOT SIGALRM raising TimeoutError.

    TimeoutError is an OSError, so a bare `except Exception` around the pool
    swallows it — spending the alarm and then re-running the whole collection
    serially with nothing armed, which can block `git checkout` indefinitely.
    Raised through the mapper rather than via a real timer so the test cannot
    flake on how fast the fixture happens to collect.
    """
    paths = _corpus(tmp_path, R._FACTS_PARALLEL_THRESHOLD + 20)

    # Raised where the real alarm fires — inside the pool context — rather than
    # from a stubbed mapper, so this exercises the actual handler and would
    # still fail if the swallow moved back down into the mapper.
    import concurrent.futures as cf

    def exploding_pool(*_a, **_k):
        raise TimeoutError("graphify rebuild exceeded 600s")

    monkeypatch.setattr(cf, "ProcessPoolExecutor", exploding_pool)
    with pytest.raises(TimeoutError):
        _collect(paths)


def test_keyboard_interrupt_propagates(tmp_path, monkeypatch):
    """Ctrl-C must stop the run, not silently downgrade it to a serial retry."""
    paths = _corpus(tmp_path, R._FACTS_PARALLEL_THRESHOLD + 20)
    import concurrent.futures as cf

    def interrupted(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(cf, "ProcessPoolExecutor", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _collect(paths)


def test_pool_failure_falls_back_without_losing_facts(tmp_path, monkeypatch, capsys):
    """A broken pool must degrade to serial, keep every fact, and SAY so."""
    paths = _corpus(tmp_path, R._FACTS_PARALLEL_THRESHOLD + 20)
    monkeypatch.delenv("GRAPHIFY_MAX_WORKERS", raising=False)
    expected = _signature(_collect(paths, parallel=False))

    def exploding(*_a, **_k):
        raise OSError("simulated broken pool")
        yield  # pragma: no cover - generator marker

    monkeypatch.setattr(R, "_map_facts_parallel", exploding)
    assert _signature(_collect(paths)) == expected
    assert "falling back to serial" in capsys.readouterr().err


def test_interrupt_does_not_wait_for_queued_work(tmp_path, monkeypatch):
    """Ctrl-C must not block until every queued chunk drains.

    `with ProcessPoolExecutor(...)` exits via shutdown(wait=True), which on a
    large corpus kept the process unresponsive for 12.6s after SIGINT (against
    0.01s for the serial path). An earlier round of this project reverted a
    pool partly for breaking Ctrl-C, so this pins the shutdown arguments
    rather than racing a real timer.
    """
    import concurrent.futures as cf

    calls = []

    class RecordingPool:
        def __init__(self, *_a, **_k):
            pass

        def map(self, _fn, _items, chunksize=None):
            raise KeyboardInterrupt

        def shutdown(self, wait=True, cancel_futures=False):
            calls.append({"wait": wait, "cancel_futures": cancel_futures})

    monkeypatch.setattr(cf, "ProcessPoolExecutor", RecordingPool)
    paths = _corpus(tmp_path, R._FACTS_PARALLEL_THRESHOLD + 20)
    with pytest.raises(KeyboardInterrupt):
        _collect(paths)

    assert calls == [{"wait": False, "cancel_futures": True}], (
        f"interrupt must cancel queued work and not wait; got {calls}"
    )


def test_clean_completion_still_waits_for_workers(tmp_path, monkeypatch):
    """The non-interrupt path must still drain normally, or results are lost."""
    import concurrent.futures as cf

    calls = []
    real = cf.ProcessPoolExecutor

    class WatchedPool(real):
        def shutdown(self, wait=True, cancel_futures=False):
            calls.append({"wait": wait, "cancel_futures": cancel_futures})
            return super().shutdown(wait=wait, cancel_futures=cancel_futures)

    monkeypatch.setattr(cf, "ProcessPoolExecutor", WatchedPool)
    paths = _corpus(tmp_path, R._FACTS_PARALLEL_THRESHOLD + 20)
    facts = _collect(paths)
    assert calls and calls[-1]["wait"] is True
    assert _signature(facts)["uses"] > 0

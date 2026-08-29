"""Tests for graphify/_minhash.py — MinHash sketch and band-LSH."""
from __future__ import annotations
import numpy as np
import pytest
from graphify._minhash import MinHash, MinHashLSH, _optimal_lsh_params


def _minhash_for(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for i in range(0, len(text) - 2):
        m.update(text[i:i + 3].encode())
    return m


# ── MinHash ───────────────────────────────────────────────────────────────────

def test_identical_texts_produce_identical_hashvalues():
    a = _minhash_for("graphextractor")
    b = _minhash_for("graphextractor")
    assert np.array_equal(a.hashvalues, b.hashvalues)


def test_similar_texts_share_most_hashvalues():
    a = _minhash_for("authentication manager")
    b = _minhash_for("authentication managers")
    overlap = np.sum(a.hashvalues == b.hashvalues) / len(a.hashvalues)
    assert overlap > 0.5


def test_unrelated_texts_share_few_hashvalues():
    a = _minhash_for("authentication manager")
    b = _minhash_for("file system watcher")
    overlap = np.sum(a.hashvalues == b.hashvalues) / len(a.hashvalues)
    assert overlap < 0.3


def test_update_mutates_hashvalues():
    m = MinHash(num_perm=64)
    before = m.hashvalues.copy()
    m.update(b"hello")
    assert not np.array_equal(m.hashvalues, before)


# ── MinHashLSH ────────────────────────────────────────────────────────────────

def test_near_duplicates_are_candidates():
    lsh = MinHashLSH(threshold=0.5, num_perm=128)
    a = _minhash_for("authentication manager")
    b = _minhash_for("authentication managers")
    lsh.insert("a", a)
    lsh.insert("b", b)
    assert "b" in lsh.query(a)


def test_unrelated_strings_not_candidates():
    lsh = MinHashLSH(threshold=0.5, num_perm=128)
    a = _minhash_for("authentication manager")
    b = _minhash_for("file system watcher")
    lsh.insert("a", a)
    lsh.insert("b", b)
    assert "b" not in lsh.query(a)


def test_query_always_returns_self():
    lsh = MinHashLSH(threshold=0.5, num_perm=128)
    m = _minhash_for("graphextractor")
    lsh.insert("x", m)
    assert "x" in lsh.query(m)


def test_duplicate_insert_raises():
    lsh = MinHashLSH(threshold=0.5, num_perm=128)
    m = _minhash_for("foo")
    lsh.insert("key", m)
    with pytest.raises(ValueError, match="already exists"):
        lsh.insert("key", m)


# ── _optimal_lsh_params ───────────────────────────────────────────────────────

def test_optimal_params_within_budget():
    b, r = _optimal_lsh_params(0.5, 128)
    assert b >= 1 and r >= 1
    assert b * r <= 128


def test_optimal_params_cached():
    result1 = _optimal_lsh_params(0.7, 128)
    result2 = _optimal_lsh_params(0.7, 128)
    assert result1 is result2


# ── EDR regression: scipy / numpy.testing must not be imported ──────────────────

def test_dedup_import_does_not_pull_scipy_or_numpy_testing():
    import sys
    for mod in ("scipy", "numpy.testing"):
        sys.modules.pop(mod, None)
    import graphify.dedup  # noqa: F401
    assert "scipy" not in sys.modules
    assert "numpy.testing" not in sys.modules


# ── update_batch ──────────────────────────────────────────────────────────────
#
# `update_batch` exists to stop paying a 128-wide numpy dispatch per VALUE: on a
# django build `update` was 394,940 calls and 1.95s, the largest single line in
# the whole `build` stage. Unlike the rest of that stage the work is not
# redundant (all 8,495 sketches are distinct), so the only lever is shape.
#
# It is therefore only correct if it is BIT-IDENTICAL to the loop it replaces,
# not merely similar: dedup thresholds are tuned against the sketch values, so a
# one-permutation drift silently changes which entities merge.

def test_update_batch_is_bit_identical_to_the_update_loop():
    vs = [f"shingle-{i}".encode() for i in range(97)]
    loop = MinHash(128)
    for v in vs:
        loop.update(v)
    batch = MinHash(128)
    batch.update_batch(vs)
    assert np.array_equal(loop.hashvalues, batch.hashvalues)
    assert loop.hashvalues.dtype == batch.hashvalues.dtype == np.uint64


@pytest.mark.parametrize("n", [0, 1, 2, 127, 128, 129, 4095, 4096, 4097, 9000])
def test_update_batch_matches_the_loop_across_the_chunk_boundary(n):
    """Sizes straddling `_BATCH` (4096) and `num_perm` (128)."""
    vs = [f"v{i}".encode() for i in range(n)]
    loop = MinHash(128)
    for v in vs:
        loop.update(v)
    batch = MinHash(128)
    batch.update_batch(vs)
    assert np.array_equal(loop.hashvalues, batch.hashvalues)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 7])
@pytest.mark.parametrize("n", [1, 5, 10, 23])
def test_update_batch_absorbs_every_value_at_a_small_chunk_width(
    monkeypatch, batch_size, n
):
    """Same equality, but with `_BATCH` shrunk so the chunk seam is crossed many
    times over few values.

    At the real width of 4096 a slicing bug that silently DROPS one value per
    chunk is invisible: 128 permutations each take a minimum over thousands of
    values, and one missing value almost never moves any of them -- an off-by-one
    stride injection passed the boundary cases above unnoticed. Shrinking the
    width makes each dropped value a large fraction of its chunk, so the minima
    move and the equality actually bites.
    """
    monkeypatch.setattr("graphify._minhash._BATCH", batch_size)
    vs = [f"v{i}".encode() for i in range(n)]
    loop = MinHash(128)
    for v in vs:
        loop.update(v)
    batch = MinHash(128)
    batch.update_batch(vs)
    assert np.array_equal(loop.hashvalues, batch.hashvalues)


def test_update_batch_is_order_independent():
    """The sketch is an elementwise min, so shingle order must not matter.

    `_make_minhash` feeds it a set, whose iteration order is not stable across
    processes when PYTHONHASHSEED varies -- if order mattered, dedup would
    disagree with itself run to run.
    """
    import random
    vs = [f"tok{i}".encode() for i in range(200)]
    a = MinHash(128)
    a.update_batch(vs)
    shuffled = vs[:]
    random.Random(11).shuffle(shuffled)
    b = MinHash(128)
    b.update_batch(shuffled)
    assert np.array_equal(a.hashvalues, b.hashvalues)


def test_update_batch_composes_with_update_and_with_itself():
    """Batches must accumulate, not replace -- `hashvalues` is a running min."""
    first = [b"aa", b"bb", b"cc"]
    second = [b"dd", b"ee"]
    loop = MinHash(128)
    for v in first + second:
        loop.update(v)
    mixed = MinHash(128)
    mixed.update_batch(first)
    mixed.update_batch(second)
    assert np.array_equal(loop.hashvalues, mixed.hashvalues)
    interleaved = MinHash(128)
    interleaved.update(first[0])
    interleaved.update_batch(first[1:])
    interleaved.update_batch(second)
    assert np.array_equal(loop.hashvalues, interleaved.hashvalues)


def test_update_batch_on_empty_leaves_the_sketch_untouched():
    m = MinHash(128)
    before = m.hashvalues.copy()
    m.update_batch([])
    assert np.array_equal(m.hashvalues, before)


def test_update_batch_accepts_a_generator():
    """`_make_minhash` may pass any iterable; the method must not require len()."""
    vs = [b"x", b"y", b"z"]
    gen = MinHash(128)
    gen.update_batch(v for v in vs)
    ref = MinHash(128)
    ref.update_batch(vs)
    assert np.array_equal(gen.hashvalues, ref.hashvalues)

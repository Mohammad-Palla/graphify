"""MinHash + band-LSH — datasketch-compatible drop-in (no scipy).

datasketch.lsh has `from scipy.integrate import quad` at module level.
scipy's array_api_compat layer then lazily loads numpy.testing, which calls
platform.machine() at import time to set test-skip decorator constants — and
that in turn spawns cmd.exe via subprocess, hanging for minutes under EDR
software in corporate Windows environments.

Covers the exact MinHash/MinHashLSH API surface used by dedup.py.
Hash family (Mersenne-prime permutations) and LSH band structure are
equivalent to datasketch so dedup quality is unchanged.
"""
from __future__ import annotations
import hashlib
import struct

import numpy as np


_MP = np.uint64((1 << 61) - 1)  # Mersenne prime for the hash family
_MH = np.uint64(0xFFFF_FFFF)    # mask to 32-bit values

# Values per vectorized `update_batch` step -- caps the intermediate at
# _BATCH x num_perm uint64 (4 MB at the 128 perms dedup uses).
_BATCH = 4096

# One (a, b) coefficient array per num_perm, shared across all instances.
_MH_COEFFS: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _mh_coeffs(num_perm: int) -> tuple[np.ndarray, np.ndarray]:
    if num_perm not in _MH_COEFFS:
        rng = np.random.RandomState(1)
        a = rng.randint(1, int(_MP), num_perm, dtype=np.uint64)
        b = rng.randint(0, int(_MP), num_perm, dtype=np.uint64)
        _MH_COEFFS[num_perm] = (a, b)
    return _MH_COEFFS[num_perm]


class MinHash:
    """MinHash sketch — same API as datasketch.MinHash for the subset used here."""

    __slots__ = ("num_perm", "hashvalues", "_a", "_b")

    def __init__(self, num_perm: int = 128) -> None:
        self.num_perm = num_perm
        self.hashvalues = np.full(num_perm, int(_MH), dtype=np.uint64)
        self._a, self._b = _mh_coeffs(num_perm)

    def update(self, v: bytes) -> None:
        hv = np.uint64(struct.unpack("<I", hashlib.sha1(v).digest()[:4])[0])
        phv = np.bitwise_and((self._a * hv + self._b) % _MP, _MH)
        self.hashvalues = np.minimum(self.hashvalues, phv)

    def update_batch(self, vs) -> None:
        """Absorb many values at once. Equivalent to `update` in a loop.

        Same permutation family, same arithmetic, same dtype -- the only change is
        the shape it runs at. `update` evaluates a 128-wide numpy expression per
        VALUE, so a dedup pass paid four numpy dispatches plus a temporary array
        per shingle: 394,940 calls and 1.95s on a django build, the single largest
        line in the whole `build` stage. Unlike everything else in that stage this
        work is not redundant -- all 8,495 sketches are distinct -- so the only
        lever is to stop paying per-element overhead on a per-element loop.

        Identical output, not merely equivalent: `min` over the permuted values is
        associative and commutative, and every intermediate is exact in uint64
        (`_MP` is 2**61-1, so `a*hv + b` wraps the same way at any shape). Asserted
        against the scalar loop in `tests/test_minhash.py`.

        Chunked because the intermediate is len(vs) x num_perm uint64 -- 1 KB per
        value at num_perm=128. Labels shingle to tens of values, but this is a
        library method and a caller with a large document should not materialize a
        multi-hundred-MB temporary.
        """
        vs = list(vs)
        if not vs:
            return
        acc = self.hashvalues
        for i in range(0, len(vs), _BATCH):
            chunk = vs[i : i + _BATCH]
            hv = np.frombuffer(
                b"".join(hashlib.sha1(v).digest()[:4] for v in chunk), dtype="<u4"
            ).astype(np.uint64)
            phv = np.bitwise_and(
                (self._a[None, :] * hv[:, None] + self._b[None, :]) % _MP, _MH
            )
            acc = np.minimum(acc, phv.min(axis=0))
        self.hashvalues = acc


def _lsh_integrate(f, lo: float, hi: float, n: int = 128) -> float:
    """Numerical integration — replaces scipy.integrate.quad for LSH param search."""
    h = (hi - lo) / n
    return h * sum(f(lo + i * h) for i in range(n))


_LSH_PARAMS_CACHE: dict[tuple[float, int], tuple[int, int]] = {}


def _optimal_lsh_params(threshold: float, num_perm: int) -> tuple[int, int]:
    """Find (bands, rows) that minimise weighted FP+FN error, without scipy."""
    key = (threshold, num_perm)
    if key in _LSH_PARAMS_CACHE:
        return _LSH_PARAMS_CACHE[key]
    best_err, best = float("inf"), (1, 1)
    for b in range(1, num_perm + 1):
        for r in range(1, num_perm // b + 1):
            fp = _lsh_integrate(
                lambda s, _b=float(b), _r=float(r): 1 - (1 - s ** _r) ** _b,
                0.0, threshold,
            )
            fn = _lsh_integrate(
                lambda s, _b=float(b), _r=float(r): 1 - (1 - (1 - s ** _r) ** _b),
                threshold, 1.0,
            )
            err = 0.5 * fp + 0.5 * fn
            if err < best_err:
                best_err, best = err, (b, r)
    _LSH_PARAMS_CACHE[key] = best
    return best


class MinHashLSH:
    """Band-hashing LSH — same API as datasketch.MinHashLSH for the subset used here."""

    def __init__(self, threshold: float = 0.5, num_perm: int = 128) -> None:
        self.b, self.r = _optimal_lsh_params(threshold, num_perm)
        self._tables: list[dict[bytes, list[str]]] = [{} for _ in range(self.b)]
        self._keys: set[str] = set()

    def insert(self, key: str, minhash: MinHash) -> None:
        if key in self._keys:
            raise ValueError(f"Key {key!r} already exists in MinHashLSH")
        self._keys.add(key)
        hv = minhash.hashvalues
        for i, table in enumerate(self._tables):
            band = hv[i * self.r : (i + 1) * self.r].tobytes()
            table.setdefault(band, []).append(key)

    def query(self, minhash: MinHash) -> list[str]:
        hv = minhash.hashvalues
        candidates: set[str] = set()
        for i, table in enumerate(self._tables):
            band = hv[i * self.r : (i + 1) * self.r].tobytes()
            candidates.update(table.get(band, []))
        return list(candidates)

"""`extract()` called twice in ONE process must not reuse filesystem state.

Every gate in this project runs a single cold build in a fresh interpreter, so a
module-level cache is always empty when they look at it. That is exactly the
blind spot a memo on `resolve_cached`'s output slipped through: it produced a
LOST call edge and a FABRICATED duplicate symbol node on the second rebuild, and
the cold-build fingerprint, the four-corpus quality gate and the whole suite all
passed it.

`graphify watch` (watch.py calls extract() in a loop) and the MCP server are the
real callers that rebuild repeatedly in one process, and `extract()` clears
`clear_resolve_cache()` / `_clear_resolution_caches()` at its top precisely so
they observe layout changes. These tests pin that contract: a second extract in
the same process must agree with a first extract of the same tree, including
when a symlink has been replaced by a real file in between.
"""

from __future__ import annotations

import json
import os
import shutil

import pytest

from graphify.extract import extract


def _write(root, rel: str, text: str):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def _sources(root):
    return sorted(root.rglob("*.ts"))


def _shape(result: dict) -> str:
    """Canonical form of the graph, order-independent for comparison."""
    return json.dumps(
        {
            "nodes": sorted(json.dumps(n, sort_keys=True, default=str)
                            for n in result.get("nodes") or []),
            "edges": sorted(json.dumps(e, sort_keys=True, default=str)
                            for e in result.get("edges") or []),
        },
        sort_keys=True,
    )


def test_symlink_replaced_by_real_file_between_rebuilds(tmp_path):
    """The shape a stale resolve-memo corrupts: symlink -> real file, same process.

    Run 2 must match what a fresh process would produce for the post-swap tree,
    which is asserted here as "no duplicate ids and the call edge survives".
    """
    root = tmp_path / "proj"
    _write(root, "vendor/util.ts", "export function helper() { return 1; }\n")
    _write(root, "src/main.ts",
           'import { helper } from "./util";\nexport function run() { return helper(); }\n')
    os.symlink("../vendor/util.ts", root / "src" / "util.ts")

    first = extract(_sources(root), root=root)
    assert first.get("nodes"), "fixture produced no nodes; the test proves nothing"

    # Replace the symlink with a real file of identical content.
    link = root / "src" / "util.ts"
    os.unlink(link)
    shutil.copy(root / "vendor" / "util.ts", link)

    second = extract(_sources(root), root=root)

    ids = [n["id"] for n in second["nodes"] if "id" in n]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, (
        f"second in-process extract fabricated duplicate node ids: {sorted(dupes)[:5]}. "
        "A cache keyed on a pre-swap resolve() stopped joining with a live "
        "path.resolve(), so ensure_symbol_node appended instead of matching."
    )


def test_repeated_extract_of_an_unchanged_tree_is_stable(tmp_path):
    """Nothing changed on disk, so rebuild N must equal rebuild 1, exactly.

    Broader than the symlink case: any per-run cache that leaks across calls
    shows up here as drift between otherwise identical rebuilds.
    """
    root = tmp_path / "proj"
    _write(root, "a.ts", "export class A { go() { return 1; } }\n")
    _write(root, "b.ts", 'import { A } from "./a";\nexport function use() { return new A().go(); }\n')

    shapes = [_shape(extract(_sources(root), root=root)) for _ in range(3)]
    assert shapes[0] == shapes[1] == shapes[2], (
        "repeated in-process extract() of an unchanged tree drifted; "
        "some per-run cache is outliving its run"
    )


def test_a_file_appearing_between_rebuilds_is_picked_up(tmp_path):
    """A cache that outlives the run can also make NEW files invisible."""
    root = tmp_path / "proj"
    _write(root, "a.ts", "export function first() { return 1; }\n")
    first = extract(_sources(root), root=root)
    n_first = len(first["nodes"])

    _write(root, "b.ts", "export function second() { return 2; }\n")
    second = extract(_sources(root), root=root)

    assert len(second["nodes"]) > n_first, (
        "a file created between in-process rebuilds contributed no nodes"
    )
    labels = {n.get("label") for n in second["nodes"]}
    assert any(l and "second" in str(l) for l in labels), (
        f"the new file's symbol never reached the graph; labels={sorted(map(str, labels))[:10]}"
    )

"""Tests for atomic JSON writes (graph.json / manifest.json).

A crash, kill, or disk-full mid-write must not leave a truncated/corrupt file
that a later load chokes on. `write_text_atomic` writes a temp file in the same
directory then `os.replace`s it into place; on failure the original is untouched.
"""
import json
import os

import pytest

from graphify.paths import write_text_atomic


def test_write_text_atomic_writes_and_leaves_no_tmp(tmp_path):
    p = tmp_path / "out" / "graph.json"  # parent doesn't exist yet
    write_text_atomic(p, '{"a": 1}')
    assert json.loads(p.read_text()) == {"a": 1}
    # No leftover temp file in the target directory.
    assert [x.name for x in p.parent.iterdir()] == ["graph.json"]


def test_write_text_atomic_preserves_existing_on_failure(tmp_path, monkeypatch):
    p = tmp_path / "graph.json"
    p.write_text("original", encoding="utf-8")

    def boom(src, dst):
        raise OSError("simulated disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        write_text_atomic(p, "content-that-must-not-land")

    # The original file is intact and the temp file was cleaned up.
    assert p.read_text() == "original"
    assert sorted(x.name for x in tmp_path.iterdir()) == ["graph.json"]


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows has no POSIX mode bits: chmod only toggles the read-only "
           "attribute, and st_mode reports 0o666 for any writable file, so "
           "chmod(0o644) followed by an equality check can never hold",
)
def test_write_text_atomic_preserves_existing_mode(tmp_path):
    # An atomic replace must not tighten a 0644 file to mkstemp's 0600 default.
    p = tmp_path / "graph.json"
    p.write_text("{}", encoding="utf-8")
    os.chmod(p, 0o644)
    write_text_atomic(p, '{"x": 1}')
    assert (os.stat(p).st_mode & 0o777) == 0o644


@pytest.mark.skipif(os.name != "nt", reason="Windows read-only attribute semantics")
def test_write_text_atomic_refuses_a_readonly_destination_without_leaking_a_temp(
    tmp_path,
):
    """The Windows analogue of the mode-preservation contract.

    There is no POSIX mode to preserve here, so the property worth pinning is
    the one Windows actually has: a read-only destination is NOT silently
    overwritten, the original survives intact, and — the part that used to be
    wrong — no temp file is left behind.

    `_atomic_replace` chmods the temp to match the destination, so against a
    read-only target the temp is read-only too; Windows then refuses to unlink
    it and the cleanup swallowed the error, dropping a `.graph.json.*.tmp` into
    the output directory on every failed write.

    Note this diverges from POSIX, where `os.replace` needs only directory write
    permission and so happily replaces a read-only file. Documented rather than
    "fixed": refusing to overwrite a file the user marked read-only is the
    defensible behaviour.
    """
    import stat as _stat

    p = tmp_path / "graph.json"
    p.write_text("original", encoding="utf-8")
    os.chmod(p, _stat.S_IREAD)
    try:
        with pytest.raises(PermissionError):
            write_text_atomic(p, "replaced")

        assert p.read_text(encoding="utf-8") == "original", "read-only file was clobbered"
        assert [x.name for x in tmp_path.iterdir()] == ["graph.json"], (
            f"failed write leaked a temp file: {[x.name for x in tmp_path.iterdir()]}"
        )
    finally:
        os.chmod(p, _stat.S_IWRITE)  # let tmp_path cleanup remove it


def test_write_text_atomic_new_file_respects_umask(tmp_path):
    # A brand-new file must land at the umask default (e.g. 0644), NOT mkstemp's
    # 0600 — otherwise every fresh graph.json would be owner-only.
    p = tmp_path / "new.json"
    write_text_atomic(p, "{}")
    umask = os.umask(0)
    os.umask(umask)
    assert (os.stat(p).st_mode & 0o777) == (0o666 & ~umask)


def test_write_text_atomic_writes_through_symlink(requires_symlinks, tmp_path):
    # Shared-output setups symlink graph.json to shared storage; the atomic write
    # must update the target and keep the link, not replace it with a real file.
    target = tmp_path / "real.json"
    target.write_text("old", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    write_text_atomic(link, "new")
    assert link.is_symlink()
    assert target.read_text() == "new"


def test_write_json_atomic_roundtrip(tmp_path):
    from graphify.paths import write_json_atomic

    p = tmp_path / "g.json"
    write_json_atomic(p, {"nodes": [1, 2], "x": "é"}, indent=2)
    assert json.loads(p.read_text()) == {"nodes": [1, 2], "x": "é"}
    assert not any(name.name.endswith(".tmp") for name in tmp_path.iterdir())


def test_to_json_writes_atomically_no_tmp_leftover(tmp_path):
    import networkx as nx
    from graphify.export import to_json

    G = nx.Graph()
    G.add_node("a", label="a", file_type="code")
    G.add_node("b", label="b", file_type="code")
    G.add_edge("a", "b")
    out = tmp_path / "graph.json"
    assert to_json(G, {}, str(out), force=True) is True
    json.loads(out.read_text())  # valid JSON
    assert not any(x.name.endswith(".tmp") for x in tmp_path.iterdir())


def test_save_manifest_writes_atomically(tmp_path):
    from graphify.detect import save_manifest

    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    mpath = tmp_path / "graphify-out" / "manifest.json"
    save_manifest({"code": [str(tmp_path / "a.py")]}, manifest_path=str(mpath),
                  kind="both", root=tmp_path)
    assert json.loads(mpath.read_text())  # non-empty, valid JSON
    assert not any(x.name.endswith(".tmp") for x in mpath.parent.iterdir())


def test_write_text_atomic_windows_permission_fallback(tmp_path, monkeypatch):
    """On Windows os.replace raises PermissionError when the destination is
    briefly locked (antivirus, an open reader); the copy-then-delete fallback
    must still land the new content and leave no temp file."""
    p = tmp_path / "graph.json"
    p.write_text("original", encoding="utf-8")

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        raise PermissionError("simulated WinError 5")

    monkeypatch.setattr(os, "replace", flaky_replace)
    write_text_atomic(p, "new-content")

    assert calls["n"] == 1  # the fallback path was actually exercised
    assert p.read_text() == "new-content"
    assert sorted(x.name for x in tmp_path.iterdir()) == ["graph.json"]


def test_write_json_atomic_ensure_ascii_false_preserves_utf8(tmp_path):
    from graphify.paths import write_json_atomic

    p = tmp_path / "g.json"
    write_json_atomic(p, {"label": "Wörker 数据"}, ensure_ascii=False)
    raw = p.read_text(encoding="utf-8")
    assert "Wörker 数据" in raw  # raw UTF-8, not \\uXXXX escapes
    assert "\\u" not in raw
    assert json.loads(raw) == {"label": "Wörker 数据"}


# ── _stream_json ──────────────────────────────────────────────────────────────
#
# `write_json_atomic` streams rather than materializing the document, which is
# right for a 62 MB graph.json. But `json.dump` streams with a Python-level
# `for chunk in iterencode(obj): fp.write(chunk)`, and the encoder yields a chunk
# per TOKEN -- 10,372,153 iterations and 10,372,153 write calls for django's
# graph.json, 3.6s of superset's 14.1s profiled json.dump spent on dispatch.
# `_stream_json` joins a bounded block of chunks so the iteration runs in C,
# keeping the memory bound (4096 chunks, a few hundred KB) that `json.dumps`
# would give up.
#
# The whole change rests on the output being byte-identical, so that is what is
# asserted -- against `json.dump` itself, not against a fixture.

import io
import json as _json

import pytest

from graphify.paths import _stream_json


_JSON_CASES = [
    {"a": 1, "b": [1, 2, {"c": None}], "d": "ünïcødé", "e": True, "f": 1.5},
    [], {}, [[]], {"k": {}}, {"x": [{"y": [1, [2, [3]]]}]},
    {"deep": {"a": {"b": {"c": {"d": [1, 2, 3]}}}}},
    list(range(5000)),
    {str(i): {"v": i, "s": "x" * 10} for i in range(2000)},
    "bare string", 42, None, True, 3.14159, -0.0,
    {"esc": '"\\\n\t\r\x00', "empty_key": ""},
]


@pytest.mark.parametrize("obj", _JSON_CASES, ids=range(len(_JSON_CASES)))
@pytest.mark.parametrize("indent", [None, 0, 2, 4])
@pytest.mark.parametrize("ensure_ascii", [True, False])
def test_stream_json_is_byte_identical_to_json_dump(obj, indent, ensure_ascii):
    ref = io.StringIO()
    _json.dump(obj, ref, indent=indent, ensure_ascii=ensure_ascii)
    got = io.StringIO()
    _stream_json(obj, got, indent=indent, ensure_ascii=ensure_ascii)
    assert got.getvalue() == ref.getvalue()


def test_stream_json_spans_many_blocks():
    """A document large enough to cross the block boundary many times -- an
    off-by-one in the islice loop would drop or duplicate a block, and a
    single-block document could never show it."""
    from graphify.paths import _JSON_BLOCK_CHUNKS
    obj = [{"i": i, "s": f"value-{i}"} for i in range(_JSON_BLOCK_CHUNKS * 3)]
    ref = io.StringIO()
    _json.dump(obj, ref, indent=2, ensure_ascii=True)
    got = io.StringIO()
    _stream_json(obj, got, indent=2, ensure_ascii=True)
    assert got.getvalue() == ref.getvalue()
    assert len(got.getvalue()) > 100_000


def test_stream_json_does_not_materialize_the_whole_document():
    """The memory bound is the reason this is not simply `f.write(json.dumps(...))`.

    The largest single string handed to `write` must stay far below the total,
    or the streaming guarantee `write_json_atomic` documents is gone.
    """
    obj = [{"i": i, "payload": "x" * 200} for i in range(20_000)]

    class Recorder(io.StringIO):
        def __init__(self):
            super().__init__()
            self.largest = 0

        def write(self, s):
            self.largest = max(self.largest, len(s))
            return super().write(s)

    rec = Recorder()
    _stream_json(obj, rec, indent=2, ensure_ascii=True)
    total = len(rec.getvalue())
    assert total > 4_000_000
    assert rec.largest < total // 10, (
        f"largest single write {rec.largest} of {total} -- not streaming"
    )


def test_write_json_atomic_round_trips_through_stream_json(tmp_path):
    from graphify.paths import write_json_atomic
    obj = {"nodes": [{"id": f"n{i}", "label": "ünïcødé"} for i in range(500)]}
    dest = tmp_path / "graph.json"
    write_json_atomic(dest, obj, indent=2, ensure_ascii=False)
    assert _json.loads(dest.read_text(encoding="utf-8")) == obj
    ref = io.StringIO()
    _json.dump(obj, ref, indent=2, ensure_ascii=False)
    assert dest.read_text(encoding="utf-8") == ref.getvalue()

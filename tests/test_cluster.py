import json
import sys
import networkx as nx
from pathlib import Path
from graphify.build import build_from_json
from graphify.cluster import cluster, cohesion_score, remap_communities_to_previous, score_all

FIXTURES = Path(__file__).parent / "fixtures"

def make_graph():
    return build_from_json(json.loads((FIXTURES / "extraction.json").read_text()))

def test_cluster_returns_dict():
    G = make_graph()
    communities = cluster(G)
    assert isinstance(communities, dict)

def test_cluster_covers_all_nodes():
    G = make_graph()
    communities = cluster(G)
    all_nodes = {n for nodes in communities.values() for n in nodes}
    assert all_nodes == set(G.nodes)

def test_cohesion_score_complete_graph():
    G = nx.complete_graph(4)
    G = nx.relabel_nodes(G, {i: str(i) for i in G.nodes})
    score = cohesion_score(G, list(G.nodes))
    assert score == 1.0

def test_cohesion_score_single_node():
    G = nx.Graph()
    G.add_node("a")
    score = cohesion_score(G, ["a"])
    assert score == 1.0

def test_cohesion_score_disconnected():
    G = nx.Graph()
    G.add_nodes_from(["a", "b", "c"])
    score = cohesion_score(G, ["a", "b", "c"])
    assert score == 0.0

def test_cohesion_score_range():
    G = make_graph()
    communities = cluster(G)
    for cid, nodes in communities.items():
        score = cohesion_score(G, nodes)
        assert 0.0 <= score <= 1.0

def test_score_all_keys_match_communities():
    G = make_graph()
    communities = cluster(G)
    scores = score_all(G, communities)
    assert set(scores.keys()) == set(communities.keys())


def test_cluster_does_not_write_to_stdout(capsys):
    """Clustering should not emit ANSI escape codes or other output.

    graspologic's leiden() can emit ANSI escape sequences that break
    PowerShell 5.1's scroll buffer on Windows (issue #19). The output
    suppression in _partition() should prevent any output from leaking.
    """
    G = make_graph()
    cluster(G)
    captured = capsys.readouterr()
    assert captured.out == "", f"cluster() wrote to stdout: {captured.out!r}"


def test_cluster_does_not_write_to_stderr(capsys):
    """Same as above but for stderr — ANSI codes can go to either stream."""
    G = make_graph()
    cluster(G)
    captured = capsys.readouterr()
    # Allow logging output (starts with [graphify]) but no raw ANSI codes
    for line in captured.err.splitlines():
        assert "\x1b" not in line, f"cluster() wrote ANSI to stderr: {line!r}"


def test_remap_communities_to_previous_reuses_old_ids():
    communities = {
        10: ["a", "b", "c"],
        11: ["d", "e"],
    }
    previous = {"a": 5, "b": 5, "c": 5, "d": 1, "e": 1}
    remapped = remap_communities_to_previous(communities, previous)
    assert set(remapped.keys()) == {1, 5}
    assert remapped[5] == ["a", "b", "c"]
    assert remapped[1] == ["d", "e"]


def test_remap_communities_to_previous_assigns_deterministic_new_ids():
    communities = {
        7: ["x", "y", "z"],
        8: ["m"],
    }
    previous = {"a": 3}
    remapped = remap_communities_to_previous(communities, previous)
    assert list(remapped.keys()) == [0, 1]
    assert remapped[0] == ["x", "y", "z"]
    assert remapped[1] == ["m"]


def _grouping(partition):
    """Canonicalize {node: community_id} into a set of frozenset node-groups,
    so two partitions compare equal regardless of the community-id labels."""
    from collections import defaultdict
    groups = defaultdict(set)
    for node, cid in partition.items():
        groups[cid].add(node)
    return {frozenset(s) for s in groups.values()}


def test_native_leiden_matches_graspologic_wrapper(monkeypatch):
    """#3104: the direct graspologic_native path must produce the SAME partition
    as the graspologic wrapper it replaces. Run _partition with the native path
    active, then with _native_leiden forced to fall through to the wrapper, and
    assert identical node groupings. Skips unless both are installed."""
    import importlib.util
    import pytest
    if not (importlib.util.find_spec("graspologic_native")
            and importlib.util.find_spec("graspologic")):
        pytest.skip("graspologic / graspologic_native not installed")
    import graphify.cluster as cl

    # Two triangles joined by a single edge: an unambiguous 2-community split.
    G = nx.Graph()
    for a, b in [("a1", "a2"), ("a1", "a3"), ("a2", "a3"),
                 ("b1", "b2"), ("b1", "b3"), ("b2", "b3"), ("a1", "b1")]:
        G.add_edge(a, b)

    native = cl._partition(G, 1.0)
    monkeypatch.setattr(cl, "_native_leiden", lambda *a, **k: None)
    wrapper = cl._partition(G, 1.0)

    assert _grouping(native) == _grouping(wrapper), (
        f"native path diverged from the wrapper: {native} vs {wrapper}"
    )


def test_native_leiden_returns_none_when_binding_absent(monkeypatch):
    """When graspologic_native cannot be imported, _native_leiden must return
    None so _partition falls through to the wrapper / Louvain, not crash."""
    import graphify.cluster as cl
    monkeypatch.setitem(sys.modules, "graspologic_native", None)  # import → ImportError
    stable = nx.Graph()
    stable.add_edge("x", "y")
    assert cl._native_leiden(stable, 1.0) is None


def test_partition_is_invariant_to_edge_endpoint_orientation():
    """#3146: for an undirected graph, (a,b) and (b,a) are the same edge, but the
    orientation networkx yields can vary across builds/machines. _partition must
    canonicalise endpoints so the ordering fed to the clusterer — and thus the
    resulting communities — is identical regardless of how edges were inserted."""
    import random
    edges = [
        ("a1", "a2"), ("a1", "a3"), ("a2", "a3"), ("a3", "a4"),
        ("b1", "b2"), ("b1", "b3"), ("b2", "b3"), ("b3", "b4"),
        ("a1", "b1"),
    ]

    def build(order, flip):
        G = nx.Graph()
        for n in order:
            G.add_node(n)
        for (u, v) in edges:
            G.add_edge(v, u) if flip else G.add_edge(u, v)
        return G

    nodes = sorted({n for e in edges for n in e})
    forward = build(nodes, flip=False)
    shuffled = list(nodes)
    random.Random(0).shuffle(shuffled)
    flipped = build(shuffled, flip=True)

    from graphify.cluster import _partition
    assert _grouping(_partition(forward, 1.0)) == _grouping(_partition(flipped, 1.0)), (
        "partition drifted with edge-endpoint orientation / insertion order"
    )


# ── _partition's edge ordering ────────────────────────────────────────────────
#
# `_partition` sorts every edge to fix a deterministic order before handing the
# graph to the partitioner (see the #1090 comment: an unstable order changed the
# cohesion-split pass from 70 communities to 69 under a different hash seed). The
# sort key used to end in a `json.dumps` of the edge attributes, computed for
# every edge -- 335,986 encodes and 1.4s of superset's 12.0s profiled `cluster`,
# for a tiebreaker that on an undirected simple graph can never be reached.
#
# It IS reachable on a DiGraph, so it is now computed only for canonical pairs
# that actually repeat. These pin that the ordering is unchanged either way.

def _reference_edge_rows(G):
    """The pre-optimization key: canonical pair, then always the attribute dump."""
    import json
    return sorted(
        G.edges(data=True),
        key=lambda row: (
            *sorted((str(row[0]), str(row[1]))),
            json.dumps(row[2], sort_keys=True, ensure_ascii=False, default=str),
        ),
    )


def _partition_edge_rows(G):
    """`_partition`'s ordering, extracted by capturing what it feeds the graph."""
    import networkx as nx
    import graphify.cluster as cl
    captured = []
    real_add_edge = nx.Graph.add_edge

    def spy(self, u, v, **attrs):
        captured.append((u, v, attrs))
        return real_add_edge(self, u, v, **attrs)

    nx.Graph.add_edge = spy
    try:
        cl._partition(G)
    finally:
        nx.Graph.add_edge = real_add_edge
    return captured


def _same_order(a, b):
    return [(str(u), str(v)) for u, v, _ in a] == [(str(u), str(v)) for u, v, _ in b]


def _same_pairs(a, b):
    """Compare the CANONICAL pair sequence, ignoring each row's orientation.

    `_partition` sorts on the canonical pair but hands `add_edge` the row's
    original orientation, and for an undirected graph that carries no meaning --
    the nodes are pre-added in sorted order, so (a, c) and (c, a) build the same
    `stable`. The invariant is the sequence of pairs, not of orientations."""
    key = lambda rows: [tuple(sorted((str(u), str(v)))) for u, v, _ in rows]
    return key(a) == key(b)


def test_partition_edge_order_matches_the_reference_key_on_an_undirected_graph():
    import networkx as nx
    G = nx.Graph()
    for u, v, w in [("b", "a", 1.0), ("a", "c", 2.0), ("z", "a", 1.0),
                    ("c", "b", 3.0), ("m", "n", 1.0)]:
        G.add_edge(u, v, weight=w, kind="calls")
    assert _same_order(_partition_edge_rows(G), _reference_edge_rows(G))


def test_partition_edge_order_is_independent_of_insertion_order():
    """The whole point of the sort (#1090): the same edge set must produce the
    same order however adjacency iteration happened to yield it."""
    import networkx as nx
    edges = [("b", "a", 1.0), ("a", "c", 2.0), ("z", "a", 1.0), ("c", "b", 3.0)]
    G1, G2 = nx.Graph(), nx.Graph()
    for u, v, w in edges:
        G1.add_edge(u, v, weight=w)
    for u, v, w in reversed(edges):
        G2.add_edge(v, u, weight=w)
    assert _same_pairs(_partition_edge_rows(G1), _partition_edge_rows(G2))


def test_attribute_tiebreaker_still_orders_a_colliding_canonical_pair():
    """A DiGraph yields (A, B) and (B, A) as two rows that collide once
    canonicalised -- exactly the case the dump exists for. It must still be
    applied there, and must still agree with the reference key."""
    import networkx as nx
    G = nx.DiGraph()
    G.add_edge("a", "b", weight=1.0, kind="zzz")
    G.add_edge("b", "a", weight=1.0, kind="aaa")
    G.add_edge("c", "d", weight=1.0, kind="mmm")
    rows = _partition_edge_rows(G)
    ref = _reference_edge_rows(G)
    assert _same_order(rows, ref)
    # The colliding pair is ordered by the dump, so "aaa" precedes "zzz".
    ab = [r for r in rows if {str(r[0]), str(r[1])} == {"a", "b"}]
    assert [r[2]["kind"] for r in ab] == ["aaa", "zzz"]


def test_partition_is_deterministic_across_repeated_calls():
    import networkx as nx
    G = nx.Graph()
    for i in range(40):
        G.add_edge(f"n{i}", f"n{(i * 7) % 40}", weight=1.0)
    from graphify.cluster import _partition
    assert _partition(G) == _partition(G)


def test_no_attribute_dump_is_computed_for_an_undirected_graph(monkeypatch):
    """Pins the optimization, not just its neutrality.

    Every other test here would still pass if the dump were computed for every
    edge -- that is the old behaviour, which was correct, just 335,986 encodes
    and 1.4s of superset's `cluster`. Only counting the calls catches a silent
    regression to it.
    """
    import json as _json
    import networkx as nx
    import graphify.cluster as cl

    calls = []
    real_dumps = _json.dumps
    monkeypatch.setattr(cl.json, "dumps",
                        lambda *a, **k: (calls.append(1), real_dumps(*a, **k))[1])
    G = nx.Graph()
    for i in range(30):
        G.add_edge(f"n{i}", f"n{(i * 11) % 30}", weight=1.0, kind="calls")
    cl._partition(G)
    assert calls == [], f"{len(calls)} attribute dumps on an undirected graph"


def test_the_dump_is_computed_only_for_the_colliding_rows(monkeypatch):
    """On a DiGraph only the rows whose canonical pair repeats need it."""
    import json as _json
    import networkx as nx
    import graphify.cluster as cl

    calls = []
    real_dumps = _json.dumps
    monkeypatch.setattr(cl.json, "dumps",
                        lambda *a, **k: (calls.append(1), real_dumps(*a, **k))[1])
    G = nx.DiGraph()
    G.add_edge("a", "b", weight=1.0)   # collides with (b, a)
    G.add_edge("b", "a", weight=1.0)
    for i in range(20):                # 20 rows that cannot collide
        G.add_edge(f"x{i}", f"y{i}", weight=1.0)
    cl._partition(G)
    assert len(calls) == 2, f"expected 2 dumps for the one colliding pair, got {len(calls)}"

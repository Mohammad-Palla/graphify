"""Regression tests for the rustworkx compatibility layer.

Every test here pins a defect found by adversarial review of the
rustworkx migration. The migration originally shipped with no test
changes at all, and the existing suite was blind to all of them - notably
tests/test_path_cli.py::test_path_deterministic_across_hash_seeds, whose
4-node diamond cannot distinguish "lexicographically smallest path" from
"whatever the traversal happened to reach first", because on that graph
the two agree.
"""
import time

import networkx as nx
import pytest
import rustworkx as rx

from graphify.rx_compat import deterministic_shortest_path, to_rustworkx


def _lexmin_reference(G, source, target, *, undirected):
    """Independent oracle: enumerate every shortest path, take the minimum."""
    H = nx.Graph() if undirected else nx.DiGraph()
    H.add_nodes_from(sorted(G.nodes))
    if undirected:
        H.add_edges_from({(min(u, v), max(u, v)) for u, v in G.edges()})
    else:
        H.add_edges_from(
            {(d.get("_src", u), d.get("_tgt", v)) for u, v, d in G.edges(data=True)}
        )
    return min(nx.all_shortest_paths(H, source, target))


# ── deterministic_shortest_path ───────────────────────────────────────────


def test_path_is_true_lexicographic_minimum_when_fork_happens_late():
    """The fork must be resolved by label order at *every* position.

    A two-hop diamond can't tell lex-min apart from first-reached, because
    the only choice is at position 1. Here the branch is three hops in, so
    an implementation that just returns some shortest path will differ.
    """
    G = nx.Graph()
    nx.add_path(G, ["a", "b", "c"])
    # Two equal-length continuations from "c"; "m_alpha" must win on label.
    nx.add_path(G, ["c", "m_beta", "z"])
    nx.add_path(G, ["c", "m_alpha", "z"])
    got = deterministic_shortest_path(G, "a", "z", undirected=True)
    assert got == ["a", "b", "c", "m_alpha", "z"]
    assert got == _lexmin_reference(G, "a", "z", undirected=True)


def test_parallel_edges_do_not_blow_up_path_search():
    """Parallel relations between one pair are a single step.

    `graphify path` deliberately loads graphs as multigraphs so parallel
    links (a `calls` and a `references` between the same nodes) survive.
    Treating each as a distinct edge multiplied shortest-path enumeration
    combinatorially - measured at 20s/2.8GB for one real query with four
    parallel relations before this was fixed.

    The chain below is deliberately long enough that the old behaviour is
    unambiguously infeasible: 11 hops x 4 parallel relations is 4**11 =
    4,194,304 distinct shortest paths, which took ~7s and ~1GB to
    enumerate. A shorter chain would NOT discriminate - at 6 hops the old
    code finished in 3ms and this test would have passed against it.
    """
    hops = 11
    G = nx.MultiGraph()
    for i in range(hops):
        for rel in ("calls", "references", "uses", "imports"):
            G.add_edge(f"n{i}", f"n{i + 1}", relation=rel)

    start = time.perf_counter()
    got = deterministic_shortest_path(G, "n0", f"n{hops}", undirected=True)
    elapsed = time.perf_counter() - start

    assert got == [f"n{i}" for i in range(hops + 1)]
    # Linear-time construction runs in well under a millisecond here; any
    # regression to per-path enumeration lands orders of magnitude above this.
    assert elapsed < 5.0, f"path search took {elapsed:.1f}s - enumeration regression?"


def test_src_tgt_marker_naming_absent_node_raises_catchable_error():
    """A `_src`/`_tgt` marker may name a node that is not in the graph.

    The codebase guards for exactly this elsewhere (analyze.py's
    `if src_id not in G.nodes`). Callers catch only NetworkXNoPath and
    NodeNotFound, so a raw KeyError would surface as a traceback instead
    of "No path found".
    """
    G = nx.DiGraph()
    G.add_nodes_from(["a", "b", "c"])
    G.add_edge("a", "b")
    G.add_edge("b", "c", _src="GHOST_NODE", _tgt="c")
    with pytest.raises((nx.NetworkXNoPath, nx.NodeNotFound)):
        deterministic_shortest_path(G, "a", "c", undirected=False)


def test_missing_endpoints_and_self_path():
    G = nx.Graph()
    nx.add_path(G, ["a", "b"])
    assert deterministic_shortest_path(G, "a", "a", undirected=True) == ["a"]
    with pytest.raises(nx.NodeNotFound):
        deterministic_shortest_path(G, "a", "nope", undirected=True)
    G.add_node("island")
    with pytest.raises(nx.NetworkXNoPath):
        deterministic_shortest_path(G, "a", "island", undirected=True)


def test_directed_respects_src_tgt_direction():
    G = nx.DiGraph()
    G.add_nodes_from(["a", "b"])
    # Arc stored reversed, with markers recording the true direction (#2309).
    G.add_edge("b", "a", _src="a", _tgt="b")
    assert deterministic_shortest_path(G, "a", "b", undirected=False) == ["a", "b"]
    with pytest.raises(nx.NetworkXNoPath):
        deterministic_shortest_path(G, "b", "a", undirected=False)


@pytest.mark.parametrize("undirected", [True, False])
def test_matches_lexmin_oracle_on_random_graphs(undirected):
    import random

    rng = random.Random(20260827)
    for _ in range(40):
        n = rng.randint(4, 9)
        labels = [f"node_{i:02d}" for i in range(n)]
        G = nx.DiGraph() if not undirected else nx.Graph()
        G.add_nodes_from(labels)
        for u in labels:
            for v in labels:
                if u != v and rng.random() < 0.3:
                    G.add_edge(u, v)
        s, t = rng.choice(labels), rng.choice(labels)
        try:
            expected = _lexmin_reference(G, s, t, undirected=undirected)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            with pytest.raises((nx.NetworkXNoPath, nx.NodeNotFound)):
                deterministic_shortest_path(G, s, t, undirected=undirected)
            continue
        assert deterministic_shortest_path(G, s, t, undirected=undirected) == expected


# ── to_rustworkx ──────────────────────────────────────────────────────────


def test_multigraph_parallel_edges_collapse_to_match_networkx():
    """NetworkX's adjacency dict collapses parallel edges; rustworkx would
    keep them and count each one, silently changing centrality scores and
    so the "bridge node" ranking Graphify reports."""
    G = nx.MultiGraph()
    G.add_edges_from([("s", "a"), ("a", "t"), ("s", "b"), ("b", "t"), ("s", "a")])
    rx_graph, index_to_label = to_rustworkx(G)
    assert rx_graph.num_edges() == 4  # not 5

    nx_scores = nx.betweenness_centrality(G)
    rx_scores = {
        index_to_label[i]: v for i, v in rx.betweenness_centrality(rx_graph).items()
    }
    for node, value in nx_scores.items():
        assert rx_scores[node] == pytest.approx(value)


def test_preserves_directedness_and_self_loops():
    G = nx.DiGraph()
    G.add_nodes_from(["a", "b"])
    G.add_edge("a", "b")
    G.add_edge("a", "a")
    rx_graph, index_to_label = to_rustworkx(G)
    assert isinstance(rx_graph, rx.PyDiGraph)
    assert rx_graph.num_nodes() == 2
    assert sorted(index_to_label) == ["a", "b"]
    a = index_to_label.index("a")
    assert rx_graph.has_edge(a, a)


def test_isolated_nodes_survive_conversion():
    G = nx.Graph()
    G.add_nodes_from(["lonely", "x", "y"])
    G.add_edge("x", "y")
    rx_graph, index_to_label = to_rustworkx(G)
    assert rx_graph.num_nodes() == 3
    assert "lonely" in index_to_label

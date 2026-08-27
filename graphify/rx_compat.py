"""Minimal NetworkX -> rustworkx conversion helpers for the in-progress
graph-engine migration (see GRAPHIFY_RUSTWORKX_MIGRATION_SCOPE.md in the
parent investigation repo for the full audit and rationale).

Kept intentionally small: a helper is added here only once a real call site
needs it, rather than building out a general-purpose bridge layer up front.
"""
from __future__ import annotations
import networkx as nx
import rustworkx as rx


def to_rustworkx(G: nx.Graph) -> tuple[rx.PyGraph | rx.PyDiGraph, list]:
    """Build a rustworkx graph mirroring G's structure and directedness.

    Node/edge attributes are not copied - callers needing them should look
    them up on G by label. Returns (rx_graph, index_to_label), where
    index_to_label[i] is the original NetworkX node id for rustworkx node
    index i.
    """
    rx_graph = rx.PyDiGraph() if G.is_directed() else rx.PyGraph()
    label_to_index: dict = {}
    index_to_label: list = []

    def _index_for(label):
        idx = label_to_index.get(label)
        if idx is None:
            idx = rx_graph.add_node(label)
            label_to_index[label] = idx
            index_to_label.append(label)
        return idx

    for n in G.nodes():
        _index_for(n)
    for u, v in G.edges():
        rx_graph.add_edge(_index_for(u), _index_for(v), None)

    return rx_graph, index_to_label


def deterministic_shortest_path(G: nx.Graph, source, target, *, undirected: bool) -> list:
    """Shortest path between two labels, via rustworkx instead of nx.shortest_path.

    #2074: NetworkX's hash-seeded neighbor views returned an arbitrary route
    among equal-length paths, varying per process. The fix's actual contract
    (see tests/test_path_cli.py::test_path_deterministic_across_hash_seeds)
    is not just "deterministic" but specifically "picks the
    lexicographically-smallest path among ties" - rustworkx's own
    Dijkstra tie-break rule does not match this (confirmed by testing: it
    depends on edge insertion order in a different way than NetworkX's BFS
    does), so instead of relying on either library's incidental internal
    tie-break, this asks rustworkx for *every* shortest path and picks the
    lexicographically-smallest one explicitly in Python - a rule this
    function controls directly rather than one that happens to fall out of
    whichever algorithm is underneath.

    Raises nx.NodeNotFound if source/target isn't a node of G, or
    nx.NetworkXNoPath if no path exists - the same exception types
    nx.shortest_path raises, so callers' existing except clauses don't need
    to change.
    """
    rx_graph = rx.PyGraph() if undirected else rx.PyDiGraph()
    label_to_idx: dict = {}
    for n in sorted(G.nodes):
        label_to_idx[n] = rx_graph.add_node(n)

    if undirected:
        edge_pairs = sorted((min(u, v), max(u, v)) for u, v in G.edges())
    else:
        edge_pairs = sorted(
            (d.get("_src", u), d.get("_tgt", v)) for u, v, d in G.edges(data=True)
        )
    for u, v in edge_pairs:
        rx_graph.add_edge(label_to_idx[u], label_to_idx[v], None)

    if source not in label_to_idx or target not in label_to_idx:
        missing = source if source not in label_to_idx else target
        raise nx.NodeNotFound(f"Node {missing} not found in graph")
    if source == target:
        return [source]

    src_idx, tgt_idx = label_to_idx[source], label_to_idx[target]
    all_paths = rx.all_shortest_paths(rx_graph, src_idx, tgt_idx)
    if not all_paths:
        raise nx.NetworkXNoPath(f"No path between {source} and {target}.")
    labeled_paths = [[rx_graph[i] for i in path] for path in all_paths]
    return min(labeled_paths)

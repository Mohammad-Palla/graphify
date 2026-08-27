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

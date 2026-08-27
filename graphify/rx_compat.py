"""Minimal NetworkX -> rustworkx conversion helpers for the in-progress
graph-engine migration (see GRAPHIFY_RUSTWORKX_MIGRATION_SCOPE.md in the
parent investigation repo for the full audit and rationale).

Kept intentionally small: a helper is added here only once a real call site
needs it, rather than building out a general-purpose bridge layer up front.
"""
from __future__ import annotations
import inspect
import networkx as nx
import rustworkx as rx

_FORK = (
    "git+https://github.com/Mohammad-Palla/rustworkx.git"
    "@feat/leiden-and-cycle-length-bound"
)


def _require_fork() -> None:
    """Fail loudly if the installed rustworkx is upstream rather than the fork.

    Graphify depends on three APIs upstream rustworkx does not provide:
    ``graph_leiden()``, ``betweenness_centrality(k=, seed=)`` and
    ``simple_cycles(length_bound=)``. Upstream installs *successfully* against
    a bare ``rustworkx`` requirement and then fails deep inside a build with a
    bare ``TypeError: unexpected keyword argument`` - or, for clustering,
    doesn't fail at all and silently swaps in a different algorithm. Checking
    once at import turns all three into one actionable message.
    """
    missing = []
    if not hasattr(rx, "graph_leiden"):
        missing.append("graph_leiden()")
    try:
        if "length_bound" not in inspect.signature(rx.simple_cycles).parameters:
            missing.append("simple_cycles(length_bound=)")
    except (TypeError, ValueError):
        # Builtins may not expose a signature; probe behaviourally instead.
        probe = rx.PyDiGraph()
        probe.add_node(None)
        try:
            rx.simple_cycles(probe, length_bound=1)
        except TypeError:
            missing.append("simple_cycles(length_bound=)")
    if missing:
        raise ImportError(
            "graphify requires a patched rustworkx providing "
            + ", ".join(missing)
            + f"; the installed rustworkx ({getattr(rx, '__version__', '?')}) "
            "does not. Install the fork with:\n"
            f"    pip install '{_FORK}'"
        )


_require_fork()


def to_rustworkx(G: nx.Graph) -> tuple[rx.PyGraph | rx.PyDiGraph, list]:
    """Build a rustworkx graph mirroring G's structure and directedness.

    Node/edge attributes are not copied - callers needing them should look
    them up on G by label. Returns (rx_graph, index_to_label), where
    index_to_label[i] is the original NetworkX node id for rustworkx node
    index i.

    Parallel edges are collapsed to one. This is required for parity, not a
    simplification: NetworkX stores adjacency as a dict keyed by neighbour,
    so its unweighted algorithms see a single edge per node pair even on a
    MultiGraph, whereas rustworkx would keep every parallel edge and count
    it. Leaving them in silently changes betweenness scores (measured: a
    MultiGraph diamond with one doubled edge gave nx {0.1667 x4} but
    rustworkx {0.2222, 0.1111, 0.1111, 0.2222}), which would reorder
    Graphify's "bridge node" rankings for no legitimate reason.
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
    seen_pairs: set = set()
    for u, v in G.edges():
        iu, iv = _index_for(u), _index_for(v)
        key = (iu, iv) if G.is_directed() else (min(iu, iv), max(iu, iv))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        rx_graph.add_edge(iu, iv, None)

    return rx_graph, index_to_label


def deterministic_shortest_path(G: nx.Graph, source, target, *, undirected: bool) -> list:
    """Shortest path between two labels, via rustworkx instead of nx.shortest_path.

    #2074: NetworkX's hash-seeded neighbour views returned an arbitrary route
    among equal-length paths, varying per process. The fix's actual contract
    (see tests/test_path_cli.py::test_path_deterministic_across_hash_seeds)
    is not merely "deterministic" but specifically "picks the
    lexicographically-smallest path among ties", and neither library's
    internal tie-break rule reproduces that on its own. So this constructs
    the lex-min path explicitly, by a rule this function owns.

    It does so WITHOUT enumerating the tied paths. Computing every shortest
    path and taking the minimum is correct but unbounded: the number of
    shortest paths can grow combinatorially (measured: 12s / 2.3GB on a
    44-node layered graph, and 20s / 2.8GB on a real corpus once parallel
    relations multiplied the count). Instead:

      1. one reverse traversal gives every node's distance to `target`;
      2. walk forward from `source`, repeatedly stepping to the
         smallest-labelled neighbour that is one step closer to `target`.

    Any neighbour whose distance-to-target is exactly one less lies on some
    shortest path, so every greedy step keeps the path shortest; taking the
    smallest label at each position therefore yields the lexicographically
    smallest shortest path. Cost is a single traversal plus one pass over
    the chosen path - linear, and immune to how many ties exist.

    Raises nx.NodeNotFound if source/target isn't a node of G, or
    nx.NetworkXNoPath if no path exists - the same exception types
    nx.shortest_path raises, so callers' existing except clauses don't need
    to change.
    """
    rx_graph = rx.PyGraph() if undirected else rx.PyDiGraph()
    label_to_idx: dict = {}

    def _idx(label):
        idx = label_to_idx.get(label)
        if idx is None:
            idx = rx_graph.add_node(label)
            label_to_idx[label] = idx
        return idx

    for n in sorted(G.nodes):
        _idx(n)

    if undirected:
        edge_pairs = {(min(u, v), max(u, v)) for u, v in G.edges()}
    else:
        # True direction is NOT raw arc order: legacy canonicalized files
        # persist a flipped arc with _src/_tgt markers (#2309). Those markers
        # can name a node absent from G (the codebase guards for exactly this
        # elsewhere, e.g. analyze.py's `if src_id not in G.nodes`), and
        # NetworkX's add_edges_from would have created it implicitly, so
        # register unknown endpoints rather than raising KeyError.
        edge_pairs = {
            (d.get("_src", u), d.get("_tgt", v)) for u, v, d in G.edges(data=True)
        }
    # Deduplicated above: parallel relations between one pair (a `calls` and a
    # `references`, say) are a single step for pathfinding, and `graphify path`
    # deliberately loads graphs as multigraphs so those parallel links survive.
    for u, v in sorted(edge_pairs):
        rx_graph.add_edge(_idx(u), _idx(v), None)

    if source not in label_to_idx or target not in label_to_idx:
        missing = source if source not in label_to_idx else target
        raise nx.NodeNotFound(f"Node {missing} not found in graph")
    if source == target:
        return [source]

    src_idx, tgt_idx = label_to_idx[source], label_to_idx[target]

    # Step 1: every node's distance to `target`. Traversing the reversed graph
    # from `target` gives that in one pass, rather than one search per
    # candidate node.
    if undirected:
        rev = rx_graph
    else:
        rev = rx_graph.copy()
        rev.reverse()
    lengths = rx.dijkstra_shortest_path_lengths(rev, tgt_idx, edge_cost_fn=lambda _: 1.0)
    dist: dict[int, int] = {tgt_idx: 0}
    for node, d in lengths.items():
        dist[node] = int(d)

    if src_idx not in dist:
        raise nx.NetworkXNoPath(f"No path between {source} and {target}.")

    # Step 2: greedily step to the smallest-labelled neighbour that is
    # strictly closer to the target.
    path = [source]
    current = src_idx
    while current != tgt_idx:
        want = dist[current] - 1
        best_idx = None
        best_label = None
        for nbr in rx_graph.neighbors(current):
            if dist.get(nbr) != want:
                continue
            label = rx_graph[nbr]
            if best_label is None or label < best_label:
                best_label, best_idx = label, nbr
        # `dist[current]` came from these same edges, so a strictly closer
        # neighbour is guaranteed to exist.
        current = best_idx
        path.append(best_label)
    return path

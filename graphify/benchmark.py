"""Token-reduction benchmark - measures how much context graphify saves vs naive full-corpus approach."""
from __future__ import annotations
import sys
import networkx as nx

from graphify.build import edge_data
from graphify.serve import _query_terms
from graphify.paths import default_graph_json as _default_graph_json


_CHARS_PER_TOKEN = 4  # standard approximation


def _safe(unicode_char: str, ascii_fallback: str) -> str:
    """Return unicode_char if stdout can encode it, else ascii_fallback.

    Windows consoles often default to cp1252 which cannot encode box-drawing
    or arrow glyphs; printing them raises UnicodeEncodeError mid-output.
    """
    encoding = getattr(sys.stdout, "encoding", None) or ""
    try:
        unicode_char.encode(encoding)
        return unicode_char
    except (UnicodeEncodeError, LookupError):
        return ascii_fallback


def _hr(width: int = 50) -> str:
    """Horizontal rule that survives non-UTF-8 stdout (e.g. Windows cp1252 console)."""
    return _safe("─", "-") * width


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _query_subgraph_tokens(G: nx.Graph, question: str, depth: int = 3) -> int:
    """Run BFS from best-matching nodes and return estimated tokens in the subgraph context."""
    terms = _query_terms(question)
    scored = []
    for nid, data in G.nodes(data=True):
        label = (data.get("label") or "").lower()
        score = sum(1 for t in terms if t in label)
        if score > 0:
            scored.append((score, nid))
    scored.sort(reverse=True)
    start_nodes = [nid for _, nid in scored[:3]]
    if not start_nodes:
        return 0

    visited: set[str] = set(start_nodes)
    frontier = set(start_nodes)
    edges_seen: list[tuple] = []
    for _ in range(depth):
        next_frontier: set[str] = set()
        for n in frontier:
            for neighbor in G.neighbors(n):
                if neighbor not in visited:
                    next_frontier.add(neighbor)
                    edges_seen.append((n, neighbor))
        visited.update(next_frontier)
        frontier = next_frontier

    lines = []
    for nid in visited:
        d = G.nodes[nid]
        lines.append(f"NODE {d.get('label', nid)} src={d.get('source_file', '')} loc={d.get('source_location', '')}")
    for u, v in edges_seen:
        if u in visited and v in visited:
            d = edge_data(G, u, v)
            lines.append(f"EDGE {G.nodes[u].get('label', u)} --{d.get('relation', '')}--> {G.nodes[v].get('label', v)}")

    return _estimate_tokens("\n".join(lines))


_SAMPLE_QUESTIONS = [
    "how does authentication work",
    "what is the main entry point",
    "how are errors handled",
    "what connects the data layer to the api",
    "what are the core abstractions",
]


def _measure_corpus_words(G, graph_path: str) -> tuple[int | None, str]:
    """Count the words in the files the graph was actually built from.

    Returns ``(words, source)``; ``(None, "unavailable")`` when the corpus
    cannot be read, so the caller declines to print a ratio rather than
    inventing one.

    The graph stores ``source_file`` repo-relative (the "no absolute paths in
    output" contract, #555/#932), so they are resolved against the repo root —
    the parent of the directory holding graph.json. A graph copied away from its
    repo therefore measures nothing, which is the honest answer: those files are
    not there to be read.
    """
    from pathlib import Path

    root = Path(graph_path).resolve().parent.parent
    rel_paths = {
        str(data.get("source_file"))
        for _, data in G.nodes(data=True)
        if data.get("source_file")
    }
    if not rel_paths:
        return None, "unavailable"

    words = 0
    read = 0
    for rel in rel_paths:
        candidate = root / rel
        try:
            words += len(candidate.read_text(
                encoding="utf-8", errors="ignore").split())
        except (OSError, ValueError):
            continue
        read += 1

    # A partial read would understate the corpus and inflate the ratio — the
    # exact failure this replaces. Require nearly all of it.
    if read < len(rel_paths) * 0.95 or words == 0:
        return None, "unavailable"
    return words, "measured"


def run_benchmark(
    graph_path: str | None = None,
    corpus_words: int | None = None,
    questions: list[str] | None = None,
) -> dict:
    """Measure token reduction: corpus tokens vs graphify query tokens.

    Args:
        graph_path: path to the built graph
        corpus_words: total word count from detect() output; if None, estimated from graph
        questions: list of questions to benchmark; defaults to _SAMPLE_QUESTIONS

    Returns dict with: corpus_tokens, avg_query_tokens, reduction_ratio, per_question
    """
    graph_path = graph_path or _default_graph_json()
    # Size-cap check + links/edges normalization + node-link parse. A raw
    # --no-cluster graph stores edges under "edges" and used to KeyError
    # here (#2212).
    from graphify.paths import load_node_link_graph
    G = load_node_link_graph(graph_path)

    corpus_source = "detect"
    if corpus_words is None:
        # NEVER estimate the denominator from the graph. This used to be
        # `G.number_of_nodes() * 50`, which made the headline "Nx fewer tokens"
        # a restatement of graph size: no file on disk contributed to it, and it
        # fired silently whenever `.graphify_detect.json` was absent, which is
        # the common case. On one 495-file repo it understated the real corpus
        # by 4.2x. A benchmark whose denominator is derived from its own
        # numerator is not a measurement, so measure the corpus or decline to
        # print a ratio.
        corpus_words, corpus_source = _measure_corpus_words(G, graph_path)

    if corpus_words is None:
        return {
            "error": "corpus size unknown: cannot compute a token-reduction "
                     "ratio without measuring the corpus. Run `graphify detect` "
                     "to write .graphify_detect.json, or pass corpus_words.",
            "corpus_words": None,
            "corpus_words_source": corpus_source,
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        }

    corpus_tokens = corpus_words * 100 // 75  # words → tokens (100 words ≈ 133 tokens)

    qs = questions or _SAMPLE_QUESTIONS
    per_question = []
    for q in qs:
        qt = _query_subgraph_tokens(G, q)
        if qt > 0:
            per_question.append({"question": q, "query_tokens": qt, "reduction": round(corpus_tokens / qt, 1)})

    if not per_question:
        return {"error": "No matching nodes found for sample questions. Build the graph first."}

    avg_query_tokens = sum(p["query_tokens"] for p in per_question) // len(per_question)
    reduction_ratio = round(corpus_tokens / avg_query_tokens, 1) if avg_query_tokens > 0 else 0

    return {
        "corpus_tokens": corpus_tokens,
        "corpus_words": corpus_words,
        "corpus_words_source": corpus_source,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "avg_query_tokens": avg_query_tokens,
        "reduction_ratio": reduction_ratio,
        "per_question": per_question,
    }


def print_benchmark(result: dict) -> None:
    """Print a human-readable benchmark report."""
    if "error" in result:
        print(f"Benchmark error: {result['error']}")
        return

    print(f"\ngraphify token reduction benchmark")
    print(_hr(50))
    arrow = _safe("→", "->")
    origin = {"detect": "from graphify detect",
              "measured": "measured from source files"}.get(
                  result.get("corpus_words_source", ""), "")
    origin = f" ({origin})" if origin else ""
    print(f"  Corpus:          {result['corpus_words']:,} words {arrow} ~{result['corpus_tokens']:,} tokens{origin}")
    print(f"  Graph:           {result['nodes']:,} nodes, {result['edges']:,} edges")
    print(f"  Avg query cost:  ~{result['avg_query_tokens']:,} tokens")
    print(f"  Reduction:       {result['reduction_ratio']}x fewer tokens per query")
    # State the baseline. "Reduction" is against reading the WHOLE corpus for
    # every question, which no agent does — an agent that greps first opens one
    # to three files. Printing the ratio without the baseline invites reading it
    # as a saving over normal tool use, which it is not.
    print(f"  Baseline:        reading every file for every question;")
    print(f"                   a grep-first agent opens far fewer")
    print(f"\n  Per question:")
    for p in result["per_question"]:
        print(f"    [{p['reduction']}x] {p['question'][:55]}")
    print()

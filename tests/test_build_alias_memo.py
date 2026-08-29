"""The pre-migration alias index derives its per-file half once per FILE.

`build_from_json`'s alias loop runs once per node, but everything it derives from
a node -- the `Path`, the absoluteness verdict, the canonical stem and the
pre-migration stem list -- is a function of `source_file` ALONE. A file
contributes one node per symbol it defines, so on django the loop ran 50,513
times over 3,047 distinct source files (16.6x), costing 152k `Path`
constructions and 1.09s in `_old_file_stems`.

Hoisting that half into a per-source_file memo is only correct if the half that
is genuinely PER NODE stays per node. These tests pin the seam: two nodes from
one file must still get different aliases when their labels and ids differ, and
the cached "skip this file" verdict for an absolute path must not leak onto a
relative one.
"""
from __future__ import annotations

from graphify.build import build_from_json


def _extraction(nodes, edges=None):
    return {"nodes": nodes, "edges": edges or [], "input_tokens": 0, "output_tokens": 0}


def _node(nid, label, sf, **extra):
    return {"id": nid, "label": label, "type": "function", "source_file": sf,
            "confidence": "EXTRACTED", **extra}


def test_two_nodes_from_one_file_keep_their_own_suffixes():
    """The file node (label == basename) takes an empty suffix; a symbol node
    takes the remainder of its id after the canonical stem. Caching the per-file
    half must not collapse those onto one shared answer.

    The edges below reference each node by its PRE-migration id, so they only
    survive if the right alias was registered for the right node.
    """
    nodes = [
        _node("pkg_mod_thing", "thing.py", "pkg/mod/thing.py", type="file"),
        _node("pkg_mod_thing_helper", "helper", "pkg/mod/thing.py"),
    ]
    edges = [
        # legacy one-parent stem forms: `mod.thing` -> mod_thing (+ suffix)
        {"source": "mod_thing", "target": "mod_thing_helper",
         "relation": "contains", "confidence": "EXTRACTED", "source_file": "pkg/mod/thing.py"},
    ]
    G = build_from_json(_extraction(nodes, edges))
    assert G.number_of_nodes() == 2
    # The legacy endpoints resolved onto the two DISTINCT nodes, not onto one.
    assert G.number_of_edges() == 1
    src, tgt = next(iter(G.edges()))
    assert {src, tgt} == {"pkg_mod_thing", "pkg_mod_thing_helper"}


def test_many_nodes_from_one_file_alias_the_same_as_one_node_would():
    """The memo is keyed on source_file, so the 2nd..Nth node of a file take the
    cached entry. They must land on exactly the aliases the 1st node produced."""
    sf = "a/b/mod.py"
    single = build_from_json(_extraction([_node("a_b_mod_one", "one", sf)]))
    many = build_from_json(_extraction([
        _node("a_b_mod_one", "one", sf),
        _node("a_b_mod_two", "two", sf),
        _node("a_b_mod_three", "three", sf),
    ]))
    # `b_mod_one` is the legacy one-parent alias for the first node either way.
    probe = {"source": "b_mod_one", "target": "b_mod_one",
             "relation": "calls", "confidence": "EXTRACTED", "source_file": "pkg/mod/thing.py"}
    g1 = build_from_json(_extraction([_node("a_b_mod_one", "one", sf)], [probe]))
    g2 = build_from_json(_extraction([
        _node("a_b_mod_one", "one", sf),
        _node("a_b_mod_two", "two", sf),
    ], [probe]))
    assert "a_b_mod_one" in g1
    assert "a_b_mod_one" in g2
    assert single.number_of_nodes() == 1 and many.number_of_nodes() == 3


def test_absolute_source_files_do_not_poison_the_relative_ones():
    """The `None` entry is the cached "skip" verdict for an absolute path (#2618:
    baking an on-disk path into ids). It must be scoped to THAT source_file.

    Several of each, interleaved: `node_set` is a set, so which arm the loop sees
    first is arbitrary, and a single absolute node would let a leaking `None`
    escape whenever it happened to sort last. With three absolute files among
    three relative ones, a leak strands at least one legacy edge whatever the
    order.
    """
    nodes, edges = [], []
    for i in range(3):
        nodes.append(_node(f"abs_node_{i}", "thing", f"/tmp/elsewhere{i}/mod.py"))
    for i in range(3):
        sf = f"a/b{i}/mod.py"
        nodes.append(_node(f"a_b{i}_mod", f"mod.py", sf, type="file"))
        nodes.append(_node(f"a_b{i}_mod_helper", "helper", sf))
        # legacy one-parent stem form: `b{i}.mod` -> b{i}_mod (+ suffix)
        edges.append({"source": f"b{i}_mod", "target": f"b{i}_mod_helper",
                      "relation": "contains", "confidence": "EXTRACTED",
                      "source_file": sf})
    G = build_from_json(_extraction(nodes, edges))
    assert G.number_of_nodes() == 9
    # Every legacy edge resolved onto that file's own two nodes.
    assert G.number_of_edges() == 3, "a relative file lost its aliases"
    for i in range(3):
        assert G.has_edge(f"a_b{i}_mod", f"a_b{i}_mod_helper")


def test_same_basename_in_two_directories_stays_distinct():
    """The memo is keyed on the whole source_file, not the basename -- #1504 is
    precisely the collision that keying on a stem would reintroduce."""
    nodes = [
        _node("v1_api_readme", "README.md", "v1/api/README.md", type="file"),
        _node("v2_api_readme", "README.md", "v2/api/README.md", type="file"),
    ]
    G = build_from_json(_extraction(nodes))
    assert G.number_of_nodes() == 2
    assert {"v1_api_readme", "v2_api_readme"} <= set(G.nodes())

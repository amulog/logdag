#!/usr/bin/env python
# coding: utf-8

"""Unit tests for logdag.pknowledge.

Regression tests for two bugs in ``ImportDAG`` rule="prune-unconnected"
found in code review:

* ``_update_edge_prune_unconnected`` called ``self._src_ugraph.has_path(...)``,
  but ``nx.Graph`` has no ``has_path`` method (the correct API is the module
  function ``nx.has_path(G, src, dst)``) -> always raised AttributeError.
* the same method had no ``return pk``, so ``update()`` overwrote ``pk`` with
  ``None`` after the first iteration and returned a broken (None) result.

These exercise the method through ``update()`` without loading a real DAG, by
stubbing the minimal collaborators (``_ldag`` / ``_src_ugraph`` / ``evmap``).
"""

import unittest

import networkx as nx

from logdag import pknowledge


class _FakeEvmap:
    """Maps prior-knowledge node ids to opaque evdefs."""

    def __init__(self, mapping):
        self._mapping = mapping

    def evdef(self, node):
        return self._mapping[node]


class _FakeLDAG:
    """Maps an evdef to a node in the source (imported) graph."""

    def __init__(self, mapping):
        self._mapping = mapping

    def evdef2node(self, evdef, graph):
        return self._mapping[evdef], None


def _make_import_dag(rule, src_ugraph, ldag):
    # bypass __init__ (which loads a real DAG from disk) and inject the
    # minimal attributes used by _update_edge_prune_unconnected / update.
    obj = pknowledge.ImportDAG.__new__(pknowledge.ImportDAG)
    obj._rule = rule
    obj._ldag = ldag
    obj._src_ugraph = src_ugraph
    obj._allow_reverse = True
    return obj


class TestPruneUnconnected(unittest.TestCase):

    def _run(self, src_ugraph):
        evmap = _FakeEvmap({0: "a", 1: "b"})
        ldag = _FakeLDAG({"a": "A", "b": "B"})
        obj = _make_import_dag("prune-unconnected", src_ugraph, ldag)
        pk = pknowledge.PriorKnowledge([0, 1])
        return obj.update(pk, evmap)

    def test_returns_pk_not_none(self):
        # regression: update() must not return None (missing return pk)
        g = nx.Graph()
        g.add_nodes_from(["A", "B"])
        result = self._run(g)
        self.assertIsNotNone(result)

    def test_unconnected_pair_is_pruned(self):
        # A and B exist but have no path -> the pair is marked no-edge.
        # also covers the has_path API bug (old code raised AttributeError).
        g = nx.Graph()
        g.add_nodes_from(["A", "B"])
        result = self._run(g)
        self.assertTrue(result.is_noedge((0, 1)))

    def test_connected_pair_is_kept(self):
        # A and B are connected -> the pair must NOT be pruned.
        g = nx.Graph()
        g.add_edge("A", "B")
        result = self._run(g)
        self.assertFalse(result.is_noedge((0, 1)))


class _FakeLDAGEdges:
    """Records how has_edge was called and reports membership."""

    def __init__(self, edges):
        self._edges = set(edges)
        self.last_original = None
        self.last_allow_reverse = None

    def has_edge(self, evdef1, evdef2, original=True, graph=None,
                 allow_reverse=False):
        self.last_original = original
        self.last_allow_reverse = allow_reverse
        return (evdef1, evdef2) in self._edges


def _make_with_ldag(rule, ldag, allow_reverse=True):
    obj = pknowledge.ImportDAG.__new__(pknowledge.ImportDAG)
    obj._rule = rule
    obj._ldag = ldag
    obj._allow_reverse = allow_reverse
    return obj


class TestPruneForce(unittest.TestCase):
    """_update_edge_prune_force must do BOTH prune and force (it used to be a
    copy of _update_edge_force and never pruned)."""

    def test_adds_edge_rule_when_edge_present(self):
        ldag = _FakeLDAGEdges(edges={("a", "b")})
        obj = _make_with_ldag("prune+force", ldag)
        pk = pknowledge.PriorKnowledge([0, 1])
        evmap = _FakeEvmap({0: "a", 1: "b"})
        obj._update_edge_prune_force(pk, evmap, 0, 1)
        self.assertTrue(pk.is_edge((0, 1)))
        self.assertFalse(pk.is_noedge((0, 1)))

    def test_adds_noedge_rule_when_edge_absent(self):
        ldag = _FakeLDAGEdges(edges=set())
        obj = _make_with_ldag("prune+force", ldag)
        pk = pknowledge.PriorKnowledge([0, 1])
        evmap = _FakeEvmap({0: "a", 1: "b"})
        obj._update_edge_prune_force(pk, evmap, 0, 1)
        self.assertTrue(pk.is_noedge((0, 1)))
        self.assertFalse(pk.is_edge((0, 1)))


class TestAllowReverseKeyword(unittest.TestCase):
    """has_edge's 3rd positional arg is `original`, not `allow_reverse`; the
    force rules must pass allow_reverse as a keyword."""

    def test_force_passes_allow_reverse_as_keyword(self):
        ldag = _FakeLDAGEdges(edges=set())
        # sentinel makes the misbinding visible
        obj = _make_with_ldag("force", ldag, allow_reverse="REV")
        pk = pknowledge.PriorKnowledge([0, 1])
        evmap = _FakeEvmap({0: "a", 1: "b"})
        obj._update_edge_force(pk, evmap, 0, 1)
        self.assertEqual(ldag.last_allow_reverse, "REV")
        self.assertIs(ldag.last_original, True)


if __name__ == "__main__":
    unittest.main()

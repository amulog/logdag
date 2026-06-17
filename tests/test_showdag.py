#!/usr/bin/env python
# coding: utf-8

"""Unit tests for logdag.showdag.

Regression test for a code-review finding: ``LogDAG.evdef2node`` returned
``node, graph.get_node_data(node)``, but networkx Graph/DiGraph has no
``get_node_data`` method (node attributes live in ``graph.nodes[node]``), so
the call raised ``AttributeError``. It is reachable via
``pknowledge.ImportDAG`` (rule="prune-unconnected") and
``visual.comparison``.
"""

import unittest

import networkx as nx

from logdag import showdag


class _FakeEvmap:
    def __init__(self, mapping):
        self._mapping = mapping

    def get_eid(self, evdef):
        return self._mapping[evdef]


class TestEvdef2Node(unittest.TestCase):

    def _make_logdag(self, graph, evmap):
        obj = showdag.LogDAG.__new__(showdag.LogDAG)
        obj.graph = graph
        obj._evmap_original = lambda: evmap
        obj._evmap_input = lambda: evmap
        return obj

    def test_returns_node_and_its_attributes(self):
        g = nx.DiGraph()
        g.add_node(3, label="x")
        obj = self._make_logdag(g, _FakeEvmap({"e": 3}))
        node, data = obj.evdef2node("e")
        self.assertEqual(node, 3)
        self.assertEqual(data, {"label": "x"})

    def test_uses_given_graph(self):
        # node attributes must come from the explicitly passed graph
        g = nx.Graph()
        g.add_node(7, kind="src")
        obj = self._make_logdag(nx.DiGraph(), _FakeEvmap({"e": 7}))
        node, data = obj.evdef2node("e", graph=g)
        self.assertEqual(node, 7)
        self.assertEqual(data, {"kind": "src"})


if __name__ == "__main__":
    unittest.main()

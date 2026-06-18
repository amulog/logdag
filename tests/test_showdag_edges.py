#!/usr/bin/env python
# coding: utf-8

"""Regression tests for logdag.showdag.LogDAG.number_of_edges / edges.

Code-review finding: with an explicit ``graph`` argument both returned the
result of ``remove_edge_duplication`` directly, which is a *generator*. So
``number_of_edges(graph)`` returned a generator instead of a count (breaking the
across-host stats in ``__main__``), and ``edges(graph)`` returned a generator
where the ``graph=None`` path returns a list.
"""

import unittest

import networkx as nx

from logdag import showdag


class _LogDAG(showdag.LogDAG):
    def __init__(self, directed):
        # bypass the real __init__; `directed` decides edge_isdirected
        self._directed = directed

    def edge_isdirected(self, edge, graph=None):
        return self._directed


class TestNumberOfEdges(unittest.TestCase):

    def test_counts_edges_with_graph(self):
        g = nx.DiGraph()
        g.add_edges_from([(0, 1), (1, 2), (2, 3)])
        obj = _LogDAG(directed=True)  # no dedup -> 3 edges
        n = obj.number_of_edges(g)
        self.assertIsInstance(n, int)
        self.assertEqual(n, 3)

    def test_dedup_counts_reverse_pair_once(self):
        g = nx.DiGraph()
        g.add_edges_from([(0, 1), (1, 0)])  # one undirected pair
        obj = _LogDAG(directed=False)       # treated undirected -> dedup to 1
        self.assertEqual(obj.number_of_edges(g), 1)

    def test_edges_with_graph_returns_list(self):
        g = nx.DiGraph()
        g.add_edges_from([(0, 1), (1, 2)])
        obj = _LogDAG(directed=True)
        e = obj.edges(graph=g)
        self.assertIsInstance(e, list)
        self.assertEqual(len(e), 2)


if __name__ == "__main__":
    unittest.main()

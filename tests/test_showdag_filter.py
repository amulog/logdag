#!/usr/bin/env python
# coding: utf-8

"""Regression tests for logdag.showdag_filter._sep_directed.

Code-review finding: ``_sep_directed`` accumulated ``edge[0:2]`` (the (u, v)
pair without the data dict) for the directed graph, so directed edges lost
their attributes (e.g. ``weight``) — and a downstream ``ate_prune`` that reads
the weight then failed. The directed graph now keeps the edge data.
"""

import unittest

import networkx as nx

from logdag import showdag_filter as f


class TestSepDirected(unittest.TestCase):

    def _graph(self):
        g = nx.DiGraph()
        g.add_edge(0, 1, weight=0.5)                       # directed only
        g.add_edge(2, 3, weight=0.7)
        g.add_edge(3, 2, weight=0.7)                       # bidirectional pair
        return g

    def test_directed_keeps_weight(self):
        gd = f.directed(self._graph())
        self.assertTrue(gd.has_edge(0, 1))
        self.assertEqual(gd[0][1]["weight"], 0.5)

    def test_separation(self):
        g = self._graph()
        gd = f.directed(g)
        gn = f.undirected(g)
        # (0,1) is one-way -> directed; (2,3)/(3,2) -> undirected
        self.assertTrue(gd.has_edge(0, 1))
        self.assertFalse(gd.has_edge(2, 3))
        self.assertTrue(gn.has_edge(2, 3))

    def test_undirected_keeps_weight(self):
        gn = f.undirected(self._graph())
        u, v, data = list(gn.edges(data=True))[0]
        self.assertEqual(data.get("weight"), 0.7)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python
# coding: utf-8

"""Regression test for logdag.visual.draw.graph_nx.

Code-review finding: ``graph_nx`` created a pygraphviz AGraph (which holds a
C-level handle) but never called ``ag.close()``, leaking the handle. The
AGraph is now closed in a ``finally`` block, even if ``draw`` raises.

pygraphviz may not be installed, so the AGraph factory is stubbed.
"""

import unittest
from unittest import mock

import networkx as nx

from logdag.visual import draw


class _FakeAGraph:
    def __init__(self):
        self.drawn = None
        self.closed = False

    def draw(self, output, prog=None):
        self.drawn = (output, prog)

    def close(self):
        self.closed = True


class _FailingAGraph(_FakeAGraph):
    def draw(self, output, prog=None):
        raise RuntimeError("boom")


class TestGraphNx(unittest.TestCase):

    def test_closes_agraph_after_draw(self):
        ag = _FakeAGraph()
        with mock.patch.object(nx.nx_agraph, "to_agraph", return_value=ag):
            out = draw.graph_nx("out.png", nx.DiGraph())
        self.assertEqual(out, "out.png")
        self.assertEqual(ag.drawn, ("out.png", "circo"))
        self.assertTrue(ag.closed)

    def test_closes_agraph_even_when_draw_fails(self):
        ag = _FailingAGraph()
        with mock.patch.object(nx.nx_agraph, "to_agraph", return_value=ag):
            with self.assertRaises(RuntimeError):
                draw.graph_nx("out.png", nx.DiGraph())
        self.assertTrue(ag.closed)


if __name__ == "__main__":
    unittest.main()

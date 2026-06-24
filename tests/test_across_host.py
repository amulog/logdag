#!/usr/bin/env python
# coding: utf-8

"""Regression test for showdag_filter._sep_across_host.

Code-review finding (F-M1): the separator compared ``src_evdef.host ==
dst_evdef.host``. A MultipleEventDefinition has no single ``host`` attribute, so
``.host`` raised AttributeError, which the broad ``except AttributeError``
swallowed -- making the whole across-host / within-host filter return
``(None, None)`` silently. It now uses ``all_attr("host")`` (a set, defined for
both single and multiple definitions).
"""

import unittest

import networkx as nx

from logdag import showdag_filter


class _Basic:
    """Single-host evdef: has both .host and all_attr (like EventDefinition)."""
    def __init__(self, host):
        self.host = host

    def all_attr(self, key):
        return {getattr(self, key)}


class _Multi:
    """Multi-host evdef: all_attr only, NO .host (like MultipleEventDefinition);
    the old code raised AttributeError on .host."""
    def __init__(self, hosts):
        self._hosts = set(hosts)

    def all_attr(self, key):
        return set(self._hosts)


class _FakeLdag:
    def __init__(self, mapping):
        self._m = mapping

    def edge_evdef(self, edge):
        return [self._m[edge[0]], self._m[edge[1]]]


class TestSepAcrossHost(unittest.TestCase):

    def _graph(self):
        g = nx.DiGraph()
        g.add_edge(0, 1, weight=1.0)
        return g

    def test_multiple_evdef_not_swallowed(self):
        # one endpoint is a MultipleEventDefinition -> old code returned (None,
        # None) via the swallowed AttributeError
        ldag = _FakeLdag({0: _Basic("h1"), 1: _Multi(["h2"])})
        g_same, g_diff = showdag_filter._sep_across_host(self._graph(), ldag=ldag)
        self.assertIsNotNone(g_diff)
        self.assertIsNotNone(g_same)
        self.assertIn((0, 1), list(g_diff.edges()))   # different hosts
        self.assertNotIn((0, 1), list(g_same.edges()))

    def test_same_host_goes_to_same(self):
        ldag = _FakeLdag({0: _Basic("h1"), 1: _Basic("h1")})
        g_same, g_diff = showdag_filter._sep_across_host(self._graph(), ldag=ldag)
        self.assertIn((0, 1), list(g_same.edges()))
        self.assertNotIn((0, 1), list(g_diff.edges()))

    def test_across_host_returns_diff_subgraph(self):
        ldag = _FakeLdag({0: _Basic("h1"), 1: _Basic("h2")})
        out = showdag_filter.across_host(self._graph(), ldag=ldag)
        self.assertIn((0, 1), list(out.edges()))


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python
# coding: utf-8

"""Regression test for logdag.showdag.apply_filter.

Code-review finding: apply_filter removed "to_undirected" / "no_isolated" from
the caller's ``l_filtername`` list in place. A reused list (e.g. a config-
derived filter list applied to several DAGs) would silently lose those entries
on the second call. apply_filter now operates on a copy.
"""

import unittest
from unittest import mock

import networkx as nx

from logdag import showdag
from logdag import showdag_filter


class TestApplyFilterNoMutation(unittest.TestCase):

    def test_input_list_not_mutated(self):
        g = nx.DiGraph()
        original = ["to_undirected", "no_isolated"]
        passed = list(original)
        with mock.patch.object(showdag_filter, "to_undirected",
                               lambda graph, **kw: graph), \
                mock.patch.object(showdag_filter, "no_isolated",
                                  lambda graph, **kw: graph):
            showdag.apply_filter(ldag=object(), l_filtername=passed, graph=g)
        # old code emptied `passed` via .remove(); it must stay intact
        self.assertEqual(passed, original)

    def test_reuse_list_twice_is_stable(self):
        g = nx.DiGraph()
        shared = ["to_undirected"]
        with mock.patch.object(showdag_filter, "to_undirected",
                               lambda graph, **kw: graph):
            showdag.apply_filter(ldag=object(), l_filtername=shared, graph=g)
            # second call must still see "to_undirected"
            showdag.apply_filter(ldag=object(), l_filtername=shared, graph=g)
        self.assertEqual(shared, ["to_undirected"])


if __name__ == "__main__":
    unittest.main()

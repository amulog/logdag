#!/usr/bin/env python
# coding: utf-8

import datetime
import unittest

from logdag.causaltestdata import variable
import networkx as nx


class TestVariable(unittest.TestCase):

    def test_value(self):
        g = nx.DiGraph()
        g.add_nodes_from([1, 2, 3, 4, 5])
        g.add_edge(1, 2, weight=0.3)
        g.add_edge(2, 3, weight=0.5)
        g.add_edge(4, 3, weight=0.1)
        g.add_edge(5, 4, weight=0.1)
        defaults = {}

        df = variable.generate_all(g, defaults)

    def test_tsevent(self):
        g = nx.DiGraph()
        g.add_nodes_from([1, 2, 3, 4, 5])
        g.add_edge(1, 2, weight=0.3)
        g.add_edge(2, 3, weight=0.5)
        g.add_edge(4, 3, weight=0.1)
        g.add_edge(5, 4, weight=0.1)
        defaults = {"default_type": "tsevent"}

        df = variable.generate_all(g, defaults)

    def test_periodic(self):
        g = nx.DiGraph()
        g.add_nodes_from([1, 2, 3, 4, 5])
        g.add_edge(1, 2, weight=0.3)
        g.add_edge(2, 3, weight=0.5)
        g.add_edge(4, 3, weight=0.1)
        g.add_edge(5, 4, weight=0.1)
        defaults = {"default_type": "periodic",
                    "periodic_interval": datetime.timedelta(minutes=10)}

        df = variable.generate_all(g, defaults)
        self.assertEqual(len(df), len(defaults["variable_index"]))

    def test_periodic_regular_spacing(self):
        # a parentless periodic node fires at exactly the given interval
        interval = datetime.timedelta(minutes=10)
        var = variable.PeriodicEventVariable(
            1, {}, [],
            {"dt_range": (datetime.datetime(2112, 9, 3),
                          datetime.datetime(2112, 9, 4)),
             "dt_interval": datetime.timedelta(minutes=1),
             "delay": datetime.timedelta(0),
             "periodic_interval": interval})
        var.generate([])
        ts = var.ts
        # 24h / 10min = 144 events, exactly interval apart
        self.assertEqual(len(ts), 24 * 6)
        diffs = {ts[i + 1] - ts[i] for i in range(len(ts) - 1)}
        self.assertEqual(diffs, {interval})

    def test_periodic_count(self):
        # periodic_count overrides the interval
        var = variable.PeriodicEventVariable(
            1, {"periodic_count": 48}, [],
            {"dt_range": (datetime.datetime(2112, 9, 3),
                          datetime.datetime(2112, 9, 4)),
             "dt_interval": datetime.timedelta(minutes=1),
             "delay": datetime.timedelta(0)})
        var.generate([])
        self.assertEqual(len(var.ts), 48)

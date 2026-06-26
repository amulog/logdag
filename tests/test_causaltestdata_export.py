#!/usr/bin/env python
# coding: utf-8

"""Unit tests for logdag.causaltestdata.amulog_export (no amulog needed)."""

import datetime
import os
import tempfile
import unittest

import networkx as nx

from logdag.causaltestdata import amulog_export


def _dag():
    # periodic node (1) + poisson tsevent nodes (2, 3) with a causal edge
    g = nx.DiGraph()
    g.add_node(1, type="periodic",
               periodic_interval=datetime.timedelta(minutes=10))
    g.add_node(2, type="tsevent", tsevent_lambd=144)
    g.add_node(3, type="tsevent", tsevent_lambd=144)
    g.add_edge(2, 3, weight=0.5)
    return g


class TestAmulogExport(unittest.TestCase):

    def setUp(self):
        self.dt_range = (datetime.datetime(2112, 9, 3),
                         datetime.datetime(2112, 9, 4))

    def test_rows_sorted_and_formatted(self):
        g = _dag()
        defaults = {"dt_range": self.dt_range}
        node_message = {1: "periodic heartbeat",
                        2: ("hostA", "poisson cause"),
                        3: "poisson effect"}
        variables, rows = amulog_export.generate_log_events(
            g, defaults, node_message)

        # rows are time-sorted
        self.assertEqual(rows, sorted(rows, key=lambda x: x[0]))
        # every row is within dt_range
        for dt, host, mes in rows:
            self.assertTrue(self.dt_range[0] <= dt < self.dt_range[1])

        # periodic node (1): exactly 144 events at 10-min spacing
        n_periodic = sum(1 for _, _, mes in rows if mes == "periodic heartbeat")
        self.assertEqual(n_periodic, 24 * 6)

        # default vs explicit host mapping
        hosts = {mes: host for _, host, mes in rows}
        self.assertEqual(hosts["poisson cause"], "hostA")
        self.assertEqual(hosts["periodic heartbeat"], amulog_export.DEFAULT_HOST)

        # ground-truth variables returned for inspection
        self.assertEqual(set(variables), {1, 2, 3})
        self.assertEqual(len(variables[1].ts), n_periodic)

    def test_line_format_matches_amulog(self):
        line = amulog_export.format_log_line(
            datetime.datetime(2112, 9, 3, 1, 2, 3), "h1", "a message")
        self.assertEqual(line, "2112-09-03 01:02:03 h1 a message")

    def test_non_event_nodes_skipped(self):
        # a continuous variable has no .ts -> produces no log lines
        g = nx.DiGraph()
        g.add_node(1, type="variable")
        g.add_node(2, type="periodic",
                   periodic_interval=datetime.timedelta(minutes=30))
        defaults = {"dt_range": self.dt_range}
        node_message = {1: "should not appear", 2: "periodic"}
        _, rows = amulog_export.generate_log_events(g, defaults, node_message)
        msgs = {mes for _, _, mes in rows}
        self.assertNotIn("should not appear", msgs)
        self.assertIn("periodic", msgs)

    def test_dump_log_writes_file(self):
        g = _dag()
        defaults = {"dt_range": self.dt_range}
        node_message = {1: "periodic", 2: "cause", 3: "effect"}
        fd, path = tempfile.mkstemp()
        os.close(fd)
        try:
            _, rows = amulog_export.dump_log(g, defaults, node_message, path)
            with open(path) as f:
                lines = [ln.rstrip("\n") for ln in f]
            self.assertEqual(len(lines), len(rows))
            # sorted, well-formed "YYYY-MM-DD HH:MM:SS host message"
            for ln in lines:
                head = ln[:19]
                datetime.datetime.strptime(head, amulog_export.LOG_DT_FORMAT)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()

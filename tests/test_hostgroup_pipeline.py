#!/usr/bin/env python
# coding: utf-8

"""Full E2E for host stratification: known DAG -> BGL-style logs -> amulog DB
-> evdb (aggregated by host_tier) -> make-dag.

Demonstrates the amulog request ("chip granularity -> midplane gives a
meaningful DAG") on *synthetic* multi-layer hosts (no loghub data, so no
CC-BY attribution / redistribution concern). The ground-truth causal structure
lets us check both the aggregation (node count drops) and that make-dag
recovers edges among the aggregated nodes.

Two chip hosts under midplane R02-M1 share one template (a common cause);
a host under R03-M0 carries the effect template. With host_tier=midplane the
two R02-M1 chip series merge into one node, and a cause->effect edge can form.
"""

import os
import datetime
import tempfile
import unittest

import numpy as np
import networkx as nx

from amulog import config as amulog_config
from logdag import arguments

from logdag.causaltestdata import amulog_export

# midplane R02-M1: two chip hosts, shared "cause" template
HOST_1A = "R02-M1-N0-C:J12-U11"
HOST_1B = "R02-M1-N1-C:J13-U01"
# midplane R03-M0: one chip host, "effect" template
HOST_2 = "R03-M0-N0-C:J01-U05"
MSG_CAUSE = "kernel ras parity event"
MSG_EFFECT = "kernel machine check abort"

# causaltestdata generates events in this (naive, wall-clock) window; amulog
# parses the log text into its configured tz. The read / make-dag side uses the
# conf's whole_term (also via getterm), so both share one tz convention.
DT_RANGE = (datetime.datetime(2112, 9, 3), datetime.datetime(2112, 9, 4))

HOST_GROUP_CONF = """\
[tiers]
order = midplane
unmatched = keep
label_tier =

[tier_midplane]
schemes = bgl
bgl.regex = ^(R\\d+-M\\d)
"""


class TestHostGroupPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        fd, cls._testlog = tempfile.mkstemp(); os.close(fd)
        fd, cls._amulogdb = tempfile.mkstemp(); os.close(fd)
        fd, cls._ltgendump = tempfile.mkstemp(); os.close(fd)
        fd, cls._amulogconf = tempfile.mkstemp(); os.close(fd)
        fd, cls._hgconf = tempfile.mkstemp(); os.close(fd)

        with open(cls._hgconf, "w") as f:
            f.write(HOST_GROUP_CONF)

        # known DAG: two cause series (same template, two R02-M1 chips) both
        # drive one effect series (R03-M0). Strong weights so the bin-level
        # correlation survives into make-dag.
        g = nx.DiGraph()
        g.add_node("c1a", type="tsevent", tsevent_lambd=120)
        g.add_node("c1b", type="tsevent", tsevent_lambd=120)
        g.add_node("e2", type="tsevent", tsevent_lambd=10)
        g.add_edge("c1a", "e2", weight=0.8)
        g.add_edge("c1b", "e2", weight=0.8)
        node_message = {
            "c1a": (HOST_1A, MSG_CAUSE),
            "c1b": (HOST_1B, MSG_CAUSE),
            "e2": (HOST_2, MSG_EFFECT),
        }
        np.random.seed(11)
        amulog_export.dump_log(g, {"dt_range": DT_RANGE}, node_message,
                               cls._testlog)

        amulog_conf = amulog_config.open_config()
        amulog_conf["general"]["src_path"] = cls._testlog
        amulog_conf["database"]["sqlite3_filename"] = cls._amulogdb
        amulog_conf["manager"]["indata_filename"] = cls._ltgendump
        amulog_conf["manager"]["host_group_filename"] = cls._hgconf
        with open(cls._amulogconf, "w") as f:
            amulog_conf.write(f)

        from amulog import __main__ as amulog_main
        from amulog import manager
        targets = amulog_main.get_targets_conf(amulog_conf)
        manager.process_files_online(amulog_conf, targets, reset_db=True)

    @classmethod
    def tearDownClass(cls):
        for p in (cls._testlog, cls._amulogdb, cls._ltgendump,
                  cls._amulogconf, cls._hgconf):
            os.remove(p)

    def _make_conf(self, host_tier):
        fd, testdb = tempfile.mkstemp(); os.close(fd)
        self.addCleanup(os.remove, testdb)
        conf = amulog_config.open_config(arguments.DEFAULT_CONFIG,
                                         base_default=False)
        conf["general"]["evdb"] = "sql"
        conf["database_sql"]["database"] = "sqlite3"
        conf["database_sql"]["sqlite3_filename"] = testdb
        conf["database_amulog"]["source_conf"] = self._amulogconf
        conf["database_amulog"]["host_tier"] = host_tier
        conf["filter"]["rules"] = ""  # keep all series; filtering tested elsewhere
        return conf

    def _read_evdefs(self, host_tier):
        from logdag.source import evgen_log
        conf = self._make_conf(host_tier)
        w_term = amulog_config.getterm(conf, "dag", "whole_term")
        el = evgen_log.LogEventLoader(conf)
        el.read(dt_range=w_term, dump_org=False)
        return conf, sorted(str(ev) for ev in el.iter_evdef(dt_range=w_term))

    def test_aggregation_reduces_node_count(self):
        _, ev_none = self._read_evdefs("")
        _, ev_mid = self._read_evdefs("midplane")
        # legacy: 3 chip-host series (two share a template but differ by host)
        self.assertEqual(len(ev_none), 3)
        # midplane: the two R02-M1 chips merge -> 2 series
        self.assertEqual(len(ev_mid), 2)
        # the aggregated identifiers are by midplane, not chip
        self.assertTrue(any("R02-M1" in s for s in ev_mid))
        self.assertTrue(any("R03-M0" in s for s in ev_mid))
        self.assertFalse(any("U11" in s or "U01" in s for s in ev_mid))

    def test_makedag_with_midplane_finds_edges(self):
        from logdag import makedag
        conf = self._make_conf("midplane")
        from logdag.source import evgen_log
        w_term = amulog_config.getterm(conf, "dag", "whole_term")
        el = evgen_log.LogEventLoader(conf)
        el.read(dt_range=w_term, dump_org=False)

        am = arguments.ArgumentManager(conf)
        am.generate(arguments.all_args)
        edge_cnt = 0
        for args in am:
            ldag = makedag.makedag_main(args, do_dump=False)
            if ldag is None:
                # unit terms with no data in this window return None
                continue
            edge_cnt += ldag.number_of_edges()
        # cause (R02-M1) -> effect (R03-M0) should surface at least one edge
        self.assertGreater(edge_cnt, 0)


if __name__ == "__main__":
    unittest.main()

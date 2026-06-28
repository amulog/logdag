#!/usr/bin/env python
# coding: utf-8

"""host stratification (amulog host_group) in AmulogLoader.

A synthetic BGL-style host hierarchy (chip -> midplane) drives logs through
amulog; with host_tier set, AmulogLoader aggregates events by host group id.
Uses *synthetic* BGL-format host strings only (no loghub data), so there is no
redistribution / attribution concern.

Backward compat: with host_tier empty the loader must behave exactly as before
(one event series per original host); that is asserted here too.
"""

import os
import datetime
import tempfile
import unittest

import numpy as np
import networkx as nx
from dateutil.tz import tzlocal

from amulog import config as amulog_config
from logdag import arguments
from logdag.causaltestdata import amulog_export

# chip hosts: two under midplane R02-M1, one under R03-M0
HOST_A = "R02-M1-N0-C:J12-U11"
HOST_B = "R02-M1-N0-C:J13-U01"
HOST_C = "R03-M0-N1-C:J01-U05"
MSG = "kernel ras parity event"  # one shared template -> one gid

DT_RANGE = (datetime.datetime(2112, 9, 3), datetime.datetime(2112, 9, 4))
DT_RANGE_TZ = tuple(dt.replace(tzinfo=tzlocal()) for dt in DT_RANGE)

HOST_GROUP_CONF = """\
[tiers]
order = midplane
unmatched = keep
label_tier =

[tier_midplane]
schemes = bgl
bgl.regex = ^(R\\d+-M\\d)
"""


class TestHostGroupAggregation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        fd, cls._testlog = tempfile.mkstemp(); os.close(fd)
        fd, cls._amulogdb = tempfile.mkstemp(); os.close(fd)
        fd, cls._ltgendump = tempfile.mkstemp(); os.close(fd)
        fd, cls._amulogconf = tempfile.mkstemp(); os.close(fd)
        fd, cls._hgconf = tempfile.mkstemp(); os.close(fd)

        with open(cls._hgconf, "w") as f:
            f.write(HOST_GROUP_CONF)

        # DAG: three independent chip-host event series, same message
        g = nx.DiGraph()
        for n in ("a", "b", "c"):
            g.add_node(n, type="tsevent", tsevent_lambd=50)
        node_message = {"a": (HOST_A, MSG), "b": (HOST_B, MSG),
                        "c": (HOST_C, MSG)}
        np.random.seed(7)
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

    def _loader(self, host_tier):
        from logdag.source import src_amulog
        amulog_conf = amulog_config.open_config(self._amulogconf)
        return src_amulog.AmulogLoader(
            amulog_conf, dt_range=DT_RANGE_TZ, gid_name="ltgid",
            host_tier=host_tier)

    def test_no_tier_keeps_original_hosts(self):
        al = self._loader("")  # legacy behaviour
        hosts = {host for host, gid in al.iter_event(dt_range=DT_RANGE_TZ)}
        # diagnostic + backward-compat: the three chip hosts are kept distinct
        self.assertEqual(hosts, {HOST_A, HOST_B, HOST_C})

    def test_midplane_aggregates_hosts(self):
        al = self._loader("midplane")
        events = list(al.iter_event(dt_range=DT_RANGE_TZ))
        hgids = {hgid for hgid, gid in events}
        # two chip hosts collapse to midplane R02-M1, one to R03-M0
        self.assertEqual(hgids, {"R02-M1", "R03-M0"})
        # one gid is shared, so exactly two (hgid, gid) events
        self.assertEqual(len(events), 2)

    def test_aggregated_load_unions_member_hosts(self):
        al_none = self._loader("")
        al_mid = self._loader("midplane")
        gid = next(gid for host, gid in al_none.iter_event(dt_range=DT_RANGE_TZ))

        n_a = len(al_none.load((HOST_A, gid), dt_range=DT_RANGE_TZ))
        n_b = len(al_none.load((HOST_B, gid), dt_range=DT_RANGE_TZ))
        n_mid = len(al_mid.load(("R02-M1", gid), dt_range=DT_RANGE_TZ))
        self.assertGreater(n_a, 0)
        self.assertGreater(n_b, 0)
        self.assertEqual(n_mid, n_a + n_b)


if __name__ == "__main__":
    unittest.main()

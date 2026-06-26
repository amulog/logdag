#!/usr/bin/env python
# coding: utf-8

"""End-to-end: causaltestdata DAG -> amulog DB -> evdb, asserting the
preprocessing filters make log_org (raw) and log_feature (filtered) differ.

This is the ground-truth-structure fixture the project is consolidating logdag
tests onto: a known DAG drives synthetic logs, so the raw vs. filtered series
measurably differ -- exactly the difference logdagviz visualizes.

Observed filter behaviour (logdag defaults), which these tests pin down:
  * a perfectly regular (periodic) series is dropped by remove_linear
    (its cumulative count is a straight line); filter_periodic alone does NOT
    drop it -- see tests/test_period.py and the project debugging notes.
  * a stationary uniform-Poisson series is also dropped by remove_linear.
  * a low-count series (< linear_count = 10 events) skips remove_linear and
    survives -- the only easy way to keep a series with causaltestdata's
    current (stationary) event variables.
"""

import os
import sqlite3
import datetime
import tempfile
import unittest

import numpy as np
import networkx as nx
from dateutil.tz import tzlocal

from amulog import config as amulog_config
from logdag import arguments
from logdag.causaltestdata import amulog_export

# plain-word messages so amulog templates them verbatim (no variable parts),
# giving each DAG node a 1:1 template gid.
MSG_PERIODIC = "alpha periodic heartbeat tick"
MSG_POISSON = "gamma poisson sparse event"

DT_RANGE = (datetime.datetime(2112, 9, 3), datetime.datetime(2112, 9, 4))
# amulog parses the log text as wall-clock in its configured timezone (default
# local), so the dt_range handed to the read/query side must carry the same tz
# to compare against the tz-aware datetimes amulog returns.
DT_RANGE_TZ = tuple(dt.replace(tzinfo=tzlocal()) for dt in DT_RANGE)


def _load_cnt_safe(el, measure, tags, dt_range):
    """load_cnt, treating a missing measure table / key as count 0.

    When every series is filtered out, the log_feature table is never created
    and sqlite raises 'no such table'; that is exactly the 'removed' outcome.
    """
    try:
        return el.load_cnt(measure, tags, dt_range) or 0
    except sqlite3.OperationalError:
        return 0


class TestCausaltestdataPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        fd, cls._path_testlog = tempfile.mkstemp(); os.close(fd)
        fd, cls._path_amulogdb = tempfile.mkstemp(); os.close(fd)
        fd, cls._path_ltgendump = tempfile.mkstemp(); os.close(fd)
        fd, cls._path_amulogconf = tempfile.mkstemp(); os.close(fd)

        # --- known DAG: a perfectly periodic node and a sparse Poisson node --
        g = nx.DiGraph()
        g.add_node("p", type="periodic",
                   periodic_interval=datetime.timedelta(minutes=10))
        # low rate -> a handful of events (< linear_count) that survive filters
        g.add_node("s", type="tsevent", tsevent_lambd=8)
        node_message = {"p": ("host0", MSG_PERIODIC),
                        "s": ("host0", MSG_POISSON)}

        np.random.seed(5)
        amulog_export.dump_log(g, {"dt_range": DT_RANGE}, node_message,
                               cls._path_testlog)

        # --- parse logs into an amulog DB ------------------------------------
        amulog_conf = amulog_config.open_config()
        amulog_conf["general"]["src_path"] = cls._path_testlog
        amulog_conf["database"]["sqlite3_filename"] = cls._path_amulogdb
        amulog_conf["manager"]["indata_filename"] = cls._path_ltgendump
        with open(cls._path_amulogconf, "w") as f:
            amulog_conf.write(f)

        from amulog import __main__ as amulog_main
        from amulog import manager
        targets = amulog_main.get_targets_conf(amulog_conf)
        manager.process_files_online(amulog_conf, targets, reset_db=True)

    @classmethod
    def tearDownClass(cls):
        for p in (cls._path_testlog, cls._path_amulogdb, cls._path_ltgendump,
                  cls._path_amulogconf):
            os.remove(p)

    def _run_evgen(self, rules):
        """Build a fresh evdb, run evgen with `rules`, return (loader, evdefs)."""
        fd, testdb = tempfile.mkstemp(); os.close(fd)
        self.addCleanup(os.remove, testdb)
        conf = amulog_config.open_config(arguments.DEFAULT_CONFIG,
                                         base_default=False)
        conf["general"]["evdb"] = "sql"
        conf["database_sql"]["database"] = "sqlite3"
        conf["database_sql"]["sqlite3_filename"] = testdb
        conf["database_amulog"]["source_conf"] = self._path_amulogconf
        conf["filter"]["rules"] = rules

        from logdag.source import evgen_log
        el = evgen_log.LogEventLoader(conf)
        el.read(dt_range=DT_RANGE_TZ, dump_org=True)

        evdefs = {}
        for evdef in el.iter_evdef(dt_range=DT_RANGE_TZ):
            evdefs[evdef] = el.instruction(evdef)  # "(host) <template text>"
        return el, evdefs

    @staticmethod
    def _find(evdefs, fragment):
        hits = [ev for ev, ins in evdefs.items() if fragment in ins]
        assert len(hits) == 1, \
            "expected exactly one gid for %r, got %d" % (fragment, len(hits))
        return hits[0]

    def test_periodic_dropped_default_rules(self):
        # default chain: sizetest, filter_periodic, remove_linear
        el, evdefs = self._run_evgen("sizetest, filter_periodic, remove_linear")
        periodic = self._find(evdefs, "periodic heartbeat")

        org = _load_cnt_safe(el, "log_org", periodic.tags(), DT_RANGE_TZ)
        feat = _load_cnt_safe(el, "log_feature", periodic.tags(), DT_RANGE_TZ)
        # 24h / 10min = 144 raw events, all removed by preprocessing
        self.assertGreater(org, 100)
        self.assertEqual(feat, 0)

    def test_nonperiodic_survives_with_remove_linear(self):
        # without filter_periodic, the periodic series still goes via
        # remove_linear, while the sparse Poisson series survives.
        el, evdefs = self._run_evgen("sizetest, remove_linear")
        periodic = self._find(evdefs, "periodic heartbeat")
        poisson = self._find(evdefs, "poisson sparse")

        feat_p = _load_cnt_safe(el, "log_feature", periodic.tags(), DT_RANGE_TZ)
        self.assertEqual(feat_p, 0, "periodic series should be removed")

        org_s = _load_cnt_safe(el, "log_org", poisson.tags(), DT_RANGE_TZ)
        feat_s = _load_cnt_safe(el, "log_feature", poisson.tags(), DT_RANGE_TZ)
        # sparse (< linear_count) -> skips remove_linear, passes through intact
        self.assertGreater(feat_s, 0, "sparse Poisson series should survive")
        self.assertLess(org_s, 10)
        self.assertEqual(feat_s, org_s)


if __name__ == "__main__":
    unittest.main()

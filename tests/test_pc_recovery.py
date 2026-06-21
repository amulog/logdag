#!/usr/bin/env python
# coding: utf-8

"""Reference test: PC recovers a known causal structure.

Generates event-count time series from a known DAG (0 -> 1 -> 2 chain) with the
causaltestdata toolkit and checks that `makedag.estimate_dag` (cause_algorithm
= pc, ci_func = gsq) recovers the skeleton {0-1, 1-2} without a spurious 0-2
edge (0 _||_ 2 | 1). This anchors the inference-input behaviour so the pc /
lingam / mixedlingam modules can be refactored (common abstraction) safely.

causaltestdata is an optional sibling repo (`pip install -e ../causaltestdata`);
the test skips when it is absent. Its `variable` module imports `Hawkes` at
import time, which we stub (only the Poisson TimeSeriesEventVariable is used).
"""

import datetime
import sys
import types
import unittest

import networkx as nx
import numpy as np
from amulog import config

from logdag import arguments, makedag

# causaltestdata.variable does `import Hawkes` at top; stub it (unused here)
sys.modules.setdefault("Hawkes", types.ModuleType("Hawkes"))
try:
    from causaltestdata import variable as ctd_variable
except ImportError:
    ctd_variable = None


def _chain_event_df(seed, weight=0.9, days=7, lambd=80):
    np.random.seed(seed)
    g = nx.DiGraph()
    for n in (0, 1, 2):
        g.add_node(n, type="tsevent")
    g.add_edge(0, 1, weight=weight)
    g.add_edge(1, 2, weight=weight)
    start = datetime.datetime(2020, 1, 1)
    defaults = {
        "default_type": "tsevent",
        "dt_range": (start, start + datetime.timedelta(days=days)),
        "dt_interval": datetime.timedelta(minutes=1),
        "tsevent_lambd": lambd,
    }
    return ctd_variable.generate_all(g, defaults)


def _pc_conf():
    conf = config.open_config(arguments.DEFAULT_CONFIG, base_default=False)
    conf["dag"]["cause_algorithm"] = "pc"
    conf["dag"]["ci_func"] = "gsq"
    return conf


@unittest.skipUnless(ctd_variable is not None, "causaltestdata not installed")
class TestPcRecovery(unittest.TestCase):

    def test_recovers_chain_skeleton(self):
        df = _chain_event_df(seed=0)
        dag = makedag.estimate_dag(_pc_conf(), df)
        skeleton = {frozenset(e[:2]) for e in dag.edges()}
        self.assertIn(frozenset({0, 1}), skeleton)
        self.assertIn(frozenset({1, 2}), skeleton)
        self.assertNotIn(frozenset({0, 2}), skeleton)  # 0 _||_ 2 | 1


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python
# coding: utf-8

"""Recovery smoke test: each causal-discovery method recovers a known structure.

Generates event-count time series from a known DAG (0 -> 1 -> 2 chain) with the
vendored logdag.causaltestdata toolkit and checks that `makedag.estimate_dag`
recovers the skeleton {0-1, 1-2} without a spurious 0-2 edge (0 _||_ 2 | 1), for
each supported method. This is the "does the method run and recover an obvious
structure" baseline. Rigorous characterization -- sensitivity of each method to
density / noise distribution / hidden confounding / discretization, after the
synthetic-data evaluation of Jarry et al. (COMPSAC'21) §IV-A using the same
causaltestdata toolkit -- is a separate, later effort.

Scope: pc, lingam. mixedlingam is intentionally excluded (dropped, unmaintained
bcause dependency); cdt is disabled. A new method is added by listing it in
ALGORITHMS once its estimate_dag branch works.

Only the Poisson TimeSeriesEventVariable is used; Hawkes is imported lazily, so
it is never loaded here.
"""

import datetime
import unittest

import networkx as nx
import numpy as np
from amulog import config

from logdag import arguments, makedag
from logdag.causaltestdata import variable as ctd_variable

# methods expected to run and recover the chain skeleton
ALGORITHMS = ["pc", "lingam"]


def _chain_event_df(seed, weight=0.9, days=7, lambd=80):
    """Generate event-count series for the chain DAG 0 -> 1 -> 2."""
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


def _conf(algorithm):
    conf = config.open_config(arguments.DEFAULT_CONFIG, base_default=False)
    conf["dag"]["cause_algorithm"] = algorithm
    conf["dag"]["ci_func"] = "gsq"
    return conf


class TestMethodRecovery(unittest.TestCase):

    def test_methods_recover_chain_skeleton(self):
        # one shared dataset so methods are compared on identical input
        df = _chain_event_df(seed=0)
        for alg in ALGORITHMS:
            with self.subTest(algorithm=alg):
                dag = makedag.estimate_dag(_conf(alg), df)
                # compare at the skeleton level: pc returns undirected pairs,
                # lingam returns directed edges; both must place 0-1 and 1-2.
                skeleton = {frozenset(e[:2]) for e in dag.edges()}
                self.assertIn(frozenset({0, 1}), skeleton,
                              "%s: missing edge 0-1" % alg)
                self.assertIn(frozenset({1, 2}), skeleton,
                              "%s: missing edge 1-2" % alg)
                self.assertNotIn(frozenset({0, 2}), skeleton,
                                 "%s: spurious edge 0-2 (0 _||_ 2 | 1)" % alg)


if __name__ == "__main__":
    unittest.main()

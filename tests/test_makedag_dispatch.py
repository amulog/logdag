#!/usr/bin/env python
# coding: utf-8

"""Dispatch tests for logdag.makedag.estimate_dag.

Pins the routing of `[dag] cause_algorithm` to the right inference input module,
plus the input-too-small and invalid-algorithm guards. The estimators are
mocked, so no inference library is exercised — this is the seam that lets the
big if/elif be refactored into a registry safely.

Non-built-in algorithms (e.g. ``mixedlingam``, now shipped out-of-tree as a
private plugin) route through the ``logdag.cause_algorithm`` entry-point lookup;
that seam is covered by mocking the loader (see the plugin test below).
"""

import unittest
from unittest import mock

import pandas as pd
from amulog import config

from logdag import arguments, makedag


def _conf(algorithm):
    c = config.open_config(arguments.DEFAULT_CONFIG, base_default=False)
    c["dag"]["cause_algorithm"] = algorithm
    return c


_DF = pd.DataFrame({0: [1.0, 2.0, 1.0], 1: [3.0, 1.0, 2.0]})  # 2 columns


class TestEstimateDagDispatch(unittest.TestCase):

    def test_pc_routes_to_pc_input(self):
        with mock.patch.object(makedag.pc_input, "pc", return_value="PC") as m:
            r = makedag.estimate_dag(_conf("pc"), _DF)
        self.assertEqual(r, "PC")
        m.assert_called_once()

    def test_pc_corr_routes_to_pc_input(self):
        with mock.patch.object(makedag.pc_input, "pc", return_value="PCCORR") as m:
            r = makedag.estimate_dag(_conf("pc-corr"), _DF)
        self.assertEqual(r, "PCCORR")
        # pc-corr forces skeleton_depth=0
        self.assertEqual(m.call_args.args[4], 0)

    def test_lingam_routes_to_estimate(self):
        from logdag import lingam_input
        with mock.patch.object(lingam_input, "estimate",
                               return_value="LINGAM") as m:
            r = makedag.estimate_dag(_conf("lingam"), _DF)
        self.assertEqual(r, "LINGAM")
        m.assert_called_once()

    def test_lingam_corr_routes_to_estimate_corr(self):
        from logdag import lingam_input
        with mock.patch.object(lingam_input, "estimate_corr",
                               return_value="LCORR") as m:
            r = makedag.estimate_dag(_conf("lingam-corr"), _DF)
        self.assertEqual(r, "LCORR")
        m.assert_called_once()

    def test_too_few_columns_returns_empty_without_estimating(self):
        one_col = pd.DataFrame({0: [1.0, 2.0]})
        with mock.patch.object(makedag.pc_input, "pc") as m:
            r = makedag.estimate_dag(_conf("pc"), one_col)
        m.assert_not_called()                 # early return, no inference
        self.assertEqual(r.number_of_edges(), 0)

    def test_plugin_algorithm_routes_to_entry_point(self):
        # an unknown algorithm is resolved via the plugin loader and called
        # with (conf, input_df, prior_knowledge)
        plugin = mock.Mock(return_value="PLUGIN")
        conf = _conf("mixedlingam")
        with mock.patch.object(makedag, "_load_algorithm_plugin",
                               return_value=plugin) as loader:
            r = makedag.estimate_dag(conf, _DF, prior_knowledge="PK")
        self.assertEqual(r, "PLUGIN")
        loader.assert_called_once_with("mixedlingam")
        plugin.assert_called_once_with(conf, _DF, "PK")

    def test_invalid_algorithm_raises(self):
        # no built-in branch and no plugin registered -> ValueError
        with self.assertRaises(ValueError):
            makedag.estimate_dag(_conf("nonexistent"), _DF)


if __name__ == "__main__":
    unittest.main()

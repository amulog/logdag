#!/usr/bin/env python
# coding: utf-8

"""Regression tests: LiNGAM graphs must hold plain Python types.

``make-dag`` with ``cause_algorithm = lingam`` and ``output_dag_format = json``
failed with "Object of type int64 is not JSON serializable". The input frame
comes from ``pd.concat`` of single-column frames, so its column Index has
dtype int64 and a label taken by position is a numpy scalar. Since an equal
Python int is already a node, networkx reuses it as the outer adjacency key
but stores the numpy scalar as the inner one: ``g.nodes()`` looks clean while
the edge target is int64. Edge weights (numpy float64) fail the same way.

The lingam package is optional, so the fitted model is stubbed out here.
"""

import json
import sys
import types
import unittest
from unittest import mock

import networkx as nx
import numpy as np
import pandas as pd

from logdag import lingam_input


ADJACENCY = np.array([[0.0, 0.0, 0.0],
                      [0.9, 0.0, 0.0],
                      [0.0, 0.7, 0.0]])


def _input_df(n_columns=3):
    """Mimic log2event.makeinput: concat of single-column frames of int eid."""
    values = np.arange(12, dtype=float).reshape(4, n_columns)
    evlist = [pd.DataFrame(values[:, i], columns=[i])
              for i in range(n_columns)]
    df = pd.concat(evlist, axis=1)
    assert isinstance(df.columns[0], np.integer)  # premise of the regression
    return df


def _assert_json_native(testcase, g):
    for node in g.nodes():
        testcase.assertIs(type(node), int)
    for from_, to, data in g.edges(data=True):
        testcase.assertIs(type(from_), int)
        testcase.assertIs(type(to), int)
        testcase.assertIs(type(data["weight"]), float)
    json.dumps(nx.node_link_data(g, edges="links"))


class TestLingamEstimateOutputTypes(unittest.TestCase):

    def test_estimate_graph_is_json_serializable(self):
        fake_lingam = types.ModuleType("lingam")
        fake_lingam.ICALiNGAM = object
        model = mock.Mock(adjacency_matrix_=ADJACENCY)
        with mock.patch.dict(sys.modules, {"lingam": fake_lingam}), \
                mock.patch.object(lingam_input, "_fit_back",
                                  return_value=model):
            g = lingam_input.estimate(_input_df(), algorithm="ica")

        self.assertEqual(2, g.number_of_edges())
        _assert_json_native(self, g)

    def test_estimate_corr_graph_is_json_serializable(self):
        class FakeLiNGAM:
            def __init__(self, **kwargs):
                self.adjacency_matrix_ = np.array([[0.0, 0.0], [0.9, 0.0]])

            def fit(self, data):
                pass

        fake_lingam = types.ModuleType("lingam")
        fake_lingam.ICALiNGAM = FakeLiNGAM
        with mock.patch.object(lingam_input, "_import_lingam",
                               return_value=fake_lingam):
            g = lingam_input.estimate_corr(_input_df(), algorithm="ica")

        self.assertEqual(3, g.number_of_edges())  # one per column pair
        _assert_json_native(self, g)


if __name__ == "__main__":
    unittest.main()

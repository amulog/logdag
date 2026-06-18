#!/usr/bin/env python
# coding: utf-8

"""Regression test for logdag.source.evpost.anomaly_if.

Code-review finding: ``IsolationForest(..., behaviour="new")`` used a parameter
that scikit-learn removed in 0.24, so the call raised ``TypeError`` on any
current sklearn. The default behaviour is now the former "new" behaviour, so
the argument is simply dropped.
"""

import unittest

import numpy as np
import pandas as pd

from logdag.source import evpost


class TestAnomalyIf(unittest.TestCase):

    def test_runs_without_behaviour_typeerror(self):
        # a non-constant series so it reaches IsolationForest
        sr = pd.Series(
            [0.0, 1.0, 0.0, 0.0, 5.0, 0.0, 0.0, 100.0, 0.0, 0.0],
            index=pd.RangeIndex(10))
        result = evpost.anomaly_if(sr)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(sr))
        # output is a 0/1 anomaly flag
        self.assertTrue(set(np.unique(result.values)).issubset({0.0, 1.0}))


if __name__ == "__main__":
    unittest.main()

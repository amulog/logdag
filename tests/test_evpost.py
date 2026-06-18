#!/usr/bin/env python
# coding: utf-8

"""Regression test for logdag.source.evpost.anomaly_if.

Code-review finding: ``IsolationForest(..., behaviour="new")`` used a parameter
that scikit-learn removed in 0.24, so the call raised ``TypeError`` on any
current sklearn. The default behaviour is now the former "new" behaviour, so
the argument is simply dropped.
"""

import unittest
import warnings

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


class TestDiffFirstElement(unittest.TestCase):
    """root_square_diff / diff_abs set the first (NaN) diff element to 0 via
    ``ret.iloc[0]`` — ``ret[0]`` is a label assignment that, on a DatetimeIndex,
    is deprecated now (FutureWarning) and will add a spurious label-0 entry on
    future pandas instead of fixing the first element."""

    def _series(self):
        idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
        return pd.Series([10.0, 12.0, 11.0], index=idx)

    def test_diff_abs_first_is_zero_no_label_warning(self):
        sr = self._series()
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            out = evpost.diff_abs(sr)
        self.assertEqual(len(out), len(sr))
        self.assertTrue(out.index.equals(sr.index))   # no spurious label 0
        self.assertEqual(out.iloc[0], 0.0)

    def test_root_square_diff_first_is_zero(self):
        sr = self._series()
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            out = evpost.root_square_diff(sr)
        self.assertEqual(len(out), len(sr))
        self.assertEqual(out.iloc[0], 0.0)


if __name__ == "__main__":
    unittest.main()

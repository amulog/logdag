#!/usr/bin/env python
# coding: utf-8

"""Tests for logdag.log2event.load_event (sequential binning path).

Pins the read-time behaviour that turns a stored (sparse) series into the dense
binned DataFrame fed to the DAG inference: an empty / all-zero series yields
None, otherwise the loaded DataFrame is returned. The EventLoader is mocked.
"""

import datetime
import unittest

import pandas as pd

from logdag import log2event


_T0 = datetime.datetime(2020, 1, 1, 0, 0, 0)
_H = datetime.timedelta(hours=1)
_RANGE = (_T0, _T0 + 3 * _H)


class _FakeEl:
    fields = ["val"]

    def __init__(self, load_ret):
        self._load_ret = load_ret

    def load(self, measure, tags, dt_range, binsize):
        return self._load_ret


def _df(values):
    idx = pd.to_datetime([_T0 + i * _H for i in range(len(values))])
    return pd.DataFrame({"val": values}, index=idx)


class TestLoadEventSequential(unittest.TestCase):

    def _call(self, el):
        return log2event.load_event("m", {}, _RANGE, _H, _H, "sequential", el=el)

    def test_returns_loaded_df(self):
        df = _df([1.0, 2.0, 0.0])
        out = self._call(_FakeEl(df))
        self.assertIs(out, df)

    def test_none_when_loader_returns_none(self):
        self.assertIsNone(self._call(_FakeEl(None)))

    def test_none_when_all_zero(self):
        # df present but sums to 0 -> treated as empty
        self.assertIsNone(self._call(_FakeEl(_df([0.0, 0.0, 0.0]))))


if __name__ == "__main__":
    unittest.main()

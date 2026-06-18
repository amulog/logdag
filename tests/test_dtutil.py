#!/usr/bin/env python
# coding: utf-8

"""Regression tests for logdag.dtutil.discretize.

Code-review finding: the inner guard ``if sum(current_idxs) > 0`` used ``sum``
on an array of bin *indices*, so a datapoint whose only active bin was index 0
(``sum([0]) == 0``) was silently dropped — i.e. the very first bin lost its
data. The guard is now ``len(current_idxs) > 0``.
"""

import datetime
import unittest

from logdag import dtutil


_T0 = datetime.datetime(2020, 1, 1, 0, 0, 0)
_H = datetime.timedelta(hours=1)
_MIN = datetime.timedelta(minutes=10)


class TestDiscretizeSequential(unittest.TestCase):

    def _range(self, nbins):
        return (_T0, _T0 + nbins * _H)

    def test_first_bin_counted(self):
        # one event in each of bins 0,1,2 -> [1,1,1] (bin 0 must not be dropped)
        l_dt = [_T0 + _MIN, _T0 + _H + _MIN, _T0 + 2 * _H + _MIN]
        a = dtutil.discretize_sequential(l_dt, self._range(3), _H,
                                         binarize=False)
        self.assertEqual(list(a), [1, 1, 1])

    def test_multiple_in_first_bin(self):
        l_dt = [_T0 + datetime.timedelta(minutes=5),
                _T0 + datetime.timedelta(minutes=20)]
        a = dtutil.discretize_sequential(l_dt, self._range(3), _H,
                                         binarize=False)
        self.assertEqual(list(a), [2, 0, 0])

    def test_binarize_first_bin(self):
        l_dt = [_T0 + datetime.timedelta(minutes=5),
                _T0 + datetime.timedelta(minutes=20)]
        a = dtutil.discretize_sequential(l_dt, self._range(3), _H,
                                         binarize=True)
        self.assertEqual(list(a), [1, 0, 0])

    def test_empty_input(self):
        a = dtutil.discretize_sequential([], self._range(3), _H,
                                         binarize=False)
        self.assertEqual(list(a), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()

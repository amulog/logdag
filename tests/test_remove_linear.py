#!/usr/bin/env python
# coding: utf-8

"""Tests for logdag.source.filter_log.LogFilter.remove_linear.

remove_linear drops a series whose cumulative event count tracks a straight
line, i.e. events arriving at a roughly uniform rate (background-like, not
bursty). It compares the cumulative count against an ideal uniform ramp and
removes the series when the normalised squared deviation falls below
``linear_th``.

Code-review note: the deviation is now computed from the resized, range-clipped
``discretize_sequential`` count (consistent with filter_periodic / remove_corr).

The sample window equals the dt_range here, so ``_resize_input`` is the identity
and the base LogFilter (whose ``_get_additional_data`` is abstract) can be used
directly.
"""

import datetime
import unittest

from logdag.source.filter_log import LogFilter


_T0 = datetime.datetime(2020, 1, 1)
_DAY = datetime.timedelta(days=1)
_HOUR = datetime.timedelta(hours=1)
_RANGE = (_T0, _T0 + _DAY)
_EVDEF = ("host", "ev")


def _filter():
    # sample length == range length -> _resize_input is identity; 24 hourly bins
    return LogFilter(rules=["remove_linear"],
                     linear_sample_rule=[(_DAY, _HOUR)],
                     linear_count=10, linear_th=0.5)


class TestRemoveLinear(unittest.TestCase):

    def test_uniform_rate_removed(self):
        # one event per hour -> cumulative count ~ straight line -> removed
        l_dt = [_T0 + i * _HOUR + datetime.timedelta(minutes=1)
                for i in range(24)]
        f = _filter()
        self.assertIsNone(f.remove_linear(list(l_dt), _RANGE, _EVDEF))
        # removal is recorded in the filter log
        self.assertIn((_RANGE, _EVDEF), f._log)
        self.assertEqual(f._log[(_RANGE, _EVDEF)][0], "remove_linear")

    def test_bursty_kept(self):
        # 24 events crammed into the first 2 hours -> far from linear -> kept
        l_dt = [_T0 + datetime.timedelta(minutes=m) for m in range(0, 120, 5)]
        out = _filter().remove_linear(list(l_dt), _RANGE, _EVDEF)
        self.assertEqual(out, l_dt)

    def test_below_count_kept(self):
        # fewer than linear_count events -> not evaluated -> kept untouched
        l_dt = [_T0 + i * _HOUR for i in range(5)]
        out = _filter().remove_linear(list(l_dt), _RANGE, _EVDEF)
        self.assertEqual(out, l_dt)


if __name__ == "__main__":
    unittest.main()

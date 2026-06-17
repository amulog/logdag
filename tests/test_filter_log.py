#!/usr/bin/env python
# coding: utf-8

"""Unit tests for logdag.source.filter_log.

Regression tests for code-review findings:

* C1: ``_resize_input`` shrink branch returned a list of booleans
  (``[dt >= cutoff for dt in l_dt]``) instead of filtering datetimes
  (``[dt for dt in l_dt if dt >= cutoff]``). The bool list was then fed to
  ``discretize_sequential`` as if it were datetimes.
* C2: ``remove_linear`` gated on the resized ``tmp_l_dt`` but built the
  cumulative curve and normalization from the original ``l_dt``. It is now
  consistent with the other filters (``discretize_sequential`` over the
  resized input), which also makes out-of-range datetimes ignored instead of
  triggering a bad negative slice / failing ``assert``.

Semantics under test: ``remove_linear`` returns ``None`` (event removed) when
the cumulative count grows ~linearly (uniform rate), and returns the input
list (kept) for bursty series.
"""

import datetime
import unittest

from logdag.source import filter_log


_T0 = datetime.datetime(2020, 1, 1, 0, 0, 0)
_DAY = datetime.timedelta(days=1)
_HOUR = datetime.timedelta(hours=1)


def _make_filter(**kwargs):
    # remove_linear/_resize_input only touch the linear_* attrs and _log.
    return filter_log.LogFilter(rules=["remove_linear"], **kwargs)


class TestResizeInput(unittest.TestCase):
    """C1 and the surrounding branches of _resize_input."""

    def setUp(self):
        self.f = _make_filter()
        # 0,1,2,...,47 hours after _T0
        self.l_dt = [_T0 + i * _HOUR for i in range(48)]

    def test_equal_returns_input_unchanged(self):
        dt_range = (_T0, _T0 + 2 * _DAY)
        out = self.f._resize_input(self.l_dt, dt_range, 2 * _DAY, evdef=None)
        self.assertEqual(out, self.l_dt)

    def test_shrink_returns_datetimes_not_bools(self):
        # dt_length (2d) > sample (1d) -> keep only the last day of datetimes.
        dt_range = (_T0, _T0 + 2 * _DAY)
        out = self.f._resize_input(self.l_dt, dt_range, _DAY, evdef=None)
        # C1: every element must be a datetime, never a bool
        self.assertTrue(all(isinstance(x, datetime.datetime) for x in out))
        self.assertFalse(any(isinstance(x, bool) for x in out))
        # cutoff = dt_range[1] - sample = _T0 + 1day; hours 24..47 remain
        cutoff = _T0 + _DAY
        self.assertEqual(out, [dt for dt in self.l_dt if dt >= cutoff])
        self.assertEqual(len(out), 24)


class TestRemoveLinear(unittest.TestCase):

    def _filter(self, linear_count=10, linear_th=0.5, binsize=_HOUR):
        return _make_filter(
            linear_sample_rule=[(_DAY, binsize)],
            linear_count=linear_count,
            linear_th=linear_th,
        )

    def test_uniform_series_is_removed(self):
        # one event per hour over a day -> cumulative count ~ linear -> remove
        f = self._filter()
        dt_range = (_T0, _T0 + _DAY)
        l_dt = [_T0 + i * _HOUR for i in range(24)]
        self.assertIsNone(f.remove_linear(l_dt, dt_range, evdef=None))

    def test_bursty_series_is_kept(self):
        # all events in the first hour -> cumulative jumps then flat -> keep
        f = self._filter()
        dt_range = (_T0, _T0 + _DAY)
        l_dt = [_T0 + datetime.timedelta(minutes=2 * i) for i in range(24)]
        out = f.remove_linear(l_dt, dt_range, evdef=None)
        self.assertEqual(out, l_dt)

    def test_too_few_events_is_kept(self):
        # fewer than linear_count -> rule skipped -> kept
        f = self._filter(linear_count=10)
        dt_range = (_T0, _T0 + _DAY)
        l_dt = [_T0 + i * _HOUR for i in range(5)]
        out = f.remove_linear(l_dt, dt_range, evdef=None)
        self.assertEqual(out, l_dt)

    def test_out_of_range_events_are_ignored(self):
        # equal branch keeps l_dt as-is, so l_dt may carry dt outside dt_range.
        # the old manual binning hit `assert cnt < len(a_stat)` (or a negative
        # slice); discretize_sequential ignores them. A uniform in-range series
        # plus a few late stragglers must still be removed without raising.
        f = self._filter()
        dt_range = (_T0, _T0 + _DAY)
        l_dt = [_T0 + i * _HOUR for i in range(24)]
        l_dt += [_T0 + _DAY + i * _HOUR for i in range(3)]  # after dt_range[1]
        self.assertIsNone(f.remove_linear(l_dt, dt_range, evdef=None))


if __name__ == "__main__":
    unittest.main()

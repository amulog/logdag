#!/usr/bin/env python
# coding: utf-8

"""Regression test for logdag.dtutil.range_dt.

Code-review finding: ``range_dt`` built each datetime with
``fromtimestamp(ut).replace(tzinfo=tzinfo)``. ``fromtimestamp(ut)`` (no tz)
interprets the epoch value in the *local* timezone, and ``replace`` then just
relabels it — so for any non-local ``tzinfo`` the wall clock was shifted by the
local UTC offset. ``fromtimestamp(ut, tz=tzinfo)`` converts correctly.
"""

import datetime
import unittest

import dateutil.tz

from logdag import dtutil


class TestRangeDt(unittest.TestCase):

    def test_utc_input_is_not_shifted(self):
        utc = dateutil.tz.tzutc()
        dts = datetime.datetime(2020, 1, 1, 0, 0, 0, tzinfo=utc)
        dte = datetime.datetime(2020, 1, 1, 3, 0, 0, tzinfo=utc)
        res = dtutil.range_dt(dts, dte, datetime.timedelta(hours=1))
        # exact instants, independent of the machine's local timezone
        self.assertEqual(
            [r.astimezone(utc) for r in res],
            [datetime.datetime(2020, 1, 1, h, 0, 0, tzinfo=utc)
             for h in (0, 1, 2)],
        )

    def test_preserves_tzinfo(self):
        utc = dateutil.tz.tzutc()
        dts = datetime.datetime(2020, 1, 1, 0, 0, 0, tzinfo=utc)
        dte = datetime.datetime(2020, 1, 1, 2, 0, 0, tzinfo=utc)
        res = dtutil.range_dt(dts, dte, datetime.timedelta(hours=1))
        self.assertEqual(res[0].utcoffset(), datetime.timedelta(0))


if __name__ == "__main__":
    unittest.main()

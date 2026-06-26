#!/usr/bin/env python
# coding: utf-8

"""Regression tests for logdag.source.sqlts.SQLTimeSeries.get_df.

Code-review findings:
* ``values.nan_to_num(fill)`` — ``values`` is a tuple (row[1:]); tuples have no
  ``nan_to_num`` method, so a non-None ``fill`` raised AttributeError. Correct
  API is ``np.nan_to_num(values, nan=fill)``.
* the ``func is None`` branch built the DataFrame from the *unsorted* ``l_dt`` /
  ``l_values`` even though sorted copies were computed; rows came back in DB
  order, not time order.

The DB layer is stubbed (``_get`` returns canned rows, ``_db.strptime`` parses
the timestamp string); the timestamp helpers are real staticmethods.
"""

import datetime
import unittest

from logdag.source import sqlts


class _FakeDB:
    def strptime(self, dtstr):
        return datetime.datetime.strptime(dtstr, "%Y-%m-%d %H:%M:%S")


def _make_sqlts(rows):
    obj = sqlts.SQLTimeSeries.__new__(sqlts.SQLTimeSeries)
    obj._db = _FakeDB()
    obj._get = lambda measure, d_tags, fields, dt_range: list(rows)
    return obj


_DT_RANGE = (datetime.datetime(2020, 1, 1), datetime.datetime(2020, 1, 2))


class TestGetDf(unittest.TestCase):

    def test_func_none_returns_time_sorted(self):
        rows = [
            ("2020-01-01 00:00:02", 2.0),
            ("2020-01-01 00:00:00", 0.0),
            ("2020-01-01 00:00:01", 1.0),
        ]
        obj = _make_sqlts(rows)
        df = obj.get_df("m", {}, ["f"], _DT_RANGE, func=None)
        # rows must come back in chronological order, not DB/insert order
        self.assertTrue(df.index.is_monotonic_increasing)
        self.assertEqual(list(df["f"]), [0.0, 1.0, 2.0])

    def test_fill_replaces_nan(self):
        rows = [("2020-01-01 00:00:00", float("nan"))]
        obj = _make_sqlts(rows)
        # old code did tuple.nan_to_num(fill) -> AttributeError
        df = obj.get_df("m", {}, ["f"], _DT_RANGE, func=None, fill=0.0)
        self.assertEqual(df["f"].iloc[0], 0.0)


class _FakeDropDB:
    def __init__(self, tables):
        self._tables = tables
        self.executed = []

    def get_table_names(self):
        return self._tables

    @staticmethod
    def drop_table_sql(name):
        return "drop table {0}".format(name)

    def execute(self, sql):
        self.executed.append(sql)


class TestDropMeasurement(unittest.TestCase):
    """drop_measurement called the non-existent _db.drop_sql (AttributeError);
    it must use amulog's drop_table_sql and skip a never-created table."""

    @staticmethod
    def _obj(tables):
        obj = sqlts.SQLTimeSeries.__new__(sqlts.SQLTimeSeries)
        obj._db = _FakeDropDB(tables)
        return obj

    def test_drops_existing_table(self):
        obj = self._obj(["log_feature", "log_org"])
        obj.drop_measurement("log_feature")
        self.assertEqual(obj._db.executed, ["drop table log_feature"])

    def test_missing_table_is_noop(self):
        obj = self._obj(["log_org"])
        obj.drop_measurement("log_feature")
        self.assertEqual(obj._db.executed, [])


if __name__ == "__main__":
    unittest.main()

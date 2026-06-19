#!/usr/bin/env python
# coding: utf-8

"""Unit tests for logdag.source.influx.InfluxDBv1 (query construction).

These mock the influxdb client, so neither the ``influxdb`` package nor a
running server is required. Regression tests for code-review findings:

* tag values were interpolated into the InfluxQL string without escaping
  (injection / broken queries for values containing a quote);
* time bounds were truncated to whole seconds (``int(ut)`` + ``"s"``), losing
  sub-second precision near the range boundary;
* ``get_count`` read a hard-coded ``"val"`` key, but the count query aliases
  each field as its own name -> KeyError for any other field name.

For an end-to-end check against a real InfluxDB 1.8 see
``tests/integration/test_influx_integration.py`` (opt-in, Docker).
"""

import datetime
import sys
import types
import unittest

import dateutil.tz

# make `import influxdb` inside influx.py resolve without the real package
sys.modules.setdefault("influxdb", types.ModuleType("influxdb"))

from logdag.source import influx  # noqa: E402


_UTC = dateutil.tz.tzutc()


class _FakeResultSet:
    def __init__(self, points):
        self._points = list(points)

    def __len__(self):
        return len(self._points)

    def get_points(self):
        return iter(self._points)


class _FakeClient:
    def __init__(self, points=None):
        self.queries = []
        self._points = points or []

    def query(self, iql, **kwargs):
        self.queries.append(iql)
        return _FakeResultSet(self._points)


def _make_db(points=None):
    obj = influx.InfluxDBv1.__new__(influx.InfluxDBv1)
    obj.dbname = "db"
    obj._rpolicy = "autogen"
    obj._precision = "n"
    obj.verbose = False
    obj.client = _FakeClient(points)
    return obj


def _dt(ut):
    return datetime.datetime.fromtimestamp(ut, tz=_UTC)


class TestEscaping(unittest.TestCase):

    def test_escape_str(self):
        self.assertEqual(influx.InfluxDBv1._escape_str("a'b"), "a\\'b")

    def test_tag_value_is_escaped_in_query(self):
        db = _make_db()
        db._get("m", {"host": "a'b"}, ["val"], (_dt(0), _dt(10)))
        iql = db.client.queries[-1]
        self.assertIn("'a\\'b'", iql)


class TestTimeBounds(unittest.TestCase):

    def test_nanosecond_precision_not_truncated(self):
        db = _make_db()
        db._get("m", {}, ["val"], (_dt(1000.5), _dt(2000.0)))
        iql = db.client.queries[-1]
        # sub-second start preserved as nanoseconds, not "1000s"
        self.assertIn("time >= 1000500000000", iql)
        self.assertIn("time < 2000000000000", iql)
        self.assertNotIn("1000s", iql)


class TestGetCount(unittest.TestCase):

    def test_uses_field_name_not_val(self):
        # count query aliases the field as its own name
        db = _make_db(points=[{"myfield": 42}])
        n = db.get_count("m", {"host": "h"}, ["myfield"], (_dt(0), _dt(10)))
        self.assertEqual(n, 42)

    def test_empty_result_returns_none(self):
        db = _make_db(points=[])
        n = db.get_count("m", {"host": "h"}, ["myfield"], (_dt(0), _dt(10)))
        self.assertIsNone(n)


if __name__ == "__main__":
    unittest.main()

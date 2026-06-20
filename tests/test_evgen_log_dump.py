#!/usr/bin/env python
# coding: utf-8

"""Tests for logdag.source.evgen_log.LogEventLoader.read dump behaviour.

Both the pre-filter (raw) and the filtered series are kept in the SAME evdb as
two measures: ``log_org`` (raw) is stored only when ``dump_org`` is on, while
``log_feature`` (filtered) is always stored. ``dump_org`` is now also driven by
the ``[general] dump_org`` config option (default true), so both are retained
by default.
"""

import unittest

from logdag.source import evgen_log


class _FakeSource:
    def iter_event(self):
        return [("h1", 1), ("h2", 2), ("h3", 3)]

    def load(self, ev):
        return ["2020-01-01 00:00:00", "2020-01-01 00:01:00"]

    def timestamp2dict(self, l_dt):
        return {dt: 1 for dt in l_dt}


class _FakeEvdb:
    def __init__(self):
        self.measures = []
        self.commits = 0

    def add(self, measure, d_tags, data, fields):
        self.measures.append(measure)

    def commit(self):
        self.commits += 1


def _make_loader():
    obj = evgen_log.LogEventLoader.__new__(evgen_log.LogEventLoader)
    obj.source = _FakeSource()
    obj._lf = None              # _apply_filters returns l_dt unchanged
    obj.evdb = _FakeEvdb()
    obj.dry = False
    obj.fields = ["val"]
    return obj


class TestReadDumpOrg(unittest.TestCase):

    def test_dump_org_true_stores_both(self):
        obj = _make_loader()
        obj.read(dump_org=True)
        self.assertIn("log_org", obj.evdb.measures)       # pre-filter
        self.assertIn("log_feature", obj.evdb.measures)   # post-filter

    def test_dump_org_false_stores_feature_only(self):
        obj = _make_loader()
        obj.read(dump_org=False)
        self.assertNotIn("log_org", obj.evdb.measures)
        self.assertIn("log_feature", obj.evdb.measures)

    def test_commit_batched_once_per_read(self):
        # commit is batched at the end of read(), not per series/dump
        obj = _make_loader()
        obj.read(dump_org=True)
        self.assertEqual(len(obj.evdb.measures), 6)   # 3 series x (org+feature)
        self.assertEqual(obj.evdb.commits, 1)         # but a single commit


if __name__ == "__main__":
    unittest.main()

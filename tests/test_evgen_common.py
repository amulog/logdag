#!/usr/bin/env python
# coding: utf-8

"""Unit tests for logdag.source.evgen_common.

Regression test for a code-review finding (M17): ``EventLoader.drop_features``
called ``self.evdb.drop_measure(measure)``, but the evdb backends (influx /
sqlts) only implement ``drop_measurement``. The ``drop-features`` CLI
subcommand therefore failed with AttributeError for any non-dry run.
"""

import unittest

from logdag.source import evgen_common


class _FakeEvdb:
    def __init__(self):
        self.dropped = []

    def drop_measurement(self, measure):
        self.dropped.append(measure)

    # deliberately NO drop_measure: the old code would AttributeError here.


def _make_loader(dry, evdb, features):
    obj = evgen_common.EventLoader.__new__(evgen_common.EventLoader)
    obj.dry = dry
    obj.evdb = evdb
    obj.all_feature = lambda: features
    return obj


class TestDropFeatures(unittest.TestCase):

    def test_drops_each_measurement(self):
        evdb = _FakeEvdb()
        loader = _make_loader(False, evdb, ["m1", "m2"])
        loader.drop_features()
        self.assertEqual(evdb.dropped, ["m1", "m2"])

    def test_dry_run_drops_nothing(self):
        evdb = _FakeEvdb()
        loader = _make_loader(True, evdb, ["m1", "m2"])
        loader.drop_features()
        self.assertEqual(evdb.dropped, [])


if __name__ == "__main__":
    unittest.main()

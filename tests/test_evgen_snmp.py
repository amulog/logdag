#!/usr/bin/env python
# coding: utf-8

"""Regression test for logdag.source.evgen_snmp.SNMPEventLoader.

Code-review finding: ``store_all_source`` used ``return`` when a (tags, df)
pair was empty, which aborted the whole method and skipped every remaining
source. The intent is to skip just that empty pair, i.e. ``continue``.
"""

import unittest

import pandas as pd

from logdag.source import evgen_snmp


class _Loader(evgen_snmp.SNMPEventLoader):
    def __init__(self, read_result):
        # bypass the heavy real __init__
        self._d_source = {"s1": object()}
        self._d_vsource = {}
        self._d_vsourcedef = {}
        self._read_result = read_result
        self.dumped = []

    def _read_source(self, name, dt_range, target_host=None):
        return list(self._read_result)

    def dump(self, measure, tags, df, fields=None):
        self.dumped.append((measure, tags))


class TestStoreAllSourceSkipsEmpty(unittest.TestCase):

    def test_empty_pair_does_not_abort_remaining(self):
        good = pd.DataFrame({"v": [1.0]})
        loader = _Loader([
            (("empty",), None),       # empty -> must be skipped, not abort
            (("ok",), good),          # must still be stored
        ])
        loader.store_all_source(dt_range=("t0", "t1"))
        # old code returned on the first empty pair and never reached ("ok",)
        self.assertEqual(loader.dumped, [("s1", ("ok",))])


if __name__ == "__main__":
    unittest.main()

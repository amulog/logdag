#!/usr/bin/env python
# coding: utf-8

"""Regression test for SNMPEventLoader._search_feature_source.

Code-review finding: the function ended with ``assert len(ret) == 1`` then
``return ret[0]``. Under ``python -O`` the assert is stripped, so a zero-match
lookup raised ``IndexError``; and even with asserts on, a zero match reported
the misleading "duplicated feature definition". It now raises explicit
ValueErrors that distinguish "not found" from "duplicated".
"""

import unittest

from logdag.source import evgen_snmp


def _make_loader(d_feature, d_source, tags_for):
    obj = evgen_snmp.SNMPEventLoader.__new__(evgen_snmp.SNMPEventLoader)
    obj._d_feature = d_feature
    obj._d_source = d_source
    obj._seriesdef2tags = lambda sdef: tags_for[sdef]
    return obj


class TestSearchFeatureSource(unittest.TestCase):

    def test_single_match_returned(self):
        sdef = "sdef_a"
        loader = _make_loader(
            d_feature=[{"name": "f1", "source": "s1"}],
            d_source={"s1": [sdef]},
            tags_for={sdef: {"host": "h"}})
        sname, seriesdef, featuredef = loader._search_feature_source(
            "f1", {"host": "h"})
        self.assertEqual(sname, "s1")
        self.assertEqual(seriesdef, sdef)

    def test_no_match_raises_valueerror(self):
        sdef = "sdef_a"
        loader = _make_loader(
            d_feature=[{"name": "f1", "source": "s1"}],
            d_source={"s1": [sdef]},
            tags_for={sdef: {"host": "h"}})
        # tags do not match -> not found (old code: IndexError under -O)
        with self.assertRaises(ValueError):
            loader._search_feature_source("f1", {"host": "other"})

    def test_duplicate_match_raises_valueerror(self):
        sdef1, sdef2 = "sdef_a", "sdef_b"
        loader = _make_loader(
            d_feature=[{"name": "f1", "source": "s1"}],
            d_source={"s1": [sdef1, sdef2]},
            tags_for={sdef1: {"host": "h"}, sdef2: {"host": "h"}})
        with self.assertRaises(ValueError):
            loader._search_feature_source("f1", {"host": "h"})


if __name__ == "__main__":
    unittest.main()

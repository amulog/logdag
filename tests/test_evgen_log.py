#!/usr/bin/env python
# coding: utf-8

"""Regression test for logdag.source.evgen_log.LogEventLoaderDirect.load_items.

Code-review finding: `_apply_filters` returns ``None`` when the filters remove
the whole event series, and ``load_items`` then did ``for dt in feature_dt``
(``for dt in None`` -> TypeError). It now returns early on ``None``.
"""

import datetime
import types
import unittest

from logdag.source import evgen_log


_DT_RANGE = (datetime.datetime(2020, 1, 1), datetime.datetime(2020, 1, 2))
_TAGS = {"host": "h", "key": "1"}


def _make_loader(load_result, lf):
    obj = evgen_log.LogEventLoaderDirect.__new__(evgen_log.LogEventLoaderDirect)
    obj.source = types.SimpleNamespace(load=lambda ev: load_result)
    obj._lf = lf
    return obj


class TestLoadItems(unittest.TestCase):

    def test_filters_remove_all_yields_nothing(self):
        # apply_filters -> None must not raise; the generator yields nothing
        lf = types.SimpleNamespace(
            apply_filters=lambda l_dt, dt_range, ev: None)
        loader = _make_loader([datetime.datetime(2020, 1, 1, 1)], lf)
        self.assertEqual(list(loader.load_items("m", _TAGS, _DT_RANGE)), [])

    def test_no_filter_yields_each_dt(self):
        dts = [datetime.datetime(2020, 1, 1, 1),
               datetime.datetime(2020, 1, 1, 2)]
        loader = _make_loader(dts, None)  # _lf=None -> _apply_filters returns l_dt
        out = [dt for dt, _ in loader.load_items("m", _TAGS, _DT_RANGE)]
        self.assertEqual(out, dts)


if __name__ == "__main__":
    unittest.main()

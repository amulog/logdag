#!/usr/bin/env python
# coding: utf-8

"""Regression test for logdag.makedag.make_input.

Code-review finding: ``log2event.makeinput`` returns ``(None, None)`` when no
data was loaded, but ``make_input`` then called ``evmap.dump(args)`` (
``None.dump`` -> AttributeError). It now returns early on a ``None`` evmap.
"""

import types
import unittest
from unittest import mock

from logdag import makedag
from logdag import log2event


_ARGS = (object(), ("t0", "t1"), "area")


class TestMakeInput(unittest.TestCase):

    def test_no_data_returns_none_without_dump(self):
        with mock.patch.object(log2event, "makeinput",
                               return_value=(None, None)):
            input_df, evmap = makedag.make_input(_ARGS)
        self.assertIsNone(input_df)
        self.assertIsNone(evmap)

    def test_with_data_dumps_evmap(self):
        dumped = []
        evmap = types.SimpleNamespace(dump=lambda args: dumped.append(args))
        sentinel_df = object()
        with mock.patch.object(log2event, "makeinput",
                               return_value=(sentinel_df, evmap)):
            input_df, out_evmap = makedag.make_input(_ARGS)
        self.assertIs(input_df, sentinel_df)
        self.assertIs(out_evmap, evmap)
        self.assertEqual(dumped, [_ARGS])


if __name__ == "__main__":
    unittest.main()

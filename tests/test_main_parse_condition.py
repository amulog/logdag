#!/usr/bin/env python
# coding: utf-8

"""Regression tests for logdag.__main__._parse_condition.

Code-review finding: a condition whose key was not node/gid/host matched no
branch and was silently dropped, so a mistyped key (e.g. ``hots=x``) produced a
wrong filter with no error. Unknown keys now raise SyntaxError.
"""

import types
import unittest

from logdag import __main__ as logdag_main


class TestParseCondition(unittest.TestCase):

    def test_parses_known_keys(self):
        d = logdag_main._parse_condition(["node=5", "gid=3", "host=h1"])
        self.assertEqual(d, {"node": 5, "gid": 3, "host": "h1"})

    def test_value_with_equals_is_kept(self):
        # partition keeps everything after the first '='
        d = logdag_main._parse_condition(["host=a=b"])
        self.assertEqual(d, {"host": "a=b"})

    def test_unknown_key_raises(self):
        # was silently ignored -> wrong filter
        with self.assertRaises(SyntaxError):
            logdag_main._parse_condition(["hots=x"])

    def test_missing_equals_raises(self):
        with self.assertRaises(SyntaxError):
            logdag_main._parse_condition(["node"])

    def test_non_int_value_raises(self):
        with self.assertRaises(ValueError):
            logdag_main._parse_condition(["node=abc"])


class TestParseOptRange(unittest.TestCase):
    """_parse_opt_range validated the range length with `assert` (stripped
    under -O); it now raises explicitly. argparse enforces nargs=2 on the CLI,
    so this guards non-CLI callers."""

    def test_none_returns_none(self):
        ns = types.SimpleNamespace(dt_range=None)
        self.assertIsNone(logdag_main._parse_opt_range(ns))

    def test_valid_range_parses(self):
        ns = types.SimpleNamespace(dt_range=["2020-01-01", "2020-01-03"])
        out = logdag_main._parse_opt_range(ns)
        self.assertEqual(len(out), 2)

    def test_wrong_length_raises(self):
        ns = types.SimpleNamespace(dt_range=["2020-01-01"])
        with self.assertRaises(ValueError):
            logdag_main._parse_opt_range(ns)


if __name__ == "__main__":
    unittest.main()

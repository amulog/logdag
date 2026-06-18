#!/usr/bin/env python
# coding: utf-8

"""Regression test for logdag.eval.__main__._safe_ratio.

Code-review finding: ``show_match_info`` computed ``valid_cnt / len(tm)``,
``detected_ticket_num / valid_cnt`` and ``match_edge_sum / valid_cnt`` with no
guard, raising ZeroDivisionError when there were no tickets or no valid
tickets (all EMPTY_GROUP or all IOError). The divisions now go through
``_safe_ratio``.
"""

import math
import unittest

from logdag.eval import __main__ as eval_main


class TestSafeRatio(unittest.TestCase):

    def test_normal_division(self):
        self.assertEqual(eval_main._safe_ratio(3, 4), 0.75)

    def test_zero_denominator_is_nan_not_error(self):
        result = eval_main._safe_ratio(5, 0)
        self.assertTrue(math.isnan(result))

    def test_zero_over_zero_is_nan(self):
        self.assertTrue(math.isnan(eval_main._safe_ratio(0, 0)))


class TestParseLids(unittest.TestCase):
    """add_lids_stdin used ``[int(v) for v in input()]``, which read one line
    and parsed each character; _parse_lids reads whitespace-separated ids."""

    def test_single_multidigit_id(self):
        self.assertEqual(eval_main._parse_lids("123"), [123])

    def test_space_separated(self):
        self.assertEqual(eval_main._parse_lids("1 2 34"), [1, 2, 34])

    def test_multiline(self):
        self.assertEqual(eval_main._parse_lids("1\n2\n3\n"), [1, 2, 3])

    def test_empty(self):
        self.assertEqual(eval_main._parse_lids(""), [])


if __name__ == "__main__":
    unittest.main()

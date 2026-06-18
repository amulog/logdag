#!/usr/bin/env python
# coding: utf-8

"""Regression tests for the ``raise Warning(...)`` anti-pattern.

``raise Warning(msg)`` raises a built-in exception and halts execution; the
intent in both sites is a non-fatal warning that should let the caller proceed
(ignoring the unsupported argument). They now use ``warnings.warn``.

* cdt_input.estimate(init_graph=...) -> warn, then run the requested category.
* lingam_input.estimate(algorithm="ica", prior_knowledge=...) -> warn, then
  build ICA-LiNGAM without the prior knowledge.

Downstream heavy work (cdt / lingam libraries) is stubbed so only the
warn-and-continue behaviour is exercised.
"""

import sys
import types
import unittest
from unittest import mock

from logdag import cdt_input
from logdag import lingam_input


class TestCdtInputWarning(unittest.TestCase):

    def test_init_graph_warns_and_continues(self):
        with mock.patch.object(cdt_input, "independence_graph",
                               return_value="GRAPH") as m:
            with self.assertWarns(UserWarning):
                result = cdt_input.estimate(
                    data=None, category="independence", algorithm="x",
                    init_graph=object())
        self.assertEqual(result, "GRAPH")
        m.assert_called_once()


class TestLingamInputWarning(unittest.TestCase):

    def test_ica_with_prior_knowledge_warns_and_continues(self):
        fake_lingam = types.ModuleType("lingam")
        fake_lingam.ICALiNGAM = object
        with mock.patch.dict(sys.modules, {"lingam": fake_lingam}), \
                mock.patch.object(lingam_input, "_fit_back",
                                  return_value=None) as m:
            with self.assertWarns(UserWarning):
                result = lingam_input.estimate(
                    data=None, algorithm="ica", prior_knowledge=object())
        # _fit_back stubbed to None -> estimate returns None without raising
        self.assertIsNone(result)
        m.assert_called_once()


if __name__ == "__main__":
    unittest.main()

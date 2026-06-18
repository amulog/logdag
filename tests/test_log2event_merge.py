#!/usr/bin/env python
# coding: utf-8

"""Regression test for logdag.log2event.merge_sync_event.

Code-review finding: the function took ``new_df = evlist[l_old_eid[0]]`` (a
reference) and did ``new_df.columns = [new_eid]``, renaming the column of the
caller's input DataFrame in place. It now operates on a ``.copy()``.
"""

import types
import unittest

import pandas as pd

from logdag import log2event


def _member(ident, source, host, group):
    return types.SimpleNamespace(identifier=ident, source=source,
                                 host=host, group=group)


class _FakeEvmap:
    def __init__(self, mapping):
        self._m = mapping

    def evdef(self, eid):
        return self._m[eid]

    def __len__(self):
        return len(self._m)


class TestMergeSyncEvent(unittest.TestCase):

    def test_does_not_mutate_input_dataframes(self):
        # column labels (99, 88) differ from the new eids (0, 1) so an in-place
        # rename would be visible
        evlist = [pd.DataFrame({99: [1, 2]}), pd.DataFrame({88: [3, 4]})]
        evmap = _FakeEvmap({0: _member("a", "s", "h1", "g"),
                            1: _member("b", "s", "h2", "g")})

        new_evlist, new_evmap = log2event.merge_sync_event(
            evlist, evmap, rules=["host"])

        # input DataFrames keep their original columns
        self.assertEqual(list(evlist[0].columns), [99])
        self.assertEqual(list(evlist[1].columns), [88])
        # and the merge still produced renamed outputs
        self.assertEqual(len(new_evlist), 2)


if __name__ == "__main__":
    unittest.main()

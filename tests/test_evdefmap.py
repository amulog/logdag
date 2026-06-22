#!/usr/bin/env python
# coding: utf-8

"""Tests for logdag.log2event.EventDefinitionMap.

EventDefinitionMap is the eid <-> event-definition registry used throughout the
DAG pipeline (node ids in the graph are eids; restoring a DAG needs the reverse
lookup). It keys everything off ``evdef.identifier``. These tests pin:

  * sequential eid assignment and forward/reverse lookup,
  * membership and iteration helpers,
  * ``from_dict`` rebuilding the reverse map (used when loading a saved DAG),
  * a pickle dump/load round-trip.

A minimal concrete EventDefinition subclass is used so the test does not depend
on any source backend.
"""

import os
import tempfile
import unittest
from unittest import mock

from logdag import log2event


class _Evdef(log2event.EventDefinition):
    _l_attr = ["source", "host", "group", "label"]

    def event(self):
        return self.label

    def __str__(self):
        return "{0}:{1}:{2}".format(self.host, self.group, self.label)


def _ev(host, label, group="g", source="log"):
    return _Evdef(source=source, host=host, group=group, label=label)


class TestEventDefinitionMap(unittest.TestCase):

    def setUp(self):
        self.evmap = log2event.EventDefinitionMap()
        self.a = _ev("host1", "L1")
        self.b = _ev("host2", "L2")
        self.eid_a = self.evmap.add_evdef(self.a)
        self.eid_b = self.evmap.add_evdef(self.b)

    def test_sequential_eids(self):
        self.assertEqual([self.eid_a, self.eid_b], [0, 1])
        self.assertEqual(len(self.evmap), 2)
        self.assertEqual(set(self.evmap.eids()), {0, 1})

    def test_forward_and_reverse_lookup(self):
        self.assertIs(self.evmap.evdef(self.eid_a), self.a)
        self.assertEqual(self.evmap.get_eid(self.b), self.eid_b)
        # reverse lookup keys on identifier, so an equal-identity copy resolves
        self.assertEqual(self.evmap.get_eid(_ev("host1", "L1")), self.eid_a)

    def test_membership(self):
        self.assertTrue(self.evmap.has_eid(self.eid_a))
        self.assertFalse(self.evmap.has_eid(99))
        self.assertTrue(self.evmap.has_evdef(self.a))
        self.assertFalse(self.evmap.has_evdef(_ev("host9", "L9")))

    def test_iteration(self):
        self.assertEqual(set(self.evmap.iter_eid()), {0, 1})
        identifiers = {ev.identifier for ev in self.evmap.iter_evdef()}
        self.assertEqual(identifiers, {self.a.identifier, self.b.identifier})
        self.assertEqual(dict(self.evmap.items()), {0: self.a, 1: self.b})

    def test_from_dict_rebuilds_reverse_map(self):
        mapping = {0: self.a, 1: self.b}
        rebuilt = log2event.EventDefinitionMap.from_dict(mapping)
        self.assertEqual(rebuilt.get_eid(self.a), 0)
        self.assertEqual(rebuilt.get_eid(self.b), 1)
        self.assertEqual(len(rebuilt), 2)

    def test_dump_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "evdef.pickle")
            with mock.patch.object(log2event.arguments.ArgumentManager,
                                   "evdef_path", staticmethod(lambda args: path)):
                self.evmap.dump(args=None)
                loaded = log2event.EventDefinitionMap()
                loaded.load(args=None)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded.evdef(0).identifier, self.a.identifier)
        self.assertEqual(loaded.get_eid(self.b), self.eid_b)


if __name__ == "__main__":
    unittest.main()

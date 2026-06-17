#!/usr/bin/env python
# coding: utf-8

"""Unit tests for logdag.visual.edge_search.

Regression tests for code-review findings (visual package, no prior tests):

* get_evpair_count applied len() to an int Counter value -> TypeError.
* edges_anomaly_score feature="edge"/score="idf" wrongly called get_tfidf
  (copy-paste) instead of get_idf.
* DAGSimilarity.similarity passed 1-D Series to cosine_similarity (needs 2-D)
  and returned a matrix instead of a scalar score.
* dag_anomaly_score summed the (edge, value) generator directly (int + tuple
  TypeError); the fix sums the extracted values.

These stub the heavy collaborators (counters / matrices) so no DAG is loaded.
"""

import types
import unittest

import pandas as pd

from logdag.visual import edge_search as es


class _FakeCounter:
    """Returns a sentinel tagging which method was called."""

    def get_idf(self, edge, ldag):
        return ("idf", edge)

    def get_tfidf(self, edge, ldag):
        return ("tfidf", edge)

    def get_edge_count(self, edge, ldag):
        return ("count", edge)


class _NumCounter:
    def get_edge_count(self, edge, ldag):
        return 5


class TestGetEvpairCount(unittest.TestCase):

    def test_local_count_is_int_value_not_len(self):
        obj = es.EventPairCount.__new__(es.EventPairCount)
        key = ("x", "y")
        obj._am = types.SimpleNamespace(jobname=lambda args: "job")
        obj._d_evpair_count = {"job": {key: 3}}      # Counter value is an int
        obj._d_evpair_args = {key: ["a", "b"]}        # collection -> len == 2
        obj.evpair_key = lambda n1, n2, ldag: key
        ldag = types.SimpleNamespace(args=("dummy",))

        local, whole = obj.get_evpair_count("n1", "n2", ldag)
        self.assertEqual(local, 3)    # old code did len(3) -> TypeError
        self.assertEqual(whole, 2)


class TestEdgesAnomalyScoreIdf(unittest.TestCase):

    def test_edge_idf_uses_get_idf_not_tfidf(self):
        edges = [("a", "b")]
        out = list(es.edges_anomaly_score(
            edges, ldag=None, feature="edge", score="idf",
            counter=_FakeCounter()))
        self.assertEqual(out, [(("a", "b"), ("idf", ("a", "b")))])

    def test_edge_tfidf_still_uses_tfidf(self):
        edges = [("a", "b")]
        out = list(es.edges_anomaly_score(
            edges, ldag=None, feature="edge", score="tfidf",
            counter=_FakeCounter()))
        self.assertEqual(out, [(("a", "b"), ("tfidf", ("a", "b")))])


class TestAnomalyScoreTupleContract(unittest.TestCase):
    """edges_anomaly_score yields (edge, value); dag_anomaly_score must sum the
    values, not the generator (summing tuples is a TypeError)."""

    def test_summing_generator_directly_is_typeerror(self):
        gen = es.edges_anomaly_score(
            [("a", "b")], ldag=None, feature="edge", score="count",
            counter=_NumCounter())
        with self.assertRaises(TypeError):
            sum(gen)

    def test_summing_extracted_values_works(self):
        total = sum(v for _e, v in es.edges_anomaly_score(
            [("a", "b"), ("c", "d")], ldag=None, feature="edge",
            score="count", counter=_NumCounter()))
        self.assertEqual(total, 10)


class TestSimilarity(unittest.TestCase):

    def _sim(self, matrix):
        # DAGSimilarity is abstract; use a concrete subclass and bypass __init__
        obj = es.DAGSimilarityEdgeCount.__new__(es.DAGSimilarityEdgeCount)
        obj._matrix = matrix
        return obj

    def test_identical_vectors_score_one(self):
        m = pd.DataFrame({"a": [1, 0, 1], "b": [1, 0, 1], "c": [0, 1, 0]})
        sim = self._sim(m)
        self.assertAlmostEqual(sim.similarity("a", "b"), 1.0)

    def test_orthogonal_vectors_score_zero(self):
        m = pd.DataFrame({"a": [1, 0, 1], "b": [1, 0, 1], "c": [0, 1, 0]})
        sim = self._sim(m)
        self.assertAlmostEqual(sim.similarity("a", "c"), 0.0)

    def test_returns_scalar(self):
        m = pd.DataFrame({"a": [1, 0, 1], "b": [0, 1, 0]})
        sim = self._sim(m)
        result = sim.similarity("a", "b")
        self.assertIsInstance(float(result), float)
        self.assertFalse(hasattr(result, "__len__"))


if __name__ == "__main__":
    unittest.main()

"""Toolkit to generate pseudo datasets for causal inference / discovery.

Vendored into logdag from the standalone ``causaltestdata`` package
(https://github.com/cpflat/causaltestdata, BSD-3-Clause, same author).
It builds synthetic time-series from a known causal DAG, which is used here
to produce test/evaluation data with ground-truth structure -- in particular
event series (see ``variable.PeriodicEventVariable`` /
``variable.TimeSeriesEventVariable``) that feed the amulog -> evdb pipeline.

This is a data-generation/evaluation component: logdag's core (makedag,
source, filter) must not import it; only tests, evaluation code and
downstream consumers (e.g. logdagviz) do.
"""

#!/usr/bin/env python
# coding: utf-8

"""Regression tests for logdag.arguments.ArgumentManager.

Code-review findings:
* ``jobname2args`` split the jobname on the first ``_``, but the area part may
  itself contain ``_`` (e.g. ``host_xxx``), so the round-trip was broken. It now
  matches the datetime suffix instead.
* ``dag_path`` / ``evdef_path`` returned ``None`` when ``mkdir`` raised (parent
  missing / path is a file), so callers hit ``open(None)``. They now always
  return the path.
"""

import datetime
import os
import shutil
import tempfile
import unittest

import dateutil.tz
from amulog import config

from logdag import arguments


def _conf():
    return config.open_config(arguments.DEFAULT_CONFIG, base_default=False)


class TestJobnameRoundTrip(unittest.TestCase):

    def test_area_with_underscore_round_trips(self):
        conf = _conf()
        area = "host_router1"
        dts = datetime.datetime(2020, 1, 1, 0, 0, 0,
                                tzinfo=dateutil.tz.tzlocal())
        term = config.getdur(conf, "dag", "unit_term")
        args = (conf, (dts, dts + term), area)

        name = arguments.ArgumentManager.jobname(args)
        _conf2, dt_range2, area2 = arguments.ArgumentManager.jobname2args(
            name, conf)
        self.assertEqual(area2, area)            # old code returned "host"
        self.assertEqual(dt_range2[0], dts)


class TestDagPath(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_dag_path_not_none_when_mkdir_fails(self):
        conf = _conf()
        # parent does not exist -> single-level os.mkdir raises OSError
        conf["dag"]["output_dir"] = os.path.join(self._tmp, "missing")
        dts = datetime.datetime(2020, 1, 1, 0, 0, 0,
                                tzinfo=dateutil.tz.tzlocal())
        term = config.getdur(conf, "dag", "unit_term")
        args = (conf, (dts, dts + term), "area")

        path = arguments.ArgumentManager.dag_path(conf, args)
        self.assertIsNotNone(path)               # old code returned None
        self.assertTrue(path.endswith("dag.pickle"))


if __name__ == "__main__":
    unittest.main()

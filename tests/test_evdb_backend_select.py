#!/usr/bin/env python
# coding: utf-8

"""Tests for EventLoader._init_evdb backend selection (general.evdb).

Pins the routing from the ``general.evdb`` config value to the backend factory,
including the InfluxDB 3 (``influx3``) branch and its ``[database_influx3]``
section. The factories are mocked, so no DB server is needed.
"""

import unittest
from unittest import mock

from amulog import config

from logdag import arguments
from logdag.source import evgen_common


class TestEvdbBackendSelect(unittest.TestCase):

    def _conf(self, evdb):
        conf = config.open_config(arguments.DEFAULT_CONFIG, base_default=False)
        conf["general"]["evdb"] = evdb
        return conf

    def _loader(self):
        # _init_evdb only touches conf/dbname_key, so a bare instance is enough
        return evgen_common.EventLoader.__new__(evgen_common.EventLoader)

    def test_influx3_routing(self):
        conf = self._conf("influx3")
        with mock.patch("logdag.source.influx3.init_influx_v3",
                        return_value="V3") as m:
            out = self._loader()._init_evdb(conf, "log_dbname")
        # dbname resolved from [database_influx3] log_dbname (= "log")
        m.assert_called_once_with(conf, "log")
        self.assertEqual(out, "V3")

    def test_influx_v1_routing(self):
        conf = self._conf("influx")
        with mock.patch("logdag.source.influx.init_influx",
                        return_value="V1") as m:
            out = self._loader()._init_evdb(conf, "log_dbname")
        m.assert_called_once()
        self.assertEqual(out, "V1")

    def test_sql_routing(self):
        conf = self._conf("sql")
        with mock.patch("logdag.source.sqlts.init_sqlts",
                        return_value="SQL") as m:
            out = self._loader()._init_evdb(conf, "log_dbname")
        m.assert_called_once()
        self.assertEqual(out, "SQL")

    def test_unknown_backend_raises(self):
        conf = self._conf("bogus")
        with self.assertRaises(NotImplementedError):
            self._loader()._init_evdb(conf, "log_dbname")


if __name__ == "__main__":
    unittest.main()

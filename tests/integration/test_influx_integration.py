#!/usr/bin/env python
# coding: utf-8

"""Integration tests for logdag.source.influx against a real InfluxDB 1.8.

OPT-IN: these need the ``influxdb`` client and a running InfluxDB 1.8. They
SKIP (never fail) when either is missing, so the normal unit run is unaffected.

Local:
    docker compose up -d            # InfluxDB 1.8 on :8086 (see docker-compose.yml)
    pip install influxdb
    pytest tests/integration
    docker compose down

CI: the `integration` job in .github/workflows/test.yml provides the service.

Override the endpoint with INFLUXDB_HOST / INFLUXDB_PORT.
"""

import datetime
import os
import unittest

import pandas as pd
from dateutil import tz

try:
    import influxdb as _influxdb
except ImportError:  # client not installed
    _influxdb = None

# NOTE: logdag.source.influx is imported lazily in setUpClass (it does a
# top-level ``import influxdb``, which is absent in the unit-only environment).


_HOST = os.environ.get("INFLUXDB_HOST", "localhost")
_PORT = int(os.environ.get("INFLUXDB_PORT", "8086"))
_DBNAME = "logdag_influxdb_v1_test"
_UTC = tz.tzutc()


def _server_available():
    if _influxdb is None:
        return False
    try:
        client = _influxdb.InfluxDBClient(host=_HOST, port=_PORT, timeout=2)
        client.ping()
        return True
    except Exception:
        return False


def _ts(ut):
    return pd.Timestamp(ut, unit="s", tz=_UTC)


def _dt(ut):
    return datetime.datetime.fromtimestamp(ut, tz=_UTC)


@unittest.skipUnless(_server_available(),
                     "InfluxDB 1.8 not reachable at {0}:{1} "
                     "(start it with `docker compose up -d`)".format(
                         _HOST, _PORT))
class TestInfluxIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from logdag.source import influx
        cls._influx = influx
        cls._client = _influxdb.InfluxDBClient(host=_HOST, port=_PORT)
        cls._client.create_database(_DBNAME)
        cls.db = influx.InfluxDBv1(_DBNAME, {"host": _HOST, "port": _PORT})

    @classmethod
    def tearDownClass(cls):
        cls._client.drop_database(_DBNAME)

    def tearDown(self):
        self.db.drop_measurement(self._measure)

    def test_count_roundtrip(self):
        self._measure = "m_count"
        base = 1600000000
        d_input = {_ts(base + i * 60): [float(i)] for i in range(5)}
        self.db.add(self._measure, {"host": "h1"}, d_input, ["val"])
        n = self.db.get_count(self._measure, {"host": "h1"}, ["val"],
                              (_dt(base), _dt(base + 5 * 60)))
        self.assertEqual(n, 5)

    def test_tag_value_with_quote_matches(self):
        # the escaping fix lets a tag value containing ' produce a valid query
        self._measure = "m_escape"
        base = 1600000000
        self.db.add(self._measure, {"host": "a'b"}, {_ts(base): [1.0]}, ["val"])
        n = self.db.get_count(self._measure, {"host": "a'b"}, ["val"],
                              (_dt(base - 1), _dt(base + 1)))
        self.assertEqual(n, 1)

    def test_get_df_roundtrip(self):
        self._measure = "m_df"
        base = 1600000000
        d_input = {_ts(base + i * 60): [float(i * 10)] for i in range(3)}
        self.db.add(self._measure, {"host": "h1"}, d_input, ["val"])
        df = self.db.get_df(self._measure, {"host": "h1"}, ["val"],
                            (_dt(base), _dt(base + 3 * 60)))
        self.assertIsNotNone(df)
        self.assertEqual(sorted(df["val"].tolist()), [0.0, 10.0, 20.0])


if __name__ == "__main__":
    unittest.main()

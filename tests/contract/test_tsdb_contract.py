#!/usr/bin/env python
# coding: utf-8

"""Conformance (contract) tests for the TimeSeriesDB backends.

sqlts.SQLTimeSeries, influx.InfluxDBv1 (and a future InfluxDB v3) all implement
the same ``TimeSeriesDB`` interface. The behaviours every backend must agree on
are written ONCE here and parametrized over the backends, so:

* a new backend (e.g. InfluxDB v3) is added by extending ``_BACKENDS`` and
  inherits the whole suite -- the safety net for the v1 -> v3 migration;
* behavioural divergences between backends surface immediately.

Per-backend availability differs, so each backend factory skips itself when it
cannot run (sqlts always works on a temp sqlite; influx_v1 needs a reachable
InfluxDB -- see docker-compose.yml). Implementation-specific details (InfluxQL
escaping, sqlts sorting/fill, ...) stay in the per-backend unit tests, not here.

CONTRACT for the empty / no-data cases (previously divergent between backends,
now unified and asserted below):
* get_count on an empty range -> 0 (a count is a number, not None).
* get_df(func=None) on an empty range -> None ("no data" sentinel that callers
  already check with ``df is None``).

(The func="sum" / fill empty-result behaviour is a separate, still-open question
and is not asserted here yet.)
"""

import datetime
import os
import tempfile

import pandas as pd
import pytest
from dateutil import tz

_UTC = tz.tzutc()
_INFLUX_HOST = os.environ.get("INFLUXDB_HOST", "localhost")
_INFLUX_PORT = int(os.environ.get("INFLUXDB_PORT", "8086"))
# v3 dev server (docker-compose `influxdb-v3`, image influxdb:3-core, port 8181)
_INFLUX_V3_URL = os.environ.get("INFLUXDB_V3_URL", "http://localhost:8181")
_INFLUX_V3_TOKEN = os.environ.get("INFLUXDB_V3_TOKEN", "")


def _ts(ut):
    return pd.Timestamp(ut, unit="s", tz=_UTC)


def _dt(ut):
    return datetime.datetime.fromtimestamp(ut, tz=_UTC)


# --- backend factories: each returns (db, teardown) or skips -----------------

def _skip_or_fail(env_name, msg):
    """Skip an unreachable influx backend -- unless it was declared REQUIRED, in
    which case fail loudly. An intended influx run (container expected up) that
    is actually blocked -- container stopped, or a sandbox cutting off localhost
    -- must not masquerade as a harmless skip. Gate per backend with
    INFLUXDB_V1_REQUIRED / INFLUXDB_V3_REQUIRED, or all at once with
    INFLUXDB_REQUIRED."""
    if os.environ.get(env_name) or os.environ.get("INFLUXDB_REQUIRED"):
        flag = env_name if os.environ.get(env_name) else "INFLUXDB_REQUIRED"
        pytest.fail("{0} [{1} set: expected a live server]".format(msg, flag))
    pytest.skip(msg)


def _make_sqlts():
    from amulog import db_sqlite
    from logdag.source.sqlts import SQLTimeSeries

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "contract.db")
    db = SQLTimeSeries(db_sqlite.Sqlite3(path))

    def teardown():
        if os.path.exists(path):
            os.remove(path)
        os.rmdir(tmpdir)

    return db, teardown


def _influx_available():
    try:
        import influxdb
    except ImportError:
        return False
    try:
        influxdb.InfluxDBClient(host=_INFLUX_HOST, port=_INFLUX_PORT,
                                timeout=2).ping()
        return True
    except Exception:
        return False


def _make_influx_v1():
    if not _influx_available():
        _skip_or_fail("INFLUXDB_V1_REQUIRED",
                      "InfluxDB v1 not reachable at {0}:{1} "
                      "(start it with `docker compose up -d`)".format(
                          _INFLUX_HOST, _INFLUX_PORT))
    import influxdb
    from logdag.source import influx

    dbname = "logdag_contract_v1_test"
    client = influxdb.InfluxDBClient(host=_INFLUX_HOST, port=_INFLUX_PORT)
    client.create_database(dbname)
    db = influx.InfluxDBv1(dbname, {"host": _INFLUX_HOST, "port": _INFLUX_PORT})

    def teardown():
        client.drop_database(dbname)

    return db, teardown


def _influx_v3_available():
    # stdlib-only probe; the v3 backend itself uses urllib (no influxdb3-python
    # dependency required for the contract run).
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.Request(_INFLUX_V3_URL + "/health", method="GET")
        if _INFLUX_V3_TOKEN:
            req.add_header("Authorization", "Bearer " + _INFLUX_V3_TOKEN)
        with urllib.request.urlopen(req, timeout=3) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as e:
        # any HTTP response means the server is up (e.g. 401 without token)
        return True
    except Exception:
        return False


def _make_influx_v3():
    if not _influx_v3_available():
        _skip_or_fail("INFLUXDB_V3_REQUIRED",
                      "InfluxDB v3 not reachable at {0} "
                      "(start it with `docker compose up -d influxdb-v3`)".format(
                          _INFLUX_V3_URL))
    from logdag.source import influx3

    dbname = "logdag_contract_v3_test"
    # v3 auto-creates a database on first write; construct with
    # create_if_missing so __init__ does not require a databases listing.
    db = influx3.InfluxDBv3(dbname, host=_INFLUX_V3_URL,
                            token=_INFLUX_V3_TOKEN or None,
                            create_if_missing=True)

    def teardown():
        # best-effort: drop every table created during the test; the dev server
        # uses object-store=memory so a missing drop is harmless (fresh per run).
        try:
            for m in db.list_measurements():
                try:
                    db.drop_measurement(m)
                except Exception:
                    pass
        except Exception:
            pass

    return db, teardown


_BACKENDS = {
    "sqlts": _make_sqlts,
    "influx_v1": _make_influx_v1,
    "influx_v3": _make_influx_v3,   # skips until the v3 backend exists
}


@pytest.fixture(params=list(_BACKENDS))
def tsdb(request):
    db, teardown = _BACKENDS[request.param]()
    try:
        yield db
    finally:
        teardown()


# --- shared contract ---------------------------------------------------------

_BASE = 1600000000  # a fixed epoch second, timezone-independent


class TestTimeSeriesDBContract:

    def test_add_get_count_roundtrip(self, tsdb):
        d_input = {_ts(_BASE + i * 60): [float(i)] for i in range(5)}
        tsdb.add("m", {"host": "h1"}, d_input, ["val"])
        tsdb.commit()
        n = tsdb.get_count("m", {"host": "h1"}, ["val"],
                           (_dt(_BASE), _dt(_BASE + 5 * 60)))
        assert n == 5

    def test_add_get_df_roundtrip(self, tsdb):
        d_input = {_ts(_BASE + i * 60): [float(i * 10)] for i in range(3)}
        tsdb.add("m", {"host": "h1"}, d_input, ["val"])
        tsdb.commit()
        df = tsdb.get_df("m", {"host": "h1"}, ["val"],
                         (_dt(_BASE), _dt(_BASE + 3 * 60)))
        assert df is not None
        assert sorted(df["val"].tolist()) == [0.0, 10.0, 20.0]

    def test_tag_filter_isolates_series(self, tsdb):
        tsdb.add("m", {"host": "h1"}, {_ts(_BASE): [1.0]}, ["val"])
        tsdb.add("m", {"host": "h2"}, {_ts(_BASE): [2.0]}, ["val"])
        tsdb.commit()
        n = tsdb.get_count("m", {"host": "h1"}, ["val"],
                           (_dt(_BASE - 1), _dt(_BASE + 1)))
        assert n == 1

    def test_get_count_empty_is_zero(self, tsdb):
        tsdb.add("m", {"host": "h1"}, {_ts(_BASE): [1.0]}, ["val"])
        tsdb.commit()
        # a range with no points -> 0 (not None)
        n = tsdb.get_count("m", {"host": "h1"}, ["val"],
                           (_dt(_BASE + 3600), _dt(_BASE + 7200)))
        assert n == 0

    def test_get_df_empty_is_none(self, tsdb):
        tsdb.add("m", {"host": "h1"}, {_ts(_BASE): [1.0]}, ["val"])
        tsdb.commit()
        # a range with no points -> None (func=None)
        df = tsdb.get_df("m", {"host": "h1"}, ["val"],
                         (_dt(_BASE + 3600), _dt(_BASE + 7200)))
        assert df is None

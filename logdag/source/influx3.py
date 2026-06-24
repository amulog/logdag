#!/usr/bin/env python
# coding: utf-8

"""InfluxDB v3 Core backend for logdag's TimeSeriesDB contract.

This mirrors :class:`logdag.source.influx.InfluxDBv1` but targets the InfluxDB
v3 (IOx) HTTP API:

* writes  -> ``POST /api/v3/write_lp?db=<database>`` (Line Protocol body, 204)
* queries -> ``GET  /api/v3/query_sql?db=<database>&q=<SQL>&format=csv`` (200)

Design notes (see docs/influxdb_migration.md sections 4-1(a)(b)(c), 4-3):

* v3 uses a single-level ``database`` namespace -- the v1 3-part
  ``"db"."autogen"."measure"`` retention-policy qualifier is dropped; tables are
  referenced by their bare (quoted) measurement name.
* SQL (DataFusion) is the primary query API. We build SQL instead of InfluxQL
  because the contract only needs SELECT / COUNT with tag-equality and a time
  window, which map cleanly onto SQL and avoid InfluxQL/SQL result-shape quirks.
* Authentication is OPTIONAL: a bearer token is read from ``INFLUXDB_V3_TOKEN``
  (or passed explicitly) and only sent when non-empty. Dev servers run
  ``--without-auth`` and need no token.
* No hard new dependency: a thin stdlib (``urllib``) HTTP client is used so the
  contract run does not require ``influxdb3-python``/``pyarrow``.

Contract semantics reproduced exactly (asserted by
tests/contract/test_tsdb_contract.py):

* ``get_count`` over an empty range -> ``0`` (not ``None``).
* ``get_df(func=None)`` over an empty range -> ``None``.
* ``get_df`` returns a wide DataFrame whose index is a tz-aware LOCAL
  ``DatetimeIndex`` (v3 returns naive UTC ``time`` -> localize utc -> convert to
  local, same as v1) and whose columns are the field names.
* tag filters isolate series.
"""

import csv
import io
import json
import logging
import os
import urllib.parse
import urllib.request
import urllib.error

import numpy as np
import pandas as pd
from dateutil import tz

from .sqlts import TimeSeriesDB

_logger = logging.getLogger(__package__)

_ENV_TOKEN = "INFLUXDB_V3_TOKEN"


class InfluxDBv3(TimeSeriesDB):
    """TimeSeriesDB backend for InfluxDB v3 Core over the v3 HTTP API."""

    def __init__(self, dbname, host="http://localhost:8181", token=None,
                 batch_size=1000, timeout=30, create_if_missing=False):
        """
        Args:
            dbname: v3 database name (single-level namespace).
            host: base URL of the v3 server, e.g. ``http://localhost:8181``.
            token: bearer token; if ``None`` falls back to the
                ``INFLUXDB_V3_TOKEN`` env var; empty -> no auth (dev).
            batch_size: max Line Protocol lines per write request.
            timeout: per-request timeout in seconds.
            create_if_missing: if True, do not error when the database is
                absent (v3 auto-creates a database on first write).
        """
        self.dbname = dbname
        self._precision = "nanosecond"
        self._batch_size = batch_size
        self._timeout = timeout
        self.verbose = False

        # normalise host to a scheme://host:port base (no trailing slash)
        if "://" not in host:
            host = "http://" + host
        self._host = host.rstrip("/")

        if token is None:
            token = os.environ.get(_ENV_TOKEN, "")
        self._token = (token or "").strip()

        # v3 auto-creates a database on first write; existence is best-effort.
        if not create_if_missing:
            try:
                dbs = self._list_database()
            except Exception:  # server may not expose the listing endpoint
                dbs = None
            if dbs is not None and dbname not in dbs:
                # create explicitly via SQL so subsequent metadata queries work
                self._create_database()

    # --- low-level HTTP --------------------------------------------------

    def _headers(self):
        h = {}
        if self._token:
            h["Authorization"] = "Bearer {0}".format(self._token)
        return h

    def _request(self, url, data=None, headers=None, method=None):
        req = urllib.request.Request(url, data=data, method=method)
        for k, v in self._headers().items():
            req.add_header(k, v)
        if headers:
            for k, v in headers.items():
                req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            body = e.read()
            raise IOError(
                "InfluxDBv3 request failed ({0}): {1} -- {2}".format(
                    e.code, url, body[:500])) from e

    def _query_sql(self, sql):
        """Run a SQL query, return list-of-dict rows (CSV parsed)."""
        params = urllib.parse.urlencode({
            "db": self.dbname,
            "q": sql,
            "format": "csv",
        })
        url = "{0}/api/v3/query_sql?{1}".format(self._host, params)
        if self.verbose:
            print(sql)
        _logger.debug("influxdb v3 sql: %s", sql)
        status, body = self._request(url, method="GET")
        text = body.decode("utf-8")
        if text.strip() == "":
            return []
        reader = csv.DictReader(io.StringIO(text))
        return [row for row in reader]

    def _write_lp(self, lines):
        """Write Line Protocol lines (a list of str)."""
        if not lines:
            return
        params = urllib.parse.urlencode({
            "db": self.dbname,
            "precision": self._precision,
        })
        url = "{0}/api/v3/write_lp?{1}".format(self._host, params)
        # batch to bound request size
        for i in range(0, len(lines), self._batch_size):
            chunk = lines[i:i + self._batch_size]
            data = "\n".join(chunk).encode("utf-8")
            status, _ = self._request(
                url, data=data,
                headers={"Content-Type": "text/plain; charset=utf-8"},
                method="POST")

    def _list_database(self):
        # v3 exposes databases via SQL system catalog; query a server-level db.
        # Use the _internal database which always exists on a v3 server.
        params = urllib.parse.urlencode({
            "db": "_internal",
            "q": "SELECT iox::database FROM system.databases",
            "format": "csv",
        })
        url = "{0}/api/v3/query_sql?{1}".format(self._host, params)
        try:
            status, body = self._request(url, method="GET")
        except Exception:
            # fall back: configure endpoint
            url2 = "{0}/api/v3/configure/database?format=json".format(self._host)
            status, body = self._request(url2, method="GET")
            data = json.loads(body.decode("utf-8"))
            return [d.get("iox::database", d.get("database", d.get("name")))
                    for d in data]
        text = body.decode("utf-8")
        if text.strip() == "":
            return []
        reader = csv.DictReader(io.StringIO(text))
        out = []
        for row in reader:
            # column name may be "iox::database" or "database"
            name = row.get("iox::database") or row.get("database") \
                or next(iter(row.values()), None)
            if name:
                out.append(name)
        return out

    def _create_database(self):
        url = "{0}/api/v3/configure/database".format(self._host)
        data = json.dumps({"db": self.dbname}).encode("utf-8")
        try:
            self._request(url, data=data,
                          headers={"Content-Type": "application/json"},
                          method="POST")
        except Exception as e:
            _logger.debug("v3 create_database best-effort failed: %s", e)

    # --- LP / SQL escaping helpers ---------------------------------------

    @staticmethod
    def _escape_lp_tag(v):
        # tag keys, tag values and field keys: escape , = and space
        return str(v).replace("\\", "\\\\").replace(",", "\\,") \
            .replace("=", "\\=").replace(" ", "\\ ")

    @staticmethod
    def _escape_lp_measure(v):
        return str(v).replace("\\", "\\\\").replace(",", "\\,") \
            .replace(" ", "\\ ")

    @staticmethod
    def _escape_sql_str(v):
        # single-quoted SQL string literal (DataFusion): double the quote
        return str(v).replace("'", "''")

    @staticmethod
    def _quote_ident(v):
        # double-quoted SQL identifier
        return '"' + str(v).replace('"', '""') + '"'

    # --- TimeSeriesDB contract -------------------------------------------

    def list_measurements(self):
        rows = self._query_sql(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'iox'")
        return [r["table_name"] for r in rows]

    def list_series(self, measure=None):
        """Return distinct tag-value combinations.

        v3 has no SHOW SERIES; reconstruct from the tag columns of the table.
        """
        if measure is None:
            measures = self.list_measurements()
        else:
            measures = [measure]

        ret = []
        for m in measures:
            tag_cols = self._tag_columns(m)
            if not tag_cols:
                continue
            sel = ", ".join(self._quote_ident(c) for c in tag_cols)
            sql = "SELECT DISTINCT {0} FROM {1}".format(
                sel, self._quote_ident(m))
            for row in self._query_sql(sql):
                d = {}
                for c in tag_cols:
                    val = row.get(c)
                    if val is not None and val != "":
                        d[c] = val
                if d:
                    ret.append(d)
        return ret

    def _column_meta(self, measure):
        """Return {column_name: data_type} for a table, via information_schema."""
        sql = ("SELECT column_name, data_type "
               "FROM information_schema.columns "
               "WHERE table_name = '{0}'".format(self._escape_sql_str(measure)))
        rows = self._query_sql(sql)
        return {r["column_name"]: r["data_type"] for r in rows}

    def _tag_columns(self, measure):
        # tags are Dictionary/Utf8 columns (excluding the special "time")
        meta = self._column_meta(measure)
        tags = []
        for name, dtype in meta.items():
            if name == "time":
                continue
            d = (dtype or "").lower()
            if "utf8" in d or "dictionary" in d or "string" in d:
                tags.append(name)
        return tags

    def list_fields(self, measure):
        # fields are the numeric columns (exclude time and tag/string columns)
        meta = self._column_meta(measure)
        fields = []
        for name, dtype in meta.items():
            if name == "time":
                continue
            d = (dtype or "").lower()
            if "utf8" in d or "dictionary" in d or "string" in d:
                continue
            fields.append(name)
        return fields

    def add(self, measure, d_tags, d_input, columns):
        """Mirror v1's NaN-exclusion; build Line Protocol; self-count writes."""
        lines = []
        m = self._escape_lp_measure(measure)
        tagstr = ""
        for k, v in d_tags.items():
            tagstr += ",{0}={1}".format(
                self._escape_lp_tag(k), self._escape_lp_tag(v))
        for t, row in d_input.items():
            fields = {key: val for key, val in zip(columns, row)
                      if not (val is None or np.isnan(val))}
            if len(fields) == 0:
                continue
            # nanosecond timestamp; t is a tz-aware (or naive) pd.Timestamp
            ts = self.pdtimestamp_naive(t)
            ns = int(pd.Timestamp(ts).value)  # ns since epoch (naive utc)
            fieldstr = ",".join(
                "{0}={1}".format(self._escape_lp_tag(k), float(v))
                for k, v in fields.items())
            lines.append("{0}{1} {2} {3}".format(m, tagstr, fieldstr, ns))
        if lines:
            self._write_lp(lines)
        return len(lines)

    def commit(self):
        # v3 writes are synchronous (204 on the write request); nothing to flush.
        pass

    def _where_clause(self, d_tags, dt_range):
        ut_range = tuple(dt.timestamp() for dt in dt_range)
        conds = []
        for k, v in d_tags.items():
            conds.append("{0} = '{1}'".format(
                self._quote_ident(k), self._escape_sql_str(v)))
        # time bounds in ns since epoch; v3 SQL accepts timestamp literals via
        # to_timestamp_nanos for robust boundary handling.
        ns_lo = int(ut_range[0] * 1e9)
        ns_hi = int(ut_range[1] * 1e9)
        conds.append("time >= to_timestamp_nanos({0})".format(ns_lo))
        conds.append("time < to_timestamp_nanos({0})".format(ns_hi))
        return " AND ".join(conds)

    def _select(self, measure, d_tags, fields, dt_range, count=False,
                limit=None):
        if fields is None:
            fields = self.list_fields(measure)
        if count:
            # count over the first field (rows with that field present)
            sel = "count({0}) AS cnt".format(self._quote_ident(fields[0]))
        else:
            sel = "time, " + ", ".join(self._quote_ident(f) for f in fields)
        sql = "SELECT {0} FROM {1} WHERE {2}".format(
            sel, self._quote_ident(measure), self._where_clause(d_tags, dt_range))
        if not count:
            sql += " ORDER BY time"
        if limit is not None:
            sql += " LIMIT {0}".format(int(limit))
        return self._query_sql(sql), fields

    @staticmethod
    def _parse_time(s):
        """Parse a v3 CSV time cell into a tz-aware LOCAL pd.Timestamp.

        v3 returns naive UTC (RFC3339 without offset, or epoch) -> localize utc
        -> convert to local, matching InfluxDBv1.get_df (lines 165-168).
        """
        dt = pd.to_datetime(s)  # naive (utc) or possibly already aware
        if dt.tzinfo is None and getattr(dt, "tz", None) is None:
            dt = dt.tz_localize(tz.tzutc())
        else:
            dt = dt.tz_convert(tz.tzutc())
        return dt.tz_convert(tz.tzlocal())

    def get_items(self, measure, d_tags, fields, dt_range):
        rows, fields = self._select(measure, d_tags, fields, dt_range)
        for row in rows:
            dt = self._parse_time(row["time"])
            array = np.array([float(row[f]) for f in fields])
            yield dt, array

    def has_data(self, measure, d_tags, fields, dt_range):
        rows, _ = self._select(measure, d_tags, fields, dt_range, limit=1)
        return len(rows) >= 1

    def get_df(self, measure, d_tags, fields, dt_range,
               str_bin=None, func=None, fill=None, limit=None):
        if fields is None:
            fields = self.list_fields(measure)
        # Densify (func/fill/str_bin) is delegated to the shared layer per
        # docs section 4-4; the backend only serves sparse reads (func=None).
        if func is not None:
            raise NotImplementedError(
                "InfluxDBv3 serves sparse reads only; densify (func/fill) is "
                "handled by the feature/analysis layer (docs 4-4).")
        rows, fields = self._select(measure, d_tags, fields, dt_range,
                                    limit=limit)
        if len(rows) == 0:
            # contract: no rows -> None (the "no data" sentinel)
            return None
        dtindex = pd.DatetimeIndex([self._parse_time(r["time"]) for r in rows])
        l_array = [np.array([float(r[f]) for f in fields]) for r in rows]
        return pd.DataFrame(l_array, index=dtindex, columns=fields)

    def get_count(self, measure, d_tags, fields, dt_range):
        if fields is None:
            fields = self.list_fields(measure)
        rows, fields = self._select(measure, d_tags, fields, dt_range,
                                    count=True)
        if len(rows) == 0:
            # contract: an empty range counts as 0, not None
            return 0
        val = rows[0].get("cnt")
        if val is None or val == "":
            return 0
        return int(float(val))

    def drop_measurement(self, measure):
        # v3 deletes a table via the configure/table DELETE endpoint.
        params = urllib.parse.urlencode({"db": self.dbname, "table": measure})
        url = "{0}/api/v3/configure/table?{1}".format(self._host, params)
        try:
            self._request(url, method="DELETE")
        except Exception:
            # fall back to JSON body form
            url2 = "{0}/api/v3/configure/table".format(self._host)
            data = json.dumps({"db": self.dbname,
                               "table": measure}).encode("utf-8")
            self._request(url2, data=data,
                          headers={"Content-Type": "application/json"},
                          method="DELETE")


def init_influx_v3(conf, dbname):
    """Factory mirroring ``influx.init_influx`` for the v3 backend.

    Reads the ``[database_influx3]`` config section:
        * ``host``  -- base URL (``http://host:port``); falls back to
          ``http://<host>:<port>`` if only ``host``/``port`` are present.
        * ``token`` -- optional; env var ``INFLUXDB_V3_TOKEN`` takes precedence
          when set, so the token need not be written into the config file.
        * ``batch_size`` -- optional write batch size.

    The ``dbname`` argument is the resolved database name (the caller passes the
    value of e.g. ``log_dbname``/``snmp_dbname`` from ``[database_influx3]``).
    """
    section = "database_influx3"

    def _get(key, default=None):
        try:
            v = conf[section][key]
        except Exception:
            return default
        v = (v or "").strip()
        return v if v else default

    host = _get("host")
    if host is None:
        # fall back to host+port (v1-style keys), default v3 port 8181
        h = _get("host", "localhost") or "localhost"
        p = _get("port", "8181") or "8181"
        host = "http://{0}:{1}".format(h, p)

    # env var wins over config for the token (security: avoid plaintext token)
    token = os.environ.get(_ENV_TOKEN, _get("token", "")) or ""

    try:
        batch_size = conf.getint(section, "batch_size")
    except Exception:
        batch_size = 1000

    return InfluxDBv3(dbname, host=host, token=token, batch_size=batch_size)

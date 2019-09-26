#!/usr/bin/env python
# coding: utf-8

### OLD FILE !!!!!


import pandas as pd

from amulog import db_common


class RRDEventDB():

    def __init__(self, conf):

        self.db = db_common.influxdb()





class RRDEventDBSQL():

    def __init__(self, conf):
        if db_type == "sqlite3":
            dbpath = conf.get("database_rrd", "sqlite3_filename")
            self.db = db_common.sqlite3(dbpath)
        elif db_type == "mysql":
            host = conf.get("database_rrd", "mysql_host")
            dbname = conf.get("database_rrd", "mysql_dbname")
            user = conf.get("database_rrd", "mysql_user")
            passwd = conf.get("database_rrd", "mysql_passwd")
            self.db = db_common.mysql(host, dbname, user, passwd)
        else:
            raise ValueError("invalid database type ({0})".format(
                    db_type))

        if self.db.db_exists():
            if not _exists_table("catalog"):
                self._init_catalog()
        else:
            self._init_catalog()

    def _tablename(self, host, name):
        return "[{0}@{1}]".format(name, host)

    def _init_catalog(self):
        table_name = "catalog"
        l_key = [db_common.tablekey("tablename", "text",
                    ("primary_key", "not_null")),
                 db_common.tablekey("host", "text"),
                 db_common.tablekey("name", "text")]
        sql = self.db.create_table_sql(table_name, l_key)
        self.db.execute(sql)

    def _init_table(self, host, name):
        table_name = self._tablename(host, name)
        l_key = [db_common.tablekey("dt", "datetime",
                    ("primary_key", "not_null")),
                 db_common.tablekey("value", "real")]
        sql = self.db.create_table_sql(table_name, l_key)
        self.db.execute(sql)

        index_name = table_name + "_index"
        l_key = [db_common.tablekey("dt", "datetime")]
        sql = self.db.create_index_sql(table_name, index_name, l_key)
        self.db.execute(sql)

    def _add_catalog(self, host, name):
        if not _exists_table("catalog"):
            self._init_catalog()
        l_ss = [db_common.setstate("tablename", "tablename"),
                db_common.setstate("host", "host"),
                db_common.setstate("name", "name")]
        sql = self.db.insert_sql("catalog", l_ss)
        args = {"tablename": self._tablename(host, name),
                "host": host,
                "name": name}
        self.db.execute(sql, args)

    def _exists_table(self, name):
        s_name = set(self.db.get_table_names())
        return name in s_name

    def add_lines(self, host, name, iterable):
        tablename = self._tablename(host, name)
        if not _exists_table(tablename):
            self._init_table(host, name)

        l_ss = [db_common.setstate("datetime", "datetime"),
                db_common.setstate("value", "value")]
        sql = self.db.insert_sql(tablename, l_ss)
        l_args = [{"datetime": dt, "value": val} for dt, val in iterable]
        self.db.executemany(sql, l_args)

    def get_catalog(self):
        table_name = "catalog"
        l_key = ["host", "name"]
        sql = self.db.select_sql(table_name, l_key)
        cursor = self.db.execute(sql)
        return [(row[0], row[1]) for row in cursor]

    def get(self, host, name, dt_range):
        tablename = self._tablename(host, name)
        if not _exists_table(tablename):
            return None

        l_key = ["datetime, value"]
        l_cond = [db_common.cond("dt", ">=", "dts"),
                  db_common.cond("dt", "<", "dte")]
        sql = self.db.select_sql(table_name, l_key, l_cond)
        args = {"dts": dt_range[0],
                "dte": dt_range[1]}
        if self.db.connect is None:
            self.db._open()
        return pd.read_sql_query(sql, con = self.db.connect, params = args)

    def commit(self):
        self.db.commit()




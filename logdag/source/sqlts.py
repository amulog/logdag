from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from dateutil import tz

from amulog import config
from amulog import db_common
from .. import dtutil


class TimeSeriesDB(ABC):

    @staticmethod
    def pdtimestamp_naive(input_dt):
        if isinstance(input_dt, pd.Timestamp):
            dt = input_dt
        else:
            dt = pd.Timestamp(input_dt)

        # If input_dt is timezone-aware,
        # convert it into naive(utc) for database input
        if dt.tz is not None:
            dt = dt.tz_convert(None)
            dt = dt.tz_localize(None)
        return dt

    @staticmethod
    def pdtimestamp(input_dt):
        if isinstance(input_dt, pd.Timestamp):
            dt = input_dt
        else:
            dt = pd.Timestamp(input_dt)

        # If input_dt is timezone-naive,
        # convert it into timezone-aware(local) for logdag use
        if dt.tz is None:
            dt = dt.tz_localize(tz.tzutc())
            dt = dt.tz_convert(tz.tzlocal())
        return dt

    @staticmethod
    def pdtimestamps(input_dts):
        dtindex = pd.to_datetime(input_dts)

        # If input_dt is timezone-naive,
        # convert it into timezone-aware(local) for logdag use
        if dtindex.tz is None:
            dtindex = dtindex.tz_localize(tz.tzutc())
            dtindex = dtindex.tz_convert(tz.tzlocal())
        return dtindex

    @abstractmethod
    def list_measurements(self):
        raise NotImplementedError

    @abstractmethod
    def list_series(self, measure):
        raise NotImplementedError

    @abstractmethod
    def list_fields(self, measure):
        raise NotImplementedError

    @abstractmethod
    def add(self, measure, d_tags, d_input, columns):
        raise NotImplementedError

    @abstractmethod
    def commit(self):
        raise NotImplementedError

    @abstractmethod
    def get_items(self, measure, d_tags, fields, dt_range):
        raise NotImplementedError

    @abstractmethod
    def get_df(self, measure, d_tags, fields, dt_range,
               str_bin=None, func=None, fill=None, limit=None):
        raise NotImplementedError

    @abstractmethod
    def get_count(self, measure, d_tags, fields, dt_range):
        raise NotImplementedError

    @abstractmethod
    def drop_measurement(self, measure):
        raise NotImplementedError


class SQLTimeSeries(TimeSeriesDB):
    _key_primary = "key"
    _key_time = "time"
    _header_length = 2
    _header_tag = "t_"
    _header_field = "f_"

    def __init__(self, database):
        self._db = database

    @classmethod
    def _field_column_name(cls, field_key):
        return cls._header_field + field_key

    @classmethod
    def _tag_column_name(cls, tag_key):
        return cls._header_tag + tag_key

    @classmethod
    def _table_keys(cls, tag_keys, field_keys):
        l_key = [db_common.TableKey(cls._key_primary, "integer",
                                    ("primary_key",
                                     "auto_increment", "not_null")),
                 db_common.TableKey(cls._key_time, "datetime", tuple())]
        for tag_key in tag_keys:
            key = cls._tag_column_name(tag_key)
            l_key.append(db_common.TableKey(key, "text", tuple()))
        for field_key in field_keys:
            key = cls._field_column_name(field_key)
            l_key.append(db_common.TableKey(key, "real", tuple()))
        return l_key

    @staticmethod
    def _index_name(measure):
        return measure + "_index"

    @classmethod
    def _index_keys(cls, tag_keys):
        l_key = [db_common.TableKey(cls._key_time, "datetime", tuple())]
        for tag_key in tag_keys:
            key = cls._header_tag + tag_key
            l_key.append(db_common.TableKey(key, "text", tuple()))
        return l_key

    def _init_table(self, measure, tag_keys, field_keys):
        table_names = self._db.get_table_names()
        if measure not in table_names:
            l_key = self._table_keys(tag_keys, field_keys)
            sql = self._db.create_table_sql(measure, l_key)
            self._db.execute(sql)

            index_name = self._index_name(measure)
            assert index_name not in table_names
            l_key = self._index_keys(tag_keys)
            sql = self._db.create_index_sql(measure, index_name, l_key)
            self._db.execute(sql)

    def _tag_column_names(self, measure):
        ret = []
        for name in self._db.get_column_names(measure):
            if name[:self._header_length] == self._header_tag:
                ret.append(name)
        return ret

    def _field_names(self, measure):
        ret = []
        for name in self._db.get_column_names(measure):
            if name[:self._header_length] == self._header_field:
                ret.append(name)
        return ret

    def list_measurements(self):
        return self._db.get_table_names()

    def list_series(self, measure):
        l_tag_key = self._tag_column_names(measure)
        sql = self._db.select_sql(measure, l_tag_key, opt=["distinct"])
        cursor = self._db.execute(sql)

        return [
            {tag_key: tag_val for tag_val, tag_key in zip(row, l_tag_key)}
            for row in cursor
        ]

    def list_fields(self, measure):
        return self._field_names(measure)

    def add(self, measure, d_tags, d_input, columns):
        tag_keys = d_tags.keys()
        field_keys = columns
        self._init_table(measure, tag_keys, field_keys)

        l_ss = [db_common.StateSet(self._key_time, self._key_time), ]
        for tag_key in tag_keys:
            key = self._header_tag + tag_key
            l_ss.append(db_common.StateSet(key, tag_key))
        for field_key in field_keys:
            key = self._header_field + field_key
            l_ss.append(db_common.StateSet(key, field_key))
        sql = self._db.insert_sql(measure, l_ss)

        l_args = []
        for t, row in d_input.items():
            dtstr = self._db.strftime(self.pdtimestamp_naive(t))
            args = {self._key_time: dtstr}
            for tag_key in tag_keys:
                args[tag_key] = d_tags[tag_key]
            for field_index, field_key in enumerate(field_keys):
                args[field_key] = row[field_index]
            l_args.append(args)

        self._db.executemany(sql, l_args)
        return len(l_args)

    def commit(self):
        self._db.commit()

    @staticmethod
    def _get_row_values(row):
        return row[0], row[1:]

    def _get(self, measure, d_tags, fields, dt_range):
        if fields is None:
            fields = self._field_names(measure)

        # dt_range is converted from aware(local) to naive(utc)
        dt_range_utc = [self.pdtimestamp_naive(dt) for dt in dt_range]

        l_key = [self._key_time] + [self._field_column_name(field_key)
                                    for field_key in fields]
        l_cond = []
        args = {}
        l_cond.append(db_common.Condition("time", ">=", "dts", True))
        args["dts"] = self._db.strftime(dt_range_utc[0])
        l_cond.append(db_common.Condition("time", "<", "dte", True))
        args["dte"] = self._db.strftime(dt_range_utc[1])
        for tag_key, tag_val in d_tags.items():
            key = self._tag_column_name(tag_key)
            l_cond.append(db_common.Condition(key, "=", tag_key, True))
            args[tag_key] = tag_val
        sql = self._db.select_sql(measure, l_key, l_cond)
        cursor = self._db.execute(sql, args)
        return cursor

    def get_items(self, measure, d_tags, fields, dt_range):
        cursor = self._get(measure, d_tags, fields, dt_range)

        for row in cursor:
            dtstr, values = self._get_row_values(row)
            # obtained as naive(utc), converted into aware(local)
            dt = self.pdtimestamp(self._db.strptime(dtstr))
            yield dt, np.array(values)

    def get_df(self, measure, d_tags, fields, dt_range,
               str_bin=None, func=None, fill=None, limit=None):
        if fields is None:
            fields = self.list_fields(measure)

        cursor = self._get(measure, d_tags, fields, dt_range)
        l_dt = []
        l_values = []
        for rid, row in enumerate(cursor):
            if limit is not None and rid >= limit:
                break
            dtstr, values = self._get_row_values(row)
            # obtained as naive(utc), converted into aware(local)
            l_dt.append(self.pdtimestamp(self._db.strptime(dtstr)))
            if fill:
                values = values.nan_to_num(fill)
            l_values.append(values)

        if func is None:
            dtindex = self.pdtimestamps(l_dt)
            return pd.DataFrame(l_values, index=dtindex, columns=fields)
        elif func == "sum":
            assert str_bin is not None
            binsize = config.str2dur(str_bin)
            dtindex = self.pdtimestamps(
                dtutil.range_dt(dt_range[0], dt_range[1], binsize)
            )

            d_values = {}
            if len(l_dt) == 0:
                for field in fields:
                    d_values[field] = [float(0)] * len(dtindex)
            else:
                for fid, series in enumerate(zip(*l_values)):
                    a_cnt = dtutil.discretize_sequential(l_dt, dt_range,
                                                         binsize, l_dt_values=series)
                    d_values[fields[fid]] = a_cnt

            return pd.DataFrame(d_values, index=dtindex)
        else:
            raise NotImplementedError

    def get_count(self, measure, d_tags, fields, dt_range):
        cursor = self._get(measure, d_tags, fields, dt_range)
        return sum(1 for _ in cursor)

    def drop_measurement(self, measure):
        sql = self._db.drop_sql(measure)
        self._db.execute(sql)


def init_sqlts(conf):
    db_type = conf["database_sql"]["database"]
    if db_type == "sqlite3":
        from amulog import db_sqlite
        dbpath = conf.get("database_sql", "sqlite3_filename")
        database = db_sqlite.Sqlite3(dbpath)
    elif db_type == "mysql":
        from amulog import db_mysql
        host = conf.get("database", "mysql_host")
        dbname = conf.get("database", "mysql_dbname")
        user = conf.get("database", "mysql_user")
        passwd = conf.get("database", "mysql_passwd")
        database = db_mysql.Mysql(host, dbname, user, passwd)
    else:
        raise NotImplementedError

    return SQLTimeSeries(database)

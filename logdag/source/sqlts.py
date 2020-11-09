from amulog import db_common


class SQLTimeSeries:
    _key_primary = "key"
    _key_time = "time"
    _header_length = 2
    _header_tag = "t_"
    _header_field = "f_"

    def __init__(self, conf):
        db_type = conf["database_sql"]["database"]
        if db_type == "sqlite3":
            from amulog import db_sqlite
            dbpath = conf.get("database_sql", "sqlite3_filename")
            self._db = db_sqlite.Sqlite3(dbpath)
        elif db_type == "mysql":
            from amulog import db_mysql
            host = conf.get("database", "mysql_host")
            dbname = conf.get("database", "mysql_dbname")
            user = conf.get("database", "mysql_user")
            passwd = conf.get("database", "mysql_passwd")
            self._db = db_mysql.Mysql(host, dbname, user, passwd)

    @classmethod
    def _keys(cls, tag_keys, field_keys):
        l_key = [db_common.TableKey(cls._key_primary, "integer",
                                    ("primary_key",
                                     "auto_increment", "not_null")),
                 db_common.TableKey(cls._key_time, "datetime", tuple())]
        for tag_key in tag_keys:
            key = cls._header_tag + tag_key
            l_key.append(db_common.TableKey(key, "text", tuple()))
        for field_key in field_keys:
            key = cls._header_field + field_key
            l_key.append(db_common.TableKey(key, "real", tuple()))
        return l_key

    @staticmethod
    def _index_name(measure):
        return measure + "_index"

    def _init_table(self, measure, tag_keys, field_keys):
        table_names = self._db.get_table_names()
        if measure not in table_names:
            l_key = self._keys(tag_keys, field_keys)
            sql = self._db.create_table_sql(measure, l_key)
            self._db.execute(sql)

            index_name = self._index_name(measure)
            assert index_name not in table_names
            l_key = ["time"] + [k for k in tag_keys]
            sql = self._db.create_index_sql(measure, index_name, l_key)
            self._db.execute(sql)

    def list_measurements(self):
        return self._db.get_table_names()

    def list_series(self):
        #TODO
        pass

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
            args = {self._key_time: t.tz_convert(None).tz_localize(None)}
            for tag_key in tag_keys:
                args[tag_key] = d_tags[tag_key]
            for field_key in field_keys:
                args[field_key] = row[field_key]
            l_args.append(args)

        self._db.executemany(sql, l_args)
        return len(l_args)

    def commit(self):
        self._db.commit()

    def _field_names(self, measure):
        ret = []
        for name in self._db.get_column_names(measure):
            if name[:self._header_length] == self._header_field:
                ret.append(name)
        return ret

    def get(self, measure, d_tags, fields, dt_range,
            str_bin=None, func=None, fill=None, limit=None):
        if fields is None:
            fields = self._field_names(measure)

        l_key = self._key_time + fields
        l_cond = []
        args = {}
        l_cond.append(db_common.Condition("time", ">=", "dts", True))
        args["dts"] = self._db.strftime(dt_range[0].tz_convert(None).tz_localize(None))
        l_cond.append(db_common.Condition("time", "<", "dte", True))
        args["dte"] = self._db.strftime(dt_range[1].tz_convert(None).tz_localize(None))
        for tag_key, tag_val in d_tags.items():
            key = self._header_tag + tag_key
            l_cond.append(db_common.Condition(key, "=", tag_key))
            args[tag_key] = tag_val
        sql = self._db.select_sql(measure, l_key, l_cond)
        cursor = self._db.execute(sql, args)

        if func is None:
            # TODO
            pass
        elif func == "count":
            return sum(1 for _ in cursor)
        elif func == "sum":
            # TODO binning and get sum
            pass



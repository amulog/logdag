#!/usr/bin/env python
# coding: utf-8

# not used in current version


import logging
import math
import numpy as np

from . import dtutil
from . import period
from amulog import config
from amulog import db_common

_logger = logging.getLogger(__package__)


class TimeSeriesDB():

    def __init__(self, conf, edit = False, reset_db = False):
        self.areafn = conf.get("database", "area_filename")

        db_type = conf.get("database_ts", "database")
        if db_type == "sqlite3":
            dbpath = conf.get("database_ts", "sqlite3_filename")
            self.db = db_common.Sqlite3(dbpath)
        elif db_type == "mysql":
            host = conf.get("database_ts", "mysql_host")
            dbname = conf.get("database_ts", "mysql_dbname")
            user = conf.get("database_ts", "mysql_user")
            passwd = conf.get("database_ts", "mysql_passwd")
            self.db = db_common.Mysql(host, dbname, user, passwd)
        else:
            raise ValueError("invalid database type ({0})".format(
                    db_type))

        if self.db.db_exists():
            if not self._exists_tables():
                self._init_tables()
        else:
            self._init_tables()

    def reset_tables(self):
        if self._exists_tables():
            self._drop_tables(self)
            self._init_tables()

    def _exists_tables(self):
        tables = ["ts"]
        s_name = set(self.db.get_table_names())
        for table in tables:
            if not table in s_name:
                return False
        else:
            return True

    def _init_tables(self):
        table_name = "ts"
        l_key = [db_common.tablekey("tid", "integer",
                    ("primary_key", "auto_increment", "not_null")),
                 db_common.tablekey("dt", "datetime"),
                 db_common.tablekey("gid", "integer"),
                 db_common.tablekey("host", "text")]
        sql = self.db.create_table_sql(table_name, l_key)
        self.db.execute(sql)

        table_name = "area"
        l_key = [db_common.tablekey("defid", "integer",
                    ("primary_key", "auto_increment", "not_null")),
                 db_common.tablekey("host", "text"),
                 db_common.tablekey("area", "text")]
        sql = self.db.create_table_sql(table_name, l_key)
        self.db.execute(sql)

        table_name = "filter"
        l_key = [db_common.tablekey("qid", "integer",
                    ("primary_key", "auto_increment", "not_null")),
                 db_common.tablekey("dts", "datetime"),
                 db_common.tablekey("dte", "datetime"),
                 db_common.tablekey("gid", "integer"),
                 db_common.tablekey("host", "text"),
                 db_common.tablekey("stat", "text"),
                 db_common.tablekey("val", "integer"),]
        sql = self.db.create_table_sql(table_name, l_key)
        self.db.execute(sql)

        self._init_index()
        self._init_area()

    def _init_index(self):
        l_table_name = self.db.get_table_names()

        table_name = "ts"
        index_name = "ts_index"
        l_key = [db_common.tablekey("tid", "integer"),
                 db_common.tablekey("dt", "datetime"),
                 db_common.tablekey("gid", "integer"),
                 db_common.tablekey("host", "text", (100, ))]
        if not index_name in l_table_name:
            sql = self.db.create_index_sql(table_name, index_name, l_key)
            self.db.execute(sql)

        table_name = "area"
        index_name = "area_index"
        l_key = [db_common.tablekey("area", "text", (100, ))]
        if not index_name in l_table_name:
            sql = self.db.create_index_sql(table_name, index_name, l_key)
            self.db.execute(sql)

        table_name = "filter"
        index_name = "filter_index"
        l_key = [db_common.tablekey("dts", "datetime"),
                 db_common.tablekey("dte", "datetime"),
                 db_common.tablekey("gid", "integer"),
                 db_common.tablekey("host", "text", (100, )),
                 db_common.tablekey("stat", "text", (100, )),
                 db_common.tablekey("val", "integer"),]
        if not index_name in l_table_name:
            sql = self.db.create_index_sql(table_name, index_name, l_key)
            self.db.execute(sql)

    def _drop_tables(self):
        table_name = "ts"
        sql = self.db.drop_sql(table_name)
        self.db.execute(sql)
        
        table_name = "area"
        sql = self.db.drop_sql(table_name)
        self.db.execute(sql)

    def _init_area(self):
        # ported from amulog db...
        if self.areafn is None or self.areafn == "":
            return
        areadict = config.GroupDef(self.areafn)
        table_name = "area"
        l_ss = [db_common.setstate("host", "host"),
                db_common.setstate("area", "area")]
        sql = self.db.insert_sql(table_name, l_ss)
        for area, host in areadict.iter_def():
            args = {
                "host" : host,
                "area" : area
            }
            self.db.execute(sql, args)
        self.commit()

    def _remove_area(self):
        table_name = "area"
        sql = self.db.delete_sql(table_name)
        self.db.execute(sql)

    def commit(self):
        self.db.commit()

    def add_line(self, dt, gid, host):
        table_name = "ts"
        d_val = {
            "dt" : self.db.strftime(dt),
            "gid" : gid,
            "host" : host,
        }
        l_ss = [db_common.setstate(k, k) for k in d_val.keys()]
        sql = self.db.insert_sql(table_name, l_ss)
        self.db.execute(sql, d_val)

    def add_filterlog(self, dt_range, gid, host, stat, val):
        table_name = "filter"
        d_val = {
            "dts" : dt_range[0],
            "dte" : dt_range[1],
            "gid" : gid,
            "host" : host,
            "stat" : stat,
            "val" : val,
        }
        l_ss = [db_common.setstate(k, k) for k in d_val.keys()]
        sql = self.db.insert_sql(table_name, l_ss)
        self.db.execute(sql, d_val)

    def iter_ts(self, **kwargs):
        if "area" in kwargs:
            if kwargs["area"] is None or kwargs["area"] == "all":
                del kwargs["area"]
            elif area[:5] == "host_":
                assert not "host" in kwargs
                kwargs["host"] = area[5:]
                del kwargs["area"]
            else:
                kwargs["area"] = area

        if len(kwargs) == 0:
            raise ValueError("More than 1 argument should NOT be None")

        for row in self._select_ts(**kwargs):
            dt = self.db.datetime(row[0])
            yield dt

    def _select_ts(self, **kwargs):
        table_name = "ts"
        l_key = ["dt"]
        l_cond = []
        args = {}
        for c in kwargs.keys():
            if c == "dts":
                l_cond.append(db_common.cond("dt", ">=", c))
                args[c] = self.db.strftime(kwargs[c])
            elif c == "dte":
                l_cond.append(db_common.cond("dt", "<", c))
                args[c] = self.db.strftime(kwargs[c])
            elif c == "area":
                sql = self.db.select_sql("area", ["host"],
                        [db_common.cond(c, "=", c)])
                l_cond.append(db_common.cond("host", "in", sql, False))
            else:
                l_cond.append(db_common.cond(c, "=", c))
                args[c] = kwargs[c]
        sql = self.db.select_sql(table_name, l_key, l_cond)
        return self.db.execute(sql, args)
    
    def iter_filter(self, **kwargs):
        if "area" in kwargs:
            if kwargs["area"] is None or kwargs["area"] == "all":
                del kwargs["area"]
            elif area[:5] == "host_":
                assert not "host" in kwargs
                kwargs["host"] = area[5:]
                del kwargs["area"]
            else:
                kwargs["area"] = area

        if len(kwargs) == 0:
            raise ValueError("More than 1 argument should NOT be None")

        for row in self._select_filter(**kwargs):
            dts = self.db.datetime(row[0])
            dte = self.db.datetime(row[1])
            gid = int(row[2])
            host = row[3]
            stat = row[4]
            val = int(row[5]) if row[5] is not None else None
            yield FilterLog((dts, dte), gid, host, stat, val)

    def _select_filter(self, **kwargs):
        table_name = "filter"
        l_key = ["dts", "dte", "gid", "host", "stat", "val"]
        l_cond = []
        args = {}
        for c in kwargs.keys():
            if c == "area":
                sql = self.db.select_sql("area", ["host"],
                        [db_common.cond(c, "=", c)])
                l_cond.append(db_common.cond("host", "in", sql, False))
            else:
                l_cond.append(db_common.cond(c, "=", c))
                args[c] = kwargs[c]

        sql = self.db.select_sql(table_name, l_key, l_cond)
        return self.db.execute(sql, args)

    def count_lines(self):
        table_name = "ts"
        l_key = ["max(tid)"]
        sql = self.db.select_sql(table_name, l_key)
        cursor = self.db.execute(sql)
        return int(cursor.fetchone()[0])

    def dt_term(self):
        table_name = "ts"
        l_key = ["min(dt)", "max(dt)"]
        sql = self.db.select_sql(table_name, l_key)
        cursor = self.db.execute(sql)
        top_dtstr, end_dtstr = cursor.fetchone()
        if None in (top_dtstr, end_dtstr):
            raise ValueError("No data found in DB")
        return self.db.datetime(top_dtstr), self.db.datetime(end_dtstr)

    def whole_gid_host(self, **kwargs):
        table_name = "ts"
        l_key = ["gid", "host"]
        l_cond = []
        args = {}
        if "dts" in kwargs:
            l_cond.append(db_common.cond("dt", ">=", "dts"))
            args["dts"] = self.db.strftime(kwargs["dts"])
        if "dte" in kwargs:
            l_cond.append(db_common.cond("dt", "<", "dte"))
            args["dte"] = self.db.strftime(kwargs["dte"])

        area = kwargs["area"] if "area" in kwargs else None
        if area is None or area == "all":
            pass
        elif area[:5] == "host_":
            l_cond.append(db_common.cond("host", "=", "host"))
            args["host"] = area[5:]
        else:
            temp_sql = self.db.select_sql(
                "area", ["host"], [db_common.cond("area", "=", "area")])
            l_cond.append(db_common.cond("host", "in", temp_sql, False))
            args["area"] = area

        sql = self.db.select_sql(table_name, l_key, l_cond, opt = ["distinct"])
        cursor = self.db.execute(sql, args)
        return [(row[0], row[1]) for row in cursor]

    @staticmethod
    def str_event(dt_range, gid, host):
        return "[{0}, gid={1}, host = {2}]".format(dt_range[0].date(),
                                                   gid, host)


class FilterLog():

    def __init__(self, dt_range, gid, host, stat, val):
        # stats: none, const, period
        # const: val = counts/day, period : val = interval(seconds)
        self.dt_range = dt_range
        self.gid = gid
        self.host = host
        self.stat = stat
        self.val = val

    def __str__(self):
        ev_name = TimeSeriesDB.str_event(self.dt_range, self.gid, self.host)
        if self.stat == "none":
            stat_name = self.stat
        else:
            stat_name = "{0}[{1}]".format(self.stat, self.val)
        return "{0}: {1}".format(ev_name, stat_name)


def log2ts(conf, dt_range):
    _logger.info("make-tsdb job start ({0[0]} - {0[1]})".format(dt_range))
    
    gid_name = conf.get("dag", "event_gid")
    usefilter = conf.getboolean("database_ts", "usefilter")
    top_dt, end_dt = dt_range
    
    from amulog import log_db
    ld = log_db.LogData(conf)
    if gid_name == "ltid":
        iterobj = ld.whole_host_lt(top_dt, end_dt, "all")
    elif gid_name == "ltgid":
        iterobj = ld.whole_host_ltg(top_dt, end_dt, "all")
    else:
        raise NotImplementedError

    for host, gid in iterobj:
        # load time-series from db
        d = {gid_name: gid,
             "host": host,
             "top_dt": top_dt,
             "end_dt": end_dt}
        iterobj = ld.iter_lines(**d)
        l_dt = sorted([line.dt for line in iterobj])
        _logger.debug("gid {0}, host {1}: {2} counts".format(gid, host,
                                                             len(l_dt)))
        assert len(l_dt) > 0

        # apply preprocessing(filter)
        evdef = (gid, host)
        stat, new_l_dt, val = apply_filter(conf, ld, l_dt, dt_range, evdef)

        # update database
        td = TimeSeriesDB(conf, edit = True)
        if new_l_dt is not None and len(new_l_dt) > 0:
            for dt in new_l_dt:
                td.add_line(dt, gid, host)
        td.add_filterlog(dt_range, gid, host, stat, val)
        td.commit()

        fl = FilterLog(dt_range, gid, host, stat, val)
        _logger.debug(str(fl))
    
    _logger.info("make-tsdb job done".format(dt_range))


def log2ts_pal(conf, dt_range, pal = 1):
    from amulog import common
    timer = common.Timer(
        "make-tsdb subtask ({0[0]} - {0[1]})".format(dt_range),
        output = _logger)
    timer.start()
    
    gid_name = conf.get("dag", "event_gid")
    usefilter = conf.getboolean("database_ts", "usefilter")

    from amulog import log_db
    ld = log_db.LogData(conf)
    if gid_name == "ltid":
        iterobj = ld.whole_host_lt(dt_range[0], dt_range[1], "all")
    elif gid_name == "ltgid":
        iterobj = ld.whole_host_ltg(dt_range[0], dt_range[1], "all")
    else:
        raise NotImplementedError

    import multiprocessing
    td = TimeSeriesDB(conf, edit = True)
    l_args = [(conf, dt_range, gid, host) for host, gid in iterobj]
    with multiprocessing.Pool(processes = pal) as pool:
        for ret in pool.imap_unordered(log2ts_elem, l_args):
            gid, host, stat, new_l_dt, val = ret
            if new_l_dt is not None and len(new_l_dt) > 0:
                for dt in new_l_dt:
                    td.add_line(dt, gid, host)
            td.add_filterlog(dt_range, gid, host, stat, val)
        pool.close()
        pool.join()
    td.commit() 
    timer.stop()
    return

    #l_args = [[conf, dt_range, gid, host] for host, gid in iterobj]
    #l_queue = [multiprocessing.Queue() for args in l_args]
    #l_process = [multiprocessing.Process(name = processname(*args),
    #                                     target = log2ts_elem,
    #                                     args = [queue] + args)
    #             for args, queue in zip(l_args, l_queue)]
    #l_pq = list(zip(l_process, l_queue))

    #td = TimeSeriesDB(conf, edit = True)
    #for ret in common.mprocess_queueing(l_pq, pal):
    #    gid, host, stat, new_l_dt, val = ret
    #    if new_l_dt is not None and len(new_l_dt) > 0:
    #        for dt in new_l_dt:
    #            td.add_line(dt, gid, host)
    #    td.add_filterlog(dt_range, gid, host, stat, val)
    #td.commit() 
    #del td
    #timer.stop()
    #return


def log2ts_elem(args):
    conf, dt_range, gid, host = args
    name = "{0}_{1}_{2}".format(dtutil.shortstr(dt_range[0]), gid, host)
    _logger.info("make-tsdb job start ({0})".format(name))
    from amulog import log_db
    ld = log_db.LogData(conf)
    gid_name = conf.get("dag", "event_gid")
    d = {gid_name: gid,
         "host": host,
         "top_dt": dt_range[0],
         "end_dt": dt_range[1]}
    iterobj = ld.iter_lines(**d)
    l_dt = sorted([line.dt for line in iterobj])
    del iterobj
    _logger.debug("gid {0}, host {1}: {2} counts".format(gid, host,
                                                         len(l_dt)))
    assert len(l_dt) > 0

    evdef = (gid, host)
    stat, new_l_dt, val = apply_filter(conf, ld, l_dt, dt_range, evdef)

    fl = FilterLog(dt_range, gid, host, stat, val)
    _logger.debug(str(fl))
    _logger.info("make-tsdb job done ({0})".format(name))

    return (gid, host, stat, new_l_dt, val)


def apply_filter(conf, ld, l_dt, dt_range, evdef):
    """Apply filter fucntions for time-series based on given configuration.
    
    Args:
        conf (configparser.ConfigParser)
        ld (amulog.log_db.LogData)
        l_dt (List[datetime.datetime])
        dt_range (datetime.datetime, datetime.datetime)
        evdef (int, str): tuple of gid and host.

    Returns:
        stat (str): 1 of ["none", "const", "period"]
        l_dt (List[datetime.datetime]): time series after filtering
        val (int): optional value that explain filtering status.
                   periodicity interval if stat is "period"
                   time-series counts per a day if stat is "const"
    """
    usefilter = conf.getboolean("database_ts", "usefilter")
    if usefilter:
        act = conf.get("filter", "action")
        if act in ("remove", "replace"):
            method = act
            pflag, remain, interval = filter_periodic(conf, ld, l_dt, dt_range,
                                                      evdef, method = method)
            if pflag:
                return ("period", remain, int(interval.total_seconds()))
            else:
                return ("none", l_dt, None)
        elif act == "linear":
            lflag = filter_linear(conf, l_dt, dt_range)
            if lflag:
                return ("const", None, len(l_dt))
            else:
                return ("none", l_dt, None)
        elif act in ("remove+linear", "replace+linear"):
            method = act.partition("+")[0]
            # periodic
            pflag, remain, interval = filter_periodic(conf, ld, l_dt, dt_range,
                                                      evdef, method = method)
            if pflag:
                return ("period", remain, int(interval.total_seconds()))
            # linear
            lflag = filter_linear(conf, l_dt, dt_range)
            if lflag:
                return ("const", None, len(l_dt))
            else:
                return ("none", l_dt, None)
        elif act in ("linear+remove", "linear+replace"):
            method = act.partition("+")[-1]
            # linear
            lflag = filter_linear(conf, l_dt, dt_range)
            if lflag:
                return ("const", None, len(l_dt))
            # periodic
            pflag, remain, interval = filter_periodic(conf, ld, l_dt,
                                                      dt_range, evdef,
                                                      method = method)
            if pflag:
                return ("period", remain, int(interval.total_seconds()))
            else:
                return ("none", l_dt, None)
        else:
            raise NotImplementedError
    else:
        return ("none", l_dt, None)


def filter_linear(conf, l_dt, dt_range):
    """Return True if a_cnt appear linearly."""
    binsize = config.getdur(conf, "filter", "linear_binsize")
    threshold = conf.getfloat("filter", "linear_threshold")
    th_count = conf.getint("filter", "linear_count")

    if len(l_dt) < th_count:
        return False

    # generate time-series cumulative sum
    length = (dt_range[1] - dt_range[0]).total_seconds()
    bin_length = binsize.total_seconds()
    bins = math.ceil(1.0 * length / bin_length)
    a_stat = np.array([0] * int(bins))
    for dt in l_dt:
        cnt = int((dt - dt_range[0]).total_seconds() / bin_length)
        assert cnt < len(a_stat)
        a_stat[cnt:] += 1

    a_linear = np.linspace(0, len(l_dt), bins, endpoint = False)
    val = sum((a_stat - a_linear) ** 2) / (bins * len(l_dt))
    return val < threshold


def filter_periodic(conf, ld, l_dt, dt_range, evdef, method):
    """Return True and the interval if a_cnt is periodic."""

    ret_false = False, None, None
    gid_name = conf.get("dag", "event_gid")
    p_cnt = conf.getint("filter", "pre_count")
    p_term = config.getdur(conf, "filter", "pre_term")

    # preliminary tests
    if len(l_dt) < p_cnt:
        _logger.debug("time-series count too small, skip")
        return ret_false
    elif max(l_dt) - min(l_dt) < p_term:
        _logger.debug("time-series range too small, skip")
        return ret_false

    # periodicity tests
    for dt_cond in config.gettuple(conf, "filter", "sample_rule"):
        dt_length, binsize = [config.str2dur(s) for s in dt_cond.split("_")]
        if (dt_range[1] - dt_range[0]) == dt_length:
            temp_l_dt = l_dt
        else:
            temp_l_dt = reload_ts(ld, evdef, dt_length, dt_range, gid_name)
        a_cnt = dtutil.discretize_sequential(temp_l_dt, dt_range,
                                             binsize, binarize = False)

        remain_dt = None
        if method == "remove":
            flag, interval = period.fourier_remove(conf, a_cnt, binsize)
        elif method == "replace":
            flag, remain_array, interval = period.fourier_replace(conf, a_cnt,
                                                                  binsize)
            if remain_array is not None:
                remain_dt = revert_event(remain_array, dt_range, binsize)
        elif method == "corr":
            flag, interval = period.periodic_corr(conf, a_cnt, binsize) 
        else:
            raise NotImplementedError
        if flag:
            return flag, remain_dt, interval
    return ret_false


def reload_ts(ld, evdef, dt_length, dt_range, gid_name):
    new_top_dt = dt_range[1] - dt_length
    d = {gid_name: evdef[0],
         "host": evdef[1],
         "top_dt": new_top_dt,
         "end_dt": dt_range[1]}
    iterobj = ld.iter_lines(**d)
    return sorted([line.dt for line in iterobj])


def revert_event(a_cnt, dt_range, binsize):
    top_dt, end_dt = dt_range
    assert top_dt + len(a_cnt) * binsize == end_dt
    return [top_dt + i * binsize for i, val in enumerate(a_cnt) if val > 0]


def reload_area(conf):
    td = TimeSeriesDB(conf, edit = True)
    td._remove_area()
    td._init_area()


def ts_filtered(conf, **kwargs):
    assert "dts" in kwargs
    assert "dte" in kwargs
    assert "gid" in kwargs
    assert "host" in kwargs
    
    from amulog import log_db
    ld = log_db.LogData(conf)
    gid_name = conf.get("dag", "event_gid")
    d = {"top_dt": kwargs["dts"],
         "end_dt": kwargs["dte"],
         gid_name: kwargs["gid"],
         "host": kwargs["host"]}
    l_dt = [line.dt for line in ld.iter_lines(**d)]

    td = TimeSeriesDB(conf)
    l_ts = [dt for dt in td.iter_ts(**kwargs)]
    l_filtered = [dt for dt in l_dt if not dt in l_ts]

    return l_filtered, l_ts


# visualize functions


def show_event(conf, **kwargs):
    td = TimeSeriesDB(conf)
    try:
        dt_range = (kwargs["dts"], kwargs["dte"])
    except KeyError:
        dt_range = td.dt_term()

    l_buf = []
    for gid, host in sorted(td.whole_gid_host(**kwargs), key = lambda x: x[0]):
        event_str = td.str_event(dt_range, gid, host)
        #num = sum(1 for i in td.iter_ts(dts = dt_range[0], dte = dt_range[1],
        #                                gid = gid, host = host))
        #l_buf.append("{0}: {1}".format(event_str, num))
        l_buf.append(event_str)
    return "\n".join(l_buf)


def show_ts(conf, **kwargs):
    assert "dts" in kwargs
    assert "dte" in kwargs
    assert "gid" in kwargs
    assert "host" in kwargs
    td = TimeSeriesDB(conf)

    l_buf = []
    for dt in td.iter_ts(**kwargs):
        l_buf.append(str(dt))
    return "\n".join(l_buf)


def show_ts_compare(conf, **kwargs):
    l_buf = []
    l_fts, l_ts = ts_filtered(conf, **kwargs)
    l_buf.append("# filtered ({0}) #".format(len(l_fts)))
    for dt in l_fts:
        l_buf.append(str(dt))
    l_buf.append("")
    l_buf.append("# remaining ({0}) #".format(len(l_ts)))
    for dt in l_ts:
        l_buf.append(str(dt))
    return "\n".join(l_buf)


def show_filterlog(conf, **kwargs):
    td = TimeSeriesDB(conf)
    buf = []
    for fl in td.iter_filter(**kwargs):
        buf.append(str(fl))
    return "\n".join(buf)



#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import rrdtool

from amulog import config


class RRDLoader():

    def __init__(self, conf):
        self._rows = conf.getint("database_rrd", "rows")
        self._cf = conf["database_rrd"]["cf"]
        self._correct_roundup = conf.getboolean("database_rrd",
                                                "correct_roundup")
        self._binsize = int(
            config.getdur(conf, "database_rrd", "binsize").total_seconds())

    def load(self, fp, dt_range):
        ut_range = [dt.timestamp() for dt in dt_range]
        return fetch(fp, ut_range, self._rows, self._cf,
                     self._binsize, self._correct_roundup)


def fetch(fp, ut_range, rows = 1, cf = "MAX", binsize = 60,
          correct_roundup = False):
    time_start = ut_range[0]
    time_end = ut_range[1] - binsize

    # correction for rounded up timestamp
    # (adjust to syslogs that is usually rounded down)
    if correct_roundup:
        time_start = time_start - binsize
        time_end = time_end - binsize

    fetch_args = ["-s", str(int(time_start)),
                  "-e", str(int(time_end)),
                  "-r", str(rows),
                  fp,  cf]
    try:
        robj = rrdtool.fetch(*fetch_args)
    except rrdtool.OperationalError:
        raise

    # correction for rounded up timestamp
    # (adjust to syslogs that is usually rounded down)
    if correct_roundup:
        time = list(range(int(robj[0][0]) + binsize,
                          int(robj[0][1]) + binsize, int(robj[0][2])))
    else:
        time = list(range(int(robj[0][0]), int(robj[0][1], int(robj[0][2]))))
    keys = list(robj[1])
    data = np.array(robj[2])
    if len(data) == 0:
        raise ValueError

    df = pd.DataFrame(data, columns = keys, dtype = float)
    #df.index = time
    df.set_index(pd.to_datetime(time, unit = 's'), inplace = True)
    return df


#def rrd2influx(rrd_fp, dbname, influx_kwargs, measurements, d_tags,
#               ut_range, rows = 1, cf = "MAX", binsize = 60,
#               correct_roundup = False):
#
#    import influx
#
#    ut2dt = lambda x: datetime.datetime.fromtimestamp(x)
#    time_start = ut_range[0] - 2 * binsize
#    time_end = ut_range[1] - 2 * binsize
#
#    # correction for rounded up timestamp
#    # (adjust to syslogs that is usually rounded down)
#    if correct_roundup:
#        time_start = time_start - binsize
#        time_end = time_end - binsize
#
#    fetch_args = ["-s", str(int(time_start)),
#                  "-e", str(int(time_end)),
#                  "-r", str(rows),
#                  rrd_fp,  cf]
#    try:
#        robj = rrdtool.fetch(*fetch_args)
#    except rrdtool.OperationalError:
#        raise
#
#    times = [int(ut) for ut in range(*robj[0])]
#    keys = robj[1]
#    data = robj[2]
#
#    inf = influx.InfluxDB(dbname, **influx_kwargs)
#    inf.add(measurements, d_tags, data, times, keys)


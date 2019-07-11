#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict

from amulog import config
from amulog import log_db


class AmulogLoader(object):

    def __init__(self, dt_range, conf_fn, gid_name):
        conf = config.open_config(conf_fn)
        self._ld = log_db.LogData(conf)
        self._gid_name = gid_name
        self.dt_range = dt_range

    def iter_evdef(self, dt_range = None, area = None):
        if dt_range is None:
            dt_range = self.dt_range
        if self._gid_name == "ltid":
            return self._ld.whole_host_lt(top_dt = dt_range[0],
                                          end_dt = dt_range[1], area = area)
        elif self._gid_name == "ltgid":
            return self._ld.whole_host_ltg(top_dt = dt_range[0],
                                           end_dt = dt_range[1], area = area)

    def iter_dt(self, evdef, dt_range = None):
        if dt_range is None:
            dt_range = self.dt_range
        host, gid = evdef
        d = {"top_dt": dt_range[0],
             "end_dt": dt_range[1],
             self._gid_name: gid,
             "host": host}
        for lm in self._ld.iter_lines(**d):
             yield lm.dt

    @staticmethod
    def timestamp2dict(iterable):
        d_dt = defaultdict(int)
        for dt in iterable:
            d_dt[dt] += 1
        return d_dt

    @staticmethod
    def timestamp2df(iterable):
        d_dt = defaultdict(int)
        for dt in iterable:
            d_dt[dt] += 1
        if len(d_dt) == 0:
            return None

        import pandas as pd
        df = pd.DataFrame(list(d_dt.items()),
                          columns = ["timestamp", "val"])
        df.set_index("timestamp", inplace = True)
        return df

    def load(self, evdef, dt_range = None):
        return sorted(self.iter_dt(evdef, dt_range))
        #return self.timestamp2df(iterable)



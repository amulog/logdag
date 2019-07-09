#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict

from amulog import log_db


class AmulogLoader(object):

    def __init__(self, conf, gid_name, dt_range):
        self._ld = log_db.LogData(conf)
        self._gid_name = gid_name
        self.dt_range = dt_range

    def iter_evdef(self, area = None):
        if self._gid_name == "ltid":
            return self._ld.whole_host_lt(top_dt = dt_range[0],
                                          end_dt = dt_range[1], area = area)
        elif self._gid_name == "ltgid":
            return self._ld.whole_host_ltg(top_dt = dt_range[0],
                                           end_dt = dt_range[1], area = area)

    def iter_dt(self, evdef):
        gid, host = evdef
        d = {"top_dt": self.dt_range[0],
             "end_dt": self.dt_range[1],
             self._gid_name: gid,
             "host": host}
        d_dt = defaultdict(int)
        return self._ld.iter_lines(**d)

    @staticmethod
    def timestamp2dict(iterable):
        d_dt = defaultdict(int)
        for line in iterable:
            d_dt[line.dt] += 1
        return d_dt

    @staticmethod
    def timestamp2df(iterable):
        d_dt = defaultdict(int)
        for line in iterable:
            d_dt[line.dt] += 1
        if len(d_dt) == 0:
            return None

        df = pd.DataFrame(list(d_dt.items()),
                          columns = ["timestamp", "val"])
        df.set_index("timestamp", inplace = True)
        return df

    def load(self, evdef):
        iterable = self.iter_dt(evdef)
        return list(iterable)
        #return self.timestamp2df(iterable)



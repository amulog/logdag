#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
from dateutil.tz import tzlocal

from amulog import config
from amulog import log_db


class AmulogLoader(object):

    def __init__(self, dt_range, conf_fn, gid_name):
        self.conf = config.open_config(conf_fn)
        self._ld = log_db.LogData(self.conf)
        self._gid_name = gid_name
        self.dt_range = dt_range
        self._ll = None

    def iter_event(self, dt_range=None, area=None):
        if dt_range is None:
            dt_range = self.dt_range
        if self._gid_name == "ltid":
            return self._ld.whole_host_lt(top_dt=dt_range[0],
                                          end_dt=dt_range[1], area=area)
        elif self._gid_name == "ltgid":
            return self._ld.whole_host_ltg(top_dt=dt_range[0],
                                           end_dt=dt_range[1], area=area)

    def _iter_lines(self, ev, dt_range=None):
        if dt_range is None:
            dt_range = self.dt_range
        host, gid = ev
        d = {"top_dt": dt_range[0],
             "end_dt": dt_range[1],
             self._gid_name: gid,
             "host": host}
        return self._ld.iter_lines(**d)

    def iter_dt(self, ev, dt_range=None):
        for lm in self._iter_lines(ev, dt_range):
            dt = lm.dt.replace(tzinfo=tzlocal())
            yield dt

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
                          columns=["timestamp", "val"])
        df.set_index("timestamp", inplace=True)
        return df

    def load(self, ev, dt_range=None):
        return sorted(self.iter_dt(ev, dt_range))

    def load_org(self, ev, dt_range):
        for lm in self._iter_lines(ev, dt_range):
            dt = lm.dt.replace(tzinfo=tzlocal())
            yield (dt, lm.host, lm.restore_message())

    def gid_instruction(self, gid):
        if self._gid_name == "ltid":
            return str(self._ld.lt(gid))
        elif self._gid_name == "ltgid":
            l_lt = self._ld.ltg_members(gid)
            if len(l_lt) == 1:
                return str(l_lt[0])
            else:
                "{0} tpls like: {1}".format(len(l_lt), l_lt[0])

    def label(self, gid):
        if self._ll is None:
            from amulog import lt_label
            self._ll = lt_label.init_ltlabel(self.conf)
        if gid is None:
            return self.conf["visual"]["ltlabel_default_group"]
        return self._ll.get_ltg_group(gid, self._ld.ltg_members(gid))


def init_amulogloader(conf, dt_range):
    args = [dt_range,
            conf["database_amulog"]["source_conf"],
            conf["database_amulog"]["event_gid"]]
    return AmulogLoader(*args)


#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
from dateutil.tz import tzlocal

from amulog import log_db


class AmulogLoader(object):

    def __init__(self, conf, dt_range=None, gid_name="ltid", use_mapping=False, ld=None):
        self.conf = conf
        if ld is not None:
            self._ld = ld
        else:
            self._ld = log_db.LogData(self.conf)
        self._gid_name = gid_name
        self.dt_range = dt_range

        self._mapper = None
        if use_mapping:
            # use if tsdb is anonymized but amulog db is original
            from amulog import anonymize
            self._mapper = anonymize.AnonymizeMapper(self.conf)
            self._mapper.load()

    @classmethod
    def from_ld(cls, ld):
        return AmulogLoader(ld.conf, ld=ld)

    def restore_host(self, host):
        if self._mapper:
            return self._mapper.restore_host(host)
        else:
            return host

    def _restore_lt(self, ltobj):
        if self._mapper:
            return self._mapper.restore_lt(ltobj)
        else:
            return ltobj

    def iter_event(self, dt_range=None):
        if dt_range is None:
            dt_range = self.dt_range
        if self._gid_name == "ltid":
            return self._ld.whole_host_lt(dts=dt_range[0],
                                          dte=dt_range[1])
        elif self._gid_name == "ltgid":
            return self._ld.whole_host_ltg(dts=dt_range[0],
                                           dte=dt_range[1])

    def _get_tags(self, gid):
        kwargs = {self._gid_name: gid}
        return self._ld.get_tags(**kwargs)

    def _iter_lines(self, ev, dt_range=None):
        if dt_range is None:
            dt_range = self.dt_range
        host, gid = ev
        d = {"dts": dt_range[0],
             "dte": dt_range[1],
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
        # restored_ev = (self.restore_host(ev[0]), ev[1])
        for lm in self._iter_lines(ev, dt_range):
            lm.dt = lm.dt.replace(tzinfo=tzlocal())
            yield lm

    def gid_instruction(self, gid):
        if self._gid_name == "ltid":
            ltobj = self._ld.lt(gid)
            return str(ltobj)
        elif self._gid_name == "ltgid":
            l_lt = self._ld.ltg_members(gid)
            repr_lt = l_lt[0]
            if len(l_lt) == 1:
                return str(repr_lt)
            else:
                return "{0} tpls: {1}".format(len(l_lt), repr_lt)

    def group(self, gid):
        tags = [tag for tag in self._get_tags(gid)]
        if len(tags) == 0:
            return None
        else:
            return "|".join(sorted(tags))


def init_amulogloader(conf, dt_range):
    args = [dt_range,
            conf["database_amulog"]["source_conf"],
            conf["database_amulog"]["event_gid"],
            conf.getboolean("database_amulog",
                            "use_anonymize_mapping")]
    return AmulogLoader(*args)

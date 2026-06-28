#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
from dateutil.tz import tzlocal

from amulog import log_db


class AmulogLoader(object):

    def __init__(self, conf, dt_range=None, gid_name="ltid", use_mapping=False,
                 ld=None, host_tier=None):
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

        # Optional host stratification (amulog host_group). When host_tier is
        # empty, self._hg stays None and every method below takes the original
        # (pre-stratification) code path -- i.e. behaviour is unchanged.
        self._host_tier = host_tier if host_tier else None
        self._hg = None
        self._host2hgid = None      # memoized host -> hgid (resolve cache)
        self._hgid_hosts = None     # hgid -> set(original hosts), built once
        if self._host_tier is not None:
            from amulog import host_group
            self._hg = host_group.init_hostgroup(self.conf)
            if self._host_tier not in self._hg.tiers():
                raise ValueError(
                    "host_tier {0!r} is not a defined amulog host_group tier "
                    "(available: {1}). Set [manager] host_group_filename in "
                    "the amulog config.".format(self._host_tier,
                                                self._hg.tiers()))

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

    def _whole_host_pairs(self, dt_range):
        if self._gid_name == "ltid":
            return self._ld.whole_host_lt(dts=dt_range[0], dte=dt_range[1])
        elif self._gid_name == "ltgid":
            return self._ld.whole_host_ltg(dts=dt_range[0], dte=dt_range[1])

    def _ensure_hg_map(self, dt_range):
        """Build hgid <-> host maps once (resolve depends only on host text,
        so the maps are dt_range-independent and computed over distinct hosts)."""
        if self._hgid_hosts is not None:
            return
        rng = self.dt_range if self.dt_range is not None else dt_range
        host2hgid = {}
        hgid_hosts = defaultdict(set)
        for host, _gid in self._whole_host_pairs(rng):
            if host in host2hgid:
                continue
            hgid = self._hg.resolve(host, self._host_tier)
            host2hgid[host] = hgid
            if hgid is not None:
                hgid_hosts[hgid].add(host)
        self._host2hgid = host2hgid
        self._hgid_hosts = hgid_hosts

    def _resolve_host(self, host):
        # memoized; covers hosts not seen during the initial map build
        if host not in self._host2hgid:
            self._host2hgid[host] = self._hg.resolve(host, self._host_tier)
        return self._host2hgid[host]

    def iter_event(self, dt_range=None):
        if dt_range is None:
            dt_range = self.dt_range
        pairs = self._whole_host_pairs(dt_range)
        if self._hg is None:
            yield from pairs
            return
        # stratified: map each host to its hgid and deduplicate (hgid, gid)
        self._ensure_hg_map(dt_range)
        seen = set()
        for host, gid in pairs:
            hgid = self._resolve_host(host)
            if hgid is None:
                continue
            ev = (hgid, gid)
            if ev not in seen:
                seen.add(ev)
                yield ev

    def _get_tags(self, gid):
        kwargs = {self._gid_name: gid}
        return self._ld.get_tags(**kwargs)

    def _iter_lines(self, ev, dt_range=None):
        if dt_range is None:
            dt_range = self.dt_range
        host, gid = ev
        if self._hg is None:
            d = {"dts": dt_range[0],
                 "dte": dt_range[1],
                 self._gid_name: gid,
                 "host": host}
            yield from self._ld.iter_lines(**d)
            return
        # stratified: ev[0] is an hgid; union the lines of its original hosts
        # (amulog's log table keeps the original host, so no DB rebuild needed)
        self._ensure_hg_map(dt_range)
        for original_host in sorted(self._hgid_hosts.get(host, ())):
            d = {"dts": dt_range[0],
                 "dte": dt_range[1],
                 self._gid_name: gid,
                 "host": original_host}
            yield from self._ld.iter_lines(**d)

    def iter_dt(self, ev, dt_range=None):
        for lm in self._iter_lines(ev, dt_range):
            # amulog returns tz-aware datetimes (in its configured timezone,
            # default local); honor that instead of forcing local. Fall back to
            # local only for a naive dt (older amulog).
            dt = lm.dt if lm.dt.tzinfo is not None \
                else lm.dt.replace(tzinfo=tzlocal())
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
            if lm.dt.tzinfo is None:
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
    from amulog import config
    # source_conf is a path to the amulog config; open it (mirrors how
    # evgen_log.LogEventLoaderBase builds the loader). The previous code passed
    # the path string and dt_range in the wrong positions.
    amulog_conf = config.open_config(conf["database_amulog"]["source_conf"])
    args = [amulog_conf,
            dt_range,
            conf["database_amulog"]["event_gid"],
            conf.getboolean("database_amulog", "use_anonymize_mapping")]
    host_tier = conf.get("database_amulog", "host_tier", fallback="")
    return AmulogLoader(*args, host_tier=host_tier)

#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import logging

from amulog import config
from logdag import log2event
from . import evgen_common
from . import filter_log
from . import convert

_logger = logging.getLogger(__package__)

FEATURE_MEASUREMENT = "log_feature"


class LogEventDefinition(log2event.EventDefinition):

    _l_attr_log = ["gid", ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for attr in self._l_attr_log:
            setattr(self, attr, kwargs[attr])

    def __str__(self):
        # bug of string None: TODO to find the reason
        # if self.group is None or self.group == "None":
        if self.group is None:
            return "{0}:{1}".format(self.host, str(self.gid))
        else:
            return "{0}:{1}:{2}".format(self.host, str(self.gid),
                                        self.group)

    @property
    def _attribute_keys(self):
        return self._l_attr + self._l_attr_log

    @property
    def identifier(self):
        return "{0}:{1}".format(self.host, str(self.gid))

    def key(self):
        return str(self.gid)

    def tags(self):
        return {"host": self.host,
                "key": self.key()}

    def series(self):
        return FEATURE_MEASUREMENT, self.tags()

    def event(self) -> str:
        # event attributes without host
        return str(self.gid)


class LogEventLoaderBase(object):

    def __init__(self, conf):
        src = conf["general"]["log_source"]
        if src == "amulog":
            from . import src_amulog
            conf_path = conf["database_amulog"]["source_conf"]
            amulog_conf = config.open_config(conf_path)
            args = [
                amulog_conf,
                config.getterm(conf, "general", "evdb_whole_term"),
                conf["database_amulog"]["event_gid"],
                conf.getboolean("database_amulog",
                                "use_anonymize_mapping")
            ]
            self.source = src_amulog.AmulogLoader(*args)
        else:
            raise NotImplementedError

        self._lf = self._init_filters(conf)

    @staticmethod
    def _evdef(host, gid, group):
        d = {"source": log2event.SRCCLS_LOG,
             "host": host,
             "group": group,
             "gid": gid}
        return LogEventDefinition(**d)

    def iter_evdef(self, dt_range=None):
        for host, gid in self.source.iter_event(dt_range=dt_range):
            group = self.source.group(gid)
            d = {"source": log2event.SRCCLS_LOG,
                 "host": host,
                 "group": group,
                 "gid": gid}
            yield LogEventDefinition(**d)

    def _init_filters(self, conf):
        self._filter_rules = config.getlist(conf, "filter", "rules")
        for method in self._filter_rules:
            assert method in filter_log.FUNCTIONS

        if len(self._filter_rules) > 0:
            return filter_log.init_logfilter(conf, mode="evdb", loader=self.source)
        else:
            return None

    def _apply_filters(self, l_dt, dt_range, ev):
        if self._lf is None:
            return l_dt
        else:
            return self._lf.apply_filters(l_dt, dt_range, ev)


class LogEventLoader(evgen_common.EventLoader, LogEventLoaderBase):
    fields = ["val", ]

    def __init__(self, conf, dry=False):
        evgen_common.EventLoader.__init__(self, conf, dry=dry)
        LogEventLoaderBase.__init__(self, conf)

        self.evdb = self._init_evdb(conf, "log_dbname")

        self._feature_unit_diff = config.getdur(conf,
                                                "general", "evdb_unit_diff")
        self._given_amulog_database = conf["database_amulog"]["given_amulog_database"]

    def read_all(self, dump_org=False):
        return self.read(dt_range=None, dump_org=dump_org)

    def read(self, dt_range=None, dump_org=False):
        if dt_range is not None:
            self.source.dt_range = dt_range

        for ev in self.source.iter_event():
            host, gid = ev
            l_dt = self.source.load(ev)
            if len(l_dt) == 0:
                _logger.info("log gid={0} host={1} is empty".format(
                    gid, host))
                continue
            if dump_org:
                self.dump("log_org", host, gid, l_dt)
                _logger.info("added org {0} size {1}".format(
                    (host, gid), len(l_dt)))
                pass
            feature_dt = self._apply_filters(l_dt, dt_range, ev)
            if feature_dt is not None:
                self.dump(FEATURE_MEASUREMENT, host, gid, feature_dt)
                _logger.info("added feature {0} size {1}".format(
                    (host, gid), len(feature_dt)))

    def dump(self, measure, host, gid, l_dt):
        if self.dry:
            return
        d_tags = {"host": host, "key": gid}
        data = {}
        for dt, cnt in self.source.timestamp2dict(l_dt).items():
            t = pd.to_datetime(dt)
            data[t] = [cnt, ]
        self.evdb.add(measure, d_tags, data, self.fields)
        self.evdb.commit()

    def all_feature(self):
        return [FEATURE_MEASUREMENT, ]

    def load_org(self, ev, dt_range):
        """Yields: LogMessage"""
        return self.source.load_org(ev, dt_range)

    def restore_host(self, host):
        return self.source.restore_host(host)

    def instruction(self, evdef):
        if isinstance(evdef, log2event.MultipleEventDefinition):
            l_buf = []
            for tmp_evdef in evdef.members:
                l_buf.append(self.instruction(tmp_evdef))
            return " | ".join(l_buf)
        else:
            instruction = self.source.gid_instruction(evdef.gid)
            return "({0}) {1}".format(evdef.host, instruction)

    def details(self, evdef, dt_range, evdef_org=None, show_org=True):
        if evdef_org:
            if isinstance(evdef, log2event.MultipleEventDefinition):
                results = []
                for tmp_evdef, tmp_evdef_org in zip(evdef.members, evdef_org.members):
                    results += self.details(tmp_evdef, dt_range,
                                            evdef_org=tmp_evdef_org, show_org=show_org)
                return sorted(results, key=lambda x: x[0])
        else:
            if isinstance(evdef, log2event.MultipleEventDefinition):
                results = []
                for tmp_evdef in evdef.members:
                    results += self.details(tmp_evdef, dt_range, show_org=show_org)
                return sorted(results, key=lambda x: x[0])
            evdef_org = evdef

        measure = "log_feature"
        if show_org:
            # It extracts timestamps on valid bins after preprocessing
            # Note: it is impossible to distinguish counts in one bin
            # if it includes periodic and aperiodic components
            s_dt = {dt for dt, values
                    in self.load_items(measure, evdef.tags(), dt_range)}
            if len(s_dt) == 0:
                import pdb; pdb.set_trace()
                msg = ("No time-series for {0}, ".format(evdef) +
                       "inconsistent with tsdb")
                raise ValueError(msg)

            if self._given_amulog_database == "anonymized":
                ev = (evdef.host, evdef.gid)
            elif self._given_amulog_database == "original":
                ev = (evdef_org.host, evdef_org.gid)
            else:
                raise ValueError
            l_org_lm = [lm for lm in self.load_org(ev, dt_range)]
            if len(l_org_lm) == 0:
                msg = ("No logs for {0}, ".format(ev) +
                       "inconsistent with source")
                raise ValueError(msg)
            ret = [(lm.dt, lm.host, lm.restore_message()) for lm in l_org_lm]
            if len(ret) == 0:
                msg = ("No matching logs for {0}, ".format(ev) +
                       "inconsistent with source")
                raise ValueError(msg)
            assert len(ret) >= len(s_dt), "sanity check failure {0}".format(ev)
        else:
            ret = [(dt, evdef.host, values[0]) for dt, values
                   in self.load_items(measure, evdef.tags(), dt_range)]
            if len(ret) == 0:
                msg = ("No time-series for {0}, ".format(evdef) +
                       "inconsistent with tsdb")
                raise ValueError(msg)

        return ret


class LogEventLoaderDirect(evgen_common.EventLoader, LogEventLoaderBase):
    fields = ["val", ]

    def __init__(self, conf, dry=False):
        evgen_common.EventLoader.__init__(self, conf, dry=dry)
        LogEventLoaderBase.__init__(self, conf)

    def load(self, measure, tags, dt_range, binsize):
        host = tags["host"]
        gid = int(tags["key"])
        ev = (host, gid)
        l_dt = self.source.load(ev)
        l_values = [[float(1)] * len(self.fields)] * len(l_dt)
        df = convert.timestamps2df(l_dt, l_values, self.fields, dt_range, binsize)
        return df

    def load_items(self, measure, tags, dt_range):
        host = tags["host"]
        gid = int(tags["key"])
        ev = (host, gid)
        l_dt = self.source.load(ev)
        feature_dt = self._apply_filters(l_dt, dt_range, ev)

        for dt in feature_dt:
            yield dt, np.array([1.0])



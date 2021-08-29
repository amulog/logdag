#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import logging

from amulog import config
from logdag import log2event
from . import evgen_common
from . import filter_log

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
        if self.group is None or self.group == "None":
            return "{0}:{1}".format(self.host, str(self.gid))
        else:
            return "{0}:{1}:{2}".format(self.host, str(self.gid),
                                        self.group)

    def key(self):
        return str(self.gid)

    def tags(self):
        return {"host": self.host,
                "key": self.key()}

    def series(self):
        return FEATURE_MEASUREMENT, self.tags()


class LogEventLoader(evgen_common.EventLoader):
    fields = ["val", ]

    def __init__(self, conf, dry=False):
        super().__init__(conf, dry=dry)
        src = conf["general"]["log_source"]
        if src == "amulog":
            from . import src_amulog
            args = [
                config.getterm(conf, "general", "evdb_whole_term"),
                conf["database_amulog"]["source_conf"],
                conf["database_amulog"]["event_gid"],
                conf.getboolean("database_amulog",
                                "use_anonymize_mapping")
            ]
            self.source = src_amulog.AmulogLoader(*args)
        else:
            raise NotImplementedError
        self._filter_rules = config.getlist(conf, "filter", "rules")
        for method in self._filter_rules:
            assert method in filter_log.FUNCTIONS

        self.evdb = self._init_evdb(conf, "log_dbname")
#        dst = conf["general"]["evdb"]
#        if dst == "influx":
#            dbname = conf["database_influx"]["log_dbname"]
#            from . import influx
#            self.evdb = influx.init_influx(conf, dbname, df=False)
#            # self.evdb_df = influx.init_influx(conf, dbname, df = True)
#        else:
#            raise NotImplementedError

        self._lf = None
        if len(self._filter_rules) > 0:
            self._lf = filter_log.init_logfilter(conf, self.source)
        self._feature_unit_diff = config.getdur(conf,
                                                "general", "evdb_unit_diff")

    @staticmethod
    def _evdef(host, gid, group):
        d = {"source": log2event.SRCCLS_LOG,
             "host": host,
             "group": group,
             "gid": gid}
        return LogEventDefinition(**d)

    def _apply_filter(self, l_dt, dt_range, ev):
        tmp_l_dt = l_dt
        for method in self._filter_rules:
            args = (tmp_l_dt, dt_range, ev)
            tmp_l_dt = getattr(self._lf, method)(*args)
            if method == "sizetest" and tmp_l_dt is None:
                # sizetest failure means skipping later tests
                # and leave all events
                return l_dt
            elif tmp_l_dt is None or len(tmp_l_dt) == 0:
                msg = "event {0} removed with {1}".format(ev, method)
                _logger.info(msg)
                return None
        return tmp_l_dt

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
            feature_dt = self._apply_filter(l_dt, dt_range, ev)
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

    def iter_evdef(self, dt_range=None):
        for host, gid in self.source.iter_event(dt_range=dt_range):
            group = self.source.group(gid)
            d = {"source": log2event.SRCCLS_LOG,
                 "host": host,
                 "group": group,
                 "gid": gid}
            yield LogEventDefinition(**d)

    def instruction(self, evdef):
        if isinstance(evdef, log2event.MultipleEventDefinition):
            l_buf = []
            for tmp_evdef in evdef.members:
                l_buf.append(self.instruction(tmp_evdef))
            return " | ".join(l_buf)
        else:
            instruction = self.source.gid_instruction(evdef.gid)
            return "({0}) {1}".format(evdef.host, instruction)

    def details(self, evdef, dt_range, org=False):
        if isinstance(evdef, log2event.MultipleEventDefinition):
            results = []
            for tmp_evdef in evdef.members:
                results += self.details(tmp_evdef, dt_range, org)
            return sorted(results, key=lambda x: x[0])

        measure = "log_feature"
        if org:
            # It extracts timestamps on valid bins after preprocessing
            # Note: it is impossible to distinguish counts in one bin
            # if it includes periodic and aperiodic components
            s_dt = {dt for dt, values
                    in self.load_items(measure, evdef.tags(), dt_range)}
            if len(s_dt) == 0:
                msg = ("No time-series for {0}, ".format(evdef) +
                       "inconsistent with tsdb")
                raise ValueError(msg)

            ev = (evdef.host, evdef.gid)
            l_org_lm = [lm for lm in self.load_org(ev, dt_range)]
            if len(l_org_lm) == 0:
                msg = ("No logs for {0}, ".format(evdef) +
                       "inconsistent with source")
                raise ValueError(msg)
            ret = [(lm.dt, lm.host, lm.restore_message()) for lm in l_org_lm]
            if len(ret) == 0:
                msg = ("No matching logs for {0}, ".format(evdef) +
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

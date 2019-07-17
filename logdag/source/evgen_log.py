#!/usr/bin/env python
# coding: utf-8

import logging
import numpy as np

from amulog import config
from . import evgen_common
from . import filter_log

_logger = logging.getLogger(__package__)


class LogEventLoader(evgen_common.EventLoader):
    fields = ["val",]

    def __init__(self, conf, dry = False):
        self.conf = conf
        self.dry = dry
        src = conf["general"]["log_source"] 
        if src == "amulog":
            from . import source_amulog
            args = [config.getterm(conf, "general", "evdb_whole_term"),
                    conf["database_amulog"]["source_conf"],
                    conf["database_amulog"]["event_gid"]]
            self.source = source_amulog.AmulogLoader(*args)
        else:
            raise NotImplementedError
        self._filter_rules = config.getlist(conf, "filter", "rules")
        for method in self._filter_rules:
            assert method in filter_log.FUNCTIONS

        dst = conf["general"]["evdb"]
        if dst == "influx":
            dbname = conf["database_influx"]["log_dbname"]
            from . import influx
            self.evdb = influx.init_influx(conf, dbname, df = False)
            #self.evdb_df = influx.init_influx(conf, dbname, df = True)
        else:
            raise NotImplementedError

        self._lf = filter_log.init_logfilter(conf, self.source)

    def _apply_filter(self, l_dt, dt_range, evdef):
        tmp_l_dt = l_dt
        for method in self._filter_rules:
            args = (tmp_l_dt, dt_range, evdef)
            tmp_l_dt = getattr(self._lf, method)(*args)
            if method == "sizetest":
                # sizetest failure means skipping later tests
                if tmp_l_dt is None:
                    return l_dt
            elif tmp_l_dt is None or len(tmp_l_dt) == 0:
                msg = "event {0} removed with {1}".format(evdef, method)
                _logger.info(msg)
                return None
        return tmp_l_dt

    def read_all(self, dump_org = False):
        return self.read(dt_range = None, dump_org = dump_org)

    def read(self, dt_range = None, dump_org = False):
        if dt_range is not None:
            self.source.dt_range = dt_range

        measure = "log"
        for evdef in self.source.iter_evdef():
            host, gid = evdef
            l_dt = self.source.load(evdef)
            if len(l_dt) == 0:
                _logger.info("log gid={0} host={1} is empty".format(
                    gid, host))
                continue
            if dump_org:
                self.dump("log_org", host, gid, l_dt)
                _logger.info("added org {0} size {1}".format(
                    (host, gid), len(l_dt)))
                pass
            feature_dt = self._apply_filter(l_dt, dt_range, evdef)
            if feature_dt is not None:
                self.dump("log_feature", host, gid, feature_dt)
                _logger.info("added feature {0} size {1}".format(
                    (host, gid), len(feature_dt)))

    def dump(self, measure, host, gid, l_dt):
        if self.dry:
            return
        d_tags = {"host": host, "key": gid}
        data = {k: [v,] for k, v in self.source.timestamp2dict(l_dt).items()}
        self.evdb.add(measure, d_tags, data, self.fields)
        self.evdb.commit()

    def all_feature(self):
        return ["log_feature",]


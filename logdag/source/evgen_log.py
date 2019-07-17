#!/usr/bin/env python
# coding: utf-8

import logging
import numpy as np

from amulog import config
from . import filter_log

_logger = logging.getLogger(__package__)


class LogEventLoader(object):
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
            self.evdb_df = influx.init_influx(conf, dbname, df = True)
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
        data = {k: [v,] for k, v in self.source.timestamp2dict(l_dt).items()}
        self.evdb.add(measure, {"host": host, "key": gid}, data, self.fields)
        self.evdb.commit()

    def all_feature(self):
        return ["log_feature",]

    def all_condition(self, dt_range = None):
        for featurename in self.all_feature():
            measure = featurename
            if dt_range is None:
                l_d_tags = self.evdb.list_series(measure = measure)
            else:
                ut_range = tuple(dt.timestamp() for dt in dt_range)
                l_d_tags = self.evdb.list_series(measure = measure,
                                                 ut_range = ut_range)
            for d_tags in l_d_tags:
                yield (measure, d_tags["host"], int(d_tags["key"]))

    def load(self, measure, host, gid, dt_range, binsize):
        d_tags = {"host": host, "key": str(gid)}
        ut_range = tuple(dt.timestamp() for dt in dt_range)
        str_bin = config.dur2str(binsize)
        df = self.evdb_df.get(measure, d_tags, self.fields,
                              ut_range, str_bin, func = "sum", fill = 0)
        import pdb; pdb.set_trace()
        return df

    def load_items(self, measure, host, gid, dt_range):
        d_tags = {"host": host, "key": str(gid)}
        ut_range = tuple(dt.timestamp() for dt in dt_range)
        rs = self.evdb.get(measure, d_tags, self.fields, ut_range)

        l_dt = [p["time"] for p in rs.get_points()]
        l_array = [np.array([p[f] for f in self.fields])
                   for p in rs.get_points()]
        return (l_dt, l_array)

    def load_all(self, dt_range, binsize):
        for featurename in self.all_feature:
            measure = featurename
            for series in self.all_series(measure):
                host, key = tuple(series)
                df = self.load(measure, host, key, dt_range, binsize)
                yield (host, key, df)



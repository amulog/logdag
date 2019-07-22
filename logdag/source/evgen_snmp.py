#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import logging
from collections import defaultdict

from amulog import config
from amulog import host_alias
from . import evgen_common
from . import evpost

_logger = logging.getLogger(__package__)


class SNMPEventLoader(evgen_common.EventLoader):
    fields = ["val", ]

    def __init__(self, conf, dry=False):
        self.conf = conf
        self.dry = dry
        src = conf["general"]["snmp_source"]
        if src == "rrd":
            from . import rrd
            self.source = rrd.RRDLoader(conf)
        else:
            raise NotImplementedError

        dst = conf["general"]["evdb"]
        if dst == "influx":
            dbname = conf["database_influx"]["snmp_dbname"]
            from . import influx
            self.evdb = influx.init_influx(conf, dbname, df=False)
        else:
            raise NotImplementedError

        ha_fn = conf["general"]["host_alias_filename"]
        self._ha = host_alias.HostAlias(ha_fn)
        self._d_source, self._d_vsource = self._init_sourcelist(conf)
        self._d_feature = self._init_featurelist(conf)

    @staticmethod
    def _read_filelist(fp):
        with open(fp, 'r') as f:
            for line in f:
                if len(line.rstrip()) == 0:
                    continue
                temp = line.split()
                path = temp[0]
                host = temp[1]
                key = temp[2]
                yield (path, host, key)

    def _hostname(self, name):
        tmp = self._ha.resolve_host(name)
        if tmp:
            return tmp
        else:
            return name

    def _init_sourcelist(self, conf):
        d_source = {}
        for name in config.getlist(conf, "snmp_source", "all"):
            fp = conf["snmp_source"][name]
            d_source[name] = [(path, self._ha.resolve_host(host), key)
                              for path, host, key in self._read_filelist(fp)]
        d_vsource = {}
        for name in config.getlist(conf, "snmp_vsource", "all"):
            tmp = [e.strip() for e in conf["snmp_vsource"][name].split(",")]
            assert len(tmp) == 2
            d_vsource[name] = tmp
        return d_source, d_vsource

    def _read_source(self, name, dt_range):
        for path, host, key in self._d_source[name]:
            _logger.info("read source {0} {1}".format(host, key))
            df = self.source.load(path, dt_range)
            yield (host, key, df)

    def _read_source_host(self, name, target_host, dt_range):
        for path, host, key in self._d_source[name]:
            if host == target_host:
                _logger.info("read source {0} {1}".format(host, key))
                df = self.source.load(path, dt_range)
                yield (key, df)

    def _read_vsource(self, sourcename, dt_range, func, dump_org=False):
        # sourcename, func = self._d_vsource[name]
        if func == "hostsum":
            return self._read_vsource_hostsum(sourcename, dt_range, dump_org)
        else:
            raise NotImplementedError

    def _read_vsource_hostsum(self, name, dt_range, dump_org):
        s_host = {host for path, host, key in self._d_source[name]}
        for host in s_host:
            ret_df = None
            for key, df in self._read_source_host(name, host, dt_range):
                if df is None or self.isallnan(df):
                    _logger.info("source {0} {1} {2} is empty".format(
                        name, host, key))
                    continue
                if dump_org:
                    self.dump(name, host, key, df)
                    _logger.info("added org {0} size {1}".format(
                        (host, key), df.shape))
                if ret_df is None:
                    ret_df = df.fillna(0)
                else:
                    ret_df = ret_df.add(df.fillna(0), fill_value=0)
            yield (host, ret_df)

    def _init_featurelist(self, conf):
        d_feature = defaultdict(list)
        for name in sorted(conf.options("snmp_feature")):
            # for name in conf["feature"]["all"]:
            # tmp = conf["feature"][name]
            tmp = config.getlist(conf, "snmp_feature", name)
            assert len(tmp) >= 2
            sourcename = tmp[0]
            keyfunc = tmp[1]
            l_postfunc = tmp[2:]
            d_feature[sourcename].append((name, keyfunc, l_postfunc))
        return d_feature

    @staticmethod
    def isallnan(df):
        tmp = df.values.flatten()
        return sum(np.isnan(tmp)) == len(tmp)

    def read(self, dt_range, dump_org=False):
        # reverse resolution by sourcenames to avoid duplicated load
        all_sourcename = set(self._d_source.keys()) | \
                         set(self._d_vsource.keys())
        for sourcename, l_feature_def in sorted(self._d_feature.items()):
            if sourcename in self._d_source:
                _logger.info("loading source {0}".format(sourcename))
                for host, key, df in self._read_source(sourcename, dt_range):
                    if df is None or self.isallnan(df):
                        _logger.info("source {0} {1} {2} is empty".format(
                            sourcename, host, key))
                        continue
                    if dump_org:
                        self.dump(sourcename, host, key, df)
                        _logger.info("added org {0} size {1}".format(
                            (host, key), df.shape))
                    self._make_feature(sourcename, host, key,
                                       df, l_feature_def)
            elif sourcename in self._d_vsource:
                _logger.info("loading vsource {0}".format(sourcename))
                key = "all"
                orgname, func = self._d_vsource[sourcename]
                for host, df in self._read_vsource(orgname, dt_range,
                                                   func, dump_org):
                    if df is None or self.isallnan(df):
                        _logger.info("vsource {0} {1} {2} is empty".format(
                            sourcename, host, key))
                        continue
                    if dump_org:
                        self.dump(sourcename, host, key, df)
                        _logger.info("added org {0} size {1}".format(
                            (host, key), df.shape))
                    self._make_feature(sourcename, host, key,
                                       df, l_feature_def)
                all_sourcename.remove(orgname)
            all_sourcename.remove(sourcename)

        if not dump_org:
            return

        # add sources without features
        _logger.info("add sources without features ({0})".format(
            all_sourcename))
        for sourcename in all_sourcename:
            if sourcename in self._d_source:
                _logger.info("loading source {0}".format(sourcename))
                for host, key, df in self._read_source(sourcename, dt_range):
                    if df is None or self.isallnan(df):
                        _logger.info("source {0} {1} {2} is empty".format(
                            sourcename, host, key))
                        continue
                    self.dump(sourcename, host, key, df)
                    _logger.info("added org {0} size {1}".format(
                        (host, key), df.shape))
            elif sourcename in self._d_vsource:
                _logger.info("loading vsource {0}".format(sourcename))
                key = "all"
                orgname, func = self._d_vsource[sourcename]
                for host, df in self._read_vsource(orgname, dt_range,
                                                   func, dump_org):
                    if df is None or self.isallnan(df):
                        _logger.info("vsource {0} {1} {2} is empty".format(
                            sourcename, host, key))
                        continue
                    self.dump(sourcename, host, key, df)
                    _logger.info("added org {0} size {1}".format(
                        (host, key), df.shape))

    def _make_feature(self, sourcename, host, key, df, l_feature_def):
        for feature_def in l_feature_def:
            feature_name = feature_def[0]
            data = self._calc_feature(df, feature_def)
            if data is None or self.isallnan(data):
                _logger.info("feature {0} {1} is empty".format(
                    feature_name, (host, key)))
            else:
                self.dump(feature_name, host, key, data)
                _logger.info("added feature {0} {1} size {2}".format(
                    feature_name, (host, key), df.shape))

    def _calc_feature(self, df, feature_def):
        featurename, keyfunc, l_postfunc = feature_def

        if "#" in keyfunc:
            raise NotImplementedError
        else:
            if not keyfunc in df.columns:
                return None
            sr = df[keyfunc]
            new_columns = self.fields

        if self.isallnan(sr):
            return None

        for postfunc in l_postfunc:
            sr = eval("evpost.{0}".format(postfunc))(sr)

        if len(sr) == 0 or self.isallnan(sr):
            return None

        ret = pd.DataFrame(sr)
        ret.columns = new_columns
        return ret

    def dump(self, measure, host, key, df):
        if self.dry:
            return
        data = {k: v for k, v
                in zip(df.index, df.itertuples(index=False, name=None))}
        d_tags = {"host": host, "key": key}
        self.evdb.add(measure, d_tags, data, self.fields)
        self.evdb.commit()

    def dump_feature(self, measure, host, key, df):
        return self.dump(measure, host, key, df[df > 0].dropna())

    def all_feature(self):
        ret = []
        for l_featuredef in self._d_feature.values():
            for name, keyfunc, l_postfunc in l_featuredef:
                ret.append(name)
        return ret

    def load_org(self, measure, host, key, dt_range):
        def search_source(featurename):
            for srcname, l_feature_def in self._d_feature[featurename]:
                for feature_def in l_feature_def:
                    if featurename == feature_def[0]:
                        return srcname
            else:
                return None

        sourcename = search_source(measure)
        if sourcename is None:
            return None
        else:
            return self.load_items(sourcename, host, key, dt_range)

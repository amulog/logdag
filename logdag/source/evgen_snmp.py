#!/usr/bin/env python
# coding: utf-8

import json
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

        self._ha = host_alias.HostAlias(conf["general"]["host_alias_filename"])

        snmp_def = conf["general"]["snmp_feature_def"]
        with open(snmp_def, "r") as f:
            jsobj = json.load(f)

        self._d_source = {d["name"]: list(self._read_filelist(d["filelist"]))
                          for d in jsobj["source"]}

        self._d_vsource = jsobj["vsource"]

        self._d_feature = jsobj["feature"]

        self._d_rfeature = defaultdict(list)
        for name in self._d_feature:
            src = self._d_feature[name]["src"]
            d = {"name": name,
                 "column": self._d_feature[name]["column"],
                 "func_list": self._d_feature[name]["func_list"]}
            self._d_rfeature[src].append(d)

    def _read_filelist(self, fp):
        with open(fp, 'r') as f:
            for line in f:
                if len(line.rstrip()) == 0:
                    continue
                temp = line.split()
                yield {"path": temp[0],
                       "host": self._hostname(temp[1]),
                       "key": temp[2]}

    def _hostname(self, name):
        tmp = self._ha.resolve_host(name)
        if tmp:
            return tmp
        else:
            return name

    def _read_source(self, name, dt_range):
        for d in self._d_source[name]:
            _logger.info("read source {0} {1}".format(d["host"], d["key"]))
            df = self.source.load(d["path"], dt_range)
            yield d["host"], d["key"], df

    def _read_source_host(self, name, target_host, dt_range):
        for d in self._d_source[name]:
            if d["host"] == target_host:
                _logger.info("read source {0} {1}".format(d["host"], d["key"]))
                df = self.source.load(d["path"], dt_range)
                yield d["key"], df

    def _read_vsource(self, vsourcename, dt_range, func, dump_org=False):
        # sourcename, func = self._d_vsource[name]
        if func == "hostsum":
            return self._read_vsource_hostsum(vsourcename, dt_range, dump_org)
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
            yield host, ret_df

    @staticmethod
    def isallnan(df):
        tmp = df.values.flatten()
        return sum(np.isnan(tmp)) == len(tmp)

    def read(self, dt_range, dump_org=False):
        # reverse resolution by sourcenames to avoid duplicated load
        all_sourcename = set(self._d_source.keys()) | \
                         set(self._d_vsource.keys())
        for sourcename, l_feature_def in sorted(self._d_rfeature.items()):
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
                    self._make_feature(self._d_rfeature[sourcename],
                                       host, key, df)
            elif sourcename in self._d_vsource:
                _logger.info("loading vsource {0}".format(sourcename))
                key = "all"
                orgname = self._d_vsource[sourcename]["src"]
                func = self._d_vsource[sourcename]["func"]
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
                    self._make_feature(self._d_rfeature[sourcename],
                                       host, key, df)
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
                orgname = self._d_vsource[sourcename]["src"]
                func = self._d_vsource[sourcename]["func"]
                for host, df in self._read_vsource(orgname, dt_range,
                                                   func, dump_org):
                    if df is None or self.isallnan(df):
                        _logger.info("vsource {0} {1} {2} is empty".format(
                            sourcename, host, key))
                        continue
                    self.dump(sourcename, host, key, df)
                    _logger.info("added org {0} size {1}".format(
                        (host, key), df.shape))

    def _make_feature(self, l_feature_def, host, key, df):
        for feature_def in l_feature_def:
            data = self._calc_feature(df, feature_def)
            if data is None or self.isallnan(data):
                _logger.info("feature {0} {1} is empty".format(
                    feature_def["name"], (host, key)))
            else:
                self.dump(feature_def["name"], host, key, data)
                _logger.info("added feature {0} {1} size {2}".format(
                    feature_def["name"], (host, key), df.shape))

    def _calc_feature(self, df, feature_def):
        sr = df[feature_def["column"]]
        if self.isallnan(sr):
            return None

        for postfunc in feature_def["func_list"]:
            sr = eval("evpost.{0}".format(postfunc))(sr)

        if len(sr) == 0 or self.isallnan(sr):
            return None

        ret = pd.DataFrame(sr)
        ret.columns = self.fields
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
        return list(self._d_feature.keys())

    def load_org(self, measure, host, key, dt_range):
        sourcename = self._d_feature[measure]["src"]
        return self.load_items(sourcename, host, key, dt_range)

    def label(self, measure):
        return self._d_feature[measure]["group"]

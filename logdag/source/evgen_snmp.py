# !/usr/bin/env python
# coding: utf-8

import json
import numpy as np
import pandas as pd
import logging
from collections import defaultdict

from amulog import host_alias
from logdag import log2event
from . import evgen_common
from . import evpost

_logger = logging.getLogger(__package__)

VSOURCE_KEY = "all"


class SNMPEventDefinition(log2event.EventDefinition):
    _l_attr_key = ["mod_cls", "mod_id", ]
    _l_attr_snmp = ["measure", "direction", ] + _l_attr_key

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for attr in self._l_attr_snmp:
            setattr(self, attr, kwargs[attr])

    def __str__(self):
        return "{0}, {1}@{2}({3})".format(self.host, self.measure,
                                          self.key(), self.group)

    def key(self):
        return "@".join([getattr(self, attr) for attr in self._l_attr_key])

    def tags(self):
        return {"host": self.host,
                "key": self.key()}

    def series(self):
        return self.measure, self.tags()

    @classmethod
    def fromfeature(cls, sourcedef, featuredef):
        d = {"source": log2event.SRCCLS_SNMP,
             "host": sourcedef["host"],
             "group": featuredef["group"],
             "measure": featuredef["name"],
             "mod_cls": sourcedef["mod_cls"],
             "mod_id": sourcedef["mod_id"],
             }
        if "direction" in featuredef:
            d["direction"] = featuredef["direction"]
        else:
            d["direction"] = None
        return cls(**d)


class SNMPVirtualEventDefinition(SNMPEventDefinition):

    def __str__(self):
        return "{0}, {1}({2})".format(self.host, self.measure,
                                      self.group)

    def key(self):
        return VSOURCE_KEY

    @classmethod
    def fromfeature(cls, host, featuredef):
        d = {"source": log2event.SRCCLS_SNMP,
             "host": host,
             "group": featuredef["group"],
             "measure": featuredef["name"],
             "mod_cls": None,
             "mod_id": None,
             }
        if "direction" in featuredef:
            d["direction"] = featuredef["direction"]
        else:
            d["direction"] = None
        return cls(**d)


class SNMPEventLoader(evgen_common.EventLoader):
    vsource_key = "all"
    fields = ["val", ]

    def __init__(self, conf, dry=False):
        self.conf = conf
        self.dry = dry
        self._srcdb = conf["general"]["snmp_source"]
        if self._srcdb == "rrd":
            from . import rrd
            self.source = rrd.RRDLoader(conf)
        elif self._srcdb == "influx":
            source_dbname = conf["database_influx"]["snmp_source_dbname"]
            from . import influx
            self.source = influx.init_influx(conf, source_dbname, df=False)
        else:
            raise NotImplementedError

        self._dstdb = conf["general"]["evdb"]
        if self._dstdb == "influx":
            dbname = conf["database_influx"]["snmp_dbname"]
            from . import influx
            self.evdb = influx.init_influx(conf, dbname, df=False)
        else:
            raise NotImplementedError

        self._ha = host_alias.HostAlias(conf["general"]["host_alias_filename"])

        snmp_def = conf["general"]["snmp_feature_def"]
        with open(snmp_def, "r") as f:
            jsobj = json.load(f)

        # self._d_source: list of dict: seriesdef
        # seriesdef keys: filelist, host, mod_cls, mod_id
        self._d_source = jsobj["source"]

        self._d_vsourcedef = jsobj["vsource"]
        self._init_vsource()

        self._d_feature = jsobj["feature"]

        self._d_rfeature = defaultdict(list)
        for name in self._d_feature:
            src = self._d_feature[name]["source"]
            d = {"name": name,
                 "column": self._d_feature[name]["column"],
                 "func_list": self._d_feature[name]["func_list"]}
            self._d_rfeature[src].append(d)

    def _init_vsource(self):
        self._d_vsource = defaultdict(list)
        for vsourcename, vsourcedef in self._d_vsourcedef.items():
            if vsourcedef["func"] == "hostsum":
                orgname = vsourcedef["src"]
                s_host = {d["host"] for d in self._d_source[orgname]}
                for host in s_host:
                    seriesdef = {"host": host, "type": "vsource"}
                    self._d_vsource[vsourcename].append(seriesdef)
            else:
                raise NotImplementedError

    @staticmethod
    def _evdef_source(seriesdef, featuredef):
        return SNMPEventDefinition.fromfeature(seriesdef, featuredef)

    @staticmethod
    def _evdef_vsource(host, featuredef):
        return SNMPVirtualEventDefinition.fromfeature(host, featuredef)

    @staticmethod
    def _seriesdef2tags(seriesdef):
        stype = seriesdef.get("type", "source")
        if stype == "source":
            return {"host": seriesdef["host"],
                    "key": "@".join((seriesdef["mod_cls"], seriesdef["mod_id"]))}
        elif stype == "vsource":
            return {"host": seriesdef["host"],
                    "key": VSOURCE_KEY}
        else:
            raise ValueError("invalid seriesdef type")

    def _hostname(self, name):
        tmp = self._ha.resolve_host(name)
        if tmp:
            return tmp
        else:
            return name

    def _read_source(self, name, dt_range, target_host=None):
        for seriesdef in self._d_source[name]:
            tags = self._seriesdef2tags(seriesdef)
            if target_host is None or tags["host"] == target_host:
                _logger.info("read source {0}".format(tags))
                ret_df = self.load_source(name, seriesdef, dt_range)
                # ret_df = None
                # for fp in seriesdef["filelist"]:
                #    df = self.source.load(fp, dt_range)
                #    if ret_df is None:
                #        ret_df = df
                #    else:
                #        ret_df = ret_df.add(df, fill_value=0)
                yield tags, ret_df

    def _read_vsource(self, name, dt_range, func, dump_org=False):
        # sourcename, func = self._d_vsourcedef[name]
        if func == "hostsum":
            return self._read_vsource_hostsum(name, dt_range, dump_org)
        else:
            raise NotImplementedError

    def _read_vsource_hostsum(self, name, dt_range, dump_org):
        if self._srcdb == "influx":
            for seriesdef in self._d_vsource[name]:
                yield self.source.load_source(name, seriesdef, dt_range)
        else:
            orgname = self._d_vsourcedef[name]["src"]
            ret_df = None
            for seriesdef in self._d_vsource[name]:
                host = seriesdef["host"]
                vsource_tags = self._seriesdef2tags(seriesdef)
                for tags, df in self._read_source(orgname, dt_range,
                                                  target_host=host):
                    if df is None or self.isallnan(df):
                        _logger.info("source {0} {1} is empty".format(
                            name, tags))
                        continue
                    if dump_org:
                        self.dump(orgname, tags, df)
                        _logger.info("added org {0} size {1}".format(
                            tags, df.shape))
                    if ret_df is None:
                        ret_df = df.fillna(0)
                    else:
                        ret_df = ret_df.add(df.fillna(0), fill_value=0)
                yield vsource_tags, ret_df

        #orgname = self._d_vsource[name]["src"]
        #s_host = {d["host"] for d in self._d_source[orgname]}
        #for host in s_host:
        #    vsource_tags = {"host": host,
        #                    "key": VSOURCE_KEY}
        #    if self._srcdb == "influx":
        #        tmp_seriesdef = {"filelist": None,
        #                         "host": host,
        #                         "mod_cls": None,
        #                         "mod_id": None}
        #        # sourcedef keys: filelist, host, mod_cls, mod_id
        #        yield self.source.load_source(name, tmp_seriesdef, dt_range)
        #    else:
        #        ret_df = None
        #        for tags, df in self._read_source(orgname, dt_range,
        #                                          target_host=host):
        #            if df is None or self.isallnan(df):
        #                _logger.info("source {0} {1} is empty".format(
        #                    name, tags))
        #                continue
        #            if dump_org:
        #                self.dump(orgname, tags, df)
        #                _logger.info("added org {0} size {1}".format(
        #                    tags, df.shape))
        #            if ret_df is None:
        #                ret_df = df.fillna(0)
        #            else:
        #                ret_df = ret_df.add(df.fillna(0), fill_value=0)
        #        vsource_tags = {"host": host,
        #                        "key": VSOURCE_KEY}
        #        yield vsource_tags, ret_df

    @staticmethod
    def isallnan(df):
        tmp = df.values.flatten()
        return sum(np.isnan(tmp)) == len(tmp)

    def store_all(self, dt_range, dump_org=False):
        # reverse resolution by sourcenames to avoid duplicated load
        all_sourcename = set(self._d_source.keys()) | \
                         set(self._d_vsource.keys())
        for sourcename, l_feature_def in sorted(self._d_rfeature.items()):
            if sourcename in self._d_source:
                _logger.info("loading source {0}".format(sourcename))
                for tags, df in self._read_source(sourcename, dt_range):
                    self._make_source(sourcename, l_feature_def, tags, df,
                                      dump_org)
            elif sourcename in self._d_vsourcedef:
                _logger.info("loading vsource {0}".format(sourcename))
                func = self._d_vsourcedef[sourcename]["func"]
                for tags, df in self._read_vsource(sourcename, dt_range,
                                                   func, dump_org):
                    self._make_source(sourcename, l_feature_def, tags, df,
                                      dump_org)
                orgsourcename = self._d_vsourcedef[sourcename]["src"]
                all_sourcename.remove(orgsourcename)
            else:
                raise ValueError("undefined source {0}".format(sourcename))
            all_sourcename.remove(sourcename)

        # add sources without features
        if not dump_org:
            return
        _logger.info("add sources without features ({0})".format(
            all_sourcename))
        for sourcename in all_sourcename:
            if sourcename in self._d_source:
                _logger.info("loading source {0}".format(sourcename))
                for tags, df in self._read_source(sourcename, dt_range):
                    self._make_source(sourcename, [], tags, df, dump_org)
            elif sourcename in self._d_vsourcedef:
                _logger.info("loading vsource {0}".format(sourcename))
                func = self._d_vsourcedef[sourcename]["func"]
                for tags, df in self._read_vsource(sourcename, dt_range,
                                                   func, dump_org):
                    self._make_source(sourcename, [], tags, df, dump_org)
            else:
                raise ValueError("undefined source {0}".format(sourcename))

    def store_feature(self, featurename, dt_range, dump_org=False):
        sourcename = self._d_feature[featurename]["source"]
        l_feature_def = [self._d_feature[featurename], ]
        if sourcename in self._d_source:
            for tags, df in self._read_source(sourcename, dt_range):
                _logger.info("loading source {0}".format(sourcename))
                self._make_source(sourcename, l_feature_def, tags, df,
                                  dump_org)
        elif sourcename in self._d_vsourcedef:
            _logger.info("loading vsource {0}".format(sourcename))
            func = self._d_vsourcedef[sourcename]["func"]
            for tags, df in self._read_vsource(sourcename, dt_range,
                                               func, dump_org):
                self._make_source(sourcename, l_feature_def, tags, df,
                                  dump_org)
        else:
            raise ValueError("undefined source {0}".format(sourcename))

    def _make_source(self, sourcename, l_feature_def, tags, df, dump_org):
        if df is None or self.isallnan(df):
            _logger.info("source {0} {1} is empty".format(
                sourcename, tags))
            return
        if dump_org:
            self.dump(sourcename, tags, df)
            _logger.info("added org {0} size {1}".format(
                tags, df.shape))
        for feature_def in l_feature_def:
            self._make_feature(feature_def, tags, df)

    def _make_feature(self, feature_def, tags, df):
        data = self._calc_feature(df, feature_def)
        if data is None or self.isallnan(data):
            _logger.info("feature {0} {1} is empty".format(
                feature_def["name"], tags))
        else:
            self.dump_feature(feature_def["name"], tags, data)
            _logger.info("added feature {0} {1} size {2}".format(
                feature_def["name"], tags, data.shape))

    def _calc_feature(self, df, feature_def):
        column = feature_def["column"]
        if column in df.columns:
            sr = df[column]
        else:
            return None
        if self.isallnan(sr):
            return None

        for postfunc in feature_def["func_list"]:
            sr = eval("evpost.{0}".format(postfunc))(sr)

        if len(sr) == 0 or self.isallnan(sr):
            return None

        ret = pd.DataFrame(sr)
        ret.columns = self.fields
        return ret

    def dump(self, measure, tags, df, fields=None):
        if self.dry:
            return
        data = {k: v for k, v
                in zip(df.index, df.itertuples(index=False, name=None))}
        if fields is None:
            fields = df.columns
        self.evdb.add(measure, tags, data, fields)
        self.evdb.commit()

    def dump_feature(self, measure, tags, df):
        return self.dump(measure, tags, df[df > 0].dropna(),
                         fields=self.fields)

    def all_feature(self):
        return list(self._d_feature.keys())

    def load_org(self, measure, tags, dt_range):
        sourcename = self._d_feature[measure]["src"]
        return self.load_items(sourcename, tags, dt_range)

    def load_source(self, sourcename, seriesdef, dt_range):
        # for read_source
        if self._srcdb == "influx":
            tags = self._seriesdef2tags(seriesdef)
            return self.source.load_orgdf(sourcename, tags, dt_range)
        else:
            assert seriesdef.get("type", "source") == "source"
            ret_df = None
            for fp in seriesdef["filelist"]:
                df = self.source.load(fp, dt_range)
                if ret_df is None:
                    ret_df = df
                else:
                    ret_df = ret_df.add(df, fill_value=0)
            return ret_df

    def iter_evdef(self, l_feature_name=None):
        if l_feature_name is None:
            l_feature_name = [n for n in self._d_feature.keys()]

        for feature_name in l_feature_name:
            featuredef = self._d_feature[feature_name]
            sourcename = featuredef["source"]
            if sourcename in self._d_source:
                for sourcedef in self._d_source[sourcename]:
                    yield self._evdef_source(sourcedef, featuredef)
            elif sourcename in self._d_vsource:
                orgname = self._d_vsourcedef[sourcename]["src"]
                hosts = {sourcedef["host"] for sourcedef in self._d_source[orgname]}
                for host in hosts:
                    yield self._evdef_vsource(host, featuredef)

    @staticmethod
    def instruction(evdef):
        return str(evdef)

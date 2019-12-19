            # !/usr/bin/env python
# coding: utf-8

import json
import numpy as np
import pandas as pd
import logging
from collections import defaultdict

from amulog import config
from amulog import host_alias
from logdag import log2event
from logdag import dtutil
from amulog import mproc_queue
from . import evgen_common

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

    def __init__(self, conf, parallel=None, dry=False):
        self.conf = conf
        self.parallel = parallel
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

        if isinstance(jsobj["feature"], list):
            self._d_feature = jsobj["feature"]
        elif isinstance(jsobj["feature"], dict):
            # for backward compatibility with configparser-style rule
            self._d_feature = []
            for name, fdef in jsobj["feature"].items():
                fdef["name"] = name
                self._d_feature.append(fdef)
        #self._d_feature = jsobj["feature"]

        self._d_rfeature = defaultdict(list)
        for fdef in self._d_feature:
            src = fdef["source"]
            #d = {"name": fdef["name"],
            #     "column": self._d_feature[name]["column"],
            #     "func_list": self._d_feature[name]["func_list"]}
            self._d_rfeature[src].append(fdef)

        self._feature_unit_term = config.getdur(conf,
                                                "general", "evdb_unit_term")
        self._feature_unit_diff = config.getdur(conf,
                                                "general", "evdb_unit_diff")
        self._feature_bin_size = config.getdur(conf, "general", "evdb_binsize")
        self._feature_convolve_radius = conf.getint("general",
                                                    "evdb_convolve_radius")
        self._mproc = None

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
                vsource_tags = self._seriesdef2tags(seriesdef)
                yield vsource_tags, self.load_source(name, seriesdef, dt_range)
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

        # orgname = self._d_vsource[name]["src"]
        # s_host = {d["host"] for d in self._d_source[orgname]}
        # for host in s_host:
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

    def _search_feature_source(self, featurename, tags):
        ret = []
        l_feature_def = [fdef for fdef in self._d_feature
                         if fdef["name"] == featurename]
        for featuredef in l_feature_def:
            sname = featuredef["source"]
            for seriesdef in self._d_source[sname]:
                tmp_tags = self._seriesdef2tags(seriesdef)
                if self._tags_equal(tags, tmp_tags):
                    ret.append((sname, seriesdef, featuredef))
        # if there are multiple definition sets for 1 measure/tag set,
        # it causes duplicated write into influxdb
        assert len(ret) == 1, "duplicated feature definition"
        return ret[0]

    @staticmethod
    def _tags_equal(tags1, tags2):
        for k in tags1:
            if k not in tags2:
                raise ValueError("tags format not equal")
            if not tags1[k] == tags2[k]:
                return False
        else:
            return True

    @staticmethod
    def isallnan(df):
        tmp = df.astype(float).values.flatten()
        return sum(np.isnan(tmp)) == len(tmp)

    def store_all_source(self, dt_range, dump_vsource_org=False):
        """Store source data into db, without storing features."""
        all_sourcename = set(self._d_source.keys()) | \
                         set(self._d_vsource.keys())
        for sourcename in all_sourcename:
            if sourcename in self._d_source:
                _logger.info("loading source {0}".format(sourcename))
                for tags, df in self._read_source(sourcename, dt_range):
                    if df is None or self.isallnan(df):
                        _logger.info("source {0} {1} is empty".format(
                            sourcename, tags))
                        return
                    self.dump(sourcename, tags, df)
                    _logger.info("added org {0} size {1}".format(
                        tags, df.shape))
            elif sourcename in self._d_vsourcedef:
                _logger.info("loading vsource {0}".format(sourcename))
                func = self._d_vsourcedef[sourcename]["func"]
                for tags, df in self._read_vsource(sourcename, dt_range,
                                                   func, dump_vsource_org):
                    if df is None or self.isallnan(df):
                        _logger.info("source {0} {1} is empty".format(
                            sourcename, tags))
                        return
                    self.dump(sourcename, tags, df)
                    _logger.info("added org {0} size {1}".format(
                        tags, df.shape))
            else:
                raise ValueError("undefined source {0}".format(sourcename))

    def store_all(self, dt_range, dump_org=False):
        """Store data of all defined features with all source data."""
        self._init_mproc_manager()
        # reverse resolution by sourcenames to avoid duplicated load
        all_sourcename = set(self._d_source.keys()) | \
                         set(self._d_vsource.keys())
        for sourcename, l_feature_def in sorted(self._d_rfeature.items()):
            if sourcename in self._d_source:
                _logger.info("loading source {0}".format(sourcename))
                for tags, df in self._read_source(sourcename, dt_range):
                    self._make_feature_source(sourcename, l_feature_def,
                                              tags, df, dt_range, dump_org)
            elif sourcename in self._d_vsourcedef:
                _logger.info("loading vsource {0}".format(sourcename))
                func = self._d_vsourcedef[sourcename]["func"]
                for tags, df in self._read_vsource(sourcename, dt_range,
                                                   func, dump_org):
                    self._make_feature_source(sourcename, l_feature_def,
                                              tags, df, dt_range, dump_org)
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
                    self._make_feature_source(sourcename, [], tags, df,
                                              dt_range, dump_org)
            elif sourcename in self._d_vsourcedef:
                _logger.info("loading vsource {0}".format(sourcename))
                func = self._d_vsourcedef[sourcename]["func"]
                for tags, df in self._read_vsource(sourcename, dt_range,
                                                   func, dump_org):
                    self._make_feature_source(sourcename, [], tags, df,
                                              dt_range, dump_org)
            else:
                raise ValueError("undefined source {0}".format(sourcename))
        self._close_mproc_manager()

    def store_feature(self, featurename, dt_range, dump_org=False):
        """Store data of a specified feature with all source data."""
        self._init_mproc_manager()
        l_feature_def = [fdef for fdef in self._d_feature
                         if fdef["name"] == featurename]
        l_sources = {fdef["source"] for fdef in l_feature_def}
        for sourcename in l_sources:
            if sourcename in self._d_source:
                for tags, df in self._read_source(sourcename, dt_range):
                    _logger.info("loading source {0}".format(sourcename))
                    self._make_feature_source(sourcename, l_feature_def, tags,
                                              df, dt_range, dump_org)
            elif sourcename in self._d_vsourcedef:
                _logger.info("loading vsource {0}".format(sourcename))
                func = self._d_vsourcedef[sourcename]["func"]
                for tags, df in self._read_vsource(sourcename, dt_range,
                                                   func, dump_org):
                    self._make_feature_source(sourcename, l_feature_def, tags,
                                              df, dt_range, dump_org)
            else:
                raise ValueError("undefined source {0}".format(sourcename))
        self._close_mproc_manager()

    def test_store_feature(self, featurename, tags, dt_range, dump_org=False):
        """Store 1 feature of a specified source. This is for testing."""
        sourcename, seriesdef, featuredef = self._search_feature_source(
            featurename, tags)
        df = self.load_source(sourcename, seriesdef, dt_range)
        self._init_mproc_manager()
        self._make_feature_source(sourcename, [featuredef],
                                  tags, df, dt_range, dump_org)
        self._close_mproc_manager()

    def _make_feature_source(self, sourcename, l_feature_def, tags, df,
                             dt_range, dump_org):
        if df is None or self.isallnan(df):
            _logger.info("source {0} {1} is empty".format(
                sourcename, tags))
            return
        if dump_org:
            self.dump(sourcename, tags, df)
            _logger.info("added org {0} size {1}".format(
                tags, df.shape))
        for feature_def in l_feature_def:
            self._make_feature(feature_def, tags, df, dt_range)

    def _make_feature(self, feature_def, tags, df, dt_range):
        data = self._calc_feature(df, feature_def, dt_range)
        if data is None:
            _logger.info("no valid data for feature {0} {1}".format(
                feature_def["name"], tags))
            return
        dump_data = data.query("{0} > 0".format(self.fields[0]))
        if dump_data.shape[0] == 0:
            _logger.info("feature {0} {1} is empty".format(
                feature_def["name"], tags))
        else:
            self.dump(feature_def["name"], tags, dump_data, fields=self.fields)
            _logger.info("added feature {0} {1} size {2}".format(
                feature_def["name"], tags, dump_data.shape))

    def _init_mproc_manager(self):

        def apply_postfunc(task, *args, **kwargs):
            # sense_term: datetime range of filtering target
            # data_term: datetime range of returned values
            feature_def, input_sr, sense_term, data_term = task

            # insert nan into sr to make filtered results time-series
            interval = args[0]
            ind = pd.to_datetime(dtutil.range_dt(sense_term[0], sense_term[1],
                                                 interval))
            sr = input_sr.reindex(ind).astype(float)

            post_kwargs = {}
            post_kwargs.update(kwargs)
            post_kwargs.update(feature_def)

            # apply filtering functions
            from . import evpost
            for postfunc in feature_def["func_list"]:
                sr = eval("evpost.{0}".format(postfunc))(sr, **post_kwargs)
                if sr is None:
                    return None
            return sr[data_term[0]:data_term[1]]

        #input_kwargs = {"interval": self._feature_bin_size}
        self._mproc = mproc_queue.Manager(target=apply_postfunc,
                                          n_proc=self.parallel,
                                          args=[self._feature_bin_size])
                                          #kwargs=input_kwargs)

    def _close_mproc_manager(self):
        if self._mproc is not None:
            self._mproc.close()

    def _iter_feature_terms(self, feature_def, dt_range):
        # Avoid convolve boundary problem
        if "convolve" in feature_def["func_list"]:
            if "convolve_radius" in feature_def:
                convolve_radius = feature_def["convolve_radius"]
            else:
                # compatibility for configparser-style rule
                convolve_radius = self.conf.getint("general",
                                                   "evdb_convolve_radius")
            sense_offset = self._feature_bin_size * convolve_radius

        # datetimeindex.get_loc includes stop time (unlike other types!)
        # dtindex_offset remove the stop time
        dtindex_offset = self._feature_bin_size

        if "data_range" in feature_def:
            unit_term = config.str2dur(feature_def["data_range"])
            if "sense_range" in feature_def:
                unit_diff = config.str2dur(feature_def["sense_range"])
            else:
                unit_diff = unit_term
        else:
            # compatibility for configparser-style rule
            unit_term = config.getdur(self.conf, "general", "evdb_unit_term")
            unit_diff = config.getdur(self.conf, "general", "evdb_unit_diff")

        for dts, dte in dtutil.iter_term(dt_range, unit_diff):
            sense_dts = max(dt_range[0], dte - unit_term)
            yield ((dts, dte - dtindex_offset),
                   (sense_dts - sense_offset,
                    dte - dtindex_offset + sense_offset))

    def _calc_feature(self, df, feature_def, dt_range):

        #def _iter_feature_terms(input_dt_range, input_func_list):
        #    # Avoid convolve boundary problem
        #    if "convolve" in input_func_list:
        #        sense_offset = self._feature_bin_size * \
        #                       self._feature_convolve_radius
        #    else:
        #        sense_offset = dtutil.empty_timedelta()
        #    # datetimeindex.get_loc includes stop time (unlike other types!)
        #    # dtindex_offset remove the stop time
        #    dtindex_offset = self._feature_bin_size

        #    for dts, dte in dtutil.iter_term(input_dt_range,
        #                                     self._feature_unit_diff):
        #        sense_dts = max(input_dt_range[0],
        #                        dte - self._feature_unit_term)
        #        yield ((dts, dte - dtindex_offset),
        #               (sense_dts - sense_offset,
        #                dte - dtindex_offset + sense_offset))

        column = feature_def["column"]
        if column not in df.columns:
            return None
        if len(df[column].dropna()) == 0:
            return None

        _logger.debug("calculating feature {0}".format(feature_def))
        # mapped function: see apply_postfunc in self._init_mproc_manager
        l_task = []
        for data_term, sense_term in self._iter_feature_terms(
                feature_def, dt_range):
        #for data_term, sense_term in _iter_feature_terms(dt_range, func_list):
            sr = df.loc[sense_term[0]:sense_term[1], column]
            # as df includes NaN term, sr can be empty
            # for getnun func, processing is necessary in that case
            task = (feature_def, sr, sense_term, data_term)
            l_task.append(task)
            _logger.debug("task: {0}".format(sense_term))
        if len(l_task) == 0:
            return None
        self._mproc.add_from(l_task)
        self._mproc.join()
        l_sr = [self._mproc.get(True, 1) for _ in l_task]
        self._mproc.is_clean()
        _logger.debug("calculating feature {0} done".format(feature_def))

        all_sr = pd.concat([sr for sr in l_sr if sr is not None], axis=0)
        ret = pd.DataFrame(all_sr, dtype=int)
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

    def all_feature(self):
        return list({fdef["name"] for fdef in self._d_feature})

    def load_org(self, featurename, tags, dt_range):
        sourcename, _, _ = self._search_feature_source(featurename, tags)
        return self.load_orgdf(sourcename, tags, dt_range)

    def load_source(self, sourcename, seriesdef, dt_range):
        # for read_source
        if self._srcdb == "influx":
            tags = self._seriesdef2tags(seriesdef)
            ut_range = tuple(dt.timestamp() for dt in dt_range)
            return self.source.get_df(sourcename, tags, None, ut_range)
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

    def load_feature_count(self, measure, tags, dt_range):
        fields = self.fields
        ut_range = tuple(dt.timestamp() for dt in dt_range)
        return self.evdb.get_count(measure, tags, fields, ut_range)

    def iter_evdef(self, l_feature_name=None):
        for featuredef in self._d_feature:
            fname = featuredef["name"]
            if l_feature_name is not None and fname not in l_feature_name:
                continue

            sourcename = featuredef["source"]
            if sourcename in self._d_source:
                for seriesdef in self._d_source[sourcename]:
                    yield self._evdef_source(seriesdef, featuredef)
            elif sourcename in self._d_vsource:
                orgname = self._d_vsourcedef[sourcename]["src"]
                hosts = {seriesdef["host"] for seriesdef in self._d_source[orgname]}
                for host in hosts:
                    yield self._evdef_vsource(host, featuredef)

        #if l_feature_name is None:
        #    l_feature_name = list({fdef["name"] for fdef in self._d_feature})

        #for feature_name in l_feature_name:
        #    featuredef = self._d_feature[feature_name]
        #    sourcename = featuredef["source"]
        #    if sourcename in self._d_source:
        #        for sourcedef in self._d_source[sourcename]:
        #            yield self._evdef_source(sourcedef, featuredef)
        #    elif sourcename in self._d_vsource:
        #        orgname = self._d_vsourcedef[sourcename]["src"]
        #        hosts = {sourcedef["host"] for sourcedef in self._d_source[orgname]}
        #        for host in hosts:
        #            yield self._evdef_vsource(host, featuredef)

    @staticmethod
    def instruction(evdef):
        return str(evdef)

    def terminate(self):
        self._close_mproc_manager()


def survey_snmp_stats(el, dt_range):
    d_host = defaultdict(list)
    d_measure = defaultdict(list)
    for evdef in el.iter_evdef():
        measure, tags = evdef.series()
        cnt = el.load_feature_count(measure, tags, dt_range)
        if cnt is not None:
            d_host[evdef.host].append(cnt)
            d_measure[evdef.measure].append(cnt)

    return d_host, d_measure


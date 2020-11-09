#!/usr/bin/env python
# coding: utf-8

"""Old script for configparser-style rule.conf"""


import sys
import os
import re
import json
import configparser
from collections import defaultdict

from amulog import config
from amulog import common
from amulog import host_alias
from logdag.source import src_rrd


def parse_filename(filepath, conf):
    name = os.path.splitext(os.path.basename(filepath))[0]
    for mod_cls in conf.options("filepath_re"):
        reobj = re.compile(conf["filepath_re"][mod_cls])
        mobj = reobj.match(name)
        if mobj:
            return mobj.groupdict()["host"], mod_cls, mobj.groupdict()["id"]
    else:
        raise ValueError("RRD filename parsing failure ({0})".format(name))


def search_valid(conf, path, th=1.0):
    import rrdtool
    import numpy as np
    from amulog import common
    for fp in common.recur_dir(path):
        ut_range = [dt.timestamp() for dt in config.getterm(conf, "general", "whole_term")]
        try:
            robj = src_rrd.fetch(fp, ut_range)
        except IOError as e:
            sys.stderr(e)
        except rrdtool.OperationalError as e:
            pass
        else:
            nanratio = np.mean([int(np.isnan(v)) for v in robj.values.reshape(-1,)])
            if nanratio < th:
                yield fp
            else:
                pass


def generate_json(iter_fp, conf):
    ha = host_alias.HostAlias(conf["general"]["host_alias_filename"])

    sources = conf.options("source")
    d_group = defaultdict(list)
    for fp in iter_fp:
        try:
            tmp_host, mod_cls, mod_id = parse_filename(fp, conf)
            host = ha.resolve_host(tmp_host)
            if host is not None:
                d_group[(host, mod_cls, mod_id)].append(fp)
        except ValueError:
            pass

    d_source = defaultdict(list)
    for (host, mod_cls, mod_id), l_fp in d_group.items():
        for source_name in sources:
            if mod_cls in config.getlist(conf, "source", source_name):
                d_source[source_name].append({"filelist": l_fp,
                                              "host": host,
                                              "mod_cls": mod_cls,
                                              "mod_id": mod_id})

    # generate vsource: dict with src, func
    d_vsource = {}
    for vsource_name in conf.options("vsource"):
        src, func = config.getlist(conf, "vsource", vsource_name)
        d = {"src": src,
             "func": func}
        d_vsource[vsource_name] = d

    # generate features: dict of feature
    # feature: name, source, column, func_list
    d_feature = {}
    for feature_name in sorted(conf.options("feature")):
        tmp = config.getlist(conf, "feature", feature_name)
        sourcename = tmp[0]
        keyfunc = tmp[1]
        l_postfunc = tmp[2:]
        d = {"name": feature_name,
             "source": sourcename,
             "column": keyfunc,
             "func_list": l_postfunc}
        if "in" in keyfunc:
            d["direction"] = "in"
            d["group"] = "interface"
        elif "out" in keyfunc:
            d["direction"] = "out"
            d["group"] = "interface"
        else:
            d["group"] = "system"
        d_feature[feature_name] = d

    js = {"source": d_source,
          "vsource": d_vsource,
          "feature": d_feature}
    return js


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("give me mode")
    mode = sys.argv[1]

    if mode == "filename2json":
        if len(sys.argv) < 3:
            sys.exit("usage: {0} {1} CONFIG".format(sys.argv[0], mode))
        conf = configparser.ConfigParser()
        assert os.path.exists(sys.argv[2])
        conf.read(sys.argv[2])

        def _iter_input_line():
            while True:
                try:
                    yield input()
                except EOFError:
                    break

        js = generate_json(_iter_input_line(), conf)
        json.dump(js, sys.stdout, **common.json_args)

    elif mode == "json":
        if len(sys.argv) < 4:
            sys.exit("usage: {0} {1} CONFIG PATH <THRESHOLD>".format(sys.argv[0],
                                                                     mode))
        elif len(sys.argv) == 4:
            d = {}
        else:
            d = {"th": sys.argv[4]}
        conf = configparser.ConfigParser()
        conf.read(sys.argv[2])
        search_path = sys.argv[3]

        js = generate_json(search_valid(conf, search_path, **d), conf)
        json.dump(js, sys.stdout, **common.json_args)

    elif mode == "search_valid":
        if len(sys.argv) < 4:
            sys.exit("usage: {0} {1} CONFIG PATH <THRESHOLD>".format(sys.argv[0],
                                                                     mode))
        elif len(sys.argv) == 4:
            d = {}
        else:
            d = {"th": sys.argv[4]}
        conf = configparser.ConfigParser()
        conf.read(sys.argv[2])
        search_path = sys.argv[3]
        for fp in search_valid(conf, search_path, **d):
            print(fp)
    else:
        raise ValueError("invalid mode {0}".format(mode))


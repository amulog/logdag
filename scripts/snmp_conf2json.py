#!/usr/bin/env python
# coding: utf-8

import sys
import json
import configparser

from amulog import common
from amulog import config
from amulog import host_alias


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: {0} CONF_FILE".format(sys.argv[0]))
    conf = configparser.ConfigParser()
    conf.read(sys.argv[1])

    root = {}
    amulog_conf = config.open_config(conf["database_amulog"]["source_conf"])
    ha = host_alias.init_hostalias(amulog_conf)

    l_source = []
    for srcname in config.getlist(conf, "snmp_source", "all"):
        fp = conf["snmp_source"][srcname]
        d = {"name": srcname,
             "filelist": fp}
        l_source.append(d)
    root["source"] = l_source

    d_vsource = {}
    for vsrcname in config.getlist(conf, "snmp_vsource", "all"):
        src, func = config.getlist(conf, "snmp_vsource", vsrcname)
        d = {"src": src,
             "func": func}
        d_vsource[vsrcname] = d
    root["vsource"] = d_vsource

    d_feature = {}
    for featurename in sorted(conf.options("snmp_feature")):
        tmp = config.getlist(conf, "snmp_feature", featurename)
        sourcename = tmp[0]
        keyfunc = tmp[1]
        l_postfunc = tmp[2:]
        d = {"name": featurename,
             "src": sourcename,
             "column": keyfunc,
             "func_list": l_postfunc}
        if "in" in featurename or "out" in featurename:
            d["group"] = "interface"
        else:
            d["group"] = "system"
        d_feature[featurename] = d
    root["feature"] = d_feature

    json.dump(root, sys.stdout, **common.json_args)

#!/usr/bin/env python
# coding: utf-8

import sys
import os
import logging
import argparse
import configparser

from amulog import config
from amulog import common
from logdag import dtutil

_logger = logging.getLogger(__package__)

DEFAULT_CONFIG = "/".join((os.path.dirname(__file__),
                           "data/loader.conf.default"))


def open_logdag_config(ns):
    from logdag import arguments
    return arguments.open_logdag_config(ns.conf_path, debug=ns.debug)


def _whole_term(conf):
    return config.getterm(conf, "general", "evdb_whole_term")


def _iter_evdb_term(conf):
    w_term = config.getterm(conf, "general", "evdb_whole_term")
    term = config.getdur(conf, "general", "evdb_unit_diff")
    return dtutil.iter_term(w_term, term)


def make_evdb_log_all(ns):
    conf = open_logdag_config(ns)
    dump_org = ns.org
    dry = ns.dry

    from . import evgen_log
    el = evgen_log.LogEventLoader(conf, dry=dry)
    for dt_range in _iter_evdb_term(conf):
        el.read(dt_range, dump_org=dump_org)


def make_evdb_snmp_all(ns):
    conf = open_logdag_config(ns)
    dump_org = ns.org
    dry = ns.dry
    parallel = ns.parallel

    from . import evgen_snmp
    el = evgen_snmp.SNMPEventLoader(conf, parallel=parallel, dry=dry)
    try:
        el.store_all(_whole_term(conf), dump_org=dump_org)
    except KeyboardInterrupt as e:
        el.terminate()


def make_evdb_snmp(ns):
    conf = open_logdag_config(ns)
    dump_org = ns.org
    dry = ns.dry
    parallel = ns.parallel
    feature_name = ns.feature_name

    from . import evgen_snmp
    el = evgen_snmp.SNMPEventLoader(conf, parallel=parallel, dry=dry)
    try:
        el.store_feature(feature_name, _whole_term(conf), dump_org=dump_org)
    except KeyboardInterrupt as e:
        el.terminate()


def make_evdb_snmp_org(ns):
    conf = open_logdag_config(ns)
    dump_vsource_org = ns.org
    dry = ns.dry
    parallel = ns.parallel

    from . import evgen_snmp
    el = evgen_snmp.SNMPEventLoader(conf, parallel=parallel, dry=dry)
    try:
        el.store_all_source(_whole_term(conf), dump_vsource_org)
    except KeyboardInterrupt as e:
        el.terminate()


def drop_features(ns):
    conf = open_logdag_config(ns)
    sources = ns.sources
    if sources is None:
        from . import evgen_common
        sources = evgen_common.source

    from logdag import log2event
    for src in sources:
        el = log2event.init_evloader(conf, src)
        el.drop_features()


# common argument settings
OPT_DEBUG = [["--debug"],
             {"dest": "debug", "action": "store_true",
              "help": "set logging level to debug (default: info)"}]
OPT_CONFIG = [["-c", "--config"],
              {"dest": "conf_path", "metavar": "CONFIG", "action": "store",
               "default": None,
               "help": "configuration file path for amulog"}]
OPT_PARALLEL = [["-p", "--parallel"],
                {"dest": "parallel", "metavar": "PARALLEL", "action": "store",
                 "default": 1, "type": int,
                 "help": "parallel processing for calculating features"}]
OPT_ORG = [["-o", "--org"],
           {"dest": "org", "action": "store_true",
            "help": "output original data to evdb"}]
OPT_DRY = [["-d", "--dry"],
           {"dest": "dry", "action": "store_true",
            "help": "do not write down to db (dry-run)"}]
ARG_ARGNAME = [["argname"],
               {"metavar": "TASKNAME", "action": "store",
                "help": "argument name"}]

# argument settings for each modes
# description, List[args, kwargs], func
# defined after functions because these settings use functions
DICT_ARGSET = {
    "make-evdb-log-all": ["Load log data from amulog and store features",
                          [OPT_CONFIG, OPT_DEBUG, OPT_ORG, OPT_DRY],
                          make_evdb_log_all],
    "make-evdb-snmp-all": ["Load telemetry data from rrd and store features",
                           [OPT_CONFIG, OPT_DEBUG, OPT_ORG, OPT_DRY, OPT_PARALLEL,],
                           make_evdb_snmp_all],
    "make-evdb-snmp": ["Load telemetry data from rrd and store features",
                       [OPT_CONFIG, OPT_DEBUG, OPT_ORG, OPT_DRY, OPT_PARALLEL,
                        [["feature_name"],
                         {"metavar": "FEATURE", "action": "store",
                          "help": "feature name"}]],
                       make_evdb_snmp],
    "make-evdb-snmp-org": ["Load telemetry data from rrd and store",
                           [OPT_CONFIG, OPT_DEBUG, OPT_ORG, OPT_DRY, OPT_PARALLEL],
                           make_evdb_snmp_org],
    "drop-features": ["Drop feature data (except original data) in feature DB",
                      [OPT_CONFIG, OPT_DEBUG, OPT_ORG,
                       [["sources"],
                        {"metavar": "DATA_SOURCES", "action": "store",
                         "nargs": "+",
                         "help": "source names (like log, snmp)"}]],
                      drop_features],
}

USAGE_COMMANDS = "\n".join(["  {0}: {1}".format(key, val[0])
                            for key, val in sorted(DICT_ARGSET.items())])
USAGE = ("usage: {0} MODE [options and arguments] ...\n\n"
         "mode:\n".format(sys.argv[0])) + USAGE_COMMANDS + \
        "\n\nsee \"{0} MODE -h\" to refer detailed usage".format(sys.argv[0])

if __name__ == "__main__":
    if len(sys.argv) < 1:
        sys.exit(USAGE)
    mode = sys.argv[1]
    if mode in ("-h", "--help"):
        sys.exit(USAGE)
    commandline = sys.argv[2:]

    desc, l_argset, func = DICT_ARGSET[mode]
    ap = argparse.ArgumentParser(prog=" ".join(sys.argv[0:2]),
                                 description=desc)
    for args, kwargs in l_argset:
        ap.add_argument(*args, **kwargs)
    command_ns = ap.parse_args(commandline)
    func(command_ns)

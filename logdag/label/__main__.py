#!/usr/bin/env python
# coding: utf-8

import sys
import logging
import argparse

from logdag import arguments
from logdag import dtutil
from logdag import showdag
from amulog import config
from amulog import common

_logger = logging.getLogger(__package__)


def count_ts_label(ns):
    conf = arguments.open_logdag_config(ns)

    from . import tslabel
    d_cnt = tslabel.count_ts_label(conf, agg_group = True)
    for k, v in sorted(d_cnt.items(), key = lambda x: x[0]):
        print("{0} {1}".format(k, v))
    print("Total: {0}".format(sum(d_cnt.values())))


def count_event_label(ns):
    conf = arguments.open_logdag_config(ns)

    from . import tslabel
    d_cnt = tslabel.count_event_label(conf, agg_group = True)
    for k, v in sorted(d_cnt.items(), key = lambda x: x[0]):
        print("{0} {1}".format(k, v))
    print("Total: {0}".format(sum(d_cnt.values())))


def count_node_label(ns):
    conf = arguments.open_logdag_config(ns)

    from . import edgelabel
    d_cnt = edgelabel.count_node_label(conf)
    for k, v in sorted(d_cnt.items(), key = lambda x: x[0]):
        print("{0} {1}".format(k, v))
    print("Total: {0}".format(sum(d_cnt.values())))


def count_edge_label(ns):
    conf = arguments.open_logdag_config(ns)

    from . import edgelabel
    d_cnt = edgelabel.count_edge_label(conf)
    for k, v in sorted(d_cnt.items(), key = lambda x: x[0]):
        print("{0} {1}".format(k, v))
    print("Total: {0}".format(sum(d_cnt.values())))


# common argument settings
OPT_DEBUG = [["--debug"],
             {"dest": "debug", "action": "store_true",
              "help": "set logging level to debug (default: info)"}]
OPT_CONFIG = [["-c", "--config"],
              {"dest": "conf_path", "metavar": "CONFIG", "action": "store",
               "default": None,
               "help": "configuration file path for amulog"}]

# argument settings for each modes
# description, List[args, kwargs], func
# defined after functions because these settings use functions
DICT_ARGSET = {
    "count-ts-label": ["Count all time-series corresponding to labels",
                       [OPT_CONFIG, OPT_DEBUG],
                       count_ts_label],
    "count-event-label": ["Count all events corresponding to labels",
                          [OPT_CONFIG, OPT_DEBUG],
                          count_event_label],
    "count-node-label": ["Count all node events corresponding to labels",
                         [OPT_CONFIG, OPT_DEBUG],
                         count_node_label],
    "count-edge-label": ["Count node labels of the ends of edges",
                         [OPT_CONFIG, OPT_DEBUG],
                         count_edge_label],
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
    ap = argparse.ArgumentParser(prog = " ".join(sys.argv[0:2]),
                                 description = desc)
    for args, kwargs in l_argset:
        ap.add_argument(*args, **kwargs)
    ns = ap.parse_args(commandline)
    func(ns)



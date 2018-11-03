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


def search_gid(ns):
    conf = arguments.open_logdag_config(ns)
    gid = ns.gid

    from . import edge_search
    l_result = edge_search.search_gid(conf, gid)
    for r, edge in l_result:
        print("{0} {1}".format(r.name, r.edge_str(edge)))


def draw_graph_common(ns):
    l_conffp = ns.confs
    assert len(l_conffp) == 2
    openconf = lambda c: config.open_config(
        c, ex_defaults = [arguments.DEFAULT_CONFIG])
    conf1, conf2 = [openconf(c) for c in l_conffp]
    lv = logging.DEBUG if ns.debug else logging.INFO
    am_logger = logging.getLogger("amulog")
    config.set_common_logging(conf1, logger = [_logger, am_logger], lv = lv)

    dts = dtutil.shortstr2dt(ns.timestr)
    dte = dts + config.getdur(conf1, "dag", "unit_term")
    output = ns.filename

    from . import comp_conf
    cevmap, cgraph = comp_conf.edge_set_common(conf1, conf2, (dts, dte))

    from . import draw
    rgraph = draw.relabel_graph(conf1, cgraph, cevmap)
    draw.graph_nx(output, rgraph)
    print(output)


def draw_graph_diff(ns):
    l_conffp = ns.confs
    assert len(l_conffp) == 2
    openconf = lambda c: config.open_config(
        c, ex_defaults = [arguments.DEFAULT_CONFIG])
    conf1, conf2 = [openconf(c) for c in l_conffp]
    lv = logging.DEBUG if ns.debug else logging.INFO
    am_logger = logging.getLogger("amulog")
    config.set_common_logging(conf1, logger = [_logger, am_logger], lv = lv)

    dts = dtutil.shortstr2dt(ns.timestr)
    dte = dts + config.getdur(conf1, "dag", "unit_term")
    output = ns.filename

    from . import comp_conf
    cevmap, cgraph = comp_conf.edge_set_diff(conf1, conf2, (dts, dte))

    from . import draw
    rgraph = draw.relabel_graph(conf1, cgraph, cevmap)
    draw.graph_nx(output, rgraph)
    print(output)


def show_graph_diff_lts(ns):
    l_conffp = ns.confs
    assert len(l_conffp) == 2
    openconf = lambda c: config.open_config(
        c, ex_defaults = [arguments.DEFAULT_CONFIG])
    conf1, conf2 = [openconf(c) for c in l_conffp]
    lv = logging.DEBUG if ns.debug else logging.INFO
    am_logger = logging.getLogger("amulog")
    config.set_common_logging(conf1, logger = [_logger, am_logger], lv = lv)

    from amulog import log_db
    ld = log_db.LogData(conf1)

    from . import comp_conf
    d_ltid = comp_conf.edge_diff_gid(conf1, conf2)
    for ltid, l_name in sorted(d_ltid.items(), key = lambda x: len(x[1]),
                               reverse = True):
        print("{0}: {1} ({2})".format(len(l_name), ltid, ld.lt(ltid)))
        if len(l_name) < 100:
            print(l_name)


def show_graph_diff_search(ns):
    l_conffp = ns.confs
    assert len(l_conffp) == 2
    openconf = lambda c: config.open_config(
        c, ex_defaults = [arguments.DEFAULT_CONFIG])
    conf1, conf2 = [openconf(c) for c in l_conffp]
    lv = logging.DEBUG if ns.debug else logging.INFO
    am_logger = logging.getLogger("amulog")
    config.set_common_logging(conf1, logger = [_logger, am_logger], lv = lv)

    gid = ns.gid
    from . import comp_conf
    comp_conf.edge_diff_gid_search(conf1, conf2, gid)


# common argument settings
OPT_DEBUG = [["--debug"],
             {"dest": "debug", "action": "store_true",
              "help": "set logging level to debug (default: info)"}]
OPT_CONFIG = [["-c", "--config"],
              {"dest": "conf_path", "metavar": "CONFIG", "action": "store",
               "default": None,
               "help": "configuration file path for amulog"}]
OPT_PARALLEL = [["-p", "--parallel"],
                {"dest": "parallel", "metavar": "PARALLEL",
                 "type": int, "default": 1,
                 "help": "number of processes in parallel"}]
OPT_FILENAME = [["-f", "--filename"],
                {"dest": "filename", "metavar": "FILENAME", "action": "store",
                 "default": "output",
                 "help": "output filename"}]
OPT_DIRNAME = [["-d", "--dirname"],
               {"dest": "dirname", "metavar": "DIRNAME", "action": "store",
                "default": ".",
                "help": "directory name for output"}]
OPT_GID = [["-g", "--gid"],
           {"dest": "gid", "metavar": "GID", "action": "store",
            "type": int, "default": None,
            "help": "log group identifier to search events"},]
OPT_HOSTNAME = [["-n", "--host"],
                {"dest": "host", "metavar": "HOST", "action": "store",
                 "default": None,
                 "help": "hostname to search events"}]
OPT_BINSIZE = [["-b", "--binsize"],
               {"dest": "binsize", "metavar": "BINSIZE",
                "action": "store", "default": None,
                "help": "binsize (like 10s)"}]
OPT_IGORPHAN = [["-i", "--ignore-orphan"],
                {"dest": "ignore_orphan", "action": "store_true",
                 "help": "ignore orphan nodes (without any adjacents)"}]
ARG_TIMESTR = [["timestr"],
               {"metavar": "TIMESTR", "action": "store",
                "help": "%%Y%%m%%d(_%%H%%M%%S) style time string"}]
ARG_ARGNAME = [["argname"],
               {"metavar": "TASKNAME", "action": "store",
                "help": "argument name"}]
ARG_DBSEARCH = [["conditions"],
                {"metavar": "CONDITION", "nargs": "+",
                 "help": ("Conditions to search log messages. "
                          "Example: command gid=24 date=2012-10-10 ..., "
                          "Keys: gid, date, top_date, end_date, "
                          "host, area")}]

# argument settings for each modes
# description, List[args, kwargs], func
# defined after functions because these settings use functions
DICT_ARGSET = {
    "search-gid": ["List DAGs with edges related to given gid",
                   [OPT_CONFIG, OPT_DEBUG,
                    [["gid"],
                     {"metavar": "GID", "action": "store", "type": int,
                      "help": "gid to search"}],],
                   search_gid],
    "draw-graph-common": ["Draw common edges of 2 DAG sets",
                          [OPT_DEBUG, OPT_FILENAME,
                           [["confs"],
                            {"metavar": "CONFIG", "nargs": 2,
                             "help": "2 config file path"}],
                           ARG_TIMESTR,],
                          draw_graph_common],
    "draw-graph-diff": ["Draw contrasting edges of 2 DAG sets",
                          [OPT_DEBUG, OPT_FILENAME,
                           [["confs"],
                            {"metavar": "CONFIG", "nargs": 2,
                             "help": "2 config file path"}],
                           ARG_TIMESTR,],
                          draw_graph_diff],
    "show-graph-diff-lts": ["List ltids found in diff graph of 2 DAG sets",
                            [OPT_DEBUG,
                             [["confs"],
                              {"metavar": "CONFIG", "nargs": 2,
                               "help": "2 config file path"}],
                             ],
                            show_graph_diff_lts],
    "show-graph-diff-search": ["Search diff graphs with given gid",
                               [OPT_DEBUG,
                                [["confs"],
                                 {"metavar": "CONFIG", "nargs": 2,
                                  "help": "2 config file path"}],
                                [["gid"],
                                 {"metavar": "GID", "action": "store",
                                  "type": int,
                                  "help": "gid to search"}],],
                               show_graph_diff_search],
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



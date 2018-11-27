#!/usr/bin/env python
# coding: utf-8

import sys
import logging
import argparse
from collections import defaultdict

from amulog import common
from amulog import config
from logdag import arguments

_logger = logging.getLogger(__package__)


def add_trouble(ns):
    conf = arguments.open_logdag_config(ns)

    from . import trouble
    dirname = conf.get("eval", "path")
    tm = trouble.TroubleManager(dirname)

    tr = tm.add(ns.date, ns.group, ns.title)
    print(tr.tid)


def add_lids(ns):
    conf = arguments.open_logdag_config(ns)

    from . import trouble
    dirname = conf.get("eval", "path")
    tm = trouble.TroubleManager(dirname)

    tm.add_lids(ns.tid, ns.lids)


def add_lids_stdin(ns):
    conf = arguments.open_logdag_config(ns)

    from . import trouble
    dirname = conf.get("eval", "path")
    tm = trouble.TroubleManager(dirname)

    lids = [int(v) for v in input()]
    tm.add_lids(ns.tid, lids)


def label_trouble(ns):
    conf = arguments.open_logdag_config(ns)

    from . import trouble
    dirname = conf.get("eval", "path")
    tm = trouble.TroubleManager(dirname)

    tm.update(ns.tid, group = ns.group)


def list_trouble(ns):
    conf = arguments.open_logdag_config(ns)

    from . import trouble
    dirname = conf.get("eval", "path")
    tm = trouble.TroubleManager(dirname)

    for tr in tm:
        print(tr)


def list_group(ns):
    conf = arguments.open_logdag_config(ns)
    from . import trouble
    dirname = conf.get("eval", "path")
    tm = trouble.TroubleManager(dirname)

    d = defaultdict(list)
    for tr in tm:
        group = tr.data["group"]
        d[group].append(tr)

    table = [["group", "ticket", "messages"]]
    num_sum = 0
    for group, l_tr in d.items():
        num = sum([len(tr.data["message"]) for tr in l_tr])
        #l_buf.append("{0}: {1} tickets ({2} messages)".format(
        #    group, len(l_tr), num))
        table.append([group, len(l_tr), num])
        num_sum += num
    table.append(["total", sum([len(v) for v in d.values()]), num_sum])
    print(common.cli_table(table))


def show_lids(ns):
    conf = arguments.open_logdag_config(ns)
    tid = ns.tid

    from . import trouble
    dirname = conf.get("eval", "path")
    tm = trouble.TroubleManager(dirname)

    tr = tm[tid]
    print(tr)
    print("\n".join(tr.get(ld)))


def show_trouble(ns):
    conf = arguments.open_logdag_config(ns)
    tid = ns.tid

    from . import trouble
    dirname = conf.get("eval", "path")
    tm = trouble.TroubleManager(dirname)
    from amulog import log_db
    ld = log_db.LogData(conf)

    tr = tm[tid]
    print(tr)
    print("\n".join(tr.get_message(ld, show_lid = ns.lid_header)))


def list_trouble_label(ns):
    conf = arguments.open_logdag_config(ns)
    from . import trouble
    dirname = conf.get("eval", "path")
    tm = trouble.TroubleManager(dirname)
    from amulog import log_db
    ld = log_db.LogData(conf)
    from amulog import lt_label
    ll = lt_label.init_ltlabel(conf)
    gid_name = conf.get("dag", "event_gid")

    for tr in tm:
        d_gid = defaultdict(int)
        for lid in tr.get():
            lm = ld.get_line(lid)
            gid = lm.lt.get(gid_name)
            d_gid[gid] += 1

        d = defaultdict(list)
        for gid in d_gid.keys():
            label = ll.get_ltg_label(gid, ld.ltg_members(gid))
            group = ll.get_group(label)
            d[group].append(gid)

        buf = "{0} ({1}): ".format(tr.tid, tr.data["group"])
        for group, l_gid in sorted(d.items(), key = lambda x: len(x[1]),
                                   reverse = True):
            num = sum([d_gid[gid] for gid in d[group]])
            buf += "{0}({1},{2}) ".format(group, len(l_gid), num)
        print(buf)


def show_match_dag(ns):
    conf = arguments.open_logdag_config(ns)
    from . import trouble
    dirname = conf.get("eval", "path")
    tm = trouble.TroubleManager(dirname)

    from logdag import showdag
    from . import match_edge
    tr = tm[ns.tid]
    d_args = match_edge.match_edges(conf, tr, rule = ns.rule)
    cnt = sum([len(l_edge) for l_edge in d_args.values()])
    print("{0[date]} ({0[group]}): {1}".format(tr.data, cnt))
    for name, l_edge in d_args.items():
        r = showdag.LogDAG(arguments.name2args(name, conf))
        r.load()
        for edge in l_edge:
            print(r.edge_str(edge, graph = r.graph.to_undirected()))


def show_match_all(ns):
    conf = arguments.open_logdag_config(ns)
    from . import trouble
    dirname = conf.get("eval", "path")
    tm = trouble.TroubleManager(dirname)

    from . import match_edge
    for tr in tm:
        d_args = match_edge.match_edges(conf, tr, rule = ns.rule)
        cnt = sum([len(l_edge) for l_edge in d_args.values()])
        print("Trouble {0.tid} {0.data[date]} ({0.data[group]}): {1}".format(
            tr, cnt))


# common argument settings
OPT_DEBUG = [["--debug"],
             {"dest": "debug", "action": "store_true",
              "help": "set logging level to debug (default: info)"}]
OPT_CONFIG = [["-c", "--config"],
              {"dest": "conf_path", "metavar": "CONFIG", "action": "store",
               "default": None,
               "help": "configuration file path for amulog"}]
OPT_LID = [["-l", "--lid"],
           {"dest": "lid_header", "action": "store_true",
            "help": "parse lid from head part of log message"}]
OPT_RULE = [["-r", "--rule"],
            {"dest": "rule", "action": "store",
             "type": str, "default": "all",
             "help": "one of [all, either, both]"}]
ARG_TID = [["tid"],
           {"metavar": "TID", "action": "store", "type": int,
            "help": "trouble identifier"}]
ARG_DATE = [["date"],
            {"metavar": "DATE", "action": "store",
             "help": "%%Y%%m%%d style date string"}]
ARG_GROUP = [["group"],
             {"metavar": "GROUP", "action": "store",
              "help": "class of the trouble"}]
ARG_TITLE = [["title"],
             {"metavar": "TITLE", "action": "store",
              "help": "recorded title of the trouble"}]
ARG_MESSAGES = [["lids"],
                {"metavar": "MESSAGE_IDS", "action": "store",
                 "type": int, "nargs": "+",
                 "help": "message IDs (lid in amulog db)"}]

# argument settings for each modes
# description, List[args, kwargs], func
# defined after functions because these settings use functions
DICT_ARGSET = {
    "add-trouble": ["Add a new trouble definition",
                    [OPT_CONFIG, OPT_DEBUG, ARG_DATE, ARG_GROUP, ARG_TITLE],
                    add_trouble],
    "add-lids": ["Add messages to a trouble",
                 [OPT_CONFIG, OPT_DEBUG, ARG_TID, ARG_MESSAGES],
                 add_lids],
    "add-lids-stdin": ["Add messages to a trouble from stdin",
                       [OPT_CONFIG, OPT_DEBUG, ARG_TID],
                       add_lids_stdin],
    "label-trouble": ["Set label to a trouble",
                      [OPT_CONFIG, OPT_DEBUG, ARG_TID, ARG_GROUP],
                      label_trouble],
    "list-trouble": ["List all troubles",
                     [OPT_CONFIG, OPT_DEBUG],
                     list_trouble],
    "list-group": ["List group information",
                   [OPT_CONFIG, OPT_DEBUG],
                   list_group],
    "list-trouble-label": ["Show corresponding label groups of tickets",
                           [OPT_CONFIG, OPT_DEBUG],
                           list_trouble_label],
    "show-lids": ["Show all lids in the given trouble",
                  [OPT_CONFIG, OPT_DEBUG, ARG_TID],
                  show_lids],
    "show-trouble": ["Show all messages corresponding to the given trouble",
                     [OPT_CONFIG, OPT_DEBUG, OPT_LID, ARG_TID],
                     show_trouble],
    "show-match-dag": ["Show matching edges in a DAG",
                       [OPT_CONFIG, OPT_DEBUG, OPT_RULE, ARG_TID],
                       show_match_dag],
    "show-match-all": ["Show matching edges in all DAG",
                       [OPT_CONFIG, OPT_DEBUG, OPT_RULE],
                       show_match_all],
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



#!/usr/bin/env python
# coding: utf-8

import sys
import logging
import argparse

from . import arguments
from amulog import config
from amulog import common

_logger = logging.getLogger(__package__)
SUBLIB = ["source", "visual", "eval", "label"]


def open_logdag_config(ns):
    from . import arguments
    return arguments.open_logdag_config(ns.conf_path, debug=ns.debug)


def test_makedag(ns):
    from . import makedag
    conf = open_logdag_config(ns)

    am = arguments.ArgumentManager(conf)
    am.generate(arguments.all_args)
    am.dump()
    makedag.makedag_main(am[0])


def make_tsdb(ns):
    from . import tsdb
    conf = open_logdag_config(ns)
    term = config.getdur(conf, "database_ts", "unit_term")
    diff = config.getdur(conf, "database_ts", "unit_diff")
    l_args = arguments.all_terms(conf, term, diff)

    timer = common.Timer("mk-tsdb task", output=_logger)
    timer.start()
    p = ns.parallel
    if p > 1:
        for args in l_args:
            tsdb.log2ts_pal(*args, pal=p)
    else:
        for args in l_args:
            tsdb.log2ts(*args)
    timer.stop()


def reload_area(ns):
    from . import tsdb
    conf = open_logdag_config(ns)
    tsdb.reload_area(conf)


def make_args(ns):
    conf = open_logdag_config(ns)

    am = arguments.ArgumentManager(conf)
    am.generate(arguments.all_args)
    am.init_dirs(conf)
    am.dump()


def make_dag(ns):
    from . import makedag

    def makedag_sprocess(am):
        timer = common.Timer("makedag task", output=_logger)
        timer.start()
        for args in am:
            makedag.makedag_main(args)
        timer.stop()

    def makedag_mprocess(am, pal=1):
        import multiprocessing
        timer = common.Timer("makedag task", output=_logger)
        timer.start()
        with multiprocessing.Pool(processes=pal) as pool:
            pool.map(makedag.makedag_main, am)
        timer.stop()

    conf = open_logdag_config(ns)

    am = arguments.ArgumentManager(conf)
    am.generate(arguments.all_args)
    am.init_dirs(conf)
    am.dump()

    p = ns.parallel
    if p > 1:
        makedag_mprocess(am, p)
    else:
        makedag_sprocess(am)


def make_dag_stdin(ns):
    from . import makedag

    conf = open_logdag_config(ns)

    am = arguments.ArgumentManager(conf)
    am.init_dirs(conf)
    args = am.jobname2args(ns.argname, conf)

    timer = common.Timer("makedag task for {0}".format(ns.argname),
                         output=_logger)
    timer.start()
    makedag.makedag_main(args)
    timer.stop()


# def show_event(ns):
#    from . import tsdb
#    conf = open_logdag_config(ns)
#    d = parse_condition(ns.conditions)
#    print(tsdb.show_event(conf, **d))
#
#
# def show_ts(ns):
#    from . import tsdb
#    conf = open_logdag_config(ns)
#    d = parse_condition(ns.conditions)
#    print(tsdb.show_ts(conf, **d))
#
#
# def show_ts_compare(ns):
#    from . import tsdb
#    conf = open_logdag_config(ns)
#    d = parse_condition(ns.conditions)
#    print(tsdb.show_ts_compare(conf, **d))


def make_dag_prune(ns):
    from . import makedag

    conf = open_logdag_config(ns)

    am = arguments.ArgumentManager(conf)
    am.init_dirs(conf)
    args = am.jobname2args(ns.argname, conf)

    timer = common.Timer("makedag_prune task for {0}".format(ns.argname),
                         output=_logger)
    timer.start()
    makedag.makedag_prune_test(args)
    timer.stop()


# def show_filterlog(ns):
#    from . import tsdb
#    conf = open_logdag_config(ns)
#    d = parse_condition(ns.conditions)
#    print(tsdb.show_filterlog(conf, **d))


def show_args(ns):
    conf = open_logdag_config(ns)

    am = arguments.ArgumentManager(conf)
    try:
        am.load()
    except IOError:
        path = am.args_filename
        sys.exit("ArgumentManager object file ({0}) not found".format(path))

    print(am.show())


def show_edge_list(ns):
    from . import showdag
    conf = open_logdag_config(ns)
    args = arguments.name2args(ns.argname, conf)

    l_buf = []
    r = showdag.LogDAG(args)
    r.load()
    g = showdag.apply_filter(r, ns.filters, th=ns.threshold)
    for edge in g.edges():
        l_buf.append(r.edge_str(edge))
    print("\n".join(l_buf))


def show_edge_detail(ns):
    from . import showdag
    conf = open_logdag_config(ns)
    args = arguments.name2args(ns.argname, conf)
    head = ns.head
    tail = ns.tail

    print(showdag.show_edge_detail(args, head, tail))


def show_list(ns):
    from . import showdag
    conf = open_logdag_config(ns)

    l_func = [lambda r: r.graph.number_of_nodes(),
              lambda r: r.graph.number_of_edges()]
    table = []
    for key, _, data in showdag.stat_groupby(conf, l_func, groupby=ns.groupby):
        table.append([key] + list(data))
    print(common.cli_table(table))


def show_stats(ns):
    import numpy as np
    from . import showdag
    conf = open_logdag_config(ns)

    msg = ["number of events (nodes)",
           "number of directed edges",
           "number of directed edges across hosts",
           "number of undirected edges",
           "number of undirected edges across hosts",
           "number of all edges"]
    l_func = [
        lambda r: r.graph.number_of_nodes(),
        lambda r: showdag.apply_filter(r, ["directed"]).number_of_edges(),
        lambda r: showdag.apply_filter(r, ["directed", "across_host"]).number_of_edges(),
        lambda r: showdag.apply_filter(r, ["undirected"]).number_of_edges(),
        lambda r: showdag.apply_filter(r, ["undirected", "across_host"]).number_of_edges(),
        lambda r: r.graph.number_of_edges()
    ]
    data = [v for _, _, v in showdag.stat_groupby(conf, l_func)]
    agg_data = np.sum(data, axis=0)
    print(common.cli_table(list(zip(msg, agg_data)), align="right"))


def show_netsize(ns):
    from . import showdag
    conf = open_logdag_config(ns)

    print(showdag.show_netsize_dist(conf))


def show_netsize_list(ns):
    from . import showdag
    conf = open_logdag_config(ns)

    print(showdag.list_netsize(conf))


def plot_dag(ns):
    from . import showdag
    # from . import showdag_filter
    conf = open_logdag_config(ns)

    args = arguments.name2args(ns.argname, conf)
    output = ns.filename

    r = showdag.LogDAG(args)
    r.load()
    # g = showdag_filter.apply(r, ns.filters, th=ns.threshold)
    g = showdag.apply_filter(r, ns.filters, th=ns.threshold)
    g = r.relabel(graph=g)
    r.graph_nx(output, graph=g)
    print(output)


# def parse_condition(conditions):
#    """
#    Args:
#        conditions (list)
#    """
#    import datetime
#    d = {}
#    for arg in conditions:
#        if not "=" in arg:
#            raise SyntaxError
#        key = arg.partition("=")[0]
#        if key == "gid":
#            d["gid"] = int(arg.partition("=")[-1])
#        elif key == "top_date":
#            date_string = arg.partition("=")[-1]
#            d["dts"] = datetime.datetime.strptime(date_string, "%Y-%m-%d")
#        elif key == "end_date":
#            date_string = arg.partition("=")[-1]
#            d["dte"] = datetime.datetime.strptime(date_string, "%Y-%m-%d")
#        elif key == "date":
#            date_string = arg.partition("=")[-1]
#            d["dts"] = datetime.datetime.strptime(date_string, "%Y-%m-%d")
#            d["dte"] = d["dts"] + datetime.timedelta(days=1)
#        elif key == "host":
#            d["host"] = arg.partition("=")[-1]
#        elif key == "area":
#            d["area"] = arg.partition("=")[-1]
#        else:
#            d[key] = arg.partition("=")[-1]
#    return d


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
OPT_THRESHOLD = [["-t", "--threshold"],
                 {"dest": "threshold", "metavar": "THRESHOLD", "action": "store",
                  "type": float, "default": None,
                  "help": "threshold for filter ate_prune"}]
OPT_GID = [["-g", "--gid"],
           {"dest": "gid", "metavar": "GID", "action": "store",
            "type": int, "default": None,
            "help": "log group identifier to search events"}, ]
OPT_HOSTNAME = [["-n", "--host"],
                {"dest": "host", "metavar": "HOST", "action": "store",
                 "default": None,
                 "help": "hostname to search events"}]
OPT_BINSIZE = [["-b", "--binsize"],
               {"dest": "binsize", "metavar": "BINSIZE",
                "action": "store", "default": None,
                "help": "binsize (like 10s)"}]
OPT_GROUPBY = [["--groupby"],
               {"dest": "groupby", "metavar": "GROUPBY",
                "action": "store", "default": None,
                "help": "aggregate results by given metrics (like day, area)"}]

ARG_ARGNAME = [["argname"],
               {"metavar": "TASKNAME", "action": "store",
                "help": "argument name"}]
ARG_DBSEARCH = [["conditions"],
                {"metavar": "CONDITION", "nargs": "+",
                 "help": ("Conditions to search log messages. "
                          "Example: command gid=24 date=2012-10-10 ..., "
                          "Keys: gid, date, top_date, end_date, "
                          "host, area")}]
ARG_FILTER = [["filters"],
              {"metavar": "FILTER", "nargs": "*",
               "help": ("filters for dag stats or plots. "
                        "see showdag_filter.py for more detail")}]

# argument settings for each modes
# description, List[args, kwargs], func
# defined after functions because these settings use functions
DICT_ARGSET = {
    "test": ["Generate DAG",
             [OPT_CONFIG, OPT_DEBUG],
             test_makedag],
    "make-tsdb": ["Generate time-series DB for make-dag input",
                  [OPT_CONFIG, OPT_DEBUG, OPT_PARALLEL],
                  make_tsdb],
    "make-args": ["Initialize arguments for pc algorithm",
                  [OPT_CONFIG, OPT_DEBUG],
                  make_args],
    "make-dag": ["Generate causal DAGs",
                 [OPT_CONFIG, OPT_DEBUG, OPT_PARALLEL],
                 make_dag],
    "make-dag-stdin": ["make-dag interface for pipeline processing",
                       [OPT_CONFIG, OPT_DEBUG, ARG_ARGNAME],
                       make_dag_stdin],
    "make-dag-prune": ["Show pruned DAGs before PC algorithm",
                       [OPT_CONFIG, OPT_DEBUG, ARG_ARGNAME],
                       make_dag_prune],
    #    "reload-area": ["Reload area definition for time-series DB",
    #                    [OPT_CONFIG, OPT_DEBUG],
    #                    reload_area],
    #    "show-event": ["Show events (list of gid and host)",
    #                   [OPT_CONFIG, OPT_DEBUG, ARG_DBSEARCH],
    #                   show_event],
    #    "show-ts": ["Show time series of given conditions",
    #                [OPT_CONFIG, OPT_DEBUG, ARG_DBSEARCH],
    #                show_ts],
    #    "show-ts-compare": ["Show filtered/remaining time series",
    #                        [OPT_CONFIG, OPT_DEBUG, ARG_DBSEARCH],
    #                        show_ts_compare],
    #    "show-filterlog": ["Show preprocessing log",
    #                       [OPT_CONFIG, OPT_DEBUG, ARG_DBSEARCH],
    #                       show_filterlog],
    "show-args": ["Show arguments recorded in argument file",
                  [OPT_CONFIG, OPT_DEBUG],
                  show_args],
    "show-edge-list": ["Show edges in a DAG",
                       [OPT_CONFIG, OPT_DEBUG, OPT_THRESHOLD,
                        ARG_ARGNAME, ARG_FILTER],
                       show_edge_list],
    "show-edge-detail": ["Show logs of detected edges in a DAG",
                         [OPT_CONFIG, OPT_DEBUG,
                          [["--head", ],
                           {"dest": "head", "action": "store",
                            "type": int, "default": 5,
                            "help": "number of lines from log head"}],
                          [["--tail", ],
                           {"dest": "tail", "action": "store",
                            "type": int, "default": 5,
                            "help": "number of lines from log tail"}],
                          ARG_ARGNAME],
                         show_edge_detail],
    "show-list": ["Show abstracted results of DAG generation",
                  [OPT_CONFIG, OPT_DEBUG, OPT_THRESHOLD, OPT_GROUPBY],
                  show_list],
    "show-stats": ["Show sum of nodes and edges",
                   [OPT_CONFIG, OPT_DEBUG],
                   show_stats],
    "show-netsize": ["Show distribution of connected subgraphs in DAGs",
                     [OPT_CONFIG, OPT_DEBUG],
                     show_netsize],
    "show-netsize-list": ["Show connected subgraphs in every DAG",
                          [OPT_CONFIG, OPT_DEBUG],
                          show_netsize_list],
    "plot-dag": ["Generate causal DAG view",
                 [OPT_CONFIG, OPT_DEBUG, OPT_FILENAME, OPT_THRESHOLD,
                  ARG_ARGNAME, ARG_FILTER],
                 plot_dag],
}

USAGE_COMMANDS = "\n".join(["  {0}: {1}".format(key, val[0])
                            for key, val in sorted(DICT_ARGSET.items())])
USAGE = ("usage: {0} MODE [options and arguments] ...\n\n"
         "mode:\n".format(sys.argv[0])) + USAGE_COMMANDS + \
        "\n\nsee \"{0} MODE -h\" to refer detailed usage".format(sys.argv[0]) + \
        "\nalso see sub-liblary {0}".format(" ".join(["logdag.{0}".format(n)
                                                      for n in SUBLIB]))

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
    for given_args, given_kwargs in l_argset:
        ap.add_argument(*given_args, **given_kwargs)
    namespace = ap.parse_args(commandline)
    func(namespace)

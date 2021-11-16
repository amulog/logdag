#!/usr/bin/env python
# coding: utf-8

import sys
import logging

from . import arguments
from amulog import cli
from amulog import common

_logger = logging.getLogger(__package__)
SUBLIB = ["source", "eval", "visual"]


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

    conf = open_logdag_config(ns)

    am = arguments.ArgumentManager(conf)
    am.generate(arguments.all_args)
    am.init_dirs(conf)
    am.dump()

    pal = ns.parallel
    if pal > 1:
        import multiprocessing
        timer = common.Timer("makedag task", output=_logger)
        timer.start()
        with multiprocessing.Pool(processes=pal) as pool:
            pool.map(makedag.makedag_pool, am)
        timer.stop()
    else:
        timer = common.Timer("makedag task", output=_logger)
        timer.start()
        for args in am:
            makedag.makedag_pool(args)
        timer.stop()


def make_dag_stdin(ns):
    from . import makedag

    conf = open_logdag_config(ns)

    am = arguments.ArgumentManager(conf)
    am.init_dirs(conf)
    args = am.jobname2args(ns.argname, conf)

    timer = common.Timer("makedag task for {0}".format(ns.argname),
                         output=_logger)
    timer.start()
    makedag.makedag_main(args, do_dump=True)
    timer.stop()


def update_event_label(ns):
    conf = open_logdag_config(ns)
    am = arguments.ArgumentManager(conf)
    am.init_dirs(conf)
    args = am.jobname2args(ns.argname, conf)

    from . import log2event
    evmap = log2event.EventDefinitionMap()
    evmap.load(args)

    from amulog import config
    from .source import src_amulog
    tmp_args = [config.getterm(conf, "general", "evdb_whole_term"),
                conf["database_amulog"]["source_conf"],
                conf["database_amulog"]["event_gid"]]
    al = src_amulog.AmulogLoader(*tmp_args)

    for evdef in evmap.iter_evdef():
        assert evdef.source == log2event.SRCCLS_LOG
        evdef.group = al.group(evdef.gid)

    evmap.dump(args)


def dump_input(ns):
    from . import makedag

    conf = open_logdag_config(ns)

    am = arguments.ArgumentManager(conf)
    am.init_dirs(conf)
    args = am.jobname2args(ns.argname, conf)

    input_df, _ = makedag.make_input(args)
    input_df.to_csv(ns.output)


def dump_events(ns):
    from . import log2event
    conf = open_logdag_config(ns)

    am = arguments.ArgumentManager(conf)
    am.init_dirs(conf)
    args = am.jobname2args(ns.argname, conf)

    evmap = log2event.EventDefinitionMap()
    evmap.load(args)

    if len(evmap) == 0:
        from . import makedag
        input_df, evmap = makedag.make_input(args)

    for eid, evdef in evmap.items():
        print("eid {0}: {1}".format(eid, evdef))


def show_args(ns):
    conf = open_logdag_config(ns)

    am = arguments.ArgumentManager(conf)
    try:
        am.load()
    except IOError:
        sys.exit("ArgumentManager object file not found")

    print(am.show())


def _parse_condition(conditions):
    d = {}
    for arg in conditions:
        if "=" not in arg:
            raise SyntaxError
        key = arg.partition("=")[0]
        if key == "node":
            d["node"] = int(arg.partition("=")[-1])
        elif key == "gid":
            d["gid"] = int(arg.partition("=")[-1])
        elif key == "host":
            d["host"] = arg.partition("=")[-1]
    return d


def show_edge(ns):
    from . import showdag
    conf = open_logdag_config(ns)
    args = arguments.name2args(ns.argname, conf)

    r = showdag.LogDAG(args)
    r.load()

    if ns.detail:
        context = "detail"
    elif ns.instruction:
        context = "instruction"
    else:
        context = "edge"
    d_cond = _parse_condition(ns.conditions)

    print(showdag.show_edge(r, d_cond, context=context,
                            load_cache=(not ns.nocache), graph=None))


def show_subgraphs(ns):
    from . import showdag
    conf = open_logdag_config(ns)
    args = arguments.name2args(ns.argname, conf)

    ldag = showdag.LogDAG(args)
    ldag.load()
    g = showdag.apply_filter(ldag, ns.filters, th=ns.threshold)

    if ns.detail:
        context = "detail"
    elif ns.instruction:
        context = "instruction"
    else:
        context = "edge"

    print(showdag.show_subgraphs(ldag, context,
                                 load_cache=(not ns.nocache), graph=g))


def show_edge_list(ns):
    from . import showdag
    conf = open_logdag_config(ns)
    args = arguments.name2args(ns.argname, conf)

    r = showdag.LogDAG(args)
    r.load()
    g = showdag.apply_filter(r, ns.filters, th=ns.threshold)

    if ns.detail:
        context = "detail"
    elif ns.instruction:
        context = "instruction"
    else:
        context = "edge"

    print(showdag.show_edge_list(r, context,
                                 load_cache=(not ns.nocache), graph=g))


def show_list(ns):
    from . import showdag
    conf = open_logdag_config(ns)

    l_func = [lambda r: r.graph.number_of_nodes(),
              lambda r: r.graph.number_of_edges()]
    table = []
    for key, _, data in showdag.stat_groupby(conf, l_func, groupby=ns.groupby):
        table.append([key] + list(data))
    print(common.cli_table(table))


def show_node_list(ns):
    from . import showdag
    conf = open_logdag_config(ns)
    args = arguments.name2args(ns.argname, conf)

    r = showdag.LogDAG(args)
    r.load()
    for node in r.graph.nodes():
        print("{0}: {1}".format(node, r.node_str(node)))


def show_stats(ns):
    import numpy as np
    from . import showdag
    conf = open_logdag_config(ns)

    msg = [
        "number of events (nodes)",
        "number of all edges",
        "number of directed edges",
        "number of undirected edges"
    ]
    l_func = [
        lambda r: r.number_of_nodes(),
        lambda r: r.number_of_edges(),
        lambda r: showdag.apply_filter(r, ["directed"]).number_of_edges(),
        lambda r: showdag.apply_filter(r, ["undirected"]).number_of_edges(),
    ]
    if ns.across_host:
        msg += [
            "number of edges across hosts",
            "number of directed edges across hosts",
            "number of undirected edges across hosts",
        ]
        l_func += [
            lambda r: r.number_of_edges(showdag.apply_filter(r, ["across_host"])),
            lambda r: showdag.apply_filter(r, ["directed", "across_host"]).number_of_edges(),
            lambda r: showdag.apply_filter(r, ["undirected", "across_host"]).number_of_edges(),
        ]

    dt_range = _parse_opt_range(ns)
    data = [v for _, _, v
            in showdag.stat_groupby(conf, l_func, dt_range=dt_range)]
    agg_data = np.sum(data, axis=0)
    print(common.cli_table(list(zip(msg, agg_data)), align="right"))


def show_stats_by_threshold(ns):
    import numpy as np
    from . import showdag
    conf = open_logdag_config(ns)

    thresholds = np.arange(0, 1, 0.1)
    dt_range = _parse_opt_range(ns)
    data = showdag.stat_by_threshold(conf, thresholds, dt_range=dt_range)
    print(common.cli_table(list(zip(thresholds, data)), align="right"))


def show_group_stats(ns):
    from . import showdag
    conf = open_logdag_config(ns)

    from collections import defaultdict
    d_node = defaultdict(int)
    d_edge = defaultdict(int)
    d_event = defaultdict(int)
    for ldag in showdag.iter_results(conf):
        l_node = list(ldag.nodes())
        df = ldag.node_ts(l_node)
        for node in l_node:
            evdef = ldag.node_evdef(node)
            d_node[evdef.group] += 1
            d_event[evdef.group] += df[node].sum()
        for edge in ldag.edges():
            src_evdef, dst_evdef = ldag.edge_evdef(edge)
            d_edge[src_evdef.group] += 1
            d_edge[dst_evdef.group] += 1

    table = [["group", "nodes", "edges", "logs"]]
    for key in d_node.keys():
        table.append([key, str(d_node[key]), str(d_edge[key]), str(d_event[key])])
    print(common.cli_table(table))


def show_node_ts(ns):
    from . import showdag
    conf = open_logdag_config(ns)

    args = arguments.name2args(ns.argname, conf)
    l_nodeid = [int(n) for n in ns.node_ids]

    ldag = showdag.LogDAG(args)
    ldag.load()
    df = ldag.node_ts(l_nodeid)
    print(df.to_csv())


def show_netsize(ns):
    from . import showdag
    conf = open_logdag_config(ns)

    print(showdag.show_netsize_dist(conf))


def show_netsize_list(ns):
    from . import showdag
    conf = open_logdag_config(ns)

    print(showdag.list_netsize(conf))


def show_full_config(ns):
    from amulog import config
    conf = open_logdag_config(ns)
    config.show_config(conf)


def show_default_config(_):
    from . import arguments
    arguments.show_logdag_default_config()


def plot_dag(ns):
    from . import showdag
    # from . import showdag_filter
    conf = open_logdag_config(ns)

    args = arguments.name2args(ns.argname, conf)
    output = ns.output

    r = showdag.LogDAG(args)
    r.load()
    # g = showdag_filter.apply(r, ns.filters, th=ns.threshold)
    g = showdag.apply_filter(r, ns.filters, th=ns.threshold)
    g = r.relabel(graph=g)
    r.graph_nx(output, graph=g)
    print(output)


def plot_node_ts(ns):
    from . import showdag
    conf = open_logdag_config(ns)

    args = arguments.name2args(ns.argname, conf)
    output = ns.output

    l_nodeid = [int(n) for n in ns.node_ids]
    showdag.plot_node_ts(args, l_nodeid, output)
    print(output)


def _parse_opt_range(ns):
    date_range_str = ns.dt_range
    if date_range_str is None:
        return None
    else:
        assert len(date_range_str) == 2
        import datetime
        return [datetime.datetime.strptime(s, "%Y-%m-%d") for s in date_range_str]


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
OPT_OUTPUT = [["-o", "--output"],
              {"dest": "output", "metavar": "OUTPUT", "action": "store",
               "default": "output",
               "help": "output filename"}]
OPT_DIRNAME = [["-d", "--dirname"],
               {"dest": "dirname", "metavar": "DIRNAME", "action": "store",
                "default": ".",
                "help": "directory name for output"}]
OPT_RANGE = [["-r", "--range"],
             {"dest": "dt_range",
              "metavar": "DATE", "nargs": 2, "default": None,
              "help": ("datetime range, start and end in YY-MM-dd style. "
                       "end date is not included."
                       "(optional; use all data in default)")}]
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
OPT_INSTRUCTION = [["--instruction"],
                   {"dest": "instruction", "action": "store_true",
                    "help": "show event definition with source information"}]
OPT_DETAIL = [["-d", "--detail"],
              {"dest": "detail", "action": "store_true",
               "help": "show event time-series samples"}]
OPT_IGNORE_CACHE = [["--nocache"],
                    {"dest": "nocache", "action": "store_true",
                     "help": "ignore existing cache"}]
# OPT_LOG_ORG = [["--log-org"],
#                {"dest": "log_org", "action": "store_true",
#                 "help": "show original logs from amulog db for log time-series"}]
OPT_GROUPBY = [["--groupby"],
               {"dest": "groupby", "metavar": "GROUPBY",
                "action": "store", "default": None,
                "help": "aggregate results by given metrics (like day, area)"}]
OPT_FILTER = [["-f", "--filter"],
              {"dest": "filters", "action": "append",
               "help": ("filters for dag stats or plots. "
                        "see showdag_filter.py for more detail")}]

ARG_ARGNAME = [["argname"],
               {"metavar": "TASKNAME", "action": "store",
                "help": "argument name"}]
ARG_EDGESEARCH = [["conditions"],
                  {"metavar": "CONDITION", "nargs": "+",
                   "help": ("Conditions to search edges."
                            "Example: MODE gid=24 host=host01 ..., "
                            "Keys: node, gid, host.")}]

# argument settings for each modes
# description, List[args, kwargs], func
# defined after functions because these settings use functions
DICT_ARGSET = {
    "make-args": ["Initialize arguments for pc algorithm",
                  [OPT_CONFIG, OPT_DEBUG],
                  make_args],
    "make-dag": ["Generate causal DAGs",
                 [OPT_CONFIG, OPT_DEBUG, OPT_PARALLEL],
                 make_dag],
    "make-dag-stdin": ["make-dag interface for pipeline processing",
                       [OPT_CONFIG, OPT_DEBUG, ARG_ARGNAME],
                       make_dag_stdin],
    "update-event-label": ["Overwrite labels for log events",
                           [OPT_CONFIG, OPT_DEBUG, ARG_ARGNAME],
                           update_event_label],
    "dump-input": ["Output causal analysis input in pandas csv format",
                   [OPT_CONFIG, OPT_DEBUG, OPT_OUTPUT,
                    [["-b", "--binary"],
                     {"dest": "binary", "action": "store_true",
                      "help": "dump binarized dataframe csv"}],
                    ARG_ARGNAME],
                   dump_input],
    "dump-events": ["Output event node definition in readable format",
                    [OPT_CONFIG, OPT_DEBUG, ARG_ARGNAME],
                    dump_events],
    #    "reload-area": ["Reload area definition for time-series DB",
    #                    [OPT_CONFIG, OPT_DEBUG],
    #                    reload_area],
    "show-args": ["Show arguments recorded in argument file",
                  [OPT_CONFIG, OPT_DEBUG],
                  show_args],
    "show-edge": ["Show edges related to given conditions",
                  [OPT_CONFIG, OPT_DEBUG, OPT_INSTRUCTION,
                   OPT_DETAIL, OPT_IGNORE_CACHE,
                   ARG_ARGNAME, ARG_EDGESEARCH],
                  show_edge],
    "show-edge-list": ["Show all edges in a DAG",
                       [OPT_CONFIG, OPT_DEBUG, OPT_THRESHOLD, OPT_INSTRUCTION,
                        OPT_DETAIL, OPT_IGNORE_CACHE, OPT_FILTER,
                        ARG_ARGNAME],
                       show_edge_list],
    "show-subgraphs": ["Show edges in each connected subgraphs",
                       [OPT_CONFIG, OPT_DEBUG, OPT_THRESHOLD, OPT_INSTRUCTION,
                        OPT_DETAIL, OPT_IGNORE_CACHE, OPT_FILTER,
                        ARG_ARGNAME],
                       show_subgraphs],
    "show-list": ["Show abstracted results of DAG generation",
                  [OPT_CONFIG, OPT_DEBUG, OPT_THRESHOLD, OPT_GROUPBY],
                  show_list],
    "show-node-list": ["Show node definitions of the input",
                       [OPT_CONFIG, OPT_DEBUG, ARG_ARGNAME],
                       show_node_list],
    "show-stats": ["Show sum of nodes and edges",
                   [OPT_CONFIG, OPT_DEBUG, OPT_RANGE,
                    [["--xhost"],
                     {"dest": "across_host", "action": "store_true",
                      "help": "show additional stats for edges across multiple hosts"}]],
                   show_stats],
    "show-stats-by-threshold": ["Show sum of edges by thresholds",
                                [OPT_CONFIG, OPT_DEBUG, OPT_RANGE],
                                show_stats_by_threshold],
    "show-group-stats": ["Show stats classified by amulog tags",
                         [OPT_CONFIG, OPT_DEBUG],
                         show_group_stats],
    "show-node-ts": ["Show time-series of specified nodes in csv format",
                     [OPT_CONFIG, OPT_DEBUG,
                      ARG_ARGNAME,
                      [["node_ids"],
                       {"metavar": "NODE_IDs", "nargs": "+",
                        "help": "nodes to show"}]],
                     show_node_ts],
    "show-netsize": ["Show distribution of connected subgraphs in DAGs",
                     [OPT_CONFIG, OPT_DEBUG],
                     show_netsize],
    "show-netsize-list": ["Show connected subgraphs in every DAG",
                          [OPT_CONFIG, OPT_DEBUG],
                          show_netsize_list],
    "show-full-config": ["Show virtual configuration considering defaults",
                         [OPT_CONFIG, OPT_DEBUG],
                         show_full_config],
    "show-deafult-config": ["Show configuration defaults",
                            [],
                            show_default_config],
    "plot-dag": ["Generate causal DAG view",
                 [OPT_CONFIG, OPT_DEBUG,
                  OPT_OUTPUT, OPT_THRESHOLD, OPT_FILTER,
                  ARG_ARGNAME],
                 plot_dag],
    "plot-node-ts": ["Generate node time-series view",
                     [OPT_CONFIG, OPT_DEBUG, OPT_OUTPUT,
                      ARG_ARGNAME,
                      [["node_ids"],
                       {"metavar": "NODE_IDs", "nargs": "+",
                        "help": "nodes to show"}]],
                     plot_node_ts],
}


def main():
    cli.main(DICT_ARGSET, sublibs=SUBLIB)


if __name__ == "__main__":
    main()

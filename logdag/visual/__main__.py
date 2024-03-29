#!/usr/bin/env python
# coding: utf-8

import logging
from collections import defaultdict

from amulog import cli
from amulog import config
from logdag import arguments
from logdag import dtutil
from logdag import showdag

_logger = logging.getLogger(__package__)


def open_logdag_config(ns):
    from logdag import arguments
    return arguments.open_logdag_config(ns.conf_path, debug=ns.debug)


def search_gid(ns):
    conf = open_logdag_config(ns)
    gid = ns.gid

    from . import edge_search
    l_result = edge_search.search_gid(conf, gid)
    for r, edge in l_result:
        print("{0} {1}".format(r.name, r.edge_str(edge)))


def show_minor_edges(ns):
    conf = open_logdag_config(ns)
    args = arguments.name2args(ns.argname, conf)

    from . import edge_search
    ldag = showdag.LogDAG(args)
    ldag.load()
    g = showdag.apply_filter(ldag, ns.filters, th=ns.threshold)

    if ns.detail:
        context = "detail"
    elif ns.instruction:
        context = "instruction"
    else:
        context = "edge"

    from ..__main__ import _parse_condition
    search_condition = _parse_condition(ns.conditions)

    print(edge_search.show_sorted_edges(
        ldag,
        search_condition=search_condition,
        feature=ns.feature,
        score=ns.score,
        reverse=False,
        view_context=context,
        load_cache=(not ns.nocache),
        graph=g
    ))


def show_major_edges(ns):
    conf = open_logdag_config(ns)
    args = arguments.name2args(ns.argname, conf)

    from . import edge_search
    ldag = showdag.LogDAG(args)
    ldag.load()
    g = showdag.apply_filter(ldag, ns.filters, th=ns.threshold)

    if ns.detail:
        context = "detail"
    elif ns.instruction:
        context = "instruction"
    else:
        context = "edge"

    from ..__main__ import _parse_condition
    search_condition = _parse_condition(ns.conditions)

    print(edge_search.show_sorted_edges(
        ldag,
        search_condition=search_condition,
        feature=ns.feature,
        score=ns.score,
        reverse=True,
        view_context=context,
        load_cache=(not ns.nocache),
        graph=g
    ))


def show_edges_temporal_sort(ns):
    conf = open_logdag_config(ns)
    args = arguments.name2args(ns.argname, conf)

    from . import edge_search
    ldag = showdag.LogDAG(args)
    ldag.load()
    g = showdag.apply_filter(ldag, ns.filters, th=ns.threshold)

    if ns.detail:
        context = "detail"
    elif ns.instruction:
        context = "instruction"
    else:
        context = "edge"

    from logdag import dtutil
    if ns.timestr_end is None:
        dt = dtutil.shortstr2dt(ns.timestr)
        time_condition = {"time": dt}
    else:
        dts = dtutil.shortstr2dt(ns.timestr)
        dte = dtutil.shortstr2dt(ns.timestr_end)
        time_condition = {"time_range": (dts, dte)}

    from ..__main__ import _parse_condition
    search_condition = _parse_condition(ns.conditions)

    print(edge_search.edge_temporal_sort(
        ldag,
        time_condition,
        search_condition=search_condition,
        reverse=ns.reverse,
        view_context=context,
        load_cache=(not ns.nocache),
        graph=g
    ))


def show_dag_anomaly_score(ns):
    conf = open_logdag_config(ns)

    from . import edge_search
    d_score = edge_search.dag_anomaly_score(conf, feature=ns.feature,
                                            score=ns.score)

    if ns.order:
        am = arguments.ArgumentManager(conf)
        am.load()
        iterobj = [(am.jobname(args), d_score[am.jobname(args)]) for args in am]
    else:
        iterobj = sorted(d_score.items(), key=lambda x: x[1],
                         reverse=(not ns.reverse))

    for jobname, score in iterobj:
        print(jobname, score)


def search_similar_dag(ns):
    conf = open_logdag_config(ns)
    args = arguments.name2args(ns.argname, conf)

    from . import edge_search
    ldag = showdag.LogDAG(args)
    ldag.load()

    buf = edge_search.search_similar_dag(ldag, feature=ns.feature,
                                         weight=ns.score_weight,
                                         dag_topn=10, cause_topn=10)
    print(buf)


def show_clusters(ns):
    conf = open_logdag_config(ns)

    from . import edge_search
    buf = edge_search.show_clusters(conf, feature=ns.feature,
                                    weight=ns.score_weight,
                                    clustering_method="kmeans",
                                    n_clusters=None, cause_topn=10)
    print(buf)


def draw_graph_common(ns):
    conf_fn1, conf_fn2 = ns.confs
    conf1 = arguments.open_logdag_config(conf_fn1)
    conf2 = arguments.open_logdag_config(conf_fn2)
    lv = logging.DEBUG if ns.debug else logging.INFO
    am_logger = logging.getLogger("amulog")
    config.set_common_logging(conf1, logger=[_logger, am_logger], lv=lv)

    dts = dtutil.shortstr2dt(ns.timestr)
    dte = dts + config.getdur(conf1, "dag", "unit_term")
    output = ns.filename

    from . import comparison
    cevmap, cgraph = comparison.edge_set_common(conf1, conf2, (dts, dte))

    from . import draw
    rgraph = draw.relabel_nodes(cgraph, cevmap)
    draw.graph_nx(output, rgraph)
    print(output)


def draw_graph_diff(ns):
    conf_fn1, conf_fn2 = ns.confs
    conf1 = arguments.open_logdag_config(conf_fn1)
    conf2 = arguments.open_logdag_config(conf_fn2)
    lv = logging.DEBUG if ns.debug else logging.INFO
    am_logger = logging.getLogger("amulog")
    config.set_common_logging(conf1, logger=[_logger, am_logger], lv=lv)

    dts = dtutil.shortstr2dt(ns.timestr)
    dte = dts + config.getdur(conf1, "dag", "unit_term")
    output = ns.filename

    from . import comparison
    cevmap, cgraph = comparison.edge_set_diff(conf1, conf2, (dts, dte))

    from . import draw
    rgraph = draw.relabel_nodes(cgraph, cevmap)
    draw.graph_nx(output, rgraph)
    print(output)


def show_diff_info(ns):
    conf_fn1, conf_fn2 = ns.confs
    conf1 = arguments.open_logdag_config(conf_fn1)
    conf2 = arguments.open_logdag_config(conf_fn2)
    lv = logging.DEBUG if ns.debug else logging.INFO
    am_logger = logging.getLogger("amulog")
    config.set_common_logging(conf1, logger=[_logger, am_logger], lv=lv)

    from . import comparison
    d = defaultdict(int)
    am = arguments.ArgumentManager(conf1)
    am.load()
    for dt_range in am.iter_dt_range():
        cevmap_common, cgraph_common = comparison.edge_set_common(
            conf1, conf2, dt_range)
        d["common"] += cgraph_common.number_of_edges()
        cevmap_lor, cgraph_lor = comparison.edge_set_lor(
            conf1, conf2, dt_range)
        d["lor"] += cgraph_lor.number_of_edges()
        cevmap_diff1, cgraph_diff1 = comparison.edge_set_diff(
            conf1, conf2, dt_range, lor=(cevmap_lor, cgraph_lor))
        d["diff1"] += cgraph_diff1.number_of_edges()
        cevmap_diff2, cgraph_diff2 = comparison.edge_set_diff(
            conf2, conf1, dt_range, lor=(cevmap_lor, cgraph_lor))
        d["diff2"] += cgraph_diff2.number_of_edges()
    print("Logical OR edges: {0}".format(d["lor"]))
    print("Common edges: {0}".format(d["common"]))
    print("Edges only found in {0}: {1}".format(ns.confs[0], d["diff1"]))
    print("Edges only found in {0}: {1}".format(ns.confs[1], d["diff2"]))


def show_diff_edges(ns):
    conf_fn1, conf_fn2 = ns.confs
    conf1 = arguments.open_logdag_config(conf_fn1)
    conf2 = arguments.open_logdag_config(conf_fn2)
    lv = logging.DEBUG if ns.debug else logging.INFO
    am_logger = logging.getLogger("amulog")
    config.set_common_logging(conf1, logger=[_logger, am_logger], lv=lv)

    from . import comparison
    am = arguments.ArgumentManager(conf1)
    am.load()
    for dt_range in sorted(am.iter_dt_range()):
        cevmap, cgraph = comparison.edge_set_diff(conf1, conf2, dt_range)

        buf_edges = []
        for edge in cgraph.edges():
            buf = ""
            src_info = cevmap.evdef(edge[0])
            buf += "[gid={0[0]}, host = {0[1]}]".format(src_info)
            if showdag.isdirected(edge, cgraph):
                buf += " -> "
            else:
                buf += " <-> "
            dst_info = cevmap.evdef(edge[1])
            buf += "[gid={0[0]}, host = {0[1]}]".format(dst_info)
            buf_edges.append(buf)

        if len(buf_edges) > 0:
            print("date: {0}".format(dt_range[0]))
            print("\n".join(buf_edges))


def show_graph_common_edges(ns):
    conf_fn1, conf_fn2 = ns.confs
    conf1 = arguments.open_logdag_config(conf_fn1)
    conf2 = arguments.open_logdag_config(conf_fn2)
    lv = logging.DEBUG if ns.debug else logging.INFO
    am_logger = logging.getLogger("amulog")
    config.set_common_logging(conf1, logger=[_logger, am_logger], lv=lv)

    from . import comparison
    edge_sum = 0
    d_edges = {}
    am = arguments.ArgumentManager(conf1)
    am.load()
    for dt_range in am.iter_dt_range():
        cevmap, cgraph = comparison.edge_set_common(conf1, conf2, dt_range)
        edge_sum += cgraph.number_of_edges()
        d_edges[dt_range[0]] = cgraph.edges()

    print("common edge num: {0}".format(edge_sum))
    for k, v in sorted(d_edges.items(), key=lambda x: x[0]):
        print("{0}: {1}".format(k, len(v)))


def show_graph_lor_edges(ns):
    conf_fn1, conf_fn2 = ns.confs
    conf1 = arguments.open_logdag_config(conf_fn1)
    conf2 = arguments.open_logdag_config(conf_fn2)
    lv = logging.DEBUG if ns.debug else logging.INFO
    am_logger = logging.getLogger("amulog")
    config.set_common_logging(conf1, logger=[_logger, am_logger], lv=lv)

    from . import comparison
    edge_sum = 0
    d_edges = {}
    am = arguments.ArgumentManager(conf1)
    am.load()
    for dt_range in am.iter_dt_range():
        cevmap, cgraph = comparison.edge_set_lor(conf1, conf2, dt_range)
        edge_sum += cgraph.number_of_edges()
        d_edges[dt_range[0]] = cgraph.edges()

    print("logical disjunction edge num: {0}".format(edge_sum))
    for k, v in sorted(d_edges.items(), key=lambda x: x[0]):
        print("{0}: {1}".format(k, len(v)))


def show_graph_diff_lts(ns):
    conf_fn1, conf_fn2 = ns.confs
    conf1 = arguments.open_logdag_config(conf_fn1)
    conf2 = arguments.open_logdag_config(conf_fn2)
    lv = logging.DEBUG if ns.debug else logging.INFO
    am_logger = logging.getLogger("amulog")
    config.set_common_logging(conf1, logger=[_logger, am_logger], lv=lv)

    from amulog import log_db
    ld = log_db.LogData(conf1)

    from . import comparison
    d_ltid = comparison.edge_diff_gid(conf1, conf2)
    for ltid, l_name in sorted(d_ltid.items(), key=lambda x: len(x[1]),
                               reverse=True):
        print("{0}: {1} ({2})".format(len(l_name), ltid, ld.lt(ltid)))
        if len(l_name) < 100:
            print(l_name)


def show_graph_diff_search(ns):
    conf_fn1, conf_fn2 = ns.confs
    conf1 = arguments.open_logdag_config(conf_fn1)
    conf2 = arguments.open_logdag_config(conf_fn2)
    lv = logging.DEBUG if ns.debug else logging.INFO
    am_logger = logging.getLogger("amulog")
    config.set_common_logging(conf1, logger=[_logger, am_logger], lv=lv)

    gid = ns.gid
    from . import comparison
    comparison.edge_diff_gid_search(conf1, conf2, gid)


def show_diff_direction(ns):
    conf_fn1, conf_fn2 = ns.confs
    conf1 = arguments.open_logdag_config(conf_fn1)
    conf2 = arguments.open_logdag_config(conf_fn2)
    lv = logging.DEBUG if ns.debug else logging.INFO
    am_logger = logging.getLogger("amulog")
    config.set_common_logging(conf1, logger=[_logger, am_logger], lv=lv)

    def _print_diff(_ret):
        for ev1, ev2, di1, di2 in _ret:
            print("{0} {1} | {2} {3}".format(ev1, di1, di2, ev2))

    cnt = 0
    from . import comparison
    am = arguments.ArgumentManager(conf1)
    am.load()
    if ns.argname is None:
        for dt_range in sorted(am.iter_dt_range()):
            ret = comparison.edge_direction_diff(conf1, conf2, dt_range)
            cnt += len(ret)
            if len(ret) > 0:
                print(dt_range)
                _print_diff(ret)
                print("")
    else:
        args = am.jobname2args(ns.argname, conf1)
        dt_range = args[2]
        ret = comparison.edge_direction_diff(conf1, conf2, dt_range)
        cnt += len(ret)
        _print_diff(ret)

    print(cnt)


# common argument settings
OPT_DEBUG = [["--debug"],
             {"dest": "debug", "action": "store_true",
              "help": "set logging level to debug (default: info)"}]
OPT_CONFIG = [["-c", "--config"],
              {"dest": "conf_path", "metavar": "CONFIG", "action": "store",
               "default": None,
               "help": "configuration file path for amulog"}]
OPT_FILENAME = [["-f", "--filename"],
                {"dest": "filename", "metavar": "FILENAME", "action": "store",
                 "default": "output",
                 "help": "output filename"}]
OPT_ARGNAME = [["--argname"],
               {"dest": "argname", "metavar": "TASKNAME",
                "action": "store", "default": None,
                "help": "argument name"}]
OPT_THRESHOLD = [["-t", "--threshold"],
                 {"dest": "threshold", "metavar": "THRESHOLD", "action": "store",
                  "type": float, "default": None,
                  "help": "threshold for filter ate_prune"}]
OPT_INSTRUCTION = [["--instruction"],
                   {"dest": "instruction", "action": "store_true",
                    "help": "show event definition with source information"}]
OPT_DETAIL = [["-d", "--detail"],
              {"dest": "detail", "action": "store_true",
               "help": "show event time-series samples"}]
OPT_IGNORE_CACHE = [["--nocache"],
                    {"dest": "nocache", "action": "store_true",
                     "help": "ignore existing cache"}]
OPT_COUNT = [["-n", "--number"],
             {"dest": "count", "action": "store", "type": int, "default": None,
              "help": 'threshold of edge counts'}]
OPT_FEATURE = [["--feature"],
               {"dest": "feature", "action": "store",
                "type": str, "default": "edge",
                "help": "one of [node, edge, tfidf]"}]
OPT_SCORE = [["--score"],
             {"dest": "score", "action": "store",
              "type": str, "default": "tfidf",
              "help": "one of [tfidf, idf, count]"}]
OPT_SCORE_WEIGHT = [["--score"],
                    {"dest": "score_weight", "action": "store",
                     "type": str, "default": "idf",
                     "help": "one of [none, idf]"}]
OPT_ORDER = [["--order"],
             {"dest": "order", "action": "store_true",
              "help": "do not sort results by score"}]
OPT_REVERSE = [["--reverse"],
               {"dest": "reverse", "action": "store_true",
                "help": "reversed sort"}]
OPT_FILTER = [["-f", "--filter"],
              {"dest": "filters", "action": "append",
               "help": ("filters for dag stats or plots. "
                        "see showdag_filter.py for more detail")}]
ARG_ARGNAME = [["argname"],
               {"metavar": "TASKNAME", "action": "store",
                "help": "argument name"}]
ARG_EDGESEARCH = [["conditions"],
                  {"metavar": "CONDITION", "nargs": "*",
                   "help": ("Conditions to search edges."
                            "Example: MODE gid=24 host=host01 ..., "
                            "Keys: node, gid, host.")}]
ARG_TIMESTR = [["timestr"],
               {"metavar": "TIMESTR", "action": "store",
                "help": "%%Y%%m%%d(_%%H%%M%%S) style time string"}]
OPT_TIMESTR_END = [["--range"],
                   {"dest": "timestr_end", "metavar": "TIMESTR_END",
                    "action": "store", "type": str, "default": None,
                    "help": "%%Y%%m%%d(_%%H%%M%%S) style time string, "
                            "used as the time range with TIMESTR"}]

# argument settings for each modes
# description, List[args, kwargs], func
# defined after functions because these settings use functions
DICT_ARGSET = {
    "search-gid": ["List DAGs with edges related to given gid",
                   [OPT_CONFIG, OPT_DEBUG,
                    [["gid"],
                     {"metavar": "GID", "action": "store", "type": int,
                      "help": "gid to search"}], ],
                   search_gid],
    "show-minor-edges": ["Show minor edges in all data",
                         [OPT_CONFIG, OPT_DEBUG,
                          OPT_FILTER, OPT_THRESHOLD,
                          OPT_FEATURE, OPT_SCORE,
                          OPT_INSTRUCTION, OPT_DETAIL, OPT_IGNORE_CACHE,
                          ARG_ARGNAME, ARG_EDGESEARCH],
                         show_minor_edges],
    "show-major-edges": ["Show major edges in all data",
                         [OPT_CONFIG, OPT_DEBUG,
                          OPT_FILTER, OPT_THRESHOLD,
                          OPT_FEATURE, OPT_SCORE,
                          OPT_INSTRUCTION, OPT_DETAIL, OPT_IGNORE_CACHE,
                          ARG_ARGNAME, ARG_EDGESEARCH],
                         show_major_edges],
    "show-edges-temporal-sort": ["Show sorted edges with difference from given time",
                                 [OPT_CONFIG, OPT_DEBUG,
                                  OPT_FILTER, OPT_THRESHOLD,
                                  OPT_INSTRUCTION, OPT_DETAIL, OPT_IGNORE_CACHE,
                                  OPT_TIMESTR_END, OPT_REVERSE,
                                  ARG_ARGNAME, ARG_TIMESTR, ARG_EDGESEARCH],
                                 show_edges_temporal_sort],
    "show-dag-anomaly-score": ["Show anomaly score of DAGs",
                               [OPT_CONFIG, OPT_DEBUG,
                                OPT_FEATURE, OPT_SCORE,
                                OPT_ORDER, OPT_REVERSE],
                               show_dag_anomaly_score],
    "search-similar-dag": ["Show similar DAG to the specified one",
                           [OPT_CONFIG, OPT_DEBUG,
                            OPT_FEATURE, OPT_SCORE_WEIGHT,
                            ARG_ARGNAME],
                           search_similar_dag],
    "show-clusters": ["Show clusters of DAGs",
                      [OPT_CONFIG, OPT_DEBUG,
                       OPT_FEATURE, OPT_SCORE_WEIGHT],
                      show_clusters],
    "draw-graph-common": ["Draw common edges of 2 DAG sets",
                          [OPT_DEBUG, OPT_FILENAME,
                           [["confs"],
                            {"metavar": "CONFIG", "nargs": 2,
                             "help": "2 config file path"}],
                           ARG_TIMESTR, ],
                          draw_graph_common],
    "draw-graph-diff": ["Draw contrasting edges of 2 DAG sets",
                        [OPT_DEBUG, OPT_FILENAME,
                         [["confs"],
                          {"metavar": "CONFIG", "nargs": 2,
                           "help": "2 config file path"}],
                         ARG_TIMESTR, ],
                        draw_graph_diff],
    "show-diff-info": ["Show information of 2 edge sets",
                       [OPT_DEBUG,
                        [["confs"],
                         {"metavar": "CONFIG", "nargs": 2,
                          "help": "2 config file path"}], ],
                       show_diff_info],
    "show-diff-edges": ["Show all different edges of 2 edge sets",
                        [OPT_DEBUG,
                         [["confs"],
                          {"metavar": "CONFIG", "nargs": 2,
                           "help": "2 config file path"}], ],
                        show_diff_edges],
    "show-graph-common-edges": ["List number of edges in common graph",
                                [OPT_DEBUG,
                                 [["confs"],
                                  {"metavar": "CONFIG", "nargs": 2,
                                   "help": "2 config file path"}], ],
                                show_graph_common_edges],
    "show-graph-lor-edges": ["List number of edges in lor graph",
                             [OPT_DEBUG,
                              [["confs"],
                               {"metavar": "CONFIG", "nargs": 2,
                                "help": "2 config file path"}], ],
                             show_graph_lor_edges],
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
                                  "help": "gid to search"}], ],
                               show_graph_diff_search],
    "show-graph-diff-direction": ["List edge direction differences",
                                  [OPT_DEBUG, OPT_ARGNAME,
                                   [["confs"],
                                    {"metavar": "CONFIG", "nargs": 2,
                                     "help": "2 config file path"}], ],
                                  show_diff_direction],
}


def main():
    cli.main(DICT_ARGSET)


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', filename='main.prof')
    main()

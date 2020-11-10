#!/usr/bin/env python
# coding: utf-8

import logging
from itertools import combinations

from . import arguments
from . import log2event
from . import pc_input
from . import showdag
from amulog import common

_logger = logging.getLogger(__package__)


def makedag_main(args):
    jobname = arguments.args2name(args)
    conf, dt_range, area = args

    timer = common.Timer("makedag job({0})".format(jobname), output=_logger)
    timer.start()

    # generate time-series nodes
    input_format = conf.get("dag", "input_format")
    ci_func = conf.get("dag", "ci_func")
    binarize = is_binarize(input_format, ci_func)
    # generate event set and evmap, and apply preprocessing
    # d_input, evmap = log2event.ts2input(conf, dt_range, area, binarize)
    input_df, evmap = log2event.makeinput(conf, dt_range, area, binarize)
    _logger.info("{0} pc input shape: {1}".format(jobname, input_df.shape))
    evmap.dump(conf, args)
    timer.lap("load-nodes")

    # generate initial graph
    init_graph = _init_graph(conf, evmap, jobname)
    timer.lap("prune-dag")

    # generate dag
    graph = estimate_dag(conf, input_df, ci_func, binarize, init_graph)
    timer.lap("estimate-dag")

    # record dag
    ldag = showdag.LogDAG(args, graph)
    ldag.dump()
    timer.stop()
    return ldag


def make_input(args, binarize):
    conf, dt_range, area = args
    input_df, evmap = log2event.makeinput(conf, dt_range, area, binarize)
    evmap.dump(conf, args)
    return input_df, evmap


def makedag_prune_test(args):
    jobname = arguments.args2name(args)
    conf, dt_range, area = args

    input_format = conf.get("dag", "input_format")
    ci_func = conf.get("dag", "ci_func")
    binarize = is_binarize(input_format, ci_func)
    input_df, evmap = log2event.makeinput(conf, dt_range, area, binarize)
    _logger.info("pc input shape: {0}".format(input_df.shape))
    evmap.dump(conf, args)

    node_ids = evmap.eids()
    g = _complete_graph(node_ids)
    if conf.getboolean("pc_prune", "do_pruning"):
        from . import prune
        n_edges_before = g.number_of_edges()
        init_graph = prune.prune_graph(g, conf, evmap)
        n_edges_after = init_graph.number_of_edges()
        _logger.info("{0} DAG edge pruning: ".format(jobname) +
                     "{0} -> {1}".format(n_edges_before, n_edges_after))
    else:
        n_edges = g.number_of_edges()
        init_graph = g
        _logger.info("{0} DAG edge candidates: ".format(jobname) +
                     "{0}".format(n_edges))

    # record dag
    ldag = showdag.LogDAG(args, init_graph)
    ldag.dump()
    return ldag


def _init_graph(conf, evmap, jobname=None):
    node_ids = evmap.eids()
    g = _complete_graph(node_ids)
    if conf.getboolean("pc_prune", "do_pruning"):
        from . import prune
        n_edges_before = g.number_of_edges()
        init_graph = prune.prune_graph(g, conf, evmap)
        n_edges_after = init_graph.number_of_edges()
        _logger.info("{0} DAG edge pruning: ".format(jobname) +
                     "{0} -> {1}".format(n_edges_before, n_edges_after))
        return init_graph
    else:
        return None


#def corr_graph(conf, input_df, ci_func, _, init_graph=None):
#    if input_df.shape[1] < 2:
#        _logger.info("input too small({0} nodes), return empty dag".format(
#            input_df.shape[1]))
#        return showdag.empty_dag()
#
#    cause_algorithm = conf.get("dag", "cause_algorithm")
#    if cause_algorithm in ("pc", "mixedlingam"):
#        skel_method = conf.get("dag", "skeleton_method")
#        skel_th = conf.getfloat("dag", "skeleton_threshold")
#        skel_depth = 0
#        skel_verbose = conf.getboolean("dag", "skeleton_verbose")
#        return pc_input.pc(input_df, skel_th, ci_func, skel_method,
#                           skel_depth, skel_verbose, init_graph)
#    elif cause_algorithm == "lingam":
#        raise NotImplementedError
#    else:
#        raise ValueError("invalid dag.cause_algorithm")


def estimate_dag(conf, input_df, ci_func, binarize, init_graph=None):
    if input_df.shape[1] < 2:
        _logger.info("input too small({0} nodes), return empty dag".format(
            input_df.shape[1]))
        return showdag.empty_dag()

    cause_algorithm = conf.get("dag", "cause_algorithm")
    if cause_algorithm == "pc":
        # apply pc algorithm to estimate dag
        skel_method = conf.get("dag", "skeleton_method")
        skel_th = conf.getfloat("dag", "skeleton_threshold")
        skel_depth = conf.getint("dag", "skeleton_depth")
        skel_verbose = conf.getboolean("dag", "skeleton_verbose")
        return pc_input.pc(input_df, skel_th, ci_func, skel_method,
                           skel_depth, skel_verbose, init_graph)
    elif cause_algorithm == "lingam":
        from . import lingam_input
        alg = conf.get("lingam", "algorithm")
        lower_limit = conf.getfloat("lingam", "lower_limit")
        return lingam_input.estimate(input_df, algorithm=alg,
                                     lower_limit=lower_limit,
                                     init_graph=init_graph)
    elif cause_algorithm == "mixedlingam":
        from . import mixedlingam_input
        skel_method = conf.get("dag", "skeleton_method")
        skel_th = conf.getfloat("dag", "skeleton_threshold")
        skel_depth = conf.getint("dag", "skeleton_depth")
        skel_verbose = conf.getboolean("dag", "skeleton_verbose")
        return mixedlingam_input.estimate(input_df, skel_th,
                                          skel_method, skel_depth,
                                          skel_verbose, init_graph,
                                          binarize)
    elif cause_algorithm == "pc-corr":
        skel_method = conf.get("dag", "skeleton_method")
        skel_th = conf.getfloat("dag", "skeleton_threshold")
        skel_depth = 0
        skel_verbose = conf.getboolean("dag", "skeleton_verbose")
        return pc_input.pc(input_df, skel_th, ci_func, skel_method,
                           skel_depth, skel_verbose, init_graph)
    elif cause_algorithm == "lingam-corr":
        from . import lingam_input
        alg = conf.get("lingam", "algorithm")
        lower_limit = conf.getfloat("lingam", "lower_limit")
        return lingam_input.estimate_corr(input_df, algorithm=alg,
                                          lower_limit=lower_limit,
                                          init_graph=init_graph)
    else:
        raise ValueError("invalid dag.cause_algorithm")


def is_binarize(input_format, ci_func):
    if input_format == "auto":
        if ci_func == "fisherz":
            return False
        elif ci_func == "fisherz_bin":
            return True
        elif ci_func == "gsq":
            return True
        elif ci_func == "gsq_rlib":
            return True
        else:
            raise NotImplementedError
    elif input_format == "binary":
        return True
    else:
        return False


def _complete_graph(node_ids):
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for (i, j) in combinations(node_ids, 2):
        g.add_edge(i, j)
    return g

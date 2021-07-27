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


def makedag_pool(args):
    return makedag_main(args, do_dump=True)


def makedag_main(args, do_dump=False):
    jobname = arguments.args2name(args)
    conf, dt_range, area = args

    if conf.getboolean("dag", "pass_dag_exists"):
        import os.path
        if os.path.exists(showdag.LogDAG.dag_path(args)):
            _logger.info("dag file for job({0}) exists, passed".format(jobname))
            return None

    timer = common.Timer("makedag job({0})".format(jobname), output=_logger)
    timer.start()

    # generate time-series nodes
#   input_format = conf.get("dag", "input_format")
    ci_func = conf.get("dag", "ci_func")
#   binarize = is_binarize(input_format, ci_func)
    # generate event set and evmap, and apply preprocessing
    # d_input, evmap = log2event.ts2input(conf, dt_range, area, binarize)
    input_df, evmap = log2event.makeinput(conf, dt_range, area, False)
    if input_df is None:
        return None
    _logger.info("{0} pc input shape: {1}".format(jobname, input_df.shape))
    if do_dump:
        evmap.dump(args)
    timer.lap("load-nodes")

    # generate prior knowledge
    from . import pknowledge
    prior_knowledge = pknowledge.init_prior_knowledge(conf, args, evmap)
#   init_graph = _init_graph(conf, evmap, jobname)
    timer.lap("make-prior-knowledge")

    # generate dag
    graph = estimate_dag(conf, input_df, ci_func, prior_knowledge)
    timer.lap("estimate-dag")
    if graph is None:
        _logger.info("job({0}) failed on causal inference".format(jobname))
        return None

    # record dag
    ldag = showdag.LogDAG(args, graph)
    if do_dump:
        ldag.dump()
    timer.stop()
    return ldag


def make_input(args, binarize):
    conf, dt_range, area = args
    input_df, evmap = log2event.makeinput(conf, dt_range, area, binarize)
    evmap.dump(args)
    return input_df, evmap


def makedag_prune_test(args):
    jobname = arguments.args2name(args)
    conf, dt_range, area = args

    input_format = conf.get("dag", "input_format")
    ci_func = conf.get("dag", "ci_func")
    #binarize = is_binarize(input_format, ci_func)
    binarize = False
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


#def _init_graph(conf, evmap, jobname=None):
#    node_ids = evmap.eids()
#    g = _complete_graph(node_ids)
#    if conf.getboolean("pc_prune", "do_pruning"):
#        from . import prune
#        n_edges_before = g.number_of_edges()
#        init_graph = prune.prune_graph(g, conf, evmap)
#        n_edges_after = init_graph.number_of_edges()
#        _logger.info("{0} DAG edge pruning: ".format(jobname) +
#                     "{0} -> {1}".format(n_edges_before, n_edges_after))
#        return init_graph
#    else:
#        return None


# def corr_graph(conf, input_df, ci_func, _, init_graph=None):
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


def estimate_dag(conf, input_df, ci_func, prior_knowledge=None):
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
                           skel_depth, skel_verbose, prior_knowledge)
    elif cause_algorithm == "lingam":
        from . import lingam_input
        alg = conf.get("lingam", "algorithm")
        lower_limit = conf.getfloat("lingam", "lower_limit")
        ica_max_iter = conf.getint("lingam", "ica_max_iter")
        return lingam_input.estimate(input_df, algorithm=alg,
                                     lower_limit=lower_limit,
                                     ica_max_iter=ica_max_iter,
                                     prior_knowledge=prior_knowledge)
    elif cause_algorithm == "mixedlingam":
        from . import mixedlingam_input
        skel_method = conf.get("dag", "skeleton_method")
        skel_th = conf.getfloat("dag", "skeleton_threshold")
        skel_depth = conf.getint("dag", "skeleton_depth")
        skel_verbose = conf.getboolean("dag", "skeleton_verbose")
        return mixedlingam_input.estimate(input_df, skel_th,
                                          skel_method, skel_depth,
                                          skel_verbose, prior_knowledge)
#    elif cause_algorithm == "cdt":
#        from . import cdt_input
#        category = conf.get("cdt", "category")
#        algorithm = conf.get("cdt", "algorithm")
#        max_iter = conf.getint("cdt", "max_iteration")
#        tolerance = conf.getfloat("cdt", "tolerance")
#        use_deconvolution = conf.getboolean("cdt", "use_deconvolution")
#        deconvolution_algorithm = conf.get("cdt", "deconvolution_algorithm")
#        return cdt_input.estimate(input_df, category, algorithm,
#                                  max_iter, tolerance,
#                                  use_deconvolution, deconvolution_algorithm,
#                                  prior_knowledge)
    elif cause_algorithm == "pc-corr":
        skel_method = conf.get("dag", "skeleton_method")
        skel_th = conf.getfloat("dag", "skeleton_threshold")
        skel_depth = 0
        skel_verbose = conf.getboolean("dag", "skeleton_verbose")
        return pc_input.pc(input_df, skel_th, ci_func, skel_method,
                           skel_depth, skel_verbose, prior_knowledge)
    elif cause_algorithm == "lingam-corr":
        from . import lingam_input
        alg = conf.get("lingam", "algorithm")
        lower_limit = conf.getfloat("lingam", "lower_limit")
        return lingam_input.estimate_corr(input_df, algorithm=alg,
                                          lower_limit=lower_limit,
                                          prior_knowledge=prior_knowledge)
    else:
        raise ValueError("invalid dag.cause_algorithm")


#def is_binarize(input_format, ci_func):
#    if input_format == "auto":
#        if ci_func == "fisherz":
#            return False
#        elif ci_func == "fisherz_bin":
#            return True
#        elif ci_func == "gsq":
#            return True
#        elif ci_func == "gsq_rlib":
#            return True
#        else:
#            raise NotImplementedError
#    elif input_format == "binary":
#        return True
#    else:
#        return False


def _complete_graph(node_ids):
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for (i, j) in combinations(node_ids, 2):
        g.add_edge(i, j)
    return g

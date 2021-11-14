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
    # generate event set and evmap, and apply preprocessing
    input_df, evmap = log2event.makeinput(conf, dt_range, area)
    if input_df is None:
        return None
    _logger.info("{0} input shape: {1}".format(jobname, input_df.shape))
    if do_dump:
        evmap.dump(args)
    timer.lap("load-nodes")

    # generate prior knowledge
    from . import pknowledge
    prior_knowledge = pknowledge.init_prior_knowledge(conf, args, evmap)
    timer.lap("make-prior-knowledge")

    # generate dag
    graph = estimate_dag(conf, input_df, prior_knowledge)
    timer.lap("estimate-dag")
    if graph is None:
        _logger.info("job({0}) failed on causal inference".format(jobname))
        return None

    # record dag
    ldag = showdag.LogDAG(args, graph=graph, evmap=evmap)
    if do_dump:
        ldag.dump()
    timer.stop()
    return ldag


def make_input(args):
    conf, dt_range, area = args
    input_df, evmap = log2event.makeinput(conf, dt_range, area)
    evmap.dump(args)
    return input_df, evmap


def estimate_dag(conf, input_df, prior_knowledge=None):
    if input_df.shape[1] < 2:
        _logger.info("input too small({0} nodes), return empty dag".format(
            input_df.shape[1]))
        return showdag.empty_dag()

    cause_algorithm = conf.get("dag", "cause_algorithm")
    ci_func = conf.get("dag", "ci_func")
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


def _complete_graph(node_ids):
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for (i, j) in combinations(node_ids, 2):
        g.add_edge(i, j)
    return g

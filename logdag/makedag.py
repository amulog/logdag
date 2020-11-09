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
    ci_func = conf.get("dag", "ci_func")
    binarize = is_binarize(ci_func)
    # generate event set and evmap, and apply preprocessing
    # d_input, evmap = log2event.ts2input(conf, dt_range, area, binarize)
    input_df, evmap = log2event.makeinput(conf, dt_range, area, binarize)
    _logger.info("{0} pc input shape: {1}".format(jobname, input_df.shape))
    evmap.dump(conf, args)
    timer.lap("load-nodes")

    # generate initial graph
    node_ids = evmap.eids()
    g = _complete_graph(node_ids)
    if conf.getboolean("pc_prune", "do_pruning"):
        from . import prune
        n_edges_before = g.number_of_edges()
        init_graph = prune.prune_graph(g, conf, evmap)
        n_edges_after = init_graph.number_of_edges()
        _logger.info("{0} DAG edge pruning: ".format(jobname) + \
                     "{0} -> {1}".format(n_edges_before, n_edges_after))
    else:
        n_edges = g.number_of_edges()
        init_graph = g
        _logger.info("{0} DAG edge candidates: ".format(jobname) + \
                     "{0}".format(n_edges))
    timer.lap("prune-dag")

    # generate dag
    graph = estimate_dag(conf, input_df, ci_func, init_graph)
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

    ci_func = conf.get("dag", "ci_func")
    binarize = is_binarize(ci_func)
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


def estimate_dag(conf, input_df, ci_func, init_graph=None):
    if input_df.shape[1] >= 2:
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
            if init_graph is not None:
                _logger.warning("init_graph not used in lingam")
            from . import lingam_input
            return lingam_input.estimate(input_df)
    else:
        _logger.info("input too small({0} nodes), return empty dag".format(
            input_df.shape[1]))
        return showdag.empty_dag()


def is_binarize(ci_func):
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


def _complete_graph(node_ids):
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for (i, j) in combinations(node_ids, 2):
        g.add_edge(i, j)
    return g

# def pruning(g, conf, evmap):
#    import networkx as nx
#    n_edges_before = g.number_of_edges()
#    methods = config.getlist(conf, "pc_prune", "methods")
#    for method in methods:
#        if method == "network":
#            import json
#            fp = conf.get("pc_prune", "network_file")
#            with open(fp, "r") as f:
#                js = json.load(f)
#            g_network = nx.node_link_graph(js)
#            g = _pruning_network(g, g_network, evmap)
#        elif method == "same_overhost":
#            g = _pruning_overhost(g, evmap, nodes = None)
#        elif method == "ext_overhost":
#            nodes_prune = [nid for nid, evdef in evmap.items()
#                           if isinstance(evdef.gid, str)]
#            g = _pruning_overhost(g, evmap, nodes = nodes_prune)
#        else:
#            raise NotImplementedError
#    pass
#    n_edges_after = g.number_of_edges()
#    _logger.debug("DAG edge pruning: "
#                  "{0} -> {1}".format(n_edges_before, n_edges_after))
#    return g
#
#
# def _pruning_network(g_base, g_net, evmap):
#    """Prune edges based on topology network of hosts (g_net)."""
#    import networkx as nx
#    g_ret = nx.Graph()
#    for edge in g_base.edges():
#        src_host, dst_host = [evmap.evdef(node).host for node in edge]
#        if src_host == dst_host or g_net.has_edge(src_host, dst_host):
#            g_ret.add_edge(*edge)
#    return g_ret
#
#
# def _pruning_overhost(g, evmap, nodes = None):
#    """Prune edges between two nodes those are same event
#    but different hosts."""
#    if nodes is None:
#        nodes = evmap.eids()
#    for i, j in combinations(nodes, 2):
#        evdef_i = evmap.evdef(i)
#        evdef_j = evmap.evdef(j)
#        if evdef_i.host == evdef_j.host:
#            pass
#        else:
#            if g.has_edge(i, j):
#                g.remove_edge(i, j)
#                _logger.debug("prune {0} - {1}".format(evmap.evdef_str(i),
#                                                       evmap.evdef_str(j)))
#    return g

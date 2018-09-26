#!/usr/bin/env python
# coding: utf-8

import sys
import logging
from itertools import combinations

from . import arguments
from . import log2event
from . import pc_input
from . import showdag
from amulog import common
from amulog import config

_logger = logging.getLogger(__package__)


def makedag_main(args):
    conf, dt_range, area = args

    _logger.info("makedag job start ({0} - {1} in {2})".format(
        dt_range[0], dt_range[1], area))

    ci_func = conf.get("dag", "ci_func")
    binarize = is_binarize(ci_func)
    # generate event set and evmap, and apply preprocessing
    d_input, evmap = log2event.ts2input(conf, dt_range, area, binarize)
    _logger.info("{0} nodes for pc input".format(len(d_input)))
    evmap.dump(args)

    if conf.getboolean("pc_prune", "do_pruning"):
        node_ids = evmap.eids()
        g = _complete_graph(node_ids)
        init_graph = pruning(g, conf, evmap)
    else:
        init_graph = None
    graph = estimate_dag(conf, d_input, ci_func, init_graph)

    # record dag
    ldag = showdag.LogDAG(args, graph)
    ldag.dump()
    _logger.info("makedag job done, output {0}".format(
        arguments.ArgumentManager.dag_filepath(args)))
    return ldag


def estimate_dag(conf, d_input, ci_func, init_graph = None):
    if len(d_input) >= 2:
        cause_algorithm = conf.get("dag", "cause_algorithm")
        if cause_algorithm == "pc":
            # apply pc algorithm to estimate dag
            skel_method = conf.get("dag", "skeleton_method")
            skel_th = conf.getfloat("dag", "skeleton_threshold")
            skel_depth = conf.getint("dag", "skeleton_depth")
            skel_verbose = conf.getboolean("dag", "skeleton_verbose")
            graph = pc_input.pc(d_input, skel_th, ci_func, skel_method,
                    skel_depth, skel_verbose, init_graph)
        elif cause_algorithm == "lingam":
            if init_graph is not None:
                _logger.warning("init_graph not used in lingam")
            from . import lingam_input
            graph = lingam_input.estimate(d_input)
    else:
        _logger.info("input too small({0} nodes), return empty dag".format(
            len(d_input)))
        graph = showdag.empty_dag()

    return graph


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


def pruning(g, conf, evmap):
    import networkx as nx
    n_edges_before = g.number_of_edges()
    methods = config.getlist(conf, "pc_prune", "methods")
    for method in methods:
        if method == "network":
            import json
            fp = conf.get("pc_prune", "network_file")
            with open(fp, "r") as f:
                js = json.load(f)
            g_network = nx.node_link_graph(js)
            g = _pruning_network(g, g_network, evmap)
        elif method == "same_overhost":
            g = _pruning_overhost(g, evmap, nodes = None)
        elif method == "ext_overhost":
            nodes_prune = [nid for nid, evdef in evmap.items()
                           if isinstance(evdef.gid, str)]
            g = _pruning_overhost(g, evmap, nodes = nodes_prune)
        else:
            raise NotImplementedError
    pass
    n_edges_after = g.number_of_edges()
    _logger.debug("DAG edge pruning: "
                  "{0} -> {1}".format(n_edges_before, n_edges_after))
    return g


def _pruning_network(g_base, g_net, evmap):
    """Prune edges based on topology network of hosts (g_net)."""
    import networkx as nx
    g_ret = nx.Graph()
    for edge in g_base.edges():
        src_host, dst_host = [evmap.evdef(node).host for node in edge]
        if src_host == dst_host or g_net.has_edge(src_host, dst_host):
            g_ret.add_edge(*edge)
    return g_ret


def _pruning_overhost(g, evmap, nodes = None):
    """Prune edges between two nodes those are same event
    but different hosts."""
    if nodes is None:
        nodes = evmap.eids()
    for i, j in combinations(nodes, 2):
        evdef_i = evmap.evdef(i)
        evdef_j = evmap.evdef(j)
        if evdef_i.host == evdef_j.host:
            pass
        else:
            if g.has_edge(i, j):
                g.remove_edge(i, j)
                _logger.debug("prune {0} - {1}".format(evmap.evdef_str(i),
                                                       evmap.evdef_str(j)))
    return g



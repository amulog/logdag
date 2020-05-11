#!/usr/bin/env python
# coding: utf-8

import logging
import numpy as np
import networkx as nx

_logger = logging.getLogger(__package__)


def estimate(data, skel_th, ci_func, skel_method, pc_depth,
             skel_verbose, init_graph):
    import pcalg
    from gsq.ci_tests import ci_test_bin

    pc_args = {"indep_test_func": ci_test_bin,
               "data_matrix": data.values,
               "alpha": skel_th,
               "method": skel_method,
               "verbose": skel_verbose}
    if pc_depth is not None and pc_depth >= 0:
        args["max_reach"] = pc_depth
    if init_graph is not None:
        args["init_graph"] = init_graph
    (g, sep_set) = pcalg.estimate_skeleton(**args)

    # TODO: Finish implementation (prune,adjust,select,relabel)
    g = pcalg.estimate_cpdag(skel_graph=g, sep_set=sep_set)
    # g,data,m1 = prune(g,data)
    # subgraphs = nx.weakly_connected_components(g)
    # for nodes in subgraphs:

		# d,subgraph,m2 = adjust(data,graph,nodes)
		# best = select(subgraph,d)
		# best = relabel(best,m2)
		# r = nx.disjoint_union(r,best)

    # return r

    return g


def pc_fisherz(data, threshold, skel_method, pc_depth=None,



    lingam = lingam_fast.LiNGAM()
    ret = lingam.fit(data, use_sklearn = True, algorithm="fast",
                     reg_type = "lasso")
    graph = lingam.visualize(lib = "networkx")
    return graph


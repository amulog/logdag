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
        pc_args["max_reach"] = pc_depth
    if init_graph is not None:
        pc_args["init_graph"] = init_graph
    (g, sep_set) = pcalg.estimate_skeleton(**pc_args)

    # TODO: to replace something with mixedlingam
    g = pcalg.estimate_cpdag(skel_graph=g, sep_set=sep_set)
    # end

    return g


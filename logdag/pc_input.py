#!/usr/bin/env python
# coding: utf-8

import logging
import numpy as np
import networkx as nx

_logger = logging.getLogger(__package__)


def pc(data, threshold, mode="gsq", skel_method="stable",
       pc_depth=None, verbose=False, prior_knowledge=None):

    if prior_knowledge:
        init_graph = prior_knowledge.pruned_initial_skeleton()
    else:
        init_graph = nx.complete_graph(data.columns)

    if mode == "gsq":
        from gsq.ci_tests import ci_test_bin
        func = ci_test_bin
        data = binarize_input(data)
    elif mode in ("fisherz", "fisherz_bin"):
        from citestfz.ci_tests import ci_test_gauss
        func = ci_test_gauss
    else:
        raise ValueError("ci_func invalid ({0})".format(mode))
    return estimate_dag(data, threshold, func, skel_method,
                        pc_depth, verbose, init_graph)


# def pc(data, threshold, mode="pylib", skel_method="default",
#       pc_depth=None, verbose=False, init_graph=None):
#    if mode == "gsq_rlib":
#        if init_graph is not None:
#            _logger.warning("init_graph not used in gsq_rlib")
#        graph = pc_rlib(data, threshold, skel_method, verbose)
#    elif mode == "gsq":
#        graph = pc_gsq(data, threshold, skel_method,
#                       pc_depth, verbose, init_graph)
#    elif mode in ("fisherz", "fisherz_bin"):
#        graph = pc_fisherz(data, threshold, skel_method,
#                           pc_depth, verbose, init_graph)
#    else:
#        raise ValueError("ci_func invalid ({0})".format(mode))
#    return graph


def binarize_input(data):
    return data.apply(lambda s: s.map(lambda x: 1 if x >= 1 else 0))


def estimate_skeleton(data, threshold, func, skel_method="stable",
                      pc_depth=None, verbose=False, init_graph=None):
    import pcalg
    args = {"indep_test_func": func,
            "data_matrix": data.values,
            "alpha": threshold,
            "method": skel_method,
            "verbose": verbose}
    if pc_depth is not None and pc_depth >= 0:
        args["max_reach"] = pc_depth
    if init_graph is not None:
        args["init_graph"] = init_graph
    g, _ = pcalg.estimate_skeleton(**args)
    return g.to_directed()


def estimate_dag(data, threshold, func, skel_method="stable",
                 pc_depth=None, verbose=False, init_graph=None):

    import pcalg
    args = {"indep_test_func": func,
            "data_matrix": data.values,
            "alpha": threshold,
            "method": skel_method,
            "verbose": verbose}
    if pc_depth is not None and pc_depth >= 0:
        args["max_reach"] = pc_depth
    if init_graph is not None:
        args["init_graph"] = init_graph
    g, sep_set = pcalg.estimate_skeleton(**args)
    g = pcalg.estimate_cpdag(skel_graph=g, sep_set=sep_set)
    return g


#def pc_gsq(data, threshold, skel_method, pc_depth=None,
#           verbose=False, init_graph=None):
#    import pcalg
#    from gsq.ci_tests import ci_test_bin
#
#    args = {"indep_test_func": ci_test_bin,
#            "data_matrix": data.values,
#            "alpha": threshold,
#            "method": skel_method,
#            "verbose": verbose}
#    if pc_depth is not None and pc_depth >= 0:
#        args["max_reach"] = pc_depth
#    if init_graph is not None:
#        args["init_graph"] = init_graph
#    (g, sep_set) = pcalg.estimate_skeleton(**args)
#    g = pcalg.estimate_cpdag(skel_graph=g, sep_set=sep_set)
#    return g
#
#
#def pc_fisherz(data, threshold, skel_method, pc_depth=None,
#               verbose=False, init_graph=None):
#    import pcalg
#    # from ci_test.ci_tests import ci_test_gauss
#    from citestfz.ci_tests import ci_test_gauss
#
#    # dm = np.array([data for nid, data in sorted(data.items())]).transpose()
#    cm = np.corrcoef(data.T)
#    args = {"indep_test_func": ci_test_gauss,
#            "data_matrix": data.values,
#            "corr_matrix": cm,
#            "alpha": threshold,
#            "method": skel_method,
#            "verbose": verbose}
#    if pc_depth is not None and pc_depth >= 0:
#        args["max_reach"] = pc_depth
#    if init_graph is not None:
#        args["init_graph"] = init_graph
#    (g, sep_set) = pcalg.estimate_skeleton(**args)
#    g = pcalg.estimate_cpdag(skel_graph=g, sep_set=sep_set)
#    return g
#
#
#def pc_rlib(data, threshold, skel_method, verbose):
#    import pandas
#    import pyper
#
#    if skel_method == "default":
#        method = "original"
#    else:
#        method = skel_method
#
#    r = pyper.R(use_pandas='True')
#    r("library(pcalg)")
#    r("library(graph)")
#
#    df = pandas.DataFrame(data)
#    r.assign("input.df", df)
#    r.assign("method", method)
#    r("evts = as.matrix(input.df)")
#    # print r("evts")
#    # r("t(evts)")
#
#    # r("save(evts, file='rtemp')")
#
#    r.assign("event.num", len(input_data))
#    r.assign("threshold", threshold)
#    r.assign("verbose.flag", verbose)
#
#    print(r("""
#        pc.result <- pc(suffStat = list(dm = evts, adaptDF = FALSE),
#            indepTest = binCItest, alpha = threshold, skel.method = method,
#            labels = as.character(seq(event.num)-1), verbose = verbose.flag)
#    """))
#    # print r("""
#    #    pc.result <- pc(suffStat = list(dm = evts, adaptDF = FALSE),
#    #        indepTest = binCItest, alpha = threshold,
#    #        labels = as.character(seq(event.num)-1), verbose = TRUE)
#    # """)
#
#    r("node.num <- length(nodes(pc.result@graph))")
#
#    g = nx.DiGraph()
#    for i in range(r.get("node.num")):
#        r.assign("i", i)
#        edges = r.get("pc.result@graph@edgeL[[as.character(i)]]$edges")
#        if edges is None:
#            pass
#        elif type(edges) == int:
#            g.add_edge(i, edges - 1)
#        elif type(edges) == np.ndarray:
#            for edge in edges:
#                g.add_edge(i, edge - 1)
#        else:
#            raise ValueError("edges is unknown type {0}".format(type(edges)))
#    return g

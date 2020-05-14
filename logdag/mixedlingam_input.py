#!/usr/bin/env python
# coding: utf-8

import logging
import numpy as np
import networkx as nx

from .bcause.bcause import select
from .bcause.graph.mixed_graph import MixedGraph

_logger = logging.getLogger(__package__)

def estimate(data, skel_th, ci_func, skel_method, pc_depth, skel_verbose, init_graph):
    import pcalg
    from gsq.ci_tests import ci_test_bin

    pc_args = {
        "indep_test_func": ci_test_bin,
        "data_matrix": data.values,
        "alpha": skel_th,
        "method": skel_method,
        "verbose": skel_verbose,
    }
    if pc_depth is not None and pc_depth >= 0:
        pc_args["max_reach"] = pc_depth
    if init_graph is not None:
        pc_args["init_graph"] = init_graph
    (g, sep_set) = pcalg.estimate_skeleton(**pc_args)

    # TODO: MixedLiNGAM integration (select function)
    g = MixedGraph(pcalg.estimate_cpdag(skel_graph=g, sep_set=sep_set))
    g, data, m1 = prune(g, data)
    r = MixedGraph()
    subgraphs = nx.weakly_connected_components(g)
    for nodes in subgraphs:
        d, subgraph, m2 = adjust(data, g, nodes)
        best = select(subgraph,d)
        best = relabel(best, m1, m2)
        r = nx.disjoint_union(r, best)

    return r


def prune(graph: MixedGraph, data):
    lone = list(nx.isolates(graph))
    graph.remove_nodes_from(lone)
    mapping = dict(zip(graph.nodes(), range(len(graph.nodes()))))
    graph = nx.relabel_nodes(graph, mapping)
    data = data.drop(data.columns[lone], axis=1)
    data.columns = [str(n) for n in graph.nodes()]
    return (graph, data, mapping)


def adjust(data, graph: MixedGraph, subgraph):
    d = data
    g = graph.copy()
    c = 0
    for N in graph.nodes():
        if N not in subgraph:
            g.remove_node(N)
            d = d.drop(d.columns[N - c], axis=1)
            c += 1
    m = dict(zip(g.nodes(), range(len(g.nodes()))))
    g = nx.relabel_nodes(g, m)
    return d, g, m


def relabel(graph: MixedGraph, map1, map2) -> MixedGraph:
    inv_map1 = {v: k for k, v in map1.items()}
    inv_map2 = {v: k for k, v in map2.items()}
    m = {k: inv_map1[inv_map2[k]] for k in graph.nodes()}
    g = nx.relabel_nodes(graph, m)
    return g

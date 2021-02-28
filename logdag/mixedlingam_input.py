#!/usr/bin/env python
# coding: utf-8

import logging
import numpy as np
import networkx as nx

from .bcause.bcause import select
from .bcause.graph.mixed_graph import MixedGraph

_logger = logging.getLogger(__package__)


def estimate(data, skel_th, skel_method, pc_depth, skel_verbose, init_graph, binary=True):
    import pcalg
    from gsq.ci_tests import ci_test_bin

    if binary:
        pc_data_matrix = data.values
    else:
        pc_data_matrix = data.apply(
            lambda s: s.map(lambda x: 1 if x >= 1 else 0)).values

    pc_args = {
        "indep_test_func": ci_test_bin,
        "data_matrix": pc_data_matrix,
        "alpha": skel_th,
        "method": skel_method,
        "verbose": skel_verbose,
    }
    if pc_depth is not None and pc_depth >= 0:
        pc_args["max_reach"] = pc_depth
    if init_graph is not None:
        pc_args["init_graph"] = init_graph

    (graph, sep_set) = pcalg.estimate_skeleton(**pc_args)
    mapping = {k: v for k, v in zip(graph.nodes(), data.columns.astype(int))}
    graph = MixedGraph(nx.relabel_nodes(graph, mapping))
    subgraphs = [
        graph.subgraph(nodes)
        for nodes in nx.weakly_connected_components(graph)
        if len(nodes) > 1
    ]
    graph_final = MixedGraph()
    for sub in subgraphs:
        graph_n = MixedGraph(sub)
        # import pdb; pdb.set_trace()
        data_n = data[list(graph_n.nodes())]
        # data_n = data[data.columns[np.array(graph_n.nodes())]]
        graph_n, data_n, mapping_n, invmap_n = normalize(graph_n, data_n)
        graph_n = select(graph_n, data_n)
        # invmap_n = {v: k for k, v in mapping_n.items()}
        graph_n = nx.relabel_nodes(graph_n, invmap_n)
        graph_final = nx.compose(graph_final, graph_n)

    return graph_final


def normalize(graph: MixedGraph, data):
    mapping = dict(zip(graph.nodes(), range(len(graph.nodes()))))
    inv_mapping = {v: k for k, v in mapping.items()}
    graph = nx.relabel_nodes(graph, mapping)
    data.columns = [str(n) for n in graph.nodes()]
    return graph, data, mapping, inv_mapping


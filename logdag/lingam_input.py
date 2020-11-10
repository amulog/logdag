# coding: utf-8

"""
Use LiNGAM https://github.com/cdt15/lingam
"""

import numpy as np
import networkx as nx
from itertools import combinations


def estimate(data, algorithm="ica", lower_limit=0.01, init_graph=None):
    """Generate DAG with LiNGAM"""
    import lingam
    if algorithm == "ica":
        if init_graph is not None:
            raise Warning("ICA-LiNGAM does not use prior knowledge")
        model = lingam.ICALiNGAM()
    elif algorithm == "direct":
        if init_graph is None:
            model = lingam.DirectLiNGAM()
        else:
            pmatrix = _convert_init_graph(init_graph)
            model = lingam.DirectLiNGAM(prior_knowledge=pmatrix)
    else:
        raise ValueError("invalid lingam algorithm name")

    model.fit(data)
    adj = np.nan_tu_num(model.adjacency_matrix_)
    g = nx.DiGraph()
    for i in range(adj.shape[0]):
        g.add_node(i)

    idx = np.abs(adj) > lower_limit
    dirs = np.where(idx)
    for to, from_, coef in zip(dirs[0], dirs[1], adj[idx]):
        g.add_edge(from_, to, weight=coef, label=str(round(coef, 2)))

    return g


def estimate_corr(data, algorithm="ica", lower_limit=0.01, init_graph=None):
    """Generate DAG of pair-wise LiNGAM coefficient"""
    import lingam

    def _model(alg):
        if alg == "ica":
            return lingam.ICALiNGAM()
        elif alg == "direct":
            return lingam.DirectLiNGAM()
        else:
            raise ValueError("invalid lingam algorithm name")

    g = nx.DiGraph()
    for i, j in combinations(data.columns, 2):
        if init_graph is not None and not init_graph.has_edge(i, j):
            # pruned in init_graph, pass
            continue

        tmp_data = data.loc([i, j])
        model = _model(algorithm)
        model.fit(tmp_data)
        adj = np.nan_to_num(model.adjacency_matrix_)

        idx = np.abs(adj) > lower_limit
        dirs = np.where(idx)
        for to, from_, coef in zip(dirs[0], dirs[1], adj[idx]):
            g.add_edge(from_, to, weight=coef, label=str(round(coef, 2)))

    return g


def _convert_init_graph(init_graph):
    from lingam.utils import make_prior_knowledge
    n_nodes = init_graph.number_of_nodes()
    no_paths = []
    for to, from_ in combinations(range(n_nodes), 2):
        if not init_graph.has_edge(to, from_):
            no_paths.append((to, from_))
    return make_prior_knowledge(n_nodes, no_paths=no_paths)

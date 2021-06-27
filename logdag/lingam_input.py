# coding: utf-8

"""
Use LiNGAM https://github.com/cdt15/lingam
"""

import numpy as np
import networkx as nx
from itertools import combinations


def _fit_back(data, cls, kwargs, limit=3):
    cnt = 0
    while True:
        try:
            model = cls(**kwargs)
            model.fit(data)
            return model
        except np.linalg.LinAlgError:
            cnt += 1
            if cnt >= limit:
                return None


def estimate(data, algorithm="ica", lower_limit=0.01,
             ica_max_iter=1000, prior_knowledge=None):
    """Generate DAG with LiNGAM"""
    import lingam
    if algorithm == "ica":
        if prior_knowledge is not None:
            raise Warning("ICA-LiNGAM does not use prior knowledge")
        kwargs = {"max_iter": ica_max_iter}
        model = _fit_back(data, lingam.ICALiNGAM, kwargs)
    elif algorithm == "direct":
        if prior_knowledge is None:
            kwargs = {}
        else:
            pmatrix = prior_knowledge.lingam_prior_knowledge()
            kwargs = {"prior_knowledge": pmatrix}
        model = _fit_back(data, lingam.DirectLiNGAM, kwargs)
    else:
        raise ValueError("invalid lingam algorithm name")

    if model is None:
        return None

    adj = np.nan_to_num(model.adjacency_matrix_)
    g = nx.DiGraph()
    for i in range(adj.shape[0]):
        g.add_node(i)

    idx = np.abs(adj) > lower_limit
    dirs = np.where(idx)
    for to_idx, from_idx, coef in zip(dirs[0], dirs[1], adj[idx]):
        to = data.columns[to_idx]
        from_ = data.columns[from_idx]
        g.add_edge(from_, to, weight=coef, label=str(round(coef, 2)))

    return g


def estimate_corr(data, algorithm="ica", lower_limit=0.01, prior_knowledge=None):
    """Generate DAG of pair-wise LiNGAM coefficient"""
    import lingam

    def _model(alg, _kwargs):
        if alg == "ica":
            return lingam.ICALiNGAM(**_kwargs)
        elif alg == "direct":
            return lingam.DirectLiNGAM(**_kwargs)
        else:
            raise ValueError("invalid lingam algorithm name")

    g = nx.DiGraph()
    g.add_nodes_from(data.columns)
    for i, j in combinations(data.columns, 2):
        if algorithm == "direct" and prior_knowledge:
            pmatrix = prior_knowledge.lingam_prior_knowledge(node_ids=[i, j])
            kwargs = {"prior_knowledge": pmatrix}
        else:
            kwargs = {}

        tmp_data = data[[i, j]]
        model = _model(algorithm, kwargs)
        model.fit(tmp_data)
        adj = np.nan_to_num(model.adjacency_matrix_)

        idx = np.abs(adj) > lower_limit
        dirs = np.where(idx)
        for to_idx, from_idx, coef in zip(dirs[0], dirs[1], adj[idx]):
            to = tmp_data.columns[to_idx]
            from_ = tmp_data.columns[from_idx]
            g.add_edge(from_, to, weight=coef, label=str(round(coef, 2)))

    return g


#def _convert_init_graph(init_graph):
#    from lingam.utils import make_prior_knowledge
#    n_nodes = init_graph.number_of_nodes()
#    no_paths = []
#    for to, from_ in combinations(range(n_nodes), 2):
#        if not init_graph.has_edge(to, from_):
#            no_paths.append((to, from_))
#    return make_prior_knowledge(n_nodes, no_paths=no_paths)

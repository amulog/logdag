# coding: utf-8

"""
Use LiNGAM https://github.com/cdt15/lingam
"""

import warnings

import numpy as np
import networkx as nx
from itertools import combinations


def _import_lingam():
    """Import the optional lingam package with an actionable error message."""
    try:
        import lingam
    except ImportError as e:
        raise ImportError(
            "cause_algorithm lingam / lingam-corr needs the optional lingam "
            "package: pip install logdag[lingam] (note that lingam pins "
            "scipy<=1.13.1, which has no wheel for Python >= 3.13)"
        ) from e
    return lingam


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


def _add_nodes(g, columns):
    """Add the input columns of ``data`` to graph ``g`` as node ids.

    The node ids of an estimated DAG are the column labels of the input
    frame, i.e. the event ids of the evmap -- the same convention pc_input
    follows through its init_graph.
    """
    g.add_nodes_from(int(column) for column in columns)


def _add_edges(g, columns, adj, lower_limit):
    """Add the edges of adjacency matrix ``adj`` to graph ``g``.

    Node ids and edge weights are cast to Python types on purpose. A pandas
    column label taken by position is a numpy scalar, which json.dumps
    cannot serialize (output_dag_format = json). networkx keeps such a numpy
    int as the key of the inner adjacency dict even when an equal Python int
    is already a node, so an uncast label stays invisible in g.nodes() and
    only surfaces as the target of an edge.
    """
    idx = np.abs(adj) > lower_limit
    dirs = np.where(idx)
    for to_idx, from_idx, coef in zip(dirs[0], dirs[1], adj[idx]):
        to = int(columns[to_idx])
        from_ = int(columns[from_idx])
        coef = float(coef)
        g.add_edge(from_, to, weight=coef, label=str(round(coef, 2)))


def estimate(data, algorithm="ica", lower_limit=0.01,
             ica_max_iter=1000, prior_knowledge=None):
    """Generate DAG with LiNGAM"""
    lingam = _import_lingam()
    if algorithm == "ica":
        if prior_knowledge is not None:
            warnings.warn("ICA-LiNGAM does not use prior knowledge")
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
    _add_nodes(g, data.columns)
    _add_edges(g, data.columns, adj, lower_limit)

    return g


def estimate_corr(data, algorithm="ica", lower_limit=0.01, prior_knowledge=None):
    """Generate DAG of pair-wise LiNGAM coefficient"""
    lingam = _import_lingam()

    def _model(alg, _kwargs):
        if alg == "ica":
            return lingam.ICALiNGAM(**_kwargs)
        elif alg == "direct":
            return lingam.DirectLiNGAM(**_kwargs)
        else:
            raise ValueError("invalid lingam algorithm name")

    g = nx.DiGraph()
    _add_nodes(g, data.columns)
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

        _add_edges(g, tmp_data.columns, adj, lower_limit)

    return g


#def _convert_init_graph(init_graph):
#    from lingam.utils import make_prior_knowledge
#    n_nodes = init_graph.number_of_nodes()
#    no_paths = []
#    for to, from_ in combinations(range(n_nodes), 2):
#        if not init_graph.has_edge(to, from_):
#            no_paths.append((to, from_))
#    return make_prior_knowledge(n_nodes, no_paths=no_paths)

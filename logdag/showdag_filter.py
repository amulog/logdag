#!/usr/bin/env python
# coding: utf-8

import networkx as nx

FUNCTIONS = ["no_isolated", "to_undirected",
             "directed", "undirected",
             "across_host", "witin_host",
             "subgraph_with_log", "subgraph_with_snmp",
             "ate_prune"]


#def apply(ldag, l_filtername, th=None):
#    g = ldag.graph
#
#    # make to_undirected the first filter
#    if "to_undirected" in l_filtername:
#        l_filtername.remove("to_undirected")
#        l_filtername = ["to_undirected"] + l_filtername
#
#    # make no_isolated the last filter
#    if "no_isolated" in l_filtername:
#        l_filtername.remove("no_isolated")
#        l_filtername.append("no_isolated")
#
#    for funcname in l_filtername:
#        assert funcname in FUNCTIONS
#        g = eval(funcname)(graph=g, ldag=ldag, th=th)
#    return g


def no_isolated(graph, **kwargs):
    ret = graph.copy()
    nodes = set(graph.nodes())
    nodes_connected = set()
    for (u, v) in graph.edges():
        nodes_connected.add(u)
        nodes_connected.add(v)
    for n in (nodes - nodes_connected):
        ret.remove_node(n)
    return ret


def to_undirected(graph, **kwargs):
    return graph.to_undirected()


def _sep_directed(graph, **kwargs):
    g_di = nx.DiGraph()
    g_nodi = nx.Graph()
    l_temp_edge = []
    for edge in graph.edges(data=True):
        rev_edge = (edge[1], edge[0])
        if rev_edge in l_temp_edge:
            g_nodi.add_edges_from([edge])
            l_temp_edge.remove(rev_edge)
        else:
            l_temp_edge.append(edge[0:2])
    g_di.add_edges_from(l_temp_edge)
    return g_di, g_nodi


def directed(graph, **kwargs):
    return _sep_directed(graph)[0]


def undirected(graph, **kwargs):
    return _sep_directed(graph)[1]


def _sep_across_host(graph, ldag=None, **kwargs):
    if ldag is None:
        raise ValueError("LogDAG object is needed for sep_across_host")
    g_same = nx.DiGraph()
    g_diff = nx.DiGraph()
    for edge in graph.edges(data=True):
        src_evdef, dst_evdef = ldag.edge_evdef(edge)
        if src_evdef._host == dst_evdef._host:
            g_same.add_edges_from([edge])
        else:
            g_diff.add_edges_from([edge])
    return g_same, g_diff


def across_host(graph, **kwargs):
    return _sep_across_host(graph, **kwargs)[1]


def within_host(graph, **kwargs):
    return _sep_across_host(graph, **kwargs)[0]


def subgraph_with_log(graph, ldag=None, **kwargs):
    ret = nx.create_empty_copy(graph)
    for comp in nx.connected_components(graph.to_undirected()):
        sg = graph.subgraph(comp)
        for edge in sg.edges():
            src_evdef, dst_evdef = ldag.edge_evdef(edge)
            if "log" in (src_evdef.source, dst_evdef.source):
                # add the subgraph
                ret.add_edges_from(sg.edges(data=True))
                break
        else:
            # pass the subgraph
            pass
    return ret


def subgraph_with_snmp(graph, ldag=None, **kwargs):
    ret = nx.create_empty_copy(graph)
    for comp in nx.connected_components(graph.to_undirected()):
        sg = graph.subgraph(comp)
        for edge in sg.edges():
            src_evdef, dst_evdef = ldag.edge_evdef(edge)
            if "snmp" in (src_evdef.source, dst_evdef.source):
                # add the subgraph
                ret.add_edges_from(sg.edges(data=True))
                break
        else:
            # pass the subgraph
            pass
    return ret


def ate_prune(graph, th=None, **kwargs):
    """Prune edges with smaller ATE (average treatment effect).
    Effective if DAG estimation algorithm is LiNGAM."""

    ret = graph.copy()
    try:
        edge_label = {(u, v): d["label"]
                      for (u, v, d) in graph.edges(data=True)}
        for (src, dst), val in edge_label.items():
            if float(val) < th:
                ret.remove_edge(src, dst)
        return ret
    except KeyError:
        return nx.create_empty_copy(graph)



#!/usr/bin/env python
# coding: utf-8

import networkx as nx


def relabel_nodes(graph, ldag):
    mapping = {}
    for node in graph.nodes():
        evdef = ldag.node_evdef(node)
        mapping[node] = str(evdef)
    return nx.relabel_nodes(graph, mapping, copy=True)


def graph_nx(output, graph):
    ag = nx.nx_agraph.to_agraph(graph)
    ag.draw(output, prog='circo')
    return output

#!/usr/bin/env python
# coding: utf-8


import networkx as nx

from logdag import showdag


def search_gid(conf, gid):
    l_result = []
    for r in showdag.iter_results(conf):
        for edge in r.graph.edges():
        #g = nx.Graph(r.graph)
        #for edge in g.edges():
            temp_gids = [evdef.gid for evdef in r.edge_info(edge)]
            if gid in temp_gids:
                l_result.append((r, edge))
    return l_result


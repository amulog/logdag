#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict


def count_node_label(conf):
    from amulog import log_db
    ld = log_db.LogData(conf)
    from amulog import lt_label
    ll = lt_label.init_ltlabel(conf)

    d_group = defaultdict(int)
    from logdag import showdag
    for r in showdag.iter_results(conf):
        r.load_ltlabel(conf, ld = ld, ll = ll)
        for node in r.graph.nodes():
            evdef = r.node_info(node)
            node_group = r._label_group_ltg(evdef.gid)
            d_group[node_group] += 1
    return d_group


def count_edge_label(conf):
    from amulog import log_db
    ld = log_db.LogData(conf)
    from amulog import lt_label
    ll = lt_label.init_ltlabel(conf)

    d_group = defaultdict(int)
    from logdag import showdag
    for r in showdag.iter_results(conf):
        r.load_ltlabel(conf, ld = ld, ll = ll)
        g = r.graph.to_undirected()
        for edge in g.edges():
            src_evdef, dst_evdef = r.edge_info(edge)
            src_group = r._label_group_ltg(src_evdef.gid)
            d_group[src_group] += 1
            dst_group = r._label_group_ltg(dst_evdef.gid)
            d_group[dst_group] += 1
    return d_group

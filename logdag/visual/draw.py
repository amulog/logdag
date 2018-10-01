#!/usr/bin/env python
# coding: utf-8

import networkx as nx


def relabel_graph(conf, graph, evmap):
    from amulog import log_db
    from amulog import lt_label
    ld = log_db.LogData(conf)
    ll = lt_label.init_ltlabel(conf)
    default_label = conf.get("visual", "ltlabel_default_label")

    def _label_ltg(ll, gid, default_label):
        label = ll.get_ltg_label(gid, ld.ltg_members(gid))
        if label is None:
            label = default_label
        return label

    mapping = {}
    for node in graph.nodes():
        info = evmap.evdef(node)
        label = _label_ltg(ll, info.gid, default_label)
        if label is None:
            mapping[node] = "{0}, {1}".format(info.gid, info.host)
        else:
            mapping[node] = "{0}({1}), {2}".format(info.gid,
                                                   label, info.host)

    return nx.relabel_nodes(graph, mapping, copy = True)


def graph_nx(output, graph):
    ag = nx.nx_agraph.to_agraph(graph)
    ag.draw(output, prog = 'circo')
    return output

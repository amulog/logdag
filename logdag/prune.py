#!/usr/bin/env python
# coding: utf-8

import json
import networkx as nx


class MultiLayerTopology():

    def __init__(self, d_topology_fp, d_gid, default_group):
        self._topology = self._load_graph(d_topology_fp)
        self._d_node_layer = d_gid
        self._default_group = default_group

    @staticmethod
    def _load_graph(d_fp):
        topo = {}
        for name, fp in d_fp.items():
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    js = json.load(f)
                topo[name] = nx.node_link_graph(js)
            except IOError:
                topo[name] = nx.Graph()
        return topo

    def _get_layer(self, gid):
        if gid in self._d_node_layer:
            return self._d_node_layer[gid]
        else:
            return self._default_group

    def _is_adjacent(self, evdef1, evdef2):
        src1, gid1, host1 = evdef1
        src2, gid2, host2 = evdef2
        if host1 == host2:
            return True

        group1 = self._get_layer(gid1)
        group2 = self._get_layer(gid2)
        for group in (group1, group2):
            if group in self._topology:
                net = self._topology[group]
                if net.has_edge(host1, host2):
                    return True
        else:
            return False

    def prune(self, g_base, evmap):
        g_ret = nx.Graph()
        g_ret.add_nodes_from(g_base.nodes())
        for edge in g_base.edges():
            src_evdef, dst_evdef = [evmap.evdef(node) for node in edge]
            if self._is_adjacent(src_evdef, dst_evdef):
                g_ret.add_edge(*edge)
        return g_ret


class SingleLayerTopology():

    def __init__(self, topology_fp):
        self._topology = self._load_graph(topology_fp)

    @staticmethod
    def _load_graph(fp):
        with open(fp, 'r', encoding='utf-8') as f:
            js = json.load(f)
        return nx.node_link_graph(js)

    def prune(self, g_base, evmap):
        g_ret = nx.Graph()
        g_ret.add_nodes_from(g_base.nodes())
        for edge in g_base.edges():
            src_host, dst_host = [evmap.evdef(node).host for node in edge]
            if src_host == dst_host or \
                    self._topology.has_edge(src_host, dst_host):
                g_ret.add_edge(*edge)
        return g_ret


class Independent():

    def __init__(self):
        pass

    def prune(self, g_base, evmap):
        g_ret = nx.Graph()
        g_ret.add_nodes_from(g_base.nodes())
        for edge in g_base.edges():
            src_host, dst_host = [evmap.evdef(node).host for node in edge]
            if src_host == dst_host:
                g_ret.add_edge(*edge)
        return g_ret


def init_gid_layer(conf, amulog_conf, d_rule):
    from amulog import log_db
    from amulog import lt_label
    ld = log_db.LogData(amulog_conf)
    ll = lt_label.init_ltlabel(amulog_conf)
    gid_name = conf.get("database_amulog", "event_gid")

    d_gid = {}
    for gid in ld.iter_gid(gid_name):
        group = ll.get_gid_group(gid, gid_name, ld)
        if group in d_rule:
            layer = d_rule[group]
            d_gid[gid] = layer
        else:
            d_gid[gid] = None
    return d_gid


def init_pruner(conf):
    from amulog import config
    l_pruner = []
    methods = config.getlist(conf, "pc_prune", "methods")
    amulog_conf = config.open_config(conf["database_amulog"]["source_conf"])
    for method in methods:
        if method == "topology":
            fp = conf.get("pc_prune", "single_network_file")
            l_pruner.append(SingleLayerTopology(fp))
        elif method == "multi-topology":
            d_fp = {}
            files = config.getlist(conf, "pc_prune", "multi_network_file")
            for group, fp in [s.split(":") for s in files]:
                d_fp[group] = fp
            rulestr = config.getlist(conf, "pc_prune", "multi_network_group")
            d_rule = {}
            for rule in rulestr:
                group, layer = rule.split(":")
                d_rule[group] = layer
            d_gid = init_gid_layer(conf, amulog_conf, d_rule)
            default_group = amulog_conf["visual"]["ltlabel_default_group"]
            l_pruner.append(MultiLayerTopology(d_fp, d_gid, default_group))
        elif method == "independent":
            l_pruner.append(Independent())
        else:
            raise NotImplementedError("invalid method {0}".format(method))

    return l_pruner


def prune_graph(g, conf, evmap):
    l_pruner = init_pruner(conf)
    for p in l_pruner:
        g = p.prune(g, evmap)
    return g



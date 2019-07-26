#!/usr/bin/env python
# coding: utf-8

import json
import networkx as nx


class LogMultiLayerTopology():
    _default_layer = "other"

    def __init__(self, d_topology_fp, d_rule):
        self._topology = self._load_graph(d_topology_fp)
        self._d_rule = d_rule

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

    def _get_layer(self, evdef):
        if evdef.group in self._d_rule:
            return self._d_rule[evdef.group]
        else:
            return self._default_layer

    def _is_adjacent(self, evdef1, evdef2):
        if evdef1.host == evdef2.host:
            return True

        layer1 = self._get_layer(evdef1)
        layer2 = self._get_layer(evdef2)
        for layer in (layer1, layer2):
            if layer in self._topology:
                net = self._topology[layer]
                if net.has_edge(evdef1.host, evdef2.host):
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
            l_pruner.append(LogMultiLayerTopology(d_fp, d_rule))
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



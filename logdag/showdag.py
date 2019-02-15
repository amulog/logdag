#!/usr/bin/env python
# coding: utf-8


import sys
import pickle
import networkx as nx
from collections import defaultdict

from . import arguments
from amulog import common

fmt_int = lambda x: "{:,d}".format(x)
fmt_ratio = lambda x: "{:.1f}".format(x)
fmt_int_ratio = lambda x, y: "{:,d}({:.1f}%)".format(x, y)


class LogDAG():

    def __init__(self, args, graph = None):
        self.args = args
        self.conf, self.dt_range, self.area = self.args
        self.name = arguments.args2name(self.args)
        self.graph = graph
        self._evmap_obj = None
        self._ll = None

    def _evmap(self):
        if self._evmap_obj is None:
            from . import log2event
            gid_name = self.conf.get("dag", "event_gid")
            evmap = log2event.EventDefinitionMap(gid_name)
            evmap.load(self.args)
            self._evmap_obj = evmap
        return self._evmap_obj

    def dump(self):
        fp = arguments.ArgumentManager.dag_filepath(self.args)
        with open(fp, 'wb') as f:
            pickle.dump(self.graph, f)

    def load(self):
        fp = arguments.ArgumentManager.dag_filepath(self.args)
        with open(fp, 'rb') as f:
            self.graph = pickle.load(f)

    def load_ltlabel(self, conf, ld = None, ll = None):
        from amulog import log_db
        from amulog import lt_label
        if ld is None:
            self._ld = log_db.LogData(conf)
        else:
            self._ld = ld
        if ll is None:
            self._ll = lt_label.init_ltlabel(conf)
        else:
            self._ll = ll
        self._default_label = conf.get("visual", "ltlabel_default_label")

    def _label_ltg(self, gid):
        if self._ll is None:
            return None
        else:
            label = self._ll.get_ltg_label(gid, self._ld.ltg_members(gid))
            if label is None:
                label = self._default_label
            return label

    def _label_group_ltg(self, gid):
        label = self._label_ltg(gid)
        group = self._ll.get_group(label)
        return group

    def number_of_nodes(self, graph = None):
        if graph is None:
            graph = self.graph
        return graph.number_of_nodes()

    def number_of_edges(self, graph = None):
        if graph is None:
            graph = self.graph
        #temp_graph = nx.Graph(graph)
        temp_graph = graph.to_undirected()
        return temp_graph.number_of_edges()

    def node_info(self, node):
        evmap = self._evmap()
        return evmap.evdef(node)

    def info2node(self, info):
        evmap = self._evmap()
        return evmap.get_eid(info)

    def edge_info(self, edge):
        evmap = self._evmap()
        return [evmap.evdef(node) for node in edge]

    def info2edge(self, t_info):
        evmap = self._evmap()
        return [evmap.get_eid(info) for info in t_info]

    def edges_directed(self, graph = None):
        """Returns subgraphs of input graph its edges by
        the availability of their directions.

        Args:
            graph (nx.DiGraph): A subgraph of self.graph.
                                If empty, self.graph is used.

        Returns:
            g_di (nx.DiGraph): A subgraph of directed edges.
            g_nodi (nx.Graph): A subgraph of undirected edges.
        """
        g_di = nx.DiGraph()
        g_nodi = nx.Graph()
        l_temp_edge = []
        if graph is None:
            graph = self.graph
        for edge in graph.edges():
            rev_edge = (edge[1], edge[0])
            if rev_edge in l_temp_edge:
                g_nodi.add_edge(*edge)
                l_temp_edge.remove(rev_edge)
            else:
                l_temp_edge.append(edge)
        g_di.add_edges_from(l_temp_edge)
        return g_di, g_nodi

    def edge_isdirected(self, edge, graph = None):
        if graph is None:
            graph = self.graph
        rev_edge = (edge[1], edge[0])
        if edge in graph.edges():
            if rev_edge in graph.edges():
                return False
            else:
                return True
        else:
            if rev_edge in graph.edges():
                raise ValueError("Edge not found, Reversed edge exists")
            else:
                raise ValueError("Edge not found")

    def edges_across_host(self, graph = None):
        """Returns subgraphs of input graph its edges by the consistency
        of the hosts of adjacent nodes.

        Args:
            graph (nx.DiGraph): A subgraph of self.graph.
                                If empty, self.graph is used.

        Returns:
            g_same (nx.DiGraph): A subgraph of edges among same hosts.
            g_diff (nx.DiGraph): A subgraph of edges across hosts.
        """
        g_same = nx.DiGraph()
        g_diff = nx.DiGraph()
        if graph is None:
            graph = self.graph
        for edge in graph.edges():
            src_info, dst_info = self.edge_info(edge)
            if src_info.host == dst_info.host:
                g_same.add_edge(*edge)
            else:
                g_diff.add_edge(*edge)
        return g_same, g_diff

    def connected_subgraphs(self, graph = None):
        if graph is None:
            graph = self.graph
        temp_graph = graph.to_undirected()
        return nx.connected_components(temp_graph)

    def edge_str(self, edge, graph = None):
        if graph is None:
            graph = self.graph
        src_node, dst_node = edge
        src_str = self.node_str(src_node)
        dst_str = self.node_str(dst_node)
        if self.edge_isdirected(edge, graph):
            return "{0} -> {1}".format(src_str, dst_str)
        else:
            return "{0} <-> {1}".format(src_str, dst_str)

    def node_str(self, node):
        info = self.node_info(node)
        label = self._label_ltg(info.gid)
        if label is None:
            return "[gid={0[0]}, host = {0[1]}]".format(info)
        else:
            return "[gid={0[0]}({1}), host = {0[1]}]".format(info, label)

    def edge_detail(self, edge, ld, head, foot):
        buf = ["## Edge {0}".format(self.edge_str(edge)),]
        for node in edge:
            buf.append(self.node_detail(node, ld, head, foot))
        return "\n".join(buf)

    def node_detail(self, node, ld, head, foot):
        gid, host = self.node_info(node)
        gid_name = self._evmap().gid_name
        if gid_name == "ltid":
            buf = ["# Node {0}: {1}".format(self.node_str(node),
                                          str(ld.lt(gid)))]
        else:
            buf = ["# Node {0}: {1}".format(self.node_str(node))]
        d = {gid_name: gid,
             "host": host,
             "top_dt": self.dt_range[0],
             "end_dt": self.dt_range[1],
             "head": head,
             "foot": foot}
        buf.append(ld.show_log_repr(**d))
        return "\n".join(buf)

    def ate_prune(self, threshold, graph = None):
        if graph is None:
            graph = self.graph
        ret = graph.copy()

        try:
            edge_label = {(u, v): d["label"]
                          for (u, v, d) in graph.edges(data = True)}
            for (src, dst), val in edge_label.items():
                if float(val) < threshold:
                    ret.remove_edge(src, dst)
            return ret
        except KeyError:
            return None

    def graph_no_orphan(self, graph = None):
        if graph is None:
            graph = self.graph
        ret = graph.copy()

        nodes = set(graph.nodes())
        no_orphan = set()
        for (u, v) in graph.edges():
            no_orphan.add(u)
            no_orphan.add(v)
        for n in (nodes - no_orphan):
            ret.remove_node(n)

        return ret

    def relabel_graph(self, graph = None):
        if graph is None:
            graph = self.graph

        mapping = {}
        for node in graph.nodes():
            info = self.node_info(node)
            label = self._label_ltg(info.gid)
            if label is None:
                mapping[node] = "{0}, {1}".format(info.gid, info.host)
            else:
                mapping[node] = "{0}({1}), {2}".format(info.gid,
                        label, info.host)
        return nx.relabel_nodes(graph, mapping, copy = True)

    def graph_nx(self, output, graph = None):
        if graph is None:
            graph = self.graph
        rgraph = self.relabel_graph(graph)

        ag = nx.nx_agraph.to_agraph(rgraph)
        ag.draw(output, prog = 'circo')
        return output

        #import matplotlib.pyplot as plt
        #plt.figure()
        #plt.axis('off')
        #pos = nx.spring_layout(g)
        #pos = nx.circular_layout(g)
        #pos = nx.nx_agraph.graphviz_layout(g, prog = 'neato')

        #node_color = [d.get("color", "black")
        #              for (n, d) in g.nodes(data = True)]
        #edge_color = [d.get("color", "black")
        #              for (u, v, d) in g.edges(data = True)]
        ##nx.draw_networkx_nodes(g, pos, node_size = 600,
        ##                       node_color = "w", edgecolors = node_color)
        #nx.draw_networkx_nodes(g, pos,
        #                       node_color = "w", edgecolors = node_color)
        ##nx.draw_networkx_edges(g, pos, width = 3.0, arrowsize = 30,
        ##                       edge_color = edge_color)
        #nx.draw_networkx_edges(g, pos, width = 3.0, arrowsize = 30,
        #                       edge_color = edge_color)
        #nx.draw_networkx_labels(g, pos, fontsize = 12, font_color = "k",
        #                        font_family = "Arial Black",
        #                        font_weight = "bold")
        ##nx.draw_networkx_labels(g, pos, fontsize = 18, font_color = "k",
        ##                        font_family = "Arial Black",
        ##                        font_weight = "bold")
        #try:
        #    edge_label = {(u, v): d["label"]
        #                  for (u, v, d) in g.edges(data = True)}
        #    #nx.draw_networkx_edge_labels(g, pos, edge_labels = edge_label,
        #    #                             fontsize = 18, font_color = "k",
        #    #                             font_family = "Arial Black")
        #    nx.draw_networkx_edge_labels(g, pos, edge_labels = edge_label,
        #                                 fontsize = 8, font_color = "k",
        #                                 font_family = "Arial Black")
        #except KeyError:
        #    pass
        #plt.savefig(output)
        #return output


# common functions

def empty_dag():
    """nx.DiGraph: Return empty graph."""
    return nx.DiGraph()


def iter_results(conf, src_dir = None, area = None):
    am = arguments.ArgumentManager(conf)
    am.load()
    for args in am:
        if area is None or args[2] == area:
            r = LogDAG(args)
            r.load()
            yield r


# functions for presentation

def show_edge_list(args):
    l_buf = []
    r = LogDAG(args)
    r.load()
    for edge in r.graph.edges():
        l_buf.append(r.edge_str(edge))
    return "\n".join(l_buf)


def show_edge_detail(args, head, tail):
    conf, dt_range, area = args
    from amulog import log_db
    ld = log_db.LogData(conf)
    l_buf = []
    r = LogDAG(args)
    r.load()
    for edge in r.graph.edges():
        l_buf.append(r.edge_detail(edge, ld, head, tail))
    return "\n\n".join(l_buf)


def show_graph(conf, args, output, lib = "networkx",
               threshold = None, ignore_orphan = False):
    if lib == "networkx":
        r = LogDAG(args)
        r.load()
        if threshold is not None:
            g = r.ate_prune(threshold)
        else:
            g = r.graph
        if ignore_orphan:
            g = r.graph_no_orphan(graph = g)
        r.load_ltlabel(conf)
        fp = r.graph_nx(output, graph = g)
        return fp
    else:
        raise NotImplementedError


def list_results(conf, src_dir = None):
    table = []
    table.append(["datetime", "area", "nodes", "edges", "name"])
    for r in iter_results(conf, src_dir):
        c, dt_range, area = r.args
        table.append([str(dt_range[0]), str(area),
                      str(r.number_of_nodes()), str(r.number_of_edges()),
                      r.name])
    return common.cli_table(table)


def list_results_byday(conf, src_dir = None):
    table = []
    table.append(["datetime", "nodes", "edges"])
    d_date = {}
    for r in iter_results(conf, src_dir):
        c, dt_range, area = r.args
        d = {"nodes": r.number_of_nodes(),
             "edges": r.number_of_edges()}
        if dt_range in d_date:
            for k in d:
                d_date[dt_range][k] += d[k]
        else:
            d_date[dt_range] = d

    for k, v in sorted(d_date.items(), key = lambda x: x[0]):
        table.append([str(k[0]), v["nodes"], v["edges"]])
    return common.cli_table(table)


def show_results_sum(conf, src_dir = None):
    node_num = 0
    edge_num = 0
    di_num = 0
    didiff_num = 0
    nodi_num = 0
    nodidiff_num = 0

    for r in iter_results(conf):
        c, dt_range, area = r.args
        g_di, g_nodi = r.edges_directed()
        node_num += r.number_of_nodes()
        edge_num += r.number_of_edges()
        di_num += r.number_of_edges(g_di)
        didiff_num += r.number_of_edges(r.edges_across_host(g_di)[1])
        nodi_num += r.number_of_edges(g_nodi)
        nodidiff_num += r.number_of_edges(r.edges_across_host(g_nodi)[1])

    table = []
    table.append(["number of events (nodes)", fmt_int(node_num), ""])
    table.append(["number of directed edges", fmt_int(di_num),
                  fmt_ratio(100.0 * di_num / edge_num)])
    table.append(["number of directed edges across hosts",
                  fmt_int(didiff_num),
                  fmt_ratio(100.0 * didiff_num / edge_num)])
    table.append(["number of undirected edges", fmt_int(nodi_num),
                  fmt_ratio(100.0 * nodi_num / edge_num)])
    table.append(["number of undirected edges across hosts",
                  fmt_int(nodidiff_num),
                  fmt_ratio(100.0 * nodidiff_num / edge_num)])
    table.append(["number of all edges", fmt_int(edge_num), ""])
    return common.cli_table(table, align = "right")


def list_netsize(conf):
    l_buf = []
    src_dir = conf.get("dag", "output_dir")
    for r in iter_results(conf):
        d_size = defaultdict(int)
        for net in r.connected_subgraphs():
            d_size[len(net)] += 1
        buf = []
        for size, cnt in sorted(d_size.items(), reverse = True):
            if cnt == 1:
                buf.append(str(size))
            else:
                buf.append("{0}x{1}".format(size, cnt))
        l_buf.append("{0} : {1}".format(r.name, ", ".join(buf)))
    return "\n".join(l_buf)


def show_netsize_dist(conf):
    l_buf = []
    src_dir = conf.get("dag", "output_dir")
    d_size = defaultdict(int)
    for r in iter_results(conf):
        for net in r.connected_subgraphs():
            d_size[len(net)] += 1
    return "\n".join(["{0} {1}".format(size, cnt)
                      for size, cnt in d_size.items()])


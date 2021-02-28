#!/usr/bin/env python
# coding: utf-8


import pickle
import networkx as nx
from collections import defaultdict

from . import arguments
from . import log2event
from . import showdag_filter
from amulog import common

fmt_int = lambda x: "{:,d}".format(x)
fmt_ratio = lambda x: "{:.1f}".format(x)
fmt_int_ratio = lambda x, y: "{:,d}({:.1f}%)".format(x, y)


class LogDAG:

    def __init__(self, args, graph=None):
        self.args = args
        self.conf, self.dt_range, self.area = self.args
        self.name = arguments.args2name(self.args)
        self.graph = graph

        self._evmap_obj = None
        self._d_el = None

    @classmethod
    def dag_path(cls, args):
        conf = args[0]
        dag_format = conf["dag"]["output_dag_format"]
        fp = arguments.ArgumentManager.dag_path(conf, args,
                                                ext=dag_format)
        return fp

    def _evmap(self):
        if self._evmap_obj is None:
            evmap = log2event.EventDefinitionMap()
            evmap.load(self.conf, self.args)
            self._evmap_obj = evmap
        return self._evmap_obj

    def _evloader(self):
        if self._d_el is None:
            self._d_el = log2event.init_evloaders(self.conf)
        return self._d_el

    def dump(self):
        dag_format = self.conf["dag"]["output_dag_format"]
        fp = self.dag_path(self.args)
        if dag_format == "pickle":
            with open(fp, 'wb') as f:
                pickle.dump(self.graph, f)
        elif dag_format == "json":
            import json
            with open(fp, 'w', encoding='utf-8') as f:
                obj = nx.node_link_data(self.graph)
                json.dump(obj, f)

    def load(self):
        dag_format = self.conf["dag"]["output_dag_format"]
        fp = arguments.ArgumentManager.dag_path(self.conf, self.args,
                                                ext=dag_format)
        try:
            if dag_format == "pickle":
                with open(fp, 'rb') as f:
                    self.graph = pickle.load(f)
            elif dag_format == "json":
                import json
                with open(fp, 'r', encoding='utf-8') as f:
                    obj = json.load(f)
                    self.graph = nx.node_link_graph(obj, directed=True)
        except:
            # compatibility
            fp = arguments.ArgumentManager.dag_path_old(self.args)
            with open(fp, 'rb') as f:
                self.graph = pickle.load(f)

    def number_of_nodes(self, graph=None):
        if graph is None:
            graph = self.graph
        return graph.number_of_nodes()

    def number_of_edges(self, graph=None):
        if graph is None:
            graph = self.graph
        # temp_graph = nx.Graph(graph)
        temp_graph = graph.to_undirected()
        return temp_graph.number_of_edges()

    def node_evdef(self, node):
        evmap = self._evmap()
        return evmap.evdef(node)

    def evdef2node(self, evdef):
        evmap = self._evmap()
        return evmap.get_eid(evdef)

    def edge_evdef(self, edge):
        evmap = self._evmap()
        return [evmap.evdef(node) for node in edge[0:2]]

    def evdef2edge(self, t_evdef):
        evmap = self._evmap()
        return [evmap.get_eid(evdef) for evdef in t_evdef]

    def edges_directed(self, graph=None):
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

    def edge_isdirected(self, edge, graph=None):
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

    def edges_across_host(self, graph=None):
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
            src_info, dst_info = self.edge_evdef(edge)
            if src_info._host == dst_info._host:
                g_same.add_edge(*edge)
            else:
                g_diff.add_edge(*edge)
        return g_same, g_diff

    def connected_subgraphs(self, graph=None):
        if graph is None:
            graph = self.graph
        temp_graph = graph.to_undirected()
        return nx.connected_components(temp_graph)

    def edge_str(self, edge, graph=None):
        if graph is None:
            graph = self.graph
        src_node, dst_node = edge
        src_str = self.node_str(src_node)
        dst_str = self.node_str(dst_node)
        if self.edge_isdirected(edge, graph):
            a = get_coefficient(edge, graph)
            return "{0} -[{2}]-> {1}".format(src_str, dst_str, a)
        else:
            return "{0} <-> {1}".format(src_str, dst_str)

    def node_str(self, node):
        return str(self.node_evdef(node))

    def edge_detail(self, edge, head, foot):
        buf = ["## Edge {0}".format(self.edge_str(edge)), ]
        for node in edge:
            buf.append(self.node_detail(node, head, foot))
        return "\n".join(buf)

    def node_detail(self, node, head, foot):
        evdef = self.node_evdef(node)
        buf = ["# Node {0}:".format(self.node_str(node)),
               log2event.evdef_instruction(self.conf, evdef,
                                           d_el=self._evloader()),
               log2event.evdef_detail(self.conf, evdef, self.dt_range,
                                      head, foot, d_el=self._evloader())]
        return "\n".join(buf)

    def ate_prune(self, threshold, graph=None):
        """Prune edges with smaller ATE (average treatment effect).
        Effective if DAG estimation algorithm is LiNGAM."""
        if graph is None:
            graph = self.graph
        ret = graph.copy()

        try:
            edge_label = {(u, v): d["weight"]
                          for (u, v, d) in graph.edges(data=True)}
            for (src, dst), val in edge_label.items():
                if float(val) < threshold:
                    ret.remove_edge(src, dst)
            return ret
        except KeyError:
            return None

    def graph_no_orphan(self, graph=None):
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

    def relabel(self, graph=None):
        if graph is None:
            graph = self.graph

        mapping = {}
        for node in graph.nodes():
            evdef = self.node_evdef(node)
            mapping[node] = str(evdef)
        return nx.relabel_nodes(graph, mapping, copy=True)

    def graph_nx(self, output, graph=None):
        if graph is None:
            graph = self.relabel(self.graph)

        ag = nx.nx_agraph.to_agraph(graph)
        ag.draw(output, prog='circo')
        return output


# common functions

def empty_dag():
    """nx.DiGraph: Return empty graph."""
    return nx.DiGraph()


def iter_results(conf, area=None):
    am = arguments.ArgumentManager(conf)
    am.load()
    for args in am:
        if area is None or args[2] == area:
            r = LogDAG(args)
            r.load()
            yield r


def isdirected(edge, graph):
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


def apply_filter(ldag, l_filtername, th=None, graph=None):
    from . import showdag_filter
    if graph is None:
        g = ldag.graph
    else:
        g = graph

    # make to_undirected the first filter
    if "to_undirected" in l_filtername:
        l_filtername.remove("to_undirected")
        l_filtername = ["to_undirected"] + l_filtername

    # make no_isolated the last filter
    if "no_isolated" in l_filtername:
        l_filtername.remove("no_isolated")
        l_filtername.append("no_isolated")

    for funcname in l_filtername:
        assert funcname in showdag_filter.FUNCTIONS
        g = eval("showdag_filter." + funcname)(graph=g, ldag=ldag, th=th)
    return g


# functions for presentation


def show_edge_detail(args, head, tail):
    l_buf = []
    r = LogDAG(args)
    r.load()
    for edge in r.graph.edges():
        l_buf.append(r.edge_detail(edge, head, tail))
    return "\n\n".join(l_buf)


#def list_subgraphs(args):
#    r = LogDAG(args)
#    r.load()
#
#    l_buf = []
#    separator = "\n\n"
#    iterobj = sorted(nx.connected_components(r.graph.to_undirected()),
#                     key=len, reverse=True)
#    for nodes in iterobj:
#        l_graph_buf = []
#        subg = nx.subgraph(r.graph, nodes)
#        for edge in subg.edges():
#            msg = r.edge_str(edge, r.graph)
#            l_graph_buf.append(msg)
#        l_buf.append("\n".join(l_graph_buf))
#    return separator.join(l_buf)


#def show_graph(conf, args, output, lib="networkx",
#               threshold=None, ignore_orphan=False):
#    if lib == "networkx":
#        r = LogDAG(args)
#        r.load()
#        if threshold is not None:
#            g = r.ate_prune(threshold)
#        else:
#            g = r.graph
#        if ignore_orphan:
#            g = r.graph_no_orphan(graph=g)
#        r.relabel()
#        fp = r.graph_nx(output, graph=g)
#        return fp
#    else:
#        raise NotImplementedError


def stat_groupby(conf, l_func, l_kwargs=None, dt_range=None, groupby=None):
    import numpy as np
    from . import dtutil
    d_group = defaultdict(list)
    am = arguments.ArgumentManager(conf)
    am.load()

    if dt_range is None:
        iterobj = am
    else:
        iterobj = am.args_in_time_range(dt_range)
    for args in iterobj:
        if groupby is None:
            key = arguments.args2name(args)
        elif groupby == "day":
            key = dtutil.shortstr(args[1][0])
        elif groupby == "area":
            key = args[2]
        else:
            raise NotImplementedError
        d_group[key].append(args)

    for key, l_args in d_group.items():
        data = []
        for args in l_args:
            ldag = LogDAG(args)
            ldag.load()
            if l_kwargs is None:
                data.append([func(ldag) for func in l_func])
            else:
                data.append([func(ldag, **kwargs)
                             for func, kwargs in zip(l_func, l_kwargs)])
        yield key, l_args, np.sum(data, axis=0)


def _apply_by_threshold(ldag, **kwargs):
    return ldag.number_of_edges(apply_filter(ldag, ["ate_prune"], **kwargs))


def stat_by_threshold(conf, thresholds, dt_range=None, groupby=None):
    import numpy as np
    l_func = []
    l_kwargs = []
    for th in thresholds:
        l_func.append(_apply_by_threshold)
        l_kwargs.append({"th": th})
    data = [v for _, _, v
            in stat_groupby(conf, l_func, l_kwargs=l_kwargs,
                            dt_range=dt_range, groupby=groupby)]
    return np.sum(data, axis=0)


#def list_results(conf, src_dir=None):
#    table = [["datetime", "area", "nodes", "edges", "name"], ]
#    for r in iter_results(conf, src_dir):
#        c, dt_range, area = r.args
#        table.append([str(dt_range[0]), str(area),
#                      str(r.number_of_nodes()), str(r.number_of_edges()),
#                      r.name])
#    return common.cli_table(table)
#
#
#def list_results_byday(conf, src_dir=None):
#    table = [["datetime", "nodes", "edges"], ]
#    d_date = {}
#    for r in iter_results(conf, src_dir):
#        c, dt_range, area = r.args
#        d = {"nodes": r.number_of_nodes(),
#             "edges": r.number_of_edges()}
#        if dt_range in d_date:
#            for k in d:
#                d_date[dt_range][k] += d[k]
#        else:
#            d_date[dt_range] = d
#
#    for k, v in sorted(d_date.items(), key=lambda x: x[0]):
#        table.append([str(k[0]), v["nodes"], v["edges"]])
#    return common.cli_table(table)


#def show_stats(conf, src_dir=None):
#    node_num = 0
#    edge_num = 0
#    di_num = 0
#    didiff_num = 0
#    nodi_num = 0
#    nodidiff_num = 0
#
#    for r in iter_results(conf):
#        c, dt_range, area = r.args
#        g_di, g_nodi = r.edges_directed()
#        node_num += r.number_of_nodes()
#        edge_num += r.number_of_edges()
#        di_num += r.number_of_edges(g_di)
#        didiff_num += r.number_of_edges(r.edges_across_host(g_di)[1])
#        nodi_num += r.number_of_edges(g_nodi)
#        nodidiff_num += r.number_of_edges(r.edges_across_host(g_nodi)[1])
#
#    table = []
#    table.append(["number of events (nodes)", fmt_int(node_num), ""])
#    table.append(["number of directed edges", fmt_int(di_num),
#                  fmt_ratio(100.0 * di_num / edge_num)])
#    table.append(["number of directed edges across hosts",
#                  fmt_int(didiff_num),
#                  fmt_ratio(100.0 * didiff_num / edge_num)])
#    table.append(["number of undirected edges", fmt_int(nodi_num),
#                  fmt_ratio(100.0 * nodi_num / edge_num)])
#    table.append(["number of undirected edges across hosts",
#                  fmt_int(nodidiff_num),
#                  fmt_ratio(100.0 * nodidiff_num / edge_num)])
#    table.append(["number of all edges", fmt_int(edge_num), ""])
#    return common.cli_table(table, align="right")


def list_netsize(conf):
    l_buf = []
    for r in iter_results(conf):
        d_size = defaultdict(int)
        for net in r.connected_subgraphs():
            d_size[len(net)] += 1
        buf = []
        for size, cnt in sorted(d_size.items(), reverse=True):
            if cnt == 1:
                buf.append(str(size))
            else:
                buf.append("{0}x{1}".format(size, cnt))
        l_buf.append("{0} : {1}".format(r.name, ", ".join(buf)))
    return "\n".join(l_buf)


def show_netsize_dist(conf):
    d_size = defaultdict(int)
    for r in iter_results(conf):
        for net in r.connected_subgraphs():
            d_size[len(net)] += 1
    return "\n".join(["{0} {1}".format(size, cnt)
                      for size, cnt in d_size.items()])


def get_coefficient(edge, graph):
    a = ''
    if 'label' in graph.edges[edge]:
        a = graph.edges[edge]['label']
    return a

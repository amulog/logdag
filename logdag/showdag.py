import pickle
import networkx as nx
from collections import defaultdict

from amulog import common
from . import arguments
from . import log2event

KEY_WEIGHT = "weight"

# fmt_int = lambda x: "{:,d}".format(x)
# fmt_ratio = lambda x: "{:.1f}".format(x)
# fmt_int_ratio = lambda x, y: "{:,d}({:.1f}%)".format(x, y)


class LogDAG:

    def __init__(self, args, graph=None, evmap=None):
        self.args = args
        self.conf, self.dt_range, self.area = self.args
        self.name = arguments.args2name(self.args)
        self.graph = graph
        self._evmap_org = evmap
        self._use_mapping = self.conf.getboolean("database_amulog",
                                                 "use_anonymize_mapping")

        # cache
        self._evmap_valid = None
        self._cache_edges_no_duplication = None
        self._d_el = None
        self._ed = None

    @classmethod
    def dag_path(cls, args):
        conf = args[0]
        dag_format = conf["dag"]["output_dag_format"]
        fp = arguments.ArgumentManager.dag_path(conf, args,
                                                ext=dag_format)
        return fp

    @property
    def _edges_no_duplication(self):
        if self._cache_edges_no_duplication is None:
            self._cache_edges_no_duplication = []
            undirected = set()
            for edge in self.graph.edges:
                if not self.edge_isdirected(edge):
                    edge_key = frozenset(edge)
                    if edge_key in undirected:
                        continue
                    else:
                        undirected.add(edge_key)
                self._cache_edges_no_duplication.append(edge)
        return self._cache_edges_no_duplication

    def _evmap_input(self):
        """EventDefinitionMap used in logdag causal analysis"""
        if self._evmap_org is None:
            evmap = log2event.EventDefinitionMap()
            evmap.load(self.args)
        else:
            evmap = self._evmap_org
        return evmap

    def _evmap_original(self):
        """EventDefinitionMap where anonymization is restored"""
        if self._evmap_valid is None:
            evmap_org = self._evmap_input()
            if self._use_mapping:
                self._evmap_valid = self._remap_evmap(evmap_org)
            else:
                self._evmap_valid = evmap_org

        return self._evmap_valid

    def _evloader(self):
        if self._d_el is None:
            self._d_el = log2event.init_evloaders(self.conf)
        return self._d_el

    def _eventdetail(self, load_cache):
        if self._ed is None:
            use_cache = self.conf.getboolean("dag", "event_detail_cache")
            if not use_cache:
                load_cache = False
            self._ed = log2event.EventDetail(
                self.conf, self._evloader(),
                load_cache=load_cache, dump_cache=use_cache
            )
        return self._ed

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

    def nodes(self):
        return self.graph.nodes()

    def edges(self, allow_duplication=False, graph=None):
        if allow_duplication:
            if graph is None:
                graph = self.graph
            return graph.edges()
        elif graph is None:
            return self._edges_no_duplication
        else:
            return remove_edge_duplication(graph.edges(), self, graph=graph)

    def number_of_nodes(self, graph=None):
        if graph is None:
            graph = self.graph
        return graph.number_of_nodes()

    def number_of_edges(self, graph=None):
        if graph is None:
            return len(self._edges_no_duplication)
        else:
            return remove_edge_duplication(graph.edges(), self, graph=graph)

    def _remap_evmap(self, evmap):
        mapping = {eid: self._remap_evdef(evdef)
                   for eid, evdef in evmap.items()}
        return log2event.EventDefinitionMap.from_dict(mapping)

    def _remap_evdef(self, evdef):
        if isinstance(evdef, log2event.MultipleEventDefinition):
            new_members = [self._remap_evdef(tmp_evdef)
                           for tmp_evdef in evdef.members]
            evdef.update_members(new_members)
            return evdef
        else:
            new_host = self._evloader()[evdef.source].restore_host(evdef.host)
            new_evdef = evdef.replaced_copy(host=new_host)
            return new_evdef

    def node_evdef(self, node, original=True):
        if original:
            evmap = self._evmap_original()
        else:
            evmap = self._evmap_input()
        evdef = evmap.evdef(node)
        return evdef

    def evdef2node(self, evdef, original=True, graph=None):
        if graph is None:
            graph = self.graph
        if original:
            evmap = self._evmap_original()
        else:
            evmap = self._evmap_input()
        node = evmap.get_eid(evdef)
        return node, graph.get_node_data(node)

    def edge_evdef(self, edge, original=True):
        return [self.node_evdef(node, original=original)
                for node in edge[0:2]]

    def evdef2edge(self, evdef1, evdef2, original=True,
                   graph=None, allow_reverse=False):
        """
        Returns:
            u (int): source node of the edge
            v (int): destination node of the edge
            w (dict): edge attribute
        """
        if graph is None:
            graph = self.graph
        if original:
            evmap = self._evmap_original()
        else:
            evmap = self._evmap_input()
        node1 = evmap.get_eid(evdef1)
        node2 = evmap.get_eid(evdef2)

        if graph.has_edge(node1, node2):
            return node1, node2, graph.get_edge_data(node1, node2)
        if allow_reverse:
            if graph.has_edge(node2, node1):
                return node2, node1, graph.get_edge_data(node2, node1)
        return None

    def has_node(self, evdef, original=True):
        if original:
            evmap = self._evmap_original()
        else:
            evmap = self._evmap_input()
        return evmap.has_evdef(evdef)

    def has_edge(self, evdef1, evdef2, original=True,
                 graph=None, allow_reverse=False):
        if graph is None:
            graph = self.graph
        if original:
            evmap = self._evmap_original()
        else:
            evmap = self._evmap_input()
        if self.has_node(evdef1) and self.has_node(evdef2):
            node1 = evmap.get_eid(evdef1)
            node2 = evmap.get_eid(evdef2)
            if allow_reverse:
                return graph.has_edge(node1, node2) or graph.has_edge(node2, node1)
            else:
                return graph.has_edge(node1, node2)
        else:
            return False

    @staticmethod
    def get_coefficient(edge, graph):
        a = None
        if KEY_WEIGHT in graph.edges[edge]:
            a = graph.edges[edge][KEY_WEIGHT]
        return a

    def edges_by_direction(self, graph=None, data=False):
        """Returns subgraphs of input graph its edges by
        the availability of their directions.

        Args:
            graph (nx.DiGraph): A subgraph of self.graph.
                                If empty, self.graph is used.
            data (boolean): Include data in EdgeView.

        Returns:
            edges_di (EdgeView): Directed edges.
            edges_nodi (EdgeView): Undirected edges.
        """
        if graph is None:
            graph = self.graph

        g_di = nx.DiGraph()
        g_nodi = nx.Graph()
        undirected = set()
        for u, v, ddict in graph.edges(data=True):
            if self.edge_isdirected((u, v), graph):
                g_di.add_edge(u, v, **ddict)
            else:
                edge_key = frozenset([u, v])
                if edge_key in undirected:
                    continue
                else:
                    undirected.add(edge_key)
                    g_nodi.add_edge(u, v, **ddict)
        return g_di.edges(data=data), g_nodi.edges(data=data)

    def edge_isdirected(self, edge, graph=None):
        if graph is None:
            graph = self.graph
        if edge in graph.edges():
            if graph.has_edge(edge[1], edge[0]):
                return False
            else:
                return True
        else:
            if graph.has_edge(edge[1], edge[0]):
                raise ValueError("Edge not found, Reversed edge exists")
            else:
                raise ValueError("Edge not found")

    def edges_across_host(self, graph=None, data=False):
        """Returns subgraphs of input graph its edges by the consistency
        of the hosts of adjacent nodes.

        Args:
            graph (nx.DiGraph): A subgraph of self.graph.
                                If empty, self.graph is used.
            data (boolean): Include data in EdgeView.

        Returns:
            edges_same (EdgeView): Edges between same hosts.
            edges_diff (EdgeView): Edges across multiple hosts.
        """
        g_same = nx.DiGraph()
        g_diff = nx.DiGraph()
        if graph is None:
            graph = self.graph
        for u, v, ddict in graph.edges(data=True):
            src_evdef, dst_evdef = self.edge_evdef((u, v))
            src_hosts = set(src_evdef.all_attr("host"))
            dst_hosts = set(dst_evdef.all_attr("host"))
            if len(src_hosts & dst_hosts) > 0:
                g_same.add_edge(u, v, **ddict)
            else:
                g_diff.add_edge(u, v, **ddict)
        return g_same.edges(data=data), g_diff.edges(data=data)

    def connected_subgraphs(self, graph=None):
        if graph is None:
            graph = self.graph
        temp_graph = graph.to_undirected()
        return nx.connected_components(temp_graph)

    def edge_org(self, edge, graph=None):
        if graph is None:
            graph = self.graph
        src_node, dst_node = edge
        if self.edge_isdirected(edge, graph):
            a = self.get_coefficient(edge, graph)
            if a is None:
                return "{0} -> {1}".format(src_node, dst_node)
            else:
                return "{0} -[{2}]-> {1}".format(src_node, dst_node, a)
        else:
            return "{0} <-> {1}".format(src_node, dst_node)

    def edge_str(self, edge, graph=None):
        if graph is None:
            graph = self.graph
        src_node, dst_node = edge
        src_str = self.node_str(src_node)
        dst_str = self.node_str(dst_node)
        if self.edge_isdirected(edge, graph):
            a = self.get_coefficient(edge, graph)
            if a is None:
                return "{0} -> {1}".format(src_str, dst_str)
            else:
                return "{0} -[{2}]-> {1}".format(src_str, dst_str, a)
        else:
            return "{0} <-> {1}".format(src_str, dst_str)

    def node_str(self, node):
        return str(node) + "@" + str(self.node_evdef(node, original=True))

    def edge_identifier(self, edge):
        return "-".join([evdef.identifier
                         for evdef in self.edge_evdef(edge, original=True)])

    def edge_instruction(self, edge):
        return "\n".join(["< " + self.node_instruction(edge[0]),
                          "> " + self.node_instruction(edge[1])])

    def node_instruction(self, node):
        evdef = self.node_evdef(node, original=True)
        return log2event.evdef_instruction(self.conf, evdef, d_el=self._evloader())

    def edge_detail(self, edge, load_cache=True, graph=None):
        buf = ["# Edge {0}".format(self.edge_str(edge, graph)),
               self.node_detail(edge[0],
                                header="< ", load_cache=load_cache),
               self.node_detail(edge[1],
                                header="> ", load_cache=load_cache)]
        return "\n".join(buf)

    def node_detail(self, node, header="# ", load_cache=True):
        evdef = self.node_evdef(node, original=False)
        if self._use_mapping:
            evdef_org = self.node_evdef(node, original=True)
        else:
            evdef_org = None
        buf = [header + "Node {0}: {1}".format(self.node_str(node),
                                               self.node_instruction(node))]
        ed = self._eventdetail(load_cache)
        detail_output = ed.get_detail(self.args, evdef, evdef_org=evdef_org)
        buf.append(common.add_indent(detail_output, indent=2))
        # buf.append(common.show_repr(
        #     data, head, foot, indent=2,
        #     strfunc=lambda x: "{0} {1}: {2}".format(x[0], x[1], x[2])))
        return "\n".join(buf)

    def node_ts(self, nodes):
        if isinstance(nodes, int):
            nodes = [nodes]
        l_evdef = [self.node_evdef(node, original=False) for node in nodes]
        df = log2event.load_merged_events(self.conf, self.dt_range, self.area,
                                          l_evdef, self._evloader())
        df.columns = nodes
        return df

    def node_count(self, node):
        df = self.node_ts([node])
        return df[node].sum(axis=0)

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


def remove_edge_duplication(edges, ldag, graph=None):
    undirected = set()
    for edge in edges:
        if ldag.edge_isdirected(edge, graph=graph):
            yield edge
        else:
            edge_key = frozenset(edge)
            if edge_key not in undirected:
                undirected.add(edge_key)
                yield edge


def edge_view(edge, ldag, context="edge",
              load_cache=True, graph=None):
    if context == "detail":
        return "\n".join([ldag.edge_detail(edge,
                                           load_cache=load_cache, graph=graph)])
    elif context == "instruction":
        return "\n".join([ldag.edge_str(edge, graph),
                         ldag.edge_instruction(edge)])
    elif context == "edge":
        return ldag.edge_str(edge, graph)
    else:
        raise NotImplementedError


def apply_filter(ldag, l_filtername, th=None, graph=None):
    from . import showdag_filter
    if graph is None:
        g = ldag.graph
    else:
        g = graph

    if l_filtername is None or len(l_filtername) == 0:
        return g

    filters = []

    # make to_undirected the first filter
    if "to_undirected" in l_filtername:
        l_filtername.remove("to_undirected")
        filters.append("to_undirected")

    has_no_isolated = False
    if "no_isolated" in l_filtername:
        l_filtername.remove("no_isolated")
        has_no_isolated = True

    for filtername in l_filtername:
        if filtername[0] == "_":
            raise ValueError
        elif "=" in filtername:
            key, _, val = filtername.partition("=")
            filters.append(("_search_edges", {key: val}))
        elif filtername == "ate_prune":
            filters.append(("ate_prune", {"threshold": th}))
        else:
            filters.append((filtername, {}))

    # make no_isolated the last filter
    if has_no_isolated:
        filters.append(("no_isolated", {}))

    for funcname, kwargs in filters:
        assert funcname in showdag_filter.FUNCTIONS
        g = eval("showdag_filter." + funcname)(graph=g, ldag=ldag, **kwargs)
    return g


# functions for presentation


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


# def list_results(conf, src_dir=None):
#    table = [["datetime", "area", "nodes", "edges", "name"], ]
#    for r in iter_results(conf, src_dir):
#        c, dt_range, area = r.args
#        table.append([str(dt_range[0]), str(area),
#                      str(r.number_of_nodes()), str(r.number_of_edges()),
#                      r.name])
#    return common.cli_table(table)
#
#
# def list_results_byday(conf, src_dir=None):
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


# def show_stats(conf, src_dir=None):
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

def check_conditions(edge, ldag, conditions):
    if "node" in conditions:
        if conditions["node"] not in edge:
            return False
    if "gid" in conditions:
        src_evdef = ldag.node_evdef(edge[0])
        dst_evdef = ldag.node_evdef(edge[1])
        s_gid = src_evdef.all_attr("gid") | dst_evdef.all_attr("gid")
        if conditions["gid"] not in s_gid:
            return False
    if "host" in conditions:
        src_evdef = ldag.node_evdef(edge[0])
        dst_evdef = ldag.node_evdef(edge[1])
        src_hosts = set(src_evdef.all_attr("host"))
        dst_hosts = set(dst_evdef.all_attr("host"))
        cnt = sum(1 for host in src_hosts | dst_hosts
                  if conditions["host"] in host)
        if cnt == 0:
            return False
    return True


def show_edge(ldag: LogDAG, conditions, context="edge",
              load_cache=True, graph=None):
    if graph is None:
        graph = ldag.graph

    edges_to_show = []
    for edge in graph.edges():
        if check_conditions(edge, ldag, conditions):
            edges_to_show.append(edge)

    l_buf = []
    for edge in remove_edge_duplication(edges_to_show, ldag, graph=graph):
        msg = edge_view(edge, ldag, context=context,
                        load_cache=load_cache, graph=graph)
        l_buf.append(msg)
    return "\n".join(l_buf)


def show_edge_list(ldag, context="edge", load_cache=True, graph=None):
    if graph is None:
        graph = ldag.graph

    l_buf = []
    for edge in remove_edge_duplication(graph.edges(), ldag, graph=graph):
        msg = edge_view(edge, ldag, context=context,
                        load_cache=load_cache, graph=graph)
        l_buf.append(msg)
    return "\n".join(l_buf)


def show_subgraphs(ldag, context="edge", load_cache=True, graph=None):
    if graph is None:
        graph = ldag.graph

    l_buf = []
    separator = "\n\n"
    iterobj = sorted(nx.connected_components(graph.to_undirected()),
                     key=len, reverse=True)
    for sgid, nodes in enumerate(iterobj):
        if len(nodes) == 1:
            continue
        l_graph_buf = ["Subgraph {0} ({1} nodes)".format(sgid, len(nodes))]
        subg = nx.subgraph(graph, nodes)

        for edge in remove_edge_duplication(subg.edges(), ldag, graph=graph):
            msg = edge_view(edge, ldag, context=context,
                            load_cache=load_cache, graph=graph)
            l_graph_buf.append(msg)
        l_buf.append("\n".join(l_graph_buf))
    return separator.join(l_buf)


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


def plot_node_ts(args, l_nodeid, output):
    import matplotlib
    matplotlib.use("Agg")

    ldag = LogDAG(args)
    ldag.load()
    df = ldag.node_ts(l_nodeid)

    import matplotlib.pyplot as plt
    df.plot(subplots=True, layout=(len(l_nodeid), 1))
    plt.savefig(output)

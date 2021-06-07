
import logging
import json
from itertools import combinations, permutations
from abc import ABC, abstractmethod
import networkx as nx

_logger = logging.getLogger(__package__)


class PriorKnowledge:

    def __init__(self, node_ids):
        self._node_ids = node_ids
        # edges
        self._edges = set()
        # noedges
        self._noedges = set()
        # paths
        self._paths = set()
        # nopaths
        self._nopaths = set()
        # exogenous_variables
        self._exogenous_variables = set()
        # sink_variables
        self._sink_variables = set()

    @property
    def node_ids(self):
        return self._node_ids

    @staticmethod
    def _reorder_edge(edge):
        return tuple(sorted(list(edge)))

    @staticmethod
    def _reverse_edge(edge):
        return edge[1], edge[0]

    def add_noedge_rule(self, edge):
        self._noedges.add(self._reorder_edge(edge))
        self._nopaths.add(edge)
        self._nopaths.add(self._reverse_edge(edge))

    def add_edge_rule(self, edge):
        self._edges.add(self._reorder_edge(edge))
        self._paths.add(edge)
        self._paths.add(self._reverse_edge(edge))

    def add_nopath_rule(self, edge):
        self._noedges.add(self._reorder_edge(edge))
        self._nopaths.add(edge)

    def add_path_rule(self, edge):
        self._edges.add(self._reorder_edge(edge))
        self._paths.add(edge)

    def add_exogenous_variable(self, node):
        self._exogenous_variables.add(node)

    def add_sink_variable(self, node):
        self._sink_variables.add(node)

    def is_edge(self, edge):
        return self._reorder_edge(edge) in self._edges

    def is_noedge(self, edge):
        return self._reorder_edge(edge) in self._noedges

    def is_path(self, edge):
        return edge in self._paths

    def is_nopath(self, edge):
        return edge in self._nopaths

    def is_exogenous_variable(self, node):
        return node in self._exogenous_variables

    def is_sink_variable(self, node):
        return node in self._sink_variables

    def pruned_initial_skeleton(self):
        # make initial graph for skeleton estimation methods
        # this is pruning-based approach: only considering no-edge rules
        # currently designed for python pcalg library and cnsm2020
        g = nx.Graph()
        g.add_nodes_from(self._node_ids)
        for i, j in combinations(self._node_ids, 2):
            if (i, j) not in self._noedges:
                g.add_edge(i, j)
        return g

    def lingam_prior_knowledge(self, node_ids=None):
        from lingam.utils import make_prior_knowledge
        if node_ids is None:
            kwargs = {"n_variables": len(self._node_ids),
                      "exogenous_variables": self._exogenous_variables,
                      "sink_variables": self._sink_variables,
                      "paths": self._paths,
                      "nopaths": self._nopaths}
        else:
            possible_paths = set(permutations(node_ids, 2))
            exv = set(node_ids) & self._exogenous_variables
            siv = set(node_ids) & self._sink_variables
            paths = possible_paths & self._paths
            nopaths = possible_paths & self._nopaths
            kwargs = {"n_variables": len(node_ids),
                      "exogenous_variables": exv,
                      "sink_variables": siv,
                      "paths": paths,
                      "nopaths": nopaths}
        return make_prior_knowledge(**kwargs)


class KnowledgeGenerator(ABC):

    def __init__(self):
        pass


class ImportDAG(KnowledgeGenerator):
    """Import DAGs generated by other logdag results"""

    def __init__(self, args):
        super().__init__()
        raise NotImplementedError


class RuleBasedPruning(KnowledgeGenerator, ABC):

    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def _load_graph(fp):
        with open(fp, 'r', encoding='utf-8') as f:
            js = json.load(f)
        return nx.node_link_graph(js)

    @abstractmethod
    def _is_adjacent(self, evdef1, evdef2):
        raise NotImplementedError

    def update(self, pk, evmap):
        for node1, node2 in combinations(pk.node_ids, 2):
            evdef1, evdef2 = [evmap.evdef(n) for n in (node1, node2)]
            # prune edges that are not topologically adjacent
            if not self._is_adjacent(evdef1, evdef2):
                pk.add_noedge_rule((node1, node2))
        return pk


class Topology(RuleBasedPruning):

    def __init__(self, topology_fp):
        super().__init__()
        self._topology = self._load_graph(topology_fp)

    def _is_adjacent(self, evdef1, evdef2):
        if evdef1.host == evdef2.host:
            return True
        elif self._topology.has_edge(evdef1.host, evdef2.host):
            return True
        else:
            return False


class LayeredTopology(RuleBasedPruning):
    _default_layer = "other"

    def __init__(self, d_topology_fp, d_rule):
        super().__init__()
        self._topology = self._load_graphs(d_topology_fp)
        self._d_rule = d_rule

    @classmethod
    def _load_graphs(cls, d_fp):
        topo = {}
        for name, fp in d_fp.items():
            try:
                topo[name] = cls._load_graph(fp)
            except IOError:
                msg = "failed to load {0} for layer {1}".format(name, fp)
                _logger.warning(msg)
                topo[name] = nx.Graph()
        return topo

    def _get_layer(self, evdef):
        if evdef.group in self._d_rule:
            return self._d_rule[evdef.group]
        else:
            return self._default_layer

    def _is_adjacent(self, evdef1, evdef2):
        # same host
        if evdef1.host == evdef2.host:
            return True

        # allow one intermediate variable (node)
        # -> connection on a layer of at least one end node
        # see cnsm2020 paper
        layer1 = self._get_layer(evdef1)
        layer2 = self._get_layer(evdef2)
        for layer in (layer1, layer2):
            if layer in self._topology:
                net = self._topology[layer]
                if net.has_edge(evdef1.host, evdef2.host):
                    return True
        else:
            return False


class HostIndependent(RuleBasedPruning):
    # no edges between events on different devices

    def _is_adjacent(self, evdef1, evdef2):
        return evdef1.host == evdef2.host


class AdditionalSource(RuleBasedPruning):
    # no edges between nodes of additional sources

    @staticmethod
    def _is_additional(evdef):
        from . import log2event
        return evdef.source in (log2event.SRCCLS_SNMP, )

    def _is_adjacent(self, evdef1, evdef2):
        return not (self._is_additional(evdef1) and
                    self._is_additional(evdef2))


def init_prior_knowledge(conf, evmap):
    from amulog import config
    l_pruner = []
    methods = config.getlist(conf, "pc_prune", "methods")

    node_ids = evmap.eids()
    pk = PriorKnowledge(node_ids)
    for method in methods:
        if method == "topology":
            fp = conf.get("pc_prune", "single_network_file")
            pk = Topology(fp).update(pk, evmap)
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
            pk = LayeredTopology(d_fp, d_rule).update(pk, evmap)
        elif method == "independent":
            l_pruner.append(Independent())
        elif method == "ext-source":
            l_pruner.append(ExternalSource())
        else:
            raise NotImplementedError("invalid method name {0}".format(method))



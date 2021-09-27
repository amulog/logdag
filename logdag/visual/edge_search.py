import logging
from collections import defaultdict, Counter
from typing import Optional, Dict, List, FrozenSet
from abc import ABC
import math
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from logdag import arguments
from logdag import showdag

_logger = logging.getLogger("logdag")


class DAGSimilarity(arguments.WholeCacheBase, ABC):

    def __init__(self, conf, am=None, use_cache=True,
                 weight="none", smooth_idf=True):
        super().__init__(conf, am=am)
        self._conf = conf
        if am is None:
            self._am = arguments.ArgumentManager(conf)
            self._am.load()
        else:
            self._am = am
        assert weight in ("none", "idf")
        self._weight = weight

        self._ec = EdgeCount(conf, am=am, use_cache=use_cache,
                             smooth_idf=smooth_idf)
        self._counter = self._ec  # to be overwritten

        self._matrix = self._space()

    def load_cache(self):
        self._matrix = self.load()
        _logger.info("load cache {0}".format(self.cache_path))

    def dump_cache(self):
        obj = self._matrix
        self.dump(obj)
        _logger.info("dump cache {0}".format(self.cache_path))

    def _unit_vec(self, ldag):
        if self._weight == "none":
            return self._counter.get_dag_vector(ldag)
        elif self._weight == "idf":
            return self._counter.get_dag_idf_vector(ldag)
        else:
            raise NotImplementedError

    def _space(self):
        l_key = sorted(list(self._ec.get_all_edges()))
        l_jobname = []
        d_vec = {}
        for args in self._am:
            jobname = self._am.jobname(args)
            ldag = showdag.LogDAG(args)
            ldag.load()
            l_jobname.append(jobname)
            d_vec[jobname] = self._unit_vec(ldag)

        return pd.DataFrame(d_vec, columns=l_jobname, index=l_key)

    def similarity(self, jobname1, jobname2):
        data1 = self._matrix[jobname1]
        data2 = self._matrix[jobname2]
        return cosine_similarity(data1, data2)


class DAGSimilarityEventPairCount(DAGSimilarity):

    def __init__(self, conf, am=None, use_cache=True,
                 weight="none", smooth_idf=True):
        super().__init__(conf, am=am, use_cache=use_cache,
                         weight=weight, smooth_idf=smooth_idf)
        self._counter = EventPairCount(conf, am=am, use_cache=use_cache,
                                       smooth_idf=smooth_idf)

    @property
    def base_name(self):
        return "evpairsim_" + self._weight


class DAGSimilarityNodeCount(DAGSimilarity):

    def __init__(self, conf, am=None, use_cache=True,
                 weight="none", smooth_idf=True):
        super().__init__(conf, am=am, use_cache=use_cache,
                         weight=weight, smooth_idf=smooth_idf)

        self._counter = NodeCount(conf, am=am, use_cache=use_cache,
                                  smooth_idf=smooth_idf)

    @property
    def base_name(self):
        return "nodesim_" + self._weight


class DAGSimilarityEdgeCount(DAGSimilarity):

    def __init__(self, conf, am=None, use_cache=True,
                 weight="none", smooth_idf=True):
        super().__init__(conf, am=am, use_cache=use_cache,
                         weight=weight, smooth_idf=smooth_idf)

        self._counter = EdgeCount(conf, am=am, use_cache=use_cache,
                                  smooth_idf=smooth_idf)

    @property
    def base_name(self):
        return "edgesim_" + self._weight


class EventPairCount(arguments.WholeCacheBase):

    def __init__(self, conf, am=None, use_cache=True, smooth_idf=True):
        super().__init__(conf, am=am)
        self._smooth_idf = smooth_idf

        # key: evpair, val: list of jobname
        self._d_evpair_args: Optional[Dict[FrozenSet[str]: List[str]]] = None

        # key: jobname, val: counter of evpair
        self._d_evpair_count: Optional[Dict[Counter[FrozenSet[str]: int]]] = None

        if use_cache:
            if self.has_cache():
                self.load_cache()
            else:
                self._count_all()
                self.dump_cache()
        else:
            self._count_all()

        self._sorted_evpair_keys = sorted(self._d_evpair_args.keys())

    @property
    def base_name(self):
        return "evpaircount"

    def load_cache(self):
        self._d_evpair_args, self._d_evpair_count = self.load()
        _logger.info("load cache {0}".format(self.cache_path))

    def dump_cache(self):
        obj = (self._d_evpair_args, self._d_evpair_count)
        self.dump(obj)
        _logger.info("dump cache {0}".format(self.cache_path))

    @staticmethod
    def evpair_key(node1, node2, ldag):
        ev1 = ldag.node_evdef(node1).event()
        ev2 = ldag.node_evdef(node2).event()
        return frozenset([ev1, ev2])

    def _count_all(self):
        self._d_evpair_args = defaultdict(list)
        self._d_evpair_count = {}
        for args in self._am:
            ldag = showdag.LogDAG(args)
            ldag.load()
            udgraph = ldag.graph.to_undirected()
            jobname = self._am.jobname(args)
            self._d_evpair_count[jobname] = Counter()
            for edge in udgraph.edges():
                key = self.evpair_key(edge[0], edge[1], ldag)
                self._d_evpair_args[key].append(jobname)
                self._d_evpair_count[jobname][key] += 1

    def get_all_pairs(self):
        return self._sorted_evpair_keys

    def get_dag_vector(self, ldag):
        udgraph = ldag.graph.to_undirected()
        ldag_keys = {self.evpair_key(edge[0], edge[1], ldag)
                     for edge in udgraph.edges()}
        return np.array([int(key in ldag_keys)
                         for key in self._sorted_evpair_keys])

    def get_dag_idf_vector(self, ldag):
        udgraph = ldag.graph.to_undirected()
        ldag_keys = {self.evpair_key(edge[0], edge[1], ldag)
                     for edge in udgraph.edges()}
        return np.array([self._get_dag_idf(key, ldag)
                         for key in self._sorted_evpair_keys
                         if key in ldag_keys])

    def get_edge_counts(self, ldag):
        ret = []
        for edge in showdag.remove_edge_duplication(ldag.graph.edges(), ldag):
            key = self.evpair_key(edge[0], edge[1], ldag)
            count = len(self._d_evpair_args[key])
            ret.append((edge, count))
        return ret

    def _get_dag_idf(self, evpair_key, smooth_idf=None):
        count = len(self._d_evpair_args[evpair_key])
        if smooth_idf is None:
            smooth_idf = self._smooth_idf
        if smooth_idf:
            idf = math.log((len(self._am) + 1) / (count + 1)) + 1
        else:
            idf = math.log(len(self._am) / count) + 1
        return idf

    def get_dag_tfidf(self, node1, node2, ldag, smooth_idf=None):
        jobname = self._am.jobname(ldag.args)
        evpair_key = self.evpair_key(node1, node2, ldag)
        tf = self._d_evpair_count[jobname][evpair_key] / ldag.number_of_edges()
        idf = self._get_dag_idf(evpair_key, smooth_idf=smooth_idf)
        return tf * idf


class NodeCount(arguments.WholeCacheBase):

    def __init__(self, conf, am=None, use_cache=True, smooth_idf=True):
        super().__init__(conf, am=am)
        self._smooth_idf = smooth_idf

        # key: evdef, val: list of jobname
        self._d_evdef_args: Optional[Dict[List[str]]] = None

        # key: jobname, val: {evdef: count}
        self._d_evdef_count: Optional[Dict[Dict[int]]] = None

        # key: jobname, val: count
        self._d_total: Optional[Dict[int]] = None

        if use_cache:
            if self.has_cache():
                self.load_cache()
            else:
                self._count_all()
                self.dump_cache()
        else:
            self._count_all()

        self._sorted_node_keys = sorted(self._d_evdef_args.keys())

    @property
    def base_name(self):
        return "nodecount"

    def load_cache(self):
        self._d_evdef_args, self._d_evdef_count, self._d_total = self.load()
        _logger.info("load cache {0}".format(self.cache_path))

    def dump_cache(self):
        obj = (self._d_evdef_args, self._d_evdef_count, self._d_total)
        self.dump(obj)
        _logger.info("dump cache {0}".format(self.cache_path))

    @staticmethod
    def node_key(node, ldag):
        evdef = ldag.node_evdef(node)
        return evdef.identifier

    def _count_all(self):
        self._d_evdef_args = defaultdict(list)
        self._d_evdef_count = defaultdict(dict)
        self._d_total = defaultdict(int)
        for args in self._am:
            ldag = showdag.LogDAG(args)
            ldag.load()
            jobname = self._am.jobname(args)
            for node in ldag.graph.nodes():
                key = self.node_key(node, ldag)
                count = ldag.node_count(node)
                self._d_evdef_args[key].append(jobname)
                self._d_evdef_count[jobname][key] = count
                self._d_total[jobname] += count

    def get_all_events(self):
        return self._sorted_node_keys

    def get_dag_vector(self, ldag):
        ldag_keys = {self.node_key(node, ldag)
                     for node in ldag.graph.nodes()}
        return np.array([int(key in ldag_keys)
                         for key in self._sorted_node_keys])

    def get_dag_idf_vector(self, ldag):
        ldag_keys = {self.node_key(node, ldag)
                     for node in ldag.graph.nodes()}
        return np.array([self._get_dag_idf(key, ldag)
                         for key in self._sorted_node_keys
                         if key in ldag_keys])

    def get_edge_counts(self, ldag):
        ret = []
        jobname = self._am.jobname(ldag.args)
        for edge in showdag.remove_edge_duplication(ldag.graph.edges(), ldag):
            count = 0
            for node in edge:
                evdef = ldag.node_evdef(node)
                count += self._d_evdef_count[jobname][evdef.identifier]
            ret.append((edge, count / 2))
        return ret

    def get_node_counts(self, ldag):
        ret = []
        jobname = self._am.jobname(ldag.args)
        for node in ldag.graph.nodes():
            evdef = ldag.node_evdef(node)
            count = self._d_evdef_count[jobname][evdef.identifier]
            ret.append((node, count))
        return ret

    def _get_dag_idf(self, node, ldag, smooth_idf=None):
        if smooth_idf is None:
            smooth_idf = self._smooth_idf
        evdef = ldag.node_evdef(node)
        doc_count = len(self._d_evdef_args[evdef.identifier])
        if smooth_idf:
            idf = math.log((len(self._am) + 1) / (doc_count + 1)) + 1
        else:
            idf = math.log(len(self._am) / doc_count) + 1
        return idf

    def get_dag_tfidf(self, node, ldag, smooth_idf=None):
        evdef = ldag.node_evdef(node)
        jobname = self._am.jobname(ldag.args)
        node_count = self._d_evdef_count[jobname][evdef.identifier]
        all_count = sum(self._d_evdef_count[jobname].values())

        tf = node_count / all_count
        idf = self._get_dag_idf(node, ldag, smooth_idf=smooth_idf)
        return tf * idf


class EdgeCount(arguments.WholeCacheBase):

    def __init__(self, conf, am=None, use_cache=True, smooth_idf=True):
        super().__init__(conf, am=am)
        self._smooth_idf = smooth_idf

        self._d_edge_args: Optional[Dict[FrozenSet[str]: List[str]]] = None

        if use_cache:
            if self.has_cache():
                self.load_cache()
            else:
                self._count_all()
                self.dump_cache()
        else:
            self._count_all()

        self._sorted_edge_keys = sorted(self._d_edge_args.keys())

    @property
    def base_name(self):
        return "edgecount"

    def load_cache(self):
        self._d_edge_args = self.load()
        _logger.info("load cache {0}".format(self.cache_path))

    def dump_cache(self):
        obj = self._d_edge_args
        self.dump(obj)
        _logger.info("dump cache {0}".format(self.cache_path))

    @staticmethod
    def edge_key(edge, ldag):
        src_evdef, dst_evdef = ldag.edge_evdef(edge)
        return frozenset([src_evdef.identifier, dst_evdef.identifier])

    def _count_all(self):
        self._d_edge_args = defaultdict(list)
        for args in self._am:
            ldag = showdag.LogDAG(args)
            ldag.load()
            udgraph = ldag.graph.to_undirected()
            jobname = self._am.jobname(args)
            for edge in udgraph.edges():
                key = self.edge_key(edge, ldag)
                self._d_edge_args[key].append(jobname)

    def get_all_edges(self):
        return self._sorted_edge_keys

    def get_dag_vector(self, ldag):
        udgraph = ldag.graph.to_undirected()
        ldag_keys = {self.edge_key(edge, ldag)
                     for edge in udgraph.edges()}
        return np.array([int(key in ldag_keys)
                        for key in self._sorted_edge_keys])

    def get_dag_idf_vector(self, ldag):
        udgraph = ldag.graph.to_undirected()
        ldag_keys = {self.edge_key(edge, ldag)
                     for edge in udgraph.edges()}
        return np.array([self._get_dag_idf(key)
                         for key in self._sorted_edge_keys
                         if key in ldag_keys])

    def get_dag_counts(self, ldag):
        ret = []
        for edge in showdag.remove_edge_duplication(ldag.graph.edges(), ldag):
            key = self.edge_key(edge, ldag)
            count = len(self._d_edge_args[key])
            ret.append((edge, count))
        return ret

    def _get_dag_idf(self, edge_key, smooth_idf=None):
        count = len(self._d_edge_args[edge_key])
        if smooth_idf is None:
            smooth_idf = self._smooth_idf
        if smooth_idf:
            idf = math.log((len(self._am) + 1) / (count + 1)) + 1
        else:
            idf = math.log(len(self._am) / count) + 1
        return idf

    def get_dag_tfidf(self, edge, ldag, smooth_idf=None):
        edge_key = self.edge_key(edge, ldag)
        tf = 1 / ldag.number_of_edges()
        idf = self._get_dag_idf(edge_key, smooth_idf=smooth_idf)
        return tf * idf


def init_counter(conf, feature="edge", **kwargs):
    if feature == "node":
        return NodeCount(conf, **kwargs)
    elif feature == "edge":
        return EdgeCount(conf, **kwargs)
    elif feature == "evpair":
        return EventPairCount(conf, **kwargs)
    else:
        raise NotImplementedError


def edges_anomaly_score(edges, ldag, feature="edge", counter=None, am=None):
    if feature == "node":
        if counter is None:
            counter = NodeCount(ldag.conf, am=am)
        for edge in edges:
            score = max(counter.get_dag_tfidf(edge[0], ldag),
                        counter.get_dag_tfidf(edge[1], ldag))
            yield score
    elif feature == "edge":
        if counter is None:
            counter = EdgeCount(ldag.conf, am=am)
        for edge in edges:
            score = counter.get_dag_tfidf(edge, ldag)
            yield score
    elif feature == "evpair":
        if counter is None:
            counter = EventPairCount(ldag.conf, am=am)
        for edge in edges:
            score = counter.get_dag_tfidf(edge[0], edge[1], ldag)
            yield score
    else:
        raise NotImplementedError


def dag_anomaly_score(conf, feature="edge"):
    am = arguments.ArgumentManager(conf)
    am.load()
    counter = init_counter(conf, feature, am=am)

    d_score = {}
    for args in am:
        jobname = am.jobname(args)
        ldag = showdag.LogDAG(args)
        ldag.load()
        edges = showdag.remove_edge_duplication(ldag.graph.edges(), ldag)
        score = sum(edges_anomaly_score(edges, ldag, feature=feature,
                                        counter=counter, am=am))
        d_score[jobname] = score
    return d_score


def show_sorted_edges(ldag, feature="edge", use_score=True, reverse=False,
                      view_context="edge", load_cache=True, graph=None):
    am = arguments.ArgumentManager(ldag.conf)
    am.load()

    edges = showdag.remove_edge_duplication(ldag.graph.edges(), ldag)
    if use_score:
        scores = edges_anomaly_score(edges, ldag, feature=feature, am=am)
        items = zip(edges, scores)
        order_reverse = not reverse
    else:
        counter = init_counter(ldag.conf, feature, am=am)
        items = counter.get_edge_counts(ldag)
        order_reverse = reverse

    l_buf = []
    prev = None
    for edge, score in sorted(items, key=lambda x: x[1],
                              reverse=order_reverse):
        if score != prev:
            if prev is not None:
                l_buf.append("")
            if use_score:
                l_buf.append("[count={0}]".format(score))
            else:
                l_buf.append("[score={0}]".format(score))
            prev = score
        msg = showdag.edge_view(edge, ldag, context=view_context,
                                load_cache=load_cache, graph=graph)
        l_buf.append(msg)
    return "\n".join(l_buf)


def search_gid(conf, gid):
    l_result = []
    for r in showdag.iter_results(conf):
        for edge in showdag.remove_edge_duplication(r.graph.edges(), r):
            temp_gids = [evdef.gid for evdef in r.edge_evdef(edge)]
            if gid in temp_gids:
                l_result.append((r, edge))
    return l_result

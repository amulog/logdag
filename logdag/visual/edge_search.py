from collections import defaultdict
from typing import Optional, Dict, List, FrozenSet
from abc import ABC, abstractmethod
import math
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from logdag import arguments
from logdag import showdag


class DAGSimilarity(arguments.WholeCacheBase, ABC):

    def __init__(self, conf, am=None, use_cache=True, smooth_idf=True):
        super().__init__(conf, am=am)
        self._conf = conf
        if am is None:
            self._am = arguments.ArgumentManager(conf)
            self._am.load()
        else:
            self._am = am

        self._nc = NodeCount(conf, am=am, use_cache=use_cache,
                             smooth_idf=smooth_idf)
        self._ec = EdgeCount(conf, am=am, use_cache=use_cache,
                             smooth_idf=smooth_idf)

        self._matrix = self._space()

    def load_cache(self):
        self._matrix = self.load()

    def dump_cache(self):
        obj = self._matrix
        self.dump(obj)

    @abstractmethod
    def _unit_vec(self, ldag):
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


class DAGSimilarityEdgeCount(DAGSimilarity):

    def __init__(self, conf, am=None, use_cache=True,
                 weight="none"):
        assert weight in ("none", "idf")
        self._weight = weight
        super().__init__(conf, am=am, use_cache=use_cache)

    @property
    def base_name(self):
        return "edgesim_" + self._weight

    def _unit_vec(self, ldag):
        if self._weight == "none":
            return self._ec.get_unit_edge_distribution(ldag)
        elif self._weight == "idf":
            return self._ec.get_unit_edge_idf(ldag)
        else:
            raise NotImplementedError


class DAGSimilarityNodeCount(DAGSimilarity):

    def __init__(self, conf, am=None, use_cache=True,
                 weight="none"):
        assert weight in ("none", "idf")
        self._weight = weight
        super().__init__(conf, am=am, use_cache=use_cache)

    @property
    def base_name(self):
        return "nodesim_" + self._weight

    def _unit_vec(self, ldag):
        if self._weight == "none":
            return self._nc.get_unit_node_distribution(ldag)
        elif self._weight == "idf":
            return self._nc.get_unit_node_idf(ldag)
        else:
            raise NotImplementedError


class NodeCount(arguments.WholeCacheBase):

    def __init__(self, conf, am=None, use_cache=True, smooth_idf=True):
        super().__init__(conf, am=am)
        self._smooth_idf = smooth_idf

        # key: evdef, val: list of jobname
        self._d_evdef_args: Optional[Dict[List[str]]] = None

        # key: jobname, val: {evdef: count}
        self._d_evdef_count: Optional[Dict[Dict[int]]] = None

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
        self._d_evdef_args, self._d_evdef_count = self.load()

    def dump_cache(self):
        obj = (self._d_evdef_args, self._d_evdef_count)
        self.dump(obj)

    @staticmethod
    def node_key(node, ldag):
        evdef = ldag.node_evdef(node)
        return evdef.identifier

    def _count_all(self):
        self._d_evdef_args = defaultdict(list)
        self._d_evdef_count = defaultdict(dict)
        for args in self._am:
            ldag = showdag.LogDAG(args)
            ldag.load()
            jobname = self._am.jobname(args)
            for node in ldag.graph.nodes():
                key = self.node_key(node, ldag)
                count = ldag.node_count(node)
                self._d_evdef_args[key].append(jobname)
                self._d_evdef_count[jobname][key] = count

    def get_all_events(self):
        return self._sorted_node_keys

    def get_unit_node_distribution(self, ldag):
        ldag_keys = {self.node_key(node, ldag)
                     for node in ldag.graph.nodes()}
        return np.array([int(key in ldag_keys)
                         for key in self._sorted_node_keys])

    def get_unit_node_idf(self, ldag):
        ldag_keys = {self.node_key(node, ldag)
                     for node in ldag.graph.nodes()}
        return np.array([self._get_node_idf(key, ldag)
                         for key in self._sorted_node_keys
                         if key in ldag_keys])

    def get_node_counts(self, ldag):
        ret = []
        jobname = self._am.jobname(ldag.args)
        for node in ldag.graph.nodes():
            evdef = ldag.node_evdef(node)
            count = self._d_evdef_count[jobname][evdef.identifier]
            ret.append((node, count))
        return ret

    def _get_node_idf(self, node, ldag, smooth_idf=None):
        if smooth_idf is None:
            smooth_idf = self._smooth_idf
        evdef = ldag.node_evdef(node)
        doc_count = len(self._d_evdef_args[evdef.identifier])
        if smooth_idf:
            idf = math.log((len(self._am) + 1) / (doc_count + 1)) + 1
        else:
            idf = math.log(len(self._am) / doc_count) + 1
        return idf

    def get_node_tfidf(self, node, ldag, smooth_idf=None):
        evdef = ldag.node_evdef(node)
        jobname = self._am.jobname(ldag.args)
        node_count = self._d_evdef_count[jobname][evdef.identifier]
        all_count = sum(self._d_evdef_count[jobname].values())

        tf = node_count / all_count
        idf = self._get_node_idf(node, ldag, smooth_idf=smooth_idf)
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

    def dump_cache(self):
        obj = self._d_edge_args
        self.dump(obj)

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

    def get_unit_edge_distribution(self, ldag):
        udgraph = ldag.graph.to_undirected()
        ldag_keys = {self.edge_key(edge, ldag)
                     for edge in udgraph.edges()}
        return np.array([int(key in ldag_keys)
                        for key in self._sorted_edge_keys])

    def get_unit_edge_idf(self, ldag):
        udgraph = ldag.graph.to_undirected()
        ldag_keys = {self.edge_key(edge, ldag)
                     for edge in udgraph.edges()}
        return np.array([self._get_edge_idf(key)
                         for key in self._sorted_edge_keys
                         if key in ldag_keys])

    def get_edge_counts(self, ldag):
        ret = []
        for edge in showdag.remove_edge_duplication(ldag.graph.edges(), ldag):
            key = self.edge_key(edge, ldag)
            count = len(self._d_edge_args[key])
            ret.append((edge, count))
        return ret

    def _get_edge_idf(self, edge_key, smooth_idf=None):
        count = len(self._d_edge_args[edge_key])
        if smooth_idf is None:
            smooth_idf = self._smooth_idf
        if smooth_idf:
            idf = math.log((len(self._am) + 1) / (count + 1)) + 1
        else:
            idf = math.log(len(self._am) / count) + 1
        return idf

    def get_edge_tfidf(self, edge_key, smooth_idf=None):
        tf = 1
        idf = self._get_edge_idf(edge_key, smooth_idf=smooth_idf)
        return tf * idf


def dag_anomaly_score(conf, context="edge"):
    am = arguments.ArgumentManager(conf)
    am.load()
    d_score = {}
    for args in am:
        jobname = am.jobname(args)
        ldag = showdag.LogDAG(args)
        ldag.load()
        score = float(0)
        if context == "node":
            nd = NodeCount(conf, am=am)
            for node in ldag.graph.nodes():
                score += nd.get_node_tfidf(node, ldag)
        elif context == "edge":
            ed = EdgeCount(conf, am=am)
            for edge in showdag.remove_edge_duplication(
                    ldag.graph.edges(), ldag):
                edge_key = ed.edge_key(edge, ldag)
                score += ed.get_edge_tfidf(edge_key)
        d_score[jobname] = score
    return d_score


def show_edges_by_count(ldag, minimum=None, maximum=None, reverse=False,
                        context="edge", load_cache=True, graph=None):
    am = arguments.ArgumentManager(ldag.conf)
    am.load()
    ec = EdgeCount(ldag.conf, am=am)

    l_buf = []
    prev = None
    for edge, count in sorted(ec.get_edge_counts(ldag), key=lambda x: x[1],
                              reverse=reverse):
        if maximum is not None and count > maximum:
            continue
        if minimum is not None and count < minimum:
            continue
        if count != prev:
            if prev is not None:
                l_buf.append("")
            l_buf.append("[{0}/{1}]".format(count, len(am)))
            prev = count
        msg = showdag.edge_view(edge, ldag, context=context,
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

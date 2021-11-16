import logging
from collections import defaultdict, Counter
from typing import Optional, Dict, List, FrozenSet
from abc import ABC
import math
import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean
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
        self._all_jobnames = [self._am.jobname(args) for args in self._am]

        assert weight in ("none", "idf")
        self._weight = weight

        self._ec = EdgeCount(conf, am=am, use_cache=use_cache,
                             smooth_idf=smooth_idf)
        self._counter = self._ec  # to be overwritten

        if use_cache:
            if self.has_cache():
                self.load_cache()
            else:
                self._matrix = self._space()
                self.dump_cache()
        else:
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
        l_jobname = []
        d_vec = {}
        for args in self._am:
            jobname = self._am.jobname(args)
            ldag = showdag.LogDAG(args)
            ldag.load()
            l_jobname.append(jobname)
            d_vec[jobname] = self._unit_vec(ldag)

        return pd.DataFrame(d_vec, columns=l_jobname,
                            index=self._counter.vector_index)

    def similarity(self, jobname1, jobname2):
        data1 = self._matrix[jobname1]
        data2 = self._matrix[jobname2]
        return cosine_similarity(data1, data2)

    @staticmethod
    def _renumber_clustering(labels):
        clusters = defaultdict(list)
        for ind, label in enumerate(labels):
            clusters[label].append(ind)

        iterable = sorted(clusters.items(),
                          key=lambda x: x[1], reverse=False)
        for cid, (label, members) in enumerate(iterable):
            yield cid, members

    def clustering(self, method="kmeans", cls_kwargs=None,
                   jobnames=None):
        if cls_kwargs is None:
            cls_kwargs = {}
        if jobnames:
            jobname2index = {jobname: ind
                             for ind, jobname in enumerate(self._all_jobnames)}
            jobnames_index = np.array([jobname2index[jobname]
                                       for jobname in jobnames])
            x_input = self._matrix[jobnames_index].T
        else:
            jobnames_index = np.array(range(len(self._all_jobnames)))
            x_input = self._matrix.T

        if method == "kmeans":
            from sklearn.cluster import KMeans
            km = KMeans(**cls_kwargs)
            labels = km.fit_predict(x_input)
        else:
            raise NotImplementedError

        assert len(labels) == len(self._all_jobnames)

        d_cluster = {}
        for cid, members in self._renumber_clustering(labels):
            members_in_all = jobnames_index[members]
            member_jobnames = [self._all_jobnames[ind] for ind in members_in_all]
            d_cluster[cid] = member_jobnames

        return d_cluster

    def cluster_similar_components(self, jobnames):
        if len(jobnames) < 2:
            raise ValueError

        vectors = [self._matrix[jobname] for jobname in jobnames]
        avg_vector = np.mean(vectors, axis=0)
        l_diff = [np.abs(self._matrix[jobname] - avg_vector)
                  for jobname in jobnames]
        diff_vector = np.mean(l_diff, axis=0) / len(jobnames)
        return np.argsort(diff_vector)

    def cluster_common_components(self, jobnames):
        if len(jobnames) < 2:
            raise ValueError

        vectors = []
        for jobname in jobnames:
            vector = self._matrix[jobname]
            size = np.linalg.norm(vector)
            vectors.append(vector / size)

        avg_vector = gmean(vectors, axis=0)
        return np.argsort(avg_vector)

    def similarity_causes(self, jobnames, topn=None):
        if len(jobnames) < 2:
            raise ValueError

        ret = []
        cnt = 0
        for ind in self.cluster_common_components(jobnames):
            key = self._counter.vector_index[ind]
            ret.append(key)
            cnt += 1
            if topn is not None and cnt >= topn:
                break
        return ret


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

    @property
    def vector_index(self):
        return self._sorted_evpair_keys

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
        return np.array([self._get_idf(key, ldag)
                         if key in ldag_keys else float(0)
                         for key in self._sorted_evpair_keys])

    def get_evpair_count(self, node1, node2, ldag):
        """
        Returns:
            local_count: number of corresponding edges in the given DAG
            whole_count: number of corresponding edges in all data
        """
        jobname = self._am.jobname(ldag.args)
        key = self.evpair_key(node1, node2, ldag)
        local_count = len(self._d_evpair_count[jobname][key])
        whole_count = len(self._d_evpair_args[key])
        return local_count, whole_count

    def _get_idf(self, evpair_key, smooth_idf=None):
        count = len(self._d_evpair_args[evpair_key])
        if smooth_idf is None:
            smooth_idf = self._smooth_idf
        if smooth_idf:
            idf = math.log((len(self._am) + 1) / (count + 1)) + 1
        else:
            idf = math.log(len(self._am) / count) + 1
        return idf

    def get_idf(self, node1, node2, ldag, smooth_idf=None):
        evpair_key = self.evpair_key(node1, node2, ldag)
        idf = self._get_idf(evpair_key, smooth_idf=smooth_idf)
        return idf

    def get_tfidf(self, node1, node2, ldag, smooth_idf=None):
        jobname = self._am.jobname(ldag.args)
        evpair_key = self.evpair_key(node1, node2, ldag)
        tf = self._d_evpair_count[jobname][evpair_key] / ldag.number_of_edges()
        idf = self._get_idf(evpair_key, smooth_idf=smooth_idf)
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

    @property
    def vector_index(self):
        return self._sorted_node_keys

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
        return np.array([self._get_idf(key, ldag)
                         if key in ldag_keys else float(0)
                         for key in self._sorted_node_keys])

    def get_edge_count(self, edge, ldag):
        jobname = self._am.jobname(ldag.args)
        count = 0
        for node in edge:
            evdef = ldag.node_evdef(node)
            count += self._d_evdef_count[jobname][evdef.identifier]
        return count / len(edge)

    def get_node_counts(self, node, ldag):
        jobname = self._am.jobname(ldag.args)
        evdef = ldag.node_evdef(node)
        count = self._d_evdef_count[jobname][evdef.identifier]
        return count

    def _get_idf(self, node, ldag, smooth_idf=None):
        if smooth_idf is None:
            smooth_idf = self._smooth_idf
        evdef = ldag.node_evdef(node)
        doc_count = len(self._d_evdef_args[evdef.identifier])
        if smooth_idf:
            idf = math.log((len(self._am) + 1) / (doc_count + 1)) + 1
        else:
            idf = math.log(len(self._am) / doc_count) + 1
        return idf

    def get_idf(self, node, ldag, smooth_idf=None):
        return self._get_idf(node, ldag, smooth_idf=smooth_idf)

    def get_tfidf(self, node, ldag, smooth_idf=None):
        evdef = ldag.node_evdef(node)
        jobname = self._am.jobname(ldag.args)
        node_count = self._d_evdef_count[jobname][evdef.identifier]
        all_count = sum(self._d_evdef_count[jobname].values())

        tf = node_count / all_count
        idf = self._get_idf(node, ldag, smooth_idf=smooth_idf)
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

    @property
    def vector_index(self):
        return self._sorted_edge_keys

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
        return np.array([self._get_idf(key)
                         if key in ldag_keys else float(0)
                         for key in self._sorted_edge_keys])

    def get_edge_count(self, edge, ldag):
        key = self.edge_key(edge, ldag)
        count = len(self._d_edge_args[key])
        return count

    def _get_idf(self, edge_key, smooth_idf=None):
        count = len(self._d_edge_args[edge_key])
        if smooth_idf is None:
            smooth_idf = self._smooth_idf
        if smooth_idf:
            idf = math.log((len(self._am) + 1) / (count + 1)) + 1
        else:
            idf = math.log(len(self._am) / count) + 1
        return idf

    def get_idf(self, edge, ldag, smooth_idf=None):
        edge_key = self.edge_key(edge, ldag)
        idf = self._get_idf(edge_key, smooth_idf=smooth_idf)
        return idf

    def get_tfidf(self, edge, ldag, smooth_idf=None):
        edge_key = self.edge_key(edge, ldag)
        tf = 1 / ldag.number_of_edges()
        idf = self._get_idf(edge_key, smooth_idf=smooth_idf)
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


def init_similarity(conf, feature="edge", **kwargs):
    if feature == "node":
        return DAGSimilarityNodeCount(conf, **kwargs)
    elif feature == "edge":
        return DAGSimilarityEdgeCount(conf, **kwargs)
    elif feature == "evpair":
        return DAGSimilarityEventPairCount(conf, **kwargs)
    else:
        raise NotImplementedError


def edges_anomaly_score(edges, ldag, feature="edge", score="tfidf",
                        counter=None, am=None):
    if feature == "node":
        if counter is None:
            counter = NodeCount(ldag.conf, am=am)
        for edge in edges:
            if score == "tfidf":
                val = max(counter.get_tfidf(edge[0], ldag),
                          counter.get_tfidf(edge[1], ldag))
                yield edge, val
            elif score == "idf":
                val = max(counter.get_idf(edge[0], ldag),
                          counter.get_idf(edge[1], ldag))
                yield edge, val
            elif score == "count":
                val = counter.get_edge_count(edge, ldag)
                yield edge, val
            else:
                raise NotImplementedError
    elif feature == "edge":
        if counter is None:
            counter = EdgeCount(ldag.conf, am=am)
        for edge in edges:
            if score == "tfidf":
                yield edge, counter.get_tfidf(edge, ldag)
            elif score == "idf":
                yield edge, counter.get_tfidf(edge, ldag)
            elif score == "count":
                yield edge, counter.get_edge_count(edge, ldag)
            else:
                raise NotImplementedError
    elif feature == "evpair":
        if counter is None:
            counter = EventPairCount(ldag.conf, am=am)
        for edge in edges:
            if score == "tfidf":
                yield edge, counter.get_tfidf(edge[0], edge[1], ldag)
            elif score == "idf":
                yield edge, counter.get_idf(edge[0], edge[1], ldag)
            elif score == "count":
                lcount, wcount = counter.get_evpair_count(edge[0], edge[1], ldag)
                yield edge, wcount
            else:
                raise NotImplementedError
    else:
        raise NotImplementedError


def dag_anomaly_score(conf, feature="edge", score="tfidf"):
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
                                        score=score,
                                        counter=counter, am=am))
        d_score[jobname] = score
    return d_score


def show_sorted_edges(ldag, search_condition=None, feature="edge", score="tfidf", reverse=False,
                      view_context="edge", load_cache=True, graph=None):
    am = arguments.ArgumentManager(ldag.conf)
    am.load()

    edges = [edge for edge
             in showdag.remove_edge_duplication(ldag.graph.edges(), ldag)
             if showdag.check_conditions(edge, ldag, search_condition)]
    items = list(edges_anomaly_score(edges, ldag,
                                     feature=feature, score=score, am=am))
    order_reverse = reverse if score == "count" else not reverse

    l_buf = []
    prev = None
    for edge, score in sorted(items, key=lambda x: x[1],
                              reverse=order_reverse):
        if score != prev:
            if prev is not None:
                l_buf.append("")
            l_buf.append("[score={0}]".format(score))
            prev = score
        msg = showdag.edge_view(edge, ldag, context=view_context,
                                load_cache=load_cache, graph=graph)
        l_buf.append(msg)
    return "\n".join(l_buf)


def edge_temporal_sort(ldag, time_condition, search_condition=None, reverse=False,
                       view_context="edge", load_cache=True, graph=None):
    assert "time" in time_condition or "time_range" in time_condition
    if graph is None:
        graph = ldag.graph

    from amulog import config
    ci_bin_size = config.getdur(ldag.conf, "dag", "ci_bin_size")

    nodes = set()  # nodes with any adjacent edges
    for edge in graph.edges():
        nodes.add(edge[0])
        nodes.add(edge[1])
    df_ts = ldag.node_ts(list(nodes))

    if "time" in time_condition:
        dt = time_condition["time"]
        sr_diff_td = (df_ts.index.to_series() + (0.5 * ci_bin_size) - dt).abs()
        sr_diff = sr_diff_td.map(lambda x: x.total_seconds())
        df_score = df_ts.apply(lambda x: x * sr_diff / sum(x))
    else:
        dts, dte = time_condition["time_range"]
        diff = []
        for tmp_ts in df_ts.index:
            ts = tmp_ts + 0.5 * ci_bin_size
            if ts < dts:
                diff.append((dts - ts).total_seconds())
            elif ts > dte:
                diff.append((ts - dte).total_seconds())
            else:  # dts <= ts <= dte
                diff.append(float(0))
        sr_diff = pd.Series(diff, index=df_ts.index)
        df_score = df_ts.apply(lambda x: x * sr_diff / sum(x))

    items = []
    edges = [edge for edge
             in showdag.remove_edge_duplication(ldag.graph.edges(), ldag)
             if showdag.check_conditions(edge, ldag, search_condition)]
    for edge in edges:
        score = (sum(df_score[edge[0]]) + sum(df_score[edge[1]])) / 2
        items.append((edge, score))

    l_buf = []
    prev = None
    for edge, score in sorted(items, key=lambda x: x[1],
                              reverse=reverse):
        if showdag.check_conditions(edge, ldag, search_condition):
            if score != prev:
                if prev is not None:
                    l_buf.append("")
                l_buf.append("[average_diff_sec={0}]".format(score))
                prev = score
            msg = showdag.edge_view(edge, ldag, context=view_context,
                                    load_cache=load_cache, graph=graph)
            l_buf.append(msg)
    return "\n".join(l_buf)


def search_similar_dag(ldag, feature="edge", weight="idf",
                       dag_topn=10, cause_topn=10):
    am = arguments.ArgumentManager(ldag.conf)
    am.load()
    ldag_jobname = am.jobname(ldag.args)

    sim = init_similarity(ldag.conf, feature, am=am, weight=weight)
    d_val = {}
    for args in am:
        jobname = am.jobname(args)
        if jobname != ldag_jobname:
            d_val[jobname] = sim.similarity(ldag_jobname, jobname)

    l_buf = []
    cnt = 0
    for jobname, val in sorted(d_val.items(), key=lambda x: x[1], reverse=True):
        jobnames = [ldag_jobname, jobname]
        causes = list(sim.similarity_causes(jobnames, topn=cause_topn))
        l_buf.append("{0} {1}: {2}".format(val, jobname, causes))
        cnt += 1
        if cnt >= dag_topn:
            break

    return "\n".join(l_buf)


def show_clusters(conf, feature="edge", weight="idf",
                  clustering_method="kmeans", n_clusters=None, cause_topn=10):
    am = arguments.ArgumentManager(conf)
    am.load()

    if n_clusters is None:
        n_clusters = int(math.sqrt(len(am)))

    sim = init_similarity(conf, feature, am=am, weight=weight)
    cls_kwargs = {"n_clusters": n_clusters}
    d_cluster = sim.clustering(clustering_method, cls_kwargs=cls_kwargs)

    l_buf = []
    for cid, jobnames in d_cluster.items():
        l_buf.append("[cluster {0}]: {1} ({2})".format(cid, jobnames, len(jobnames)))
        if len(jobnames) > 2:
            causes = list(sim.similarity_causes(jobnames, topn=cause_topn))
            l_buf.append("main components: {0}".format(causes))
        l_buf.append("")
    return "\n".join(l_buf)


def search_gid(conf, gid):
    l_result = []
    for r in showdag.iter_results(conf):
        for edge in showdag.remove_edge_duplication(r.graph.edges(), r):
            temp_gids = [evdef.gid for evdef in r.edge_evdef(edge)]
            if gid in temp_gids:
                l_result.append((r, edge))
    return l_result

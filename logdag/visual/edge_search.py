import os
import math
import pickle
from collections import defaultdict

from logdag import arguments
from logdag import showdag


class EdgeCount:
    cache_name = "edgecount.pickle"

    def __init__(self, conf, am=None, nocache=False):
        self._conf = conf
        if am is None:
            self._am = arguments.ArgumentManager(conf)
            self._am.load()
        else:
            self._am = am
        self._cache_path = self._am.whole_cache_path(conf, self.cache_name)

        if nocache:
            self.remove_cache()

        if self.has_cache():
            self.load()
        else:
            self._d_edge_args = self._count_all()
            self.dump()

    def has_cache(self):
        return os.path.exists(self._cache_path)

    def remove_cache(self):
        if os.path.exists(self._cache_path):
            os.remove(self._cache_path)

    def load(self):
        with open(self._cache_path, 'rb') as f:
            self._d_edge_args = pickle.load(f)

    def dump(self):
        obj = self._d_edge_args
        with open(self._cache_path, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def key(edge, ldag):
        src_evdef, dst_evdef = ldag.edge_evdef(edge)
        return frozenset([src_evdef.identifier, dst_evdef.identifier])

    def _count_all(self):
        d_edge_args = defaultdict(list)
        for args in self._am:
            ldag = showdag.LogDAG(args)
            ldag.load()
            udgraph = ldag.graph.to_undirected()
            for edge in udgraph.edges():
                d_edge_args[self.key(edge, ldag)].append(args)
        return d_edge_args

    def edge_counts(self, ldag):
        ret = []
        for edge in showdag.remove_edge_duplication(ldag.graph.edges(), ldag):
            count = len(self._d_edge_args[self.key(edge, ldag)])
            ret.append((count, edge))
        return ret

    def edge_tfidf(self, ldag, edge, smooth_idf=False):
        count = len(self._d_edge_args[self.key(edge, ldag)])
        tf = 1
        if smooth_idf:
            idf = math.log((len(self._am) + 1) / (count + 1)) + 1
        else:
            idf = math.log(len(self._am) / count) + 1
        return tf * idf


def show_edges_by_count(ldag, minimum=None, maximum=None, reverse=False,
                        context="edge", head=5, foot=5,
                        log_org=False, graph=None):
    am = arguments.ArgumentManager(ldag.conf)
    am.load()
    ec = EdgeCount(ldag.conf, am=am)

    l_buf = []
    prev = None
    for count, edge in sorted(ec.edge_counts(ldag), reverse=reverse):
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
                                head=head, foot=foot,
                                log_org=log_org, graph=graph)
        l_buf.append(msg)
    return "\n".join(l_buf)


def search_gid(conf, gid):
    l_result = []
    for r in showdag.iter_results(conf):
        for edge in r.graph.edges():
            temp_gids = [evdef.gid for evdef in r.edge_evdef(edge)]
            if gid in temp_gids:
                l_result.append((r, edge))
    return l_result


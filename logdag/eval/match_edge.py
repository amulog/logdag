#!/usr/bin/env python
# coding: utf-8


from collections import defaultdict

from logdag import showdag
from logdag.eval import trouble


def separate_args(conf, tr):
    from logdag import arguments
    am = arguments.ArgumentManager(conf)
    am.load()
    from amulog import log_db
    ld = log_db.LogData(conf)

    d_args = defaultdict(list)
    for lid in tr.data["message"]:
        lm = ld.get_line(lid)
        for args in am.args_from_time(lm.dt):
            name = arguments.args2name(args)
            d_args[name].append(lm)
    return [(arguments.name2args(name, conf), l_lm)
            for name, l_lm in d_args.items()]


def match_edges(conf, tr, rule = "all", cond = None):

    def _match_edge(input_evdef, edgeinfo, rule):
        src_evdef, dst_evdef = edgeinfo
        src_bool = src_evdef in input_evdef
        dst_bool = dst_evdef in input_evdef

        if rule == "all":
            return src_bool or dst_bool
        elif rule == "both":
            return src_bool and dst_bool
        elif rule == "either":
            return (src_bool or dst_bool) and not (src_bool and dst_bool)
        else:
            raise ValueError

    def _pass_condition(edgeinfo, cond):
        if cond is None:
            return True
        elif cond == "xhost":
            src_evdef, dst_evdef = edgeinfo
            return not src_evdef.host == dst_evdef.host
        else:
            raise NotImplementedError

    def _lm2ev(l_lm, gid_name):
        d = defaultdict(list)
        for lm in l_lm:
            gid = lm.lt.get(gid_name)
            host = lm.host
            d[(gid, host)].append(lm)
        return d

    from amulog import log_db
    ld = log_db.LogData(conf)
    gid_name = conf.get("dag", "event_gid")

    d = defaultdict(list)
    for args, l_lm in separate_args(conf, tr):
        r = showdag.LogDAG(args)
        r.load()
        g = r.graph.to_undirected()
        for edge in g.edges():
            edgeinfo = r.edge_info(edge)
            if not _pass_condition(edgeinfo, cond):
                continue
            l_evdef = [evdef for evdef in _lm2ev(l_lm, gid_name).keys()]
            if _match_edge(l_evdef, edgeinfo, rule):
                d[r.name].append(edge)

    return d




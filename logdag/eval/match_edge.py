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
    ld = log_db.LogData(arguments.open_amulog_config(conf))

    d_args = defaultdict(list)
    for lid in tr.data["message"]:
        lm = ld.get_line(lid)
        for args in am.args_from_time(lm.dt):
            name = arguments.args2name(args)
            d_args[name].append(lm)
    return [(arguments.name2args(name, conf), l_lm)
            for name, l_lm in d_args.items()]


def match_edges(conf, tr, rule="all", cond=None):

    def _match_edge(s_evdef, edge_evdef, rule):
        src_evdef, dst_evdef = edge_evdef
        src_bool = str(src_evdef) in s_evdef
        dst_bool = str(dst_evdef) in s_evdef

        if rule == "all":
            return src_bool or dst_bool
        elif rule == "both":
            return src_bool and dst_bool
        elif rule == "either":
            return (src_bool or dst_bool) and not (src_bool and dst_bool)
        else:
            raise ValueError

    def _pass_condition(edge_evdef, cond):
        if cond is None:
            return True
        elif cond == "xhost":
            src_evdef, dst_evdef = edge_evdef
            return not src_evdef.host == dst_evdef.host
        else:
            raise NotImplementedError

    def _lm2ev(lm, gid_name):
        gid = lm.lt.get(gid_name)
        d = {"source": "log",
             "gid": gid,
             "host": lm.host,
             "group": al.label(gid)}
        return evgen_log.LogEventDefinition(**d)

    from amulog import config
    from logdag.source import source_amulog
    from logdag.source import evgen_log
    dt_range = config.getterm(conf, "dag", "whole_term")
    al = source_amulog.init_amulogloader(conf, dt_range)
    gid_name = conf.get("database_amulog", "event_gid")

    d = defaultdict(list)
    for args, l_lm in separate_args(conf, tr):
        r = showdag.LogDAG(args)
        r.load()
        g = r.graph.to_undirected()
        for edge in g.edges():
            edevdef = r.edge_evdef(edge)
            if not _pass_condition(edevdef, cond):
                continue

            s_evdef = {str(_lm2ev(lm, gid_name)) for lm in l_lm}
            if _match_edge(s_evdef, edevdef, rule):
                d[r.name].append(edge)

    return d


# TODO log and snmp match

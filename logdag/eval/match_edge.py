#!/usr/bin/env python
# coding: utf-8


from collections import defaultdict

from logdag import showdag


def separate_args(conf, tr):
    """Some troubles can appear among multiple days.
    This function separates DAG arguments and corresponding logs.
    """
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


def _match_edge(s_evdef, edge_evdef, rule):
    src_evdef, dst_evdef = edge_evdef
    src_bool = len(set(src_evdef.member_identifiers()) & s_evdef) > 0
    dst_bool = len(set(dst_evdef.member_identifiers()) & s_evdef) > 0

    if rule == "all":
        return src_bool or dst_bool
    elif rule == "both":
        return src_bool and dst_bool
    elif rule == "either":
        return (src_bool or dst_bool) and not (src_bool and dst_bool)
    elif rule == "log-snmp":
        from logdag import log2event
        src_bool_snmp = (src_evdef.source == log2event.SRCCLS_SNMP)
        dst_bool_snmp = (dst_evdef.source == log2event.SRCCLS_SNMP)
        return (src_bool and dst_bool) or (src_bool and dst_bool_snmp) or\
               (src_bool_snmp and dst_bool)
    else:
        raise ValueError


def match_edges(conf, tr, rule="all", cond=None):

    def _pass_condition(edge_evdef, condition):
        if condition is None:
            return True
        elif condition == "xhost":
            src_evdef, dst_evdef = edge_evdef
            return not src_evdef.host == dst_evdef.host
        else:
            raise NotImplementedError

    from amulog import config
    from logdag.source import src_amulog
    from logdag.source import evgen_log
    dt_range = config.getterm(conf, "dag", "whole_term")
    al = src_amulog.init_amulogloader(conf, dt_range)
    gid_name = conf.get("database_amulog", "event_gid")

    d_results = defaultdict(list)
    for args, l_lm in separate_args(conf, tr):
        s_evdef = set()
        for lm in l_lm:
            gid = lm.lt.get(gid_name)
            evdef = evgen_log.LogEventDefinition(
                source="log", gid=gid, host=lm.host, group=al.group(gid)
            )
            s_evdef = s_evdef | set(evdef.member_identifiers())

        r = showdag.LogDAG(args)
        r.load()
        g = r.graph.to_undirected()
        for edge in g.edges():
            edevdef = r.edge_evdef(edge)
            if _pass_condition(edevdef, cond) and \
                    _match_edge(s_evdef, edevdef, rule):
                d_results[r.name].append(edge)

    return d_results

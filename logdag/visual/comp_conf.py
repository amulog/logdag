#!/usr/bin/env python
# coding: utf-8


import networkx as nx
from collections import defaultdict

from amulog import config
from logdag import arguments
from logdag import showdag
from logdag import log2event
from logdag import dtutil


def _set_of_term(am):
    s_term = set()
    for args in am:
        dt_range = args[1]
        s_term.add(dt_ragne)
    return s_term


def _add_nodes(evmap, r):
    for node in r.graph.nodes():
        evdef = r.node_info(node)
        if evmap.has_evdef(evdef):
            eid = evmap.get_eid(evdef)
        else:
            eid = evmap.add_evdef(evdef)
    return evmap


def _add_edges(evmap, cgraph, r):
    g = r.graph.to_undirected()
    for edge in g.edges():
        src_evdef, dst_evdef = r.edge_info(edge)
        try:
            src_eid = evmap.get_eid(src_evdef)
            dst_eid = evmap.get_eid(dst_evdef)
        except KeyError:
            raise
        cgraph.add_edge(src_eid, dst_eid)
    return cgraph


def edge_set_common(conf1, conf2, dt_range):
    gid_name = conf1.get("dag", "event_gid")
    am1 = arguments.ArgumentManager(conf1)
    am1.load()
    am2 = arguments.ArgumentManager(conf2)
    am2.load()

    temp_cevmap = log2event.EventDefinitionMap(gid_name)
    temp_cgraph = nx.Graph()
    for args in am1.args_in_time(dt_range):
        r1 = showdag.LogDAG(args)
        r1.load()
        temp_cevmap = _add_nodes(temp_cevmap, r1)
        temp_cgraph = _add_edges(temp_cevmap, temp_cgraph, r1)

    cevmap = log2event.EventDefinitionMap(gid_name)
    cgraph = nx.Graph()
    for args in am2.args_in_time(dt_range):
        r2 = showdag.LogDAG(args)
        r2.load()
        g = r2.graph.to_undirected()
        for edge in g.edges():
            src_evdef, dst_evdef = r2.edge_info(edge)
            if temp_cevmap.has_evdef(src_evdef) \
                    and temp_cevmap.has_evdef(dst_evdef):
                temp_src_eid = temp_cevmap.get_eid(src_evdef)
                temp_dst_eid = temp_cevmap.get_eid(dst_evdef)
                if temp_cgraph.has_edge(temp_src_eid, temp_dst_eid):
                    if cevmap.has_evdef(src_evdef):
                        new_src_eid = cevmap.get_eid(src_evdef)
                    else:
                        new_src_eid = cevmap.add_evdef(src_evdef)
                    if cevmap.has_evdef(dst_evdef):
                        new_dst_eid = cevmap.get_eid(dst_evdef)
                    else:
                        new_dst_eid = cevmap.add_evdef(dst_evdef)
                    cgraph.add_edge(new_src_eid, new_dst_eid)
    return cevmap, cgraph


def edge_set_lor(conf1, conf2, dt_range):
    gid_name = conf1.get("dag", "event_gid")
    am1 = arguments.ArgumentManager(conf1)
    am1.load()
    am2 = arguments.ArgumentManager(conf2)
    am2.load()

    cevmap = log2event.EventDefinitionMap(gid_name)
    cgraph = nx.Graph()
    for args in am1.args_in_time(dt_range):
        r1 = showdag.LogDAG(args)
        r1.load()
        cevmap = _add_nodes(cevmap, r1)
        cgraph = _add_edges(cevmap, cgraph, r1)

    for args in am2.args_in_time(dt_range):
        r2 = showdag.LogDAG(args)
        r2.load()
        cevmap = _add_nodes(cevmap, r2)
        cgraph = _add_edges(cevmap, cgraph, r2)
    return cevmap, cgraph


def edge_set_diff(conf1, conf2, dt_range, lor = None):
    """Edges exist in conf1, but not in conf2"""
    if lor is None:
        cevmap, cgraph_lor = edge_set_lor(conf1, conf2, dt_range)
    else:
        cevmap, cgraph_lor = lor

    gid_name = conf1.get("dag", "event_gid")
    am2 = arguments.ArgumentManager(conf2)
    am2.load()

    cgraph2 = nx.Graph()
    for args in am2.args_in_time(dt_range):
        r2 = showdag.LogDAG(args)
        r2.load()
        cgraph2 = _add_edges(cevmap, cgraph2, r2)

    cgraph_diff = nx.Graph()
    for edge in cgraph_lor.edges():
        if cgraph2.has_edge(*edge):
            pass
        else:
            cgraph_diff.add_edge(*edge)

    return cevmap, cgraph_diff


def edge_diff_gid(conf1, conf2):
    d_ltid = defaultdict(list)
    am = arguments.ArgumentManager(conf1)
    am.load()
    for dt_range in am.iter_dt_range():
        cevmap, cgraph = edge_set_diff(conf1, conf2, dt_range)
        for edge in cgraph.edges():
            timestr = dtutil.shortstr(dt_range[0])
            src_evdef = cevmap.evdef(edge[0])
            d_ltid[src_evdef.gid].append(timestr)
            dst_evdef = cevmap.evdef(edge[1])
            d_ltid[dst_evdef.gid].append(timestr)
    return d_ltid


def edge_diff_gid_search(conf1, conf2, gid):
    # processing time too long!!!
    d_ltid = defaultdict(int)
    am = arguments.ArgumentManager(conf1)
    am.load()
    for dt_range in am.iter_dt_range():
        cevmap, cgraph = edge_set_diff(conf1, conf2, dt_range)
        for edge in cgraph.edges():
            src_evdef = cevmap.evdef(edge[0])
            dst_evdef = cevmap.evdef(edge[1])
            if gid in (src_evdef.gid, dst_evdef.gid):
                timestr = dtutil.shortstr(dt_range[0])
                print("{0}: {1} - {2}".format(timestr, src_evdef, dst_evdef))


def edge_direction_diff(conf1, conf2, dt_range):

    def _get_direction(am, ev1, ev2, dt_range):
        for args in am.args_in_time(dt_range):
            r = showdag.LogDAG(args)
            r.load()
            if not r._evmap().has_evdef(ev1):
                continue
            if not r._evmap().has_evdef(ev2):
                continue
            n1, n2 = [r.info2node(ev) for ev in (ev1, ev2)]
            if (n1, n2) in r.graph.edges():
                if (n2, n1) in r.graph.edges():
                    di = "-"
                else:
                    di = "->"
            elif (n2, n1) in r.graph.edges():
                di = "<-"
            return di
        else:
            raise ValueError("Edge {0} - {1} not found in {2}".format(
                ev1, ev2, r.name))

    ret = []
    gid_name = conf1.get("dag", "event_gid")
    am1 = arguments.ArgumentManager(conf1)
    am1.load()
    am2 = arguments.ArgumentManager(conf2)
    am2.load()
    for args in am1.args_in_time(dt_range):
        r2 = showdag.LogDAG(args)

    cevmap, cgraph = edge_set_common(conf1, conf2, dt_range)
    for edge in cgraph.edges():
        ev1, ev2 = [cevmap.evdef(node) for node in edge]
        di1 = _get_direction(am1, ev1, ev2, dt_range)
        di2 = _get_direction(am2, ev1, ev2, dt_range)
        if di1 == di2:
            pass
        else:
            ret.append([ev1, ev2, di1, di2])
    return ret



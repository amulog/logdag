#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict


#!!! something bad?
def count_event_label(conf, agg_group = True):
    from amulog import log_db
    ld = log_db.LogData(conf)
    from amulog import lt_label
    ll = lt_label.init_ltlabel(conf)
    from logdag import tsdb
    td = tsdb.TimeSeriesDB(conf)

    from amulog import config
    term = config.getdur(conf, "dag", "unit_term")
    diff = config.getdur(conf, "dag", "unit_diff")
    from logdag import arguments
    l_dt_range = list(zip(*arguments.all_terms(conf, term, diff)))[1]

    d_cnt = defaultdict(int)
    for dt_range in l_dt_range:
        dts, dte = dt_range
        kwargs = {"dts": dts, "dte": dte}
        for gid, host in td.whole_gid_host(**kwargs):
            label = ll.get_ltg_label(gid, ld.ltg_members(gid))
            group = ll.get_group(label)
            if agg_group:
                d_cnt[group] += 1
            else:
                d_cnt[label] += 1
    return d_cnt


def count_ts_label(conf, agg_group = True):
    from amulog import log_db
    ld = log_db.LogData(conf)
    from amulog import lt_label
    ll = lt_label.init_ltlabel(conf)
    from logdag import tsdb
    td = tsdb.TimeSeriesDB(conf)
    from amulog import config
    dts, dte = config.getterm(conf, "dag", "whole_term")

    d_cnt = defaultdict(int)
    for gid in ld.iter_ltgid():
        cnt = sum(1 for dt in td.iter_ts(gid = gid, dts = dts, dte = dte))
        label = ll.get_ltg_label(gid, ld.ltg_members(gid))
        group = ll.get_group(label)
        if agg_group:
            d_cnt[group] += cnt
        else:
            d_cnt[label] += cnt
    return d_cnt


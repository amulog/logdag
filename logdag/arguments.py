#!/usr/bin/env python
# coding: utf-8

import os
import logging

from . import dtutil
from amulog import config
from amulog import common

DEFAULT_CONFIG = "/".join((os.path.dirname(__file__),
                           "data/config.conf.default"))
_logger = logging.getLogger(__package__)
_amulog_logger = logging.getLogger("amulog")


class ArgumentManager(object):

    def __init__(self, conf):
        self.conf = conf
        self.args_filename = conf.get("dag", "args_fn")
        self.l_args = []
        if self.args_filename.strip() == "":
            confname = conf.get("general", "base_filename").split("/")[-1]
            self.args_filename = "args_{0}".format(confname)

    def __iter__(self):
        return self.l_args.__iter__()

    def __getitem__(self, i):
        return self.l_args[i]

    def __len__(self):
        return len(self.l_args)

    def generate(self, func):
        self.l_args = func(self.conf)

    def add(self, args):
        self.l_args.append(args)

    def areas(self):
        return set([args[2] for args in self.l_args])

    def args_in_area(self, area):
        return [args for args in self.l_args if args[2] == area]

    def show(self):
        table = []
        table.append(["name", "datetime", "area"])
        for args in self.l_args:
            conf, dt_range, area = args
            temp = []
            temp.append(self.jobname(args))
            temp.append("{0} - {1}".format(dt_range[0], dt_range[1]))
            temp.append(area)
            table.append(temp)
        return common.cli_table(table, spl = " | ")

    def dump(self):
        with open(self.args_filename, 'w') as f:
            f.write("\n".join([self.jobname(args) for args in self.l_args]))

    def load(self):
        self.l_args = []
        with open(self.args_filename, 'r') as f:
            for line in f:
                args = self.jobname2args(line.rstrip(), self.conf)
                self.l_args.append(args)

    @staticmethod
    def jobname(args):
        def dt_filename(dt_range):
            top_dt, end_dt = dt_range
            if dtutil.is_intdate(top_dt) and dtutil.is_intdate(end_dt):
                return top_dt.strftime("%Y%m%d")
            else:
                return top_dt.strftime("%Y%m%d_%H%M%S")

        print(args)
        conf, dt_range, area = args
        return "_".join([area, dt_filename(dt_range)])

    @staticmethod
    def jobname2args(name, conf):
        area, dtstr = name.split("_", 1)
        top_dt = dtutil.shortstr2dt(dtstr)
        term = config.getdur(conf, "dag", "unit_term")
        end_dt = top_dt + term
        return conf, (top_dt, end_dt), area

    @classmethod
    def output_filename(cls, dirname, args):
        return "{0}/{1}".format(dirname, cls.jobname(args))

    @staticmethod
    def evdef_dir(conf):
        dirname = conf.get("dag", "evmap_dir")
        if dirname == "":
            dirname = conf.get("dag", "output_dir")
        else:
            common.mkdir(dirname)

    @classmethod
    def evdef_filepath(cls, args):
        conf, dt_range, area = args
        dirname = conf.get("dag", "evmap_dir")
        filename = cls.jobname(args)
        if dirname == "":
            dirname = conf.get("dag", "output_dir")
            filename = filename + "_def"
        else:
            common.mkdir(dirname)
        return "{0}/{1}".format(dirname, filename)

    @staticmethod
    def event_dir(conf):
        dirname = conf.get("dag", "evts_dir")
        if dirname == "":
            dirname = conf.get("dag", "output_dir")
        else:
            common.mkdir(dirname)

    @classmethod
    def event_filepath(cls, args):
        conf, dt_range, area = args
        dirname = conf.get("dag", "evts_dir")
        filename = cls.jobname(args)
        if dirname == "":
            dirname = conf.get("dag", "output_dir")
            filename = filename + "_ev"
        else:
            common.mkdir(dirname)
        return "{0}/{1}".format(dirname, filename)

    @staticmethod
    def dag_dir(conf):
        dirname = conf.get("dag", "output_dir")
        common.mkdir(dirname)

    @classmethod
    def dag_filepath(cls, args):
        conf, dt_range, area = args
        dirname = conf.get("dag", "output_dir")
        common.mkdir(dirname)
        filename = cls.jobname(args)
        return "{0}/{1}".format(dirname, filename)

    def init_dirs(self, conf):
        self.event_dir(conf)
        self.evdef_dir(conf)
        self.dag_dir(conf)


def args2name(args):
    return ArgumentManager.jobname(args)


def name2args(name, conf):
    return ArgumentManager.jobname2args(name, conf)


def open_logdag_config(ns):
    fp = ns.conf_path if ns is not None else None
    conf = config.open_config(ns.conf_path, ex_defaults = [DEFAULT_CONFIG])
    lv = logging.DEBUG if ns.debug else logging.INFO
    am_logger = logging.getLogger("amulog")
    config.set_common_logging(conf, logger = [_logger, am_logger], lv = lv)
    return conf


def whole_term(conf, ld = None):
    w_term = config.getterm(conf, "dag", "whole_term")
    if w_term is None:
        if ld is None:
            from amulog import log_db
            ld = log_db.LogData(conf)
        return ld.whole_term()
    else:
        return w_term


def all_args(conf):
    from amulog import log_db
    ld = log_db.LogData(conf)
    w_top_dt, w_end_dt = whole_term(conf, ld)
    term = config.getdur(conf, "dag", "unit_term")
    diff = config.getdur(conf, "dag", "unit_diff")

    l_args = []
    top_dt = w_top_dt
    while top_dt < w_end_dt:
        end_dt = top_dt + term
        l_area = config.getlist(conf, "dag", "area")
        if "each" in l_area:
            l_area.pop(l_area.index("each"))
            l_area += ["host_" + host for host 
                       in ld.whole_host(top_dt, end_dt)]
        for area in l_area:
            l_args.append((conf, (top_dt, end_dt), area))
        top_dt = top_dt + diff
    return l_args



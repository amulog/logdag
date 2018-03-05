#!/usr/bin/env python
# coding: utf-8

from . import dtutil
from amulog import config
from amulog import common


class ArgumentManager(object):

    def __init__(self, conf):
        self.conf = conf
        self.args_filename = conf.get("dag", "args_fn")
        if self.args_filename.strip() == "":
            confname = conf.get("general", "base_filename").split("/")[-1]
            self.args_filename = "args_{0}".format(confname)

    def __iter__(self):
        return self.l_args

    def __getitem__(self, i):
        return self.l_args[i]

    def __len__(self):
        return len(self.l_args)

    def generate(self, func):
        self.l_args = func(self.conf)
        self.evdef_dir(self.conf)

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
        return common.cli_table(table)

    def dump(self):
        with open(self.args_filename, 'w') as f:
            f.write("\n".join([self.jobname(args) for args in self.l_args]))

    def load(self):
        l_args = []
        with open(self.args_filename, 'r') as f:
            for line in f:
                args = cls.jobname2args(line.rstrip(), self.conf)
                self.l_args.append(args)

    @staticmethod
    def jobname(args):

        def dt_filename(dt_range):
            top_dt, end_dt = dt_range
            if dtutil.is_intdate(top_dt) and dtutil.is_intdate(end_dt):
                return top_dt.strftime("%Y%m%d")
            else:
                return top_dt.strftime("%Y%m%d_%H%M%S")

        conf, dt_range, area = args
        return "_".join([area, dt_filename(dt_range)])

    @staticmethod
    def jobname2args(cls, name, conf):
        area, dtstr = name.split("_", 1)
        if "_" in dtstr:
            top_dt = datetime.strptime(dtstr, "%Y%m%d_%H%M%S")
        else:
            top_dt = datetime.strptime(dtstr, "%Y%m%d")
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



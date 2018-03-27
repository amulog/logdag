#!/usr/bin/env python
# coding: utf-8

import sys
import logging
import argparse

from . import arguments
from amulog import config
from amulog import common

_logger = logging.getLogger(__package__)


def test_makedag(ns):
    from . import makedag
    from . import arguments
    conf = arguments.open_logdag_config(ns)

    am = arguments.ArgumentManager(conf)
    am.generate(arguments.all_args)
    am.dump()
    makedag.makedag_main(am[0])


def make_args(ns):
    from . import arguments
    conf = arguments.open_logdag_config(ns)

    am = arguments.ArgumentManager(conf)
    am.generate(arguments.all_args)
    am.init_dirs(conf)
    am.dump()


def make_input(ns):
    from . import makedag
    from . import arguments

    def mkinput_sprocess(am):
        timer = common.Timer("mkinput task", output = _logger)
        timer.start()
        for args in am:
            makedag.make_input(args)
        timer.stop()

    def mkinput_mprocess(am, pal=1):
        import multiprocessing
        timer = common.Timer("mkinput task", output = _logger)
        timer.start()
        l_process = [multiprocessing.Process(name = am.jobname(args),
                                             target = makedag.make_input,
                                             args = [args,])
                     for args in am]
        common.mprocess(l_process, pal)
        timer.stop()

    conf = arguments.open_logdag_config(ns)

    am = arguments.ArgumentManager(conf)
    am.generate(arguments.all_args)
    am.init_dirs(conf)
    am.dump()

    p = ns.parallel
    if p > 1:
        mkinput_mprocess(am, p)
    else:
        mkinput_sprocess(am)


def make_input_stdin(ns):
    from . import makedag
    from . import arguments

    conf = arguments.open_logdag_config(ns)

    am = arguments.ArgumentManager(conf)
    am.init_dirs(conf)
    args = am.jobname2args(ns.argname, conf)

    timer = common.Timer("mkinput task for {0}".format(ns.argname),
                         output = _logger)
    timer.start()
    makedag.make_input(args)
    timer.stop()


def make_dag(ns):
    from . import makedag
    from . import arguments

    def makedag_sprocess(am):
        timer = common.Timer("makedag task", output = _logger)
        timer.start()
        for args in am:
            makedag.makedag_main(args)
        timer.stop()

    def makedag_mprocess(am, pal=1):
        import multiprocessing
        timer = common.Timer("makedag task", output = _logger)
        timer.start()
        l_process = [multiprocessing.Process(name = am.jobname(args),
                                             target = makedag.makedag_main,
                                             args = [args,])
                     for args in am]
        common.mprocess(l_process, pal)
        timer.stop()

    conf = arguments.open_logdag_config(ns)

    am = arguments.ArgumentManager(conf)
    am.generate(arguments.all_args)
    am.init_dirs(conf)
    am.dump()

    p = ns.parallel
    if p > 1:
        makedag_mprocess(am, p)
    else:
        makedag_sprocess(am)


def make_dag_stdin(ns):
    from . import makedag
    from . import arguments

    conf = arguments.open_logdag_config(ns)

    am = arguments.ArgumentManager(conf)
    am.init_dirs(conf)
    args = am.jobname2args(ns.argname, conf)

    timer = common.Timer("makedag task for {0}".format(ns.argname),
                         output = _logger)
    timer.start()
    makedag.makedag_main(args)
    timer.stop()


def make_dag_large(ns):
    from . import makedag
    from . import arguments

    def makedag_large_sprocess(ll_args, am):
        timer = common.Timer("makedag_large task", output = _logger)
        timer.start()
        for l_args in ll_args:
            makedag.makedag_large(l_args)
        timer.stop()

    def makedag_large_mprocess(ll_args, am, pal=1):
        import multiprocessing
        timer = common.Timer("makedag_large task", output = _logger)
        timer.start()
        l_process = [multiprocessing.Process(name = am.jobname(l_args[0]),
                                             target = makedag.makedag_large,
                                             args = [l_args,])
                     for l_args in ll_args]
        common.mprocess(l_process, pal)
        timer.stop()

    conf = arguments.open_logdag_config(ns)

    am = arguments.ArgumentManager(conf)
    am.generate(arguments.all_args)
    am.init_dirs(conf)
    am.dump()

    ll_args = []
    length = ns.length
    from itertools import zip_longest
    for area in am.areas():
        temp_l_args = sorted(am.args_in_area(area), key = lambda x: x[1][0])
        for l_args in [l for l
                       in zip_longest(*[iter(temp_l_args[::-1])] * length)]:
            ll_args.append([a for a in l_args if a is not None][::-1])

    p = ns.parallel
    if p > 1:
        makedag_large_mprocess(ll_args, am, p)
    else:
        makedag_large_sprocess(ll_args, am)



def show_args(ns):
    from . import arguments
    conf = arguments.open_logdag_config(ns)
    
    am = arguments.ArgumentManager(conf)
    try:
        am.load()
    except IOError:
        path = am.args_filename
        sys.exit("ArgumentManager object file ({0}) not found".format(path))
    except:
        raise

    print(am.show())


def show_list(ns):
    from . import arguments
    from . import showdag
    conf = arguments.open_logdag_config(ns)
    
    print(showdag.list_results(conf))


def show_results_sum(ns):
    from . import arguments
    from . import showdag
    conf = arguments.open_logdag_config(ns)

    print(showdag.show_results_sum(conf))


def show_netsize(ns):
    from . import arguments
    from . import showdag
    conf = arguments.open_logdag_config(ns)

    print(showdag.show_netsize_dist(conf))


def show_netsize_list(ns):
    from . import arguments
    from . import showdag
    conf = arguments.open_logdag_config(ns)

    print(showdag.list_netsize(conf))


def plot_filter(ns):
    from . import arguments
    from . import log2event
    conf = arguments.open_logdag_config(ns)
    if ns.conf_nofilter is not None:
        conf_nofilter = config.open_config(
            ns.conf_nofilter, ex_defaults = [arguments.DEFAULT_CONFIG])
    else:
        conf_nofilter = None

    args = arguments.name2args(ns.argname, conf)
    gid = ns.gid
    host = ns.host
    if ns.binsize is None:
        binsize = None
    else:
        binsize = config.str2dur(ns.binsize)
    dirname = ns.dirname
    log2event.graph_filter(args, gid = ns.gid, host = ns.host,
                           binsize = binsize, conf_nofilter = conf_nofilter,
                           dirname = ns.dirname)


def plot_discretize(ns):
    from . import arguments
    from . import log2event
    conf = arguments.open_logdag_config(ns)

    args = arguments.name2args(ns.argname, conf)
    gid = ns.gid
    host = ns.host
    if ns.binsize is None:
        binsize = None
    else:
        binsize = config.str2dur(ns.binsize)
    dirname = ns.dirname
    log2event.graph_dis(args, gid = ns.gid, host = ns.host,
                        binsize = binsize,
                        dirname = ns.dirname)


# common argument settings
OPT_DEBUG = [["--debug"],
             {"dest": "debug", "action": "store_true",
              "help": "set logging level to debug (default: info)"}]
OPT_CONFIG = [["-c", "--config"],
              {"dest": "conf_path", "metavar": "CONFIG", "action": "store",
               "default": None,
               "help": "configuration file path for amulog"}]
OPT_PARALLEL = [["-p", "--parallel"],
                {"dest": "parallel", "metavar": "PARALLEL",
                 "type": int, "default": 1,
                 "help": "number of processes in parallel"}]
OPT_DIRNAME = [["-d", "--dirname"],
               {"dest": "dirname", "metavar": "DIRNAME", "action": "store",
                "default": ".",
                "help": "directory name for output"}]
OPT_GID = [["-g", "--gid"],
           {"dest": "gid", "metavar": "GID", "action": "store",
            "type": int, "default": None,
            "help": "log group identifier to search events"},]
OPT_HOSTNAME = [["-n", "--host"],
                {"dest": "host", "metavar": "HOST", "action": "store",
                 "default": None,
                 "help": "hostname to search events"}]
OPT_BINSIZE = [["-b", "--binsize"],
               {"dest": "binsize", "metavar": "BINSIZE",
                "action": "store", "default": None,
                "help": "binsize (like 10s)"}]
ARG_ARGNAME = [["argname"],
               {"metavar": "TASKNAME", "action": "store",
                "help": "argument name"}]

# argument settings for each modes
# description, List[args, kwargs], func
# defined after functions because these settings use functions
DICT_ARGSET = {
    "test": ["Generate DAG",
             [OPT_CONFIG, OPT_DEBUG],
             test_makedag],
    "make-args": ["Initialize arguments for pc algorithm",
                  [OPT_CONFIG, OPT_DEBUG],
                  make_args],
    "make-input": ["Generate input time-series for pc algorithm",
                   [OPT_CONFIG, OPT_DEBUG, OPT_PARALLEL],
                   make_input],
    "make-input-stdin": ["make-input interface for pipeline processing",
                         [OPT_CONFIG, OPT_DEBUG,
                          [["argname"],
                           {"metavar": "TASKNAME", "action": "store",
                            "help": "argument name"}]],
                         make_input_stdin],
    "make-dag": ["Generate causal DAGs",
                   [OPT_CONFIG, OPT_DEBUG, OPT_PARALLEL],
                   make_dag],
    "make-dag-stdin": ["make-dag interface for pipeline processing",
                       [OPT_CONFIG, OPT_DEBUG, ARG_ARGNAME],
                       make_dag_stdin],
    "make-dag-large": ["Generate causal DAGs from multiple terms",
                       [OPT_CONFIG, OPT_DEBUG, OPT_PARALLEL,
                        [["-l", "--length"],
                         {"dest": "length", "metavar": "LENGTH",
                          "action": "store", "type": int, "default": 1,
                          "help": "number of unit terms"}],],
                       make_dag_large],
    "show-args": ["Show arguments recorded in argument file",
                  [OPT_CONFIG, OPT_DEBUG],
                  show_args],
    "show-list": ["Show abstracted results of DAG generation",
                  [OPT_CONFIG, OPT_DEBUG],
                  show_list],
    "show-results-sum": ["Show abstracted results of DAG generation",
                         [OPT_CONFIG, OPT_DEBUG],
                         show_results_sum],
    "show-netsize": ["Show distribution of connected subgraphs in DAGs",
                     [OPT_CONFIG, OPT_DEBUG],
                     show_netsize],
    "show-netsize-list": ["Show connected subgraphs in every DAG",
                          [OPT_CONFIG, OPT_DEBUG],
                          show_netsize_list],
    "plot-filter": ["Generate plots to compare filtered time-series",
                    [OPT_CONFIG, OPT_DEBUG, OPT_DIRNAME, OPT_BINSIZE,
                     OPT_GID, OPT_HOSTNAME, ARG_ARGNAME,
                     [["-t", "--target", "--conf-nofilter"],
                      {"dest": "conf_nofilter", "metavar": "NOFILTER_CONF",
                       "action": "store", "default": None,
                       "help": "config file without filtering"}],],
                    plot_filter],
    "plot-discretize": ["Generate plots to compare discretizing functions",
                        [OPT_CONFIG, OPT_DEBUG, OPT_DIRNAME, OPT_BINSIZE,
                         OPT_GID, OPT_HOSTNAME, ARG_ARGNAME],
                        plot_discretize],
}

USAGE_COMMANDS = "\n".join(["  {0}: {1}".format(key, val[0])
                            for key, val in DICT_ARGSET.items()])
USAGE = ("usage: {0} MODE [options and arguments] ...\n\n"
         "mode:\n".format(sys.argv[0])) + USAGE_COMMANDS + \
    "\n\nsee \"{0} MODE -h\" to refer detailed usage".format(sys.argv[0])

if __name__ == "__main__":
    if len(sys.argv) < 1:
        sys.exit(USAGE)
    mode = sys.argv[1]
    if mode in ("-h", "--help"):
        sys.exit(USAGE)
    commandline = sys.argv[2:]

    desc, l_argset, func = DICT_ARGSET[mode]
    ap = argparse.ArgumentParser(prog = " ".join(sys.argv[0:2]),
                                 description = desc)
    for args, kwargs in l_argset:
        ap.add_argument(*args, **kwargs)
    ns = ap.parse_args(commandline)
    func(ns)



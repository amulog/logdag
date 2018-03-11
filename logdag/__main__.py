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
    conf = arguments.open_logdag_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger = _logger, lv = lv)

    from . import makedag
    from . import arguments
    am = arguments.ArgumentManager(conf)
    am.generate(arguments.all_args)
    am.dump()
    makedag.makedag_main(am[0])


def make_args(ns):
    from . import arguments
    conf = arguments.open_logdag_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger = _logger, lv = lv)

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

    conf = arguments.open_logdag_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger = _logger, lv = lv)

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

    conf = arguments.open_logdag_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger = _logger, lv = lv)

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

    conf = arguments.open_logdag_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger = _logger, lv = lv)

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

    conf = arguments.open_logdag_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger = _logger, lv = lv)

    am = arguments.ArgumentManager(conf)
    am.init_dirs(conf)
    args = am.jobname2args(ns.argname, conf)

    timer = common.Timer("makedag task for {0}".format(ns.argname),
                         output = _logger)
    timer.start()
    makedag.makedag_main(args)
    timer.stop()


def show_args(ns):
    from . import arguments
    conf = arguments.open_logdag_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger = _logger, lv = lv)
    
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
    conf = arguments.open_logdag_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger = _logger, lv = lv)
    
    print(showdag.list_results(conf))


def show_results_sum(ns):
    from . import arguments
    from . import showdag
    conf = arguments.open_logdag_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger = _logger, lv = lv)

    print(showdag.show_results_sum(conf))


def show_netsize(ns):
    from . import arguments
    from . import showdag
    conf = arguments.open_logdag_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger = _logger, lv = lv)

    print(showdag.show_netsize_dist(conf))


def show_netsize_list(ns):
    from . import arguments
    from . import showdag
    conf = arguments.open_logdag_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger = _logger, lv = lv)

    print(showdag.list_netsize(conf))


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
                       [OPT_CONFIG, OPT_DEBUG,
                        [["argname"],
                         {"metavar": "TASKNAME", "action": "store",
                          "help": "argument name"}]],
                       make_dag_stdin],
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



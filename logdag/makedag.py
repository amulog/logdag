#!/usr/bin/env python
# coding: utf-8

import sys
import logging

from . import arguments
from . import log2event
from . import pc_input
from . import showdag
from amulog import common
from amulog import config

_logger = logging.getLogger(__package__)


def makedag_main(args):
    conf, dt_range, area = args

    _logger.info("makedag job start ({0} - {1} in {2})".format(
        dt_range[0], dt_range[1], area))

    # generate event set and evmap, and apply preprocessing
    evts, evmap = log2event.get_event(args)
    _logger.info("{0} nodes for pc input".format(len(evts)))

    if len(evts) >= 2:
        # convert event set to pc algorithm input
        ci_bin_method = conf.get("dag", "ci_bin_method")
        ci_bin_size = config.getdur(conf, "dag", "ci_bin_size")
        ci_bin_diff = config.getdur(conf, "dag", "ci_bin_diff")
        ci_func = conf.get("dag", "ci_func")
        binarize = is_binarize(ci_func)
        data = log2event.event2input(evts, ci_bin_method, ci_bin_size,
                                     ci_bin_diff, dt_range, binarize)

        # apply pc algorithm to estimate dag
        skel_method = conf.get("dag", "skeleton_method")
        skel_th = conf.getfloat("dag", "skeleton_threshold")
        skel_depth = conf.getint("dag", "skeleton_depth")
        skel_verbose = conf.getboolean("dag", "skeleton_verbose")
        graph = pc_input.pc(data, skel_th, ci_func, skel_method,
                skel_depth, skel_verbose)
    else:
        _logger.info("input too small({0} nodes), return empty dag".format(
            len(evts)))
        graph = showdag.empty_dag()

    # record dag
    ldag = showdag.LogDAG(args, graph)
    ldag.dump()
    _logger.info("makedag job done, output {0}".format(
        arguments.dag_filepath(args)))
    return ldag


def make_input(args):
    conf, dt_range, area = args
    _logger.info("make_input job start ({0} - {1} in {2})".format(
        dt_range[0], dt_range[1], area))

    evts, evmap = log2event.get_event(args)
    _logger.info("{0} nodes for pc input".format(len(evts)))
    _logger.info("make_input job done")


def is_binarize(ci_func):
    if ci_func == "fisherz":
        return False
    elif ci_func == "fisherz_bin":
        return True
    elif ci_func == "gsq":
        return True
    elif ci_func == "gsq_rlib":
        return True
    else:
        raise NotImplementedError



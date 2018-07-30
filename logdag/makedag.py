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

    ci_func = conf.get("dag", "ci_func")
    binarize = is_binarize(ci_func)
    # generate event set and evmap, and apply preprocessing
    d_input, evmap = log2event.ts2input(conf, dt_range, area, binarize)
    _logger.info("{0} nodes for pc input".format(len(d_input)))
    evmap.dump(args)

    graph = estimate_dag(conf, d_input, ci_func)

    # record dag
    ldag = showdag.LogDAG(args, graph)
    ldag.dump()
    _logger.info("makedag job done, output {0}".format(
        arguments.ArgumentManager.dag_filepath(args)))
    return ldag


def estimate_dag(conf, d_input, ci_func):
    if len(d_input) >= 2:
        cause_algorithm = conf.get("dag", "cause_algorithm")
        if cause_algorithm == "pc":
            # apply pc algorithm to estimate dag
            skel_method = conf.get("dag", "skeleton_method")
            skel_th = conf.getfloat("dag", "skeleton_threshold")
            skel_depth = conf.getint("dag", "skeleton_depth")
            skel_verbose = conf.getboolean("dag", "skeleton_verbose")
            graph = pc_input.pc(d_input, skel_th, ci_func, skel_method,
                    skel_depth, skel_verbose)
        elif cause_algorithm == "lingam":
            from . import lingam_input
            graph = lingam_input.estimate(d_input)
    else:
        _logger.info("input too small({0} nodes), return empty dag".format(
            len(d_input)))
        graph = showdag.empty_dag()

    return graph


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



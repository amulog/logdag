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
    d_input, evmap = log2event.ts2input(conf, dt_range, area)
    _logger.info("{0} nodes for pc input".format(len(d_input)))
    evmap.dump(args)

    graph = estimate_dag(conf, d_input)

    # record dag
    ldag = showdag.LogDAG(args, graph)
    ldag.dump()
    _logger.info("makedag job done, output {0}".format(
        arguments.ArgumentManager.dag_filepath(args)))
    return ldag


#def make_input(args):
#    conf, dt_range, area = args
#    _logger.info("make_input job start ({0} - {1} in {2})".format(
#        dt_range[0], dt_range[1], area))
#    
#    def ts2input(conf, dt_range, area):
#    evts, evmap = log2event.get_event(args)
#    _logger.info("{0} nodes for pc input".format(len(evts)))
#    _logger.info("make_input job done")
#
#
#def makedag_large(l_args):
#    conf = l_args[0][0]
#    top_dt = min([args[1][0] for args in l_args])
#    end_dt = max([args[1][1] for args in l_args])
#    dt_range = (top_dt, end_dt)
#    area = l_args[0][2]
#
#    _logger.info("makedag_large job start ({0} - {1} in {2})".format(
#        dt_range[0], dt_range[1], area))
#
#    evts, evmap = log2event.merge_events(l_args, conf, dt_range, area)
#    _logger.info("{0} nodes for pc input".format(len(evts)))
#
#    graph = estimate_dag(conf, evts, dt_range)
#
#    # record dag
#    args = (conf, dt_range, area)
#    ldag = showdag.LogDAG(args, graph)
#    ldag.dump()
#    _logger.info("makedag_large job done, output {0}".format(
#        arguments.ArgumentManager.dag_filepath(args)))
#    return ldag
#
#
#def makedag_small(l_args):
#    args, ext_dt_range = l_args
#    conf, dt_range, area = args
#
#    _logger.info("makedag_small job start ({0} - {1} in {2})".format(
#        ext_dt_range[0], ext_dt_range[1], area))
#
#    evts, evmap = log2event.extract_events(args, ext_dt_range)
#    _logger.info("{0} nodes for pc input".format(len(evts)))
#
#    graph = estimate_dag(conf, evts, ext_dt_range)
#
#    # record dag
#    args = (conf, dt_range, area)
#    ldag = showdag.LogDAG(args, graph)
#    ldag.dump()
#    _logger.info("makedag_small job done, output {0}".format(
#        arguments.ArgumentManager.dag_filepath(args)))
#    return ldag


def estimate_dag(conf, d_input):
    if len(d_input) >= 2:
        # apply pc algorithm to estimate dag
        skel_method = conf.get("dag", "skeleton_method")
        skel_th = conf.getfloat("dag", "skeleton_threshold")
        skel_depth = conf.getint("dag", "skeleton_depth")
        skel_verbose = conf.getboolean("dag", "skeleton_verbose")
        graph = pc_input.pc(d_input, skel_th, ci_func, skel_method,
                skel_depth, skel_verbose)
    else:
        _logger.info("input too small({0} nodes), return empty dag".format(
            len(d_input)))
        graph = showdag.empty_dag()

    return graph


#def is_binarize(ci_func):
#    if ci_func == "fisherz":
#        return False
#    elif ci_func == "fisherz_bin":
#        return True
#    elif ci_func == "gsq":
#        return True
#    elif ci_func == "gsq_rlib":
#        return True
#    else:
#        raise NotImplementedError



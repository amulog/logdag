#!/usr/bin/env python
# coding: utf-8

import datetime
import logging
import math
import numpy as np

from amulog import config
from logdag import dtutil
from . import period

_logger = logging.getLogger(__package__)

FUNCTIONS = ["sizetest", "filter_periodic",
             "remove_periodic", "remove_corr", "remove_linear"]


class LogFilter(object):
    """Filter log time-series and record their periodicity."""

    defaults = {"pre_count": 5,
                "pre_term": datetime.timedelta(hours=6),
                "fourier_sample_rule": [(datetime.timedelta(days=1),
                                         datetime.timedelta(seconds=10)),
                                        (datetime.timedelta(days=7),
                                         datetime.timedelta(minutes=1)), ],
                "fourier_th_spec": 0.4,
                "fourier_th_eval": 0.1,
                "fourier_th_restore": 0.5,
                "fourier_peak_order": 200,
                "corr_sample_rule": [(datetime.timedelta(days=1),
                                      datetime.timedelta(seconds=10)), ],
                "corr_th": 0.5,
                "corr_diff": [datetime.timedelta(hours=1),
                              datetime.timedelta(days=1), ],
                "linear_sample_rule": [(datetime.timedelta(days=1),
                                        datetime.timedelta(seconds=10)), ],
                "linear_count": 10,
                "linear_th": 0.5,
                }

    def __init__(self,
                 rules=None,
                 pre_count=defaults["pre_count"],
                 pre_term=defaults["pre_term"],
                 fourier_sample_rule=defaults["fourier_sample_rule"],
                 fourier_th_spec=defaults["fourier_th_spec"],
                 fourier_th_eval=defaults["fourier_th_eval"],
                 fourier_th_restore=defaults["fourier_th_restore"],
                 fourier_peak_order=defaults["fourier_peak_order"],
                 corr_sample_rule=defaults["corr_sample_rule"],
                 corr_th=defaults["corr_th"],
                 corr_diff=defaults["corr_diff"],
                 linear_sample_rule=defaults["linear_sample_rule"],
                 linear_count=defaults["linear_count"],
                 linear_th=defaults["linear_th"],
                 ):
        self._rules = [] if rules is None else rules
        self._log = {}
        self._pre_count = pre_count
        self._pre_term = pre_term
        self._fourier_sample_rule = fourier_sample_rule
        self._fourier_th_spec = fourier_th_spec
        self._fourier_th_eval = fourier_th_eval
        self._fourier_th_restore = fourier_th_restore
        self._fourier_peak_order = fourier_peak_order
        self._corr_sample_rule = corr_sample_rule
        self._corr_th = corr_th
        self._corr_diff = corr_diff
        self._linear_sample_rule = linear_sample_rule
        self._linear_count = linear_count
        self._linear_th = linear_th

        #for k in self.defaults:
        #    if k in kwargs:
        #        setattr(self, "_" + k, kwargs[k])
        #    else:
        #        setattr(self, "_" + k, self.defaults[k])

    def sizetest(self, l_dt, dt_range, evdef):
        if len(l_dt) < self._pre_count or \
                max(l_dt) - min(l_dt) < self._pre_term:
            self._log[(dt_range, evdef)] = ("sizetest", None, None)
            return None
        else:
            return l_dt

    def _get_additional_data(self, dt_range, evdef):
        raise NotImplementedError

    def _resize_input(self, l_dt, dt_range, sample_dt_length, evdef):
        dt_length = dt_range[1] - dt_range[0]
        if dt_length == sample_dt_length:
            return l_dt
        elif dt_length > sample_dt_length:
            new_dt_range = (dt_range[1] - sample_dt_length, dt_range[1])
            new_l_dt = [dt >= new_dt_range[0] for dt in l_dt]
            return new_l_dt
        else:
            add_dt_range = (dt_range[1] - sample_dt_length, dt_range[0])
            add_dt = list(self._get_additional_data(evdef, add_dt_range))
            return sorted(add_dt + l_dt)

    @staticmethod
    def _revert_event(a_cnt, dt_range, binsize):
        l_dt = []
        top_dt, end_dt = dt_range
        assert top_dt + len(a_cnt) * binsize == end_dt
        for i, val in enumerate(a_cnt):
            if val > 0:
                dt = top_dt + i * binsize
                l_dt += [dt] * int(val)
        return l_dt
        # return [top_dt + i * binsize for i, val in enumerate(a_cnt) if val > 0]

    def filter_periodic(self, l_dt, dt_range, evdef):
        for sample_dt_length, binsize in self._fourier_sample_rule:
            tmp_l_dt = self._resize_input(l_dt, dt_range, sample_dt_length, evdef)
            a_cnt = dtutil.discretize_sequential(tmp_l_dt, dt_range,
                                                 binsize, binarize=False)
            args = (a_cnt, binsize, self._fourier_th_spec,
                    self._fourier_th_eval, self._fourier_th_restore,
                    self._fourier_peak_order)
            is_periodic, a_remain, interval = period.fourier_replace(*args)
            if a_remain is not None:
                l = ("filter_periodic", (sample_dt_length, binsize), interval)
                self._log[(dt_range, evdef)] = l
                return self._revert_event(a_remain, dt_range, binsize)
        else:
            return l_dt

    def remove_periodic(self, l_dt, dt_range, evdef):
        for sample_dt_length, binsize in self._fourier_sample_rule:
            tmp_l_dt = self._resize_input(l_dt, dt_range, sample_dt_length, evdef)
            a_cnt = dtutil.discretize_sequential(tmp_l_dt, dt_range,
                                                 binsize, binarize=False)
            args = (a_cnt, binsize, self._fourier_th_spec,
                    self._fourier_th_eval, self._fourier_peak_order)
            is_periodic, interval = period.fourier_remove(*args)
            if is_periodic:
                l = ("remove_periodic", (sample_dt_length, binsize), interval)
                self._log[(dt_range, evdef)] = l
                return None
        else:
            return l_dt

    def remove_corr(self, l_dt, dt_range, evdef):
        for sample_dt_length, binsize in self._corr_sample_rule:
            tmp_l_dt = self._resize_input(l_dt, dt_range, sample_dt_length, evdef)
            a_cnt = dtutil.discretize_sequential(tmp_l_dt, dt_range,
                                                 binsize, binarize=False)
            args = (a_cnt, binsize, self._corr_th, self._corr_diff)
            is_periodic, interval = period.periodic_corr(*args)
            if is_periodic:
                l = ("remove_corr", (sample_dt_length, binsize), interval)
                self._log[(dt_range, evdef)] = l
                return None
        else:
            return l_dt

    def remove_linear(self, l_dt, dt_range, evdef):
        for sample_dt_length, binsize in self._linear_sample_rule:
            tmp_l_dt = self._resize_input(l_dt, dt_range, sample_dt_length, evdef)
            if len(tmp_l_dt) < self._linear_count:
                continue

            # generate time-series cumulative sum
            length = (dt_range[1] - dt_range[0]).total_seconds()
            bin_length = binsize.total_seconds()
            bins = math.ceil(1.0 * length / bin_length)
            a_stat = np.array([0] * int(bins))
            for dt in l_dt:
                cnt = int((dt - dt_range[0]).total_seconds() / bin_length)
                assert cnt < len(a_stat)
                a_stat[cnt:] += 1

            a_linear = np.linspace(0, len(l_dt), bins, endpoint=False)
            val = sum((a_stat - a_linear) ** 2) / (bins * len(l_dt))
            if val < self._linear_th:
                l = ("remove_linear", (sample_dt_length, binsize), None)
                self._log[(dt_range, evdef)] = l
                return None
        else:
            return l_dt

    def apply_filters(self, l_dt, dt_range, ev):
        tmp_l_dt = l_dt
        for method in self._rules:
            args = (tmp_l_dt, dt_range, ev)
            # import pdb; pdb.set_trace()
            tmp_l_dt = getattr(self, method)(*args)
            if method == "sizetest" and tmp_l_dt is None:
                # sizetest failure means skipping later tests
                # and leave all events
                return l_dt
            elif tmp_l_dt is None or len(tmp_l_dt) == 0:
                msg = "event {0} removed with {1}".format(ev, method)
                _logger.info(msg)
                return None
        return tmp_l_dt


class LogFilterEVDB(LogFilter):

    def __init__(self, source, **kwargs):
        super().__init__(**kwargs)
        self._source = source

    def _get_additional_data(self, evdef, dt_range):
        return self._source.load(evdef, dt_range=dt_range)


class LogFilterAmulogLoader(LogFilter):

    def __init__(self, al, **kwargs):
        super().__init__(**kwargs)
        self._al = al

    def _get_additional_data(self, evdef, dt_range):
        ev = (evdef.host, evdef.gid)
        return self._al.iter_dt(ev, dt_range=dt_range)


def init_logfilter(conf, mode="direct", loader=None):
    # kwargs = dict(conf["filter"])
    kwargs = {}
    kwargs["rules"] = config.getlist(conf, "filter", "rules")
    kwargs["pre_count"] = conf.getint("filter", "pre_count")
    kwargs["pre_term"] = config.getdur(conf, "filter", "pre_term")
    kwargs["fourier_sample_rule"] = [
        tuple(config.str2dur(s) for s in dt_cond.split("_"))
        for dt_cond in config.gettuple(conf, "filter",
                                       "fourier_sample_rule")]
    kwargs["fourier_th_spec"] = conf.getfloat("filter", "fourier_th_spec")
    kwargs["fourier_th_eval"] = conf.getfloat("filter", "fourier_th_eval")
    kwargs["fourier_th_restore"] = conf.getfloat("filter",
                                                 "fourier_th_restore")
    kwargs["fourier_peak_order"] = conf.getint("filter", "fourier_peak_order")

    kwargs["corr_sample_rule"] = [
        tuple(config.str2dur(s) for s in dt_cond.split("_"))
        for dt_cond in config.gettuple(conf, "filter", "corr_sample_rule")]
    kwargs["corr_th"] = conf.getfloat("filter", "corr_th")
    kwargs["corr_diff"] = [config.str2dur(diffstr) for diffstr
                           in config.gettuple(conf, "filter", "corr_diff")]

    kwargs["linear_sample_rule"] = [
        tuple(config.str2dur(s) for s in dt_cond.split("_"))
        for dt_cond in config.gettuple(conf, "filter",
                                       "linear_sample_rule")]
    kwargs["linear_count"] = conf.getint("filter", "linear_count")
    kwargs["linear_th"] = conf.getfloat("filter", "linear_th")

    if mode == "evdb":
        return LogFilterEVDB(loader, **kwargs)
    elif mode == "direct":
        return LogFilterAmulogLoader(loader, **kwargs)

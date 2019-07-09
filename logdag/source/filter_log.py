#!/usr/bin/env python
# coding: utf-8

from amulog import config
from logdag import dtutil
from . import period

FUNCTIONS = ["sizetest", "filter_periodic",
             "remove_periodic", "remove_corr", "remove_linear"]


class LogFilter():
    """Filter log time-series and record their periodicity."""

    defaults = {"fourier_sample_rule": [(datetime.timedelta(days = 1),
                                         datetime.timedelta(seconds = 10)),
                                        (datetime.timedelta(days = 7),
                                         datetime.timedelta(minutes = 1)),],
                "fourier_th_spec": 0.4,
                "fourier_th_eval": 0.1,
                "fourier_th_restore": 0.5,
                "fourier_peak_order": 200,
                "corr_sample_rule": (datetime.timedelta(days = 1),
                                     datetime.timedelta(seconds = 10)),
                "corr_th": 0.5,
                "corr_diff": (datetime.timedelta(hours = 1),
                              datetime.timedelta(days = 1),),
                "linear_sample_rule": (datetime.timedelta(days = 1),
                                       datetime.timedelta(seconds = 10)),
                "linear_count": 10,
                "linear_threshold": 0.5,
                }

    def __init__(self, source, **kwargs):
        self._log = {}
        for k in self.defaults:
            if k in kwargs:
                setattr(self, "_" + k, kwargs[k])
            else:
                setattr(self, "_" + k, self.defaults[k])

    def _parse_kwargs(self, kwargs):
        return [kwargs[key] for key in self.arguments]

    def sizetest(self, source, l_dt, dt_range, evdef):
        l_dt = self._parse_kwargs(kwargs)[1]
        if len(l_dt) < p_cnt:
            return None
        elif max(l_dt) - min(l_dt) < p_term:
            return None
        else:
            return l_dt

    def _resize_input(self, source, l_dt, dt_range, rule, evdef):
        dt_length = dt_range[1] - dt_range[0]
        sample_dt_length = rule[0]
        if dt_length == sample_dt_length:
            return l_dt
        elif dt_length > sample_dt_length:
            new_dt_range = (dt_range[1] - sample_dt_length, dt_range[1])
            new_l_dt = [dt >= new_dt_range[0] for dt in l_dt]
            return new_lt_dt
        else:
            add_dt_range = (dt_range[1] - sample_dt_length, dt_range[0])
            add_dt = list(self.source.iter_dt(evdef))
            return add_dt + l_dt

    @staticmethod
    def _revert_event(a_cnt, dt_range, binsize):
        l_dt = []
        top_dt, end_dt = dt_range
        assert top_dt + len(a_cnt) * binsize == end_dt
        for i, val in enumerate(a_cnt):
            if val > 0:
                dt = top_dt + i * binsize
                l_dt += [dt] * val
        return l_dt
        #return [top_dt + i * binsize for i, val in enumerate(a_cnt) if val > 0]

    def filter_periodic(self, source, l_dt, dt_range, evdef):
        source, l_dt, dt_range, evdef = self._parse_kwargs(kwargs)
        for rule in self._periodic_sample_rule:
            tmp_l_dt = self._resize_input(source, l_dt, dt_range, rule, evdef)
            a_cnt = dtutil.discretize_sequential(tmp_l_dt, dt_range,
                                                 binsize, binarize = False)
            args = (a_cnt, binsize, self._fourier_th_spec,
                    self._fourier_th_eval, self._fourier_th_restore,
                    self._fourier_peak_order)
            is_periodic, a_remain, interval = period.fourier_replace(*args)
            if a_remain is not None:
                self._log[(dt_range, evdef)] = ("filter_periodic",
                                                rule, interval)
                return self._revert_event(a_remain, dt_range, binsize)
        else:
            return l_dt

    def remove_periodic(self, source, l_dt, dt_range, evdef):
        source, l_dt, dt_range, evdef = self._parse_kwargs(kwargs)
        for rule in self._periodic_sample_rule:
            tmp_l_dt = self._resize_input(source, l_dt, dt_range, rule, evdef)
            a_cnt = dtutil.discretize_sequential(tmp_l_dt, dt_range,
                                                 binsize, binarize = False)
            args = (a_cnt, binsize, self._fourier_th_spec,
                    self._fourier_th_eval, self._fourier_peak_order)
            is_periodic, interval = period.fourier_remove(*args)
            if is_periodic:
                self._log[(dt_range, evdef)] = ("remove_periodic",
                                                rule, interval)
                return None
        else:
            return l_dt

    def remove_corr(self, source, l_dt, dt_range, evdef):
        source, l_dt, dt_range, evdef = self._parse_kwargs(kwargs)
        for rule in self._corr_sample_rule:
            tmp_l_dt = self._resize_input(source, l_dt, dt_range, rule, evdef)
            a_cnt = dtutil.discretize_sequential(tmp_l_dt, dt_range,
                                                 binsize, binarize = False)
            args = (a_cnt, binsize, self._corr_th, self._corr_diff)
            is_periodic, interval = period.periodic_corr(*args) 
            if is_periodic:
                self._log[(dt_range, evdef)] = ("remove_corr", rule, None)
                return None
        else:
            return l_dt

    def remove_linear(self, source, l_dt, dt_range, evdef):
        source, l_dt, dt_range, evdef = self._parse_kwargs(kwargs)
        for rule in self._linear_sample_rule:
            tmp_l_dt = self._resize_input(source, l_dt, dt_range, rule, evdef)
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

            a_linear = np.linspace(0, len(l_dt), bins, endpoint = False)
            val = sum((a_stat - a_linear) ** 2) / (bins * len(l_dt))
            if val < self._linear_th:
                self._log[(dt_range, evdef)] = ("remove_linear",
                                                rule, None)
                return None
        else:
            return l_dt


def init_logfilter(conf, source):
    kwargs = dict(conf["filter"])
    kwargs["filter_sample_rule"] = [
        (config.str2dur(s) for s in dt_cond.split("_"))
        for dt_cond in config.gettuple(conf, "filter",
                                       "fourier_sample_rule")]
    kwargs["fourier_th_spec"] = conf.getfloat("filter", "fourier_th_spec")
    kwargs["fourier_th_eval"] = conf.getfloat("filter", "fourier_th_eval")
    kwargs["fourier_th_restore"] = conf.getfloat("filter",
                                                 "fourier_th_restore")
    kwargs["fourier_peak_order"] = conf.getint("filter", "fourier_peak_order")

    kwargs["corr_sample_rule"] = [
        (config.str2dur(s) for s in dt_cond.split("_"))
        for dt_cond in config.gettuple(conf, "filter", "corr_sample_rule")]
    kwargs["corr_th"] = conf.getfloat("filter", "corr_th")
    kwargs["corr_diff"] = [config.str2dur(diffstr) for diffstr
                           in conf.gettuple("filter", "corr_diff")]

    kwargs["linear_sample_rule"] = [
        (config.str2dur(s) for s in dt_cond.split("_"))
        for dt_cond in config.gettuple(conf, "filter",
                                       "linear_sample_rule")]
    kwargs["linear_count"] = conf.getint("filter", "linear_count")
    kwargs["linear_th"] = conf.getint("filter", "linear_th")

    return LogFilter(source, **kwargs)


#!/usr/bin/env python
# coding: utf-8

import os
import logging
import pickle
#import math
#import numpy as np
from collections import namedtuple
#from collections import UserDict

from . import dtutil
from . import arguments
#from . import period
from amulog import config

_logger = logging.getLogger(__package__)
EvDef = namedtuple("EvDef", ["gid", "host"])


class EventDefinitionMap(object):
    """Before analyze system log messages, we need to classify them with
    multiple criterions like log template IDs and hostnames.
    This class defines classified groups as "Event", and provide
    interconvirsion functions between Event IDs and their
    classifying criterions.
    
    This class allows 2 types of classifying criterion combinations.
        ltgid-host: Use log template grouping IDs and hostnames.
        ltid-host: Use log template IDs and hostnames.
    
    In addition, their can be virtual Events, not a group of log messages
    but a set of symptoms found in the data sequence of log messages.
    This class allows 2 types of virtual Events.
        periodic_top: The start of periodic appearance of an Event.
        periodic_end: The end of periodic appearance of an Event.

    The definition of Event is saved as a nametuple EvDef.
    Evdef has following attributes.
        type (int): An event type identifier.
            0: Normal event that comes from a raw log message appearance.
            1: periodic_top event.
            2: periodic_end event.
        note (any): Some paramaters to show the event characteristics.
            In the case of periodic_*, this attributes requires the interval
            of periodic appearance (seconds(int)) of log messages.
        gid (int): 
        host (str):

    Attributes:
        type_normal (int): 0.
        type_periodic_top (int): 1.
        type_periodic_end (int): 2.
        type_periodic_remainder (int): 3.
        type_mixed (int): 9.
    """
    type_normal = 0
    type_periodic_top = 1
    type_periodic_end = 2
    type_periodic_remainder = 3
    type_mixed = 9
    l_attr = ["gid", "host"]

    def __init__(self, gid_name = "ltgid"):
        """
        Args:
            gid_name (str): A string to assign classifying criterion
                            of log messages. 1 of [ltgid, ltid].
        """
        assert gid_name in ("ltid", "ltgid")
        self.gid_name = gid_name
        #self.l_attr = ["gid", "host"]

        self._emap = {} # key : eid, val : evdef
        self._ermap = {} # key : evdef, val : eid

    def __len__(self):
        return len(self._emap)

    def _eids(self):
        return self._emap.keys()

    def _next_eid(self):
        eid = len(self._emap)
        while eid in self._emap:
            eid += 1
        else:
            return eid

    @staticmethod
    def form_evdef(gid, host):
        d = {"gid" : gid,
             "host" : host}
        return EvDef(**d)

    def add_event(self, gid, host):
        evdef = self.form_evdef(gid, host)
        return self.add_evdef(evdef)

    def add_evdef(self, evdef):
        eid = self._next_eid()
        self._emap[eid] = evdef
        self._ermap[evdef] = eid
        return eid

    def has_eid(self, eid):
        return eid in self._emap

    def has_evdef(self, evdef):
        return evdef in self._ermap

    def evdef(self, eid):
        return self._emap[eid]

    def items(self):
        return self._emap.items()

    def evdef_str(self, eid):
        return self.get_str(self.evdef(eid))

    @classmethod
    def get_str(cls, evdef):
        string = ", ".join(["{0}={1}".format(key, getattr(evdef, key))
                for key in cls.l_attr])
        return "[{0}]".format(string)

#    def search_event(self, gid, host):
#        ret = []
#        for eid, evdef in self._emap.items():
#            if evdef.gid == gid and evdef.host == host:
#                ret.append(eid)
#        if len(ret) > 1:
#            _logger.error("multiple events found for ({0},{1})".format(
#                gid, host))
#            return ret
#        elif len(ret) == 1:
#            return ret[0]
#        else:
#            return None

    def search_cond(self):
        return {key : getattr(self, key) for key in [self.gid_name, "host"]}

    def evdef_repr(self, ld, eid, dt_range, limit = 5):
        info = self._emap[eid]
        top_dt, end_dt = self.dt_range
        d = {"head" : limit, "foot" : limit,
             "top_dt" : top_dt, "end_dt" : end_dt}
        d[self.gid_name] = info.gid
        d["host"] = info.host
        return ld.show_log_repr(**d)

    def get_eid(self, info):
        return self._ermap[info]

    def iter_eid(self):
        return self._emap.keys()

    def iter_evdef(self):
        return self._ermap.keys()

    def dump(self, args):
        fp = arguments.ArgumentManager.evdef_filepath(args)
        obj = (self.gid_name, self._emap, self._ermap)
        with open(fp, "wb") as f:
            pickle.dump(obj, f)

    def load(self, args):
        fp = arguments.ArgumentManager.evdef_filepath(args)
        with open(fp, "rb") as f:
            obj = pickle.load(f)
        self.gid_name, self._emap, self._ermap = obj


#class EventTimeSeries(UserDict):
#
#    def __init__(self, dt_range):
#        UserDict.__init__(self)
#        self.dt_range = dt_range
#
#    def items(self):
#        return self.data.items()
#
#    def add(self, eid, l_dt):
#        if eid in self.data:
#            assert isinstance(self.data[eid], list)
#            self.data[eid] += list(l_dt)
#        else:
#            self.data[eid] = list(l_dt)
#
#    #def add_array(self, eid, array):
#    #    self.data[eid] = array
#
#    @staticmethod
#    def exists(args):
#        fp = arguments.ArgumentManager.event_filepath(args)
#        return os.path.exists(fp)
#
#    def sort(self):
#        for eid in self.data:
#            self.data[eid] = sorted(self.data[eid])
#
#    def dump(self, args, load_org = False):
#        fp = arguments.ArgumentManager.event_filepath(args, load_org)
#        obj = (self.data, self.dt_range)
#        with open(fp, "wb") as f:
#            pickle.dump(obj, f)
#
#    def load(self, args, load_org = False):
#        fp = arguments.ArgumentManager.event_filepath(args, load_org)
#        with open(fp, "rb") as f:
#            obj = pickle.load(f)
#        self.data, self.dt_range = obj


#def get_event(args, load_org = False):
#    conf, dt_range, area = args
#    if EventTimeSeries.exists(args):
#        _logger.debug("using recorded time-series data")
#        gid_name = conf.get("dag", "event_gid")
#        evts = EventTimeSeries(dt_range)
#        evmap = EventDefinitionMap(gid_name)
#        evts.load(args, load_org)
#        evmap.load(args, load_org)
#    else:
#        _logger.debug("generating time-series data from db")
#        from amulog import log_db
#        ld = log_db.LogData(conf)
#        evts, evmap = log2event(conf, ld, dt_range, area)
#        evts.dump(args, load_org)
#        evmap.dump(args, load_org)
#
#    return evts, evmap


#def log2event(conf, ld, dt_range, area):
#    gid_name = conf.get("dag", "event_gid")
#    usefilter = conf.getboolean("dag", "usefilter")
#    top_dt, end_dt = dt_range
#    evmap = EventDefinitionMap(gid_name)
#    evts = EventTimeSeries(dt_range)
#
#    if gid_name == "ltid":
#        iterobj = ld.whole_host_lt(top_dt, end_dt, area)
#    elif gid_name == "ltgid":
#        iterobj = ld.whole_host_ltg(top_dt, end_dt, area)
#    else:
#        raise NotImplementedError
#
#    for host, gid in iterobj:
#        # load time-series from db
#        d = {gid_name: gid,
#             "host": host,
#             "top_dt": top_dt,
#             "end_dt": end_dt}
#        iterobj = ld.iter_lines(**d)
#        l_dt = [line.dt for line in iterobj]
#        del iterobj
#        _logger.debug("gid {0}, host {1}: {2} counts".format(gid, host,
#                                                             len(l_dt)))
#        assert len(l_dt) > 0
#
#        # apply preprocessing
#        interval = None
#        if usefilter:
#            temp_evdef = evmap.form_evdef(gid, host)
#            ret = apply_filter(conf, ld, l_dt, dt_range, temp_evdef)
#            if ret is None:
#                _logger.debug("gid {0}, host {1} -> "
#                              "all filtered".format(gid, host))
#                continue
#            l_dt, interval = ret
#            if len(l_dt) == 0:
#                _logger.debug("gid {0}, host {1} -> "
#                              "all filtered".format(gid, host))
#                continue
#
#        # update evts and evmap
#        if interval is None:
#            _logger.debug("gid {0}, host {1} -> "
#                          "left as is".format(gid, host))
#            type_id = EventDefinitionMap.type_normal
#            note = None
#        else:
#            _logger.debug("gid {0}, host {1} -> "
#                          "{2} counts left".format(gid, host, len(l_dt)))
#            type_id = EventDefinitionMap.type_periodic_remainder
#            note = interval
#        eid = evmap.add_event(gid, host, type_id, note)
#        evts.add(eid, l_dt)
#
#    return evts, evmap
#
#
#def apply_filter(conf, ld, l_dt, dt_range, evdef):
#    """Return remaining time-series after preprocessing."""
#    usefilter = conf.getboolean("dag", "usefilter")
#    if usefilter:
#        act = conf.get("filter", "action")
#        if act in ("remove", "replace"):
#            pflag, remain, interval = filter_periodic(conf, ld, l_dt, dt_range,
#                                                      evdef, method = method)
#            if pflag:
#                return (remain, interval)
#            else:
#                return (l_dt, None)
#        elif act == "linear":
#            lflag = filter_linear(conf, l_dt, dt_range)
#            if lflag:
#                return None
#            else:
#                return (l_dt, None)
#        elif act in ("remove+linear", "replace+linear"):
#            method = act.partition("+")[0]
#            # periodic
#            pflag, remain, interval = filter_periodic(conf, ld, l_dt, dt_range,
#                                                      evdef, method = method)
#            temp_l_dt = remain if pflag else l_dt
#            # linear
#            lflag = filter_linear(conf, temp_l_dt, dt_range)
#            if lflag:
#                return None
#            else:
#                return temp_l_dt, interval
#        elif act in ("linear+remove", "linear+replace"):
#            method = act.partition("+")[-1]
#            # linear
#            lflag = filter_linear(conf, l_dt, dt_range)
#            if lflag:
#                return None
#            else:
#                # periodic
#                pflag, remain, interval = filter_periodic(conf, ld, l_dt,
#                                                          dt_range, evdef,
#                                                          method = method)
#                return remain, interval
#        else:
#            raise NotImplementedError
#    else:
#        return (l_dt, None)
#
#
#def filter_linear(conf, l_dt, dt_range):
#    """Return True if a_cnt appear linearly."""
#    binsize = config.getdur(conf, "filter", "linear_binsize")
#    threshold = conf.getfloat("filter", "linear_threshold")
#    th_count = conf.getint("filter", "linear_count")
#
#    if len(l_dt) < th_count:
#        return False
#
#    # generate time-series cumulative sum
#    length = (dt_range[1] - dt_range[0]).total_seconds()
#    bin_length = binsize.total_seconds()
#    bins = math.ceil(1.0 * length / bin_length)
#    a_stat = np.array([0] * int(bins))
#    for dt in l_dt:
#        cnt = int((dt - dt_range[0]).total_seconds() / bin_length)
#        assert cnt < len(a_stat)
#        a_stat[cnt:] += 1
#
#    a_linear = np.linspace(0, len(l_dt), bins, endpoint = False)
#    val = sum((a_stat - a_linear) ** 2) / (bins * len(l_dt))
#    return val < threshold
#
#
#def filter_periodic(conf, ld, l_dt, dt_range, evdef, method):
#    """Return True and the interval if a_cnt is periodic."""
#
#    ret_false = False, None, None
#    gid_name = conf.get("dag", "event_gid")
#    p_cnt = conf.getint("filter", "pre_count")
#    p_term = config.getdur(conf, "filter", "pre_term")
#    
#    # preliminary test
#    if len(l_dt) < p_cnt:
#        _logger.debug("time-series count too small, skip")
#        return ret_false
#    elif max(l_dt) - min(l_dt) < p_term:
#        _logger.debug("time-series range too small, skip")
#        return ret_false
#
#    # periodicity test
#    for dt_cond in config.gettuple(conf, "filter", "sample_rule"):
#        dt_length, binsize = [config.str2dur(s) for s in dt_cond.split("_")]
#        if (dt_range[1] - dt_range[0]) == dt_length:
#            temp_l_dt = l_dt
#        else:
#            temp_l_dt = reload_ts(ld, evdef, dt_length, dt_range, gid_name)
#        a_cnt = dtutil.discretize_sequential(temp_l_dt, dt_range,
#                                             binsize, binarize = False)
#
#        remain_dt = None
#        if method == "remove":
#            flag, interval = period.fourier_remove(conf, a_cnt, binsize)
#        elif method == "replace":
#            flag, remain_array, interval = period.fourier_replace(conf, a_cnt,
#                                                                  binsize)
#            if remain_array is not None:
#                remain_dt = revert_event(remain_array, dt_range, binsize)
#        elif method == "corr":
#            flag, interval = period.periodic_corr(conf, a_cnt, binsize) 
#        else:
#            raise NotImplementedError
#        if flag:
#            return flag, remain_dt, interval
#    return ret_false
#
#
#def reload_ts(ld, evdef, dt_length, dt_range, gid_name):
#    new_top_dt = dt_range[1] - dt_length
#    d = {gid_name: evdef.gid,
#         "host": evdef.host,
#         "top_dt": new_top_dt,
#         "end_dt": dt_range[1]}
#    iterobj = ld.iter_lines(**d)
#    return [line.dt for line in iterobj]
#
#
#def revert_event(a_cnt, dt_range, binsize):
#    top_dt, end_dt = dt_range
#    assert top_dt + len(a_cnt) * binsize == end_dt
#    return [top_dt + i * binsize for i, val in enumerate(a_cnt) if val > 0]


def ts2input(conf, dt_range, area):
    from . import tsdb
    gid_name = conf.get("dag", "event_gid")
    method = conf.get("dag", "ci_bin_method")
    ci_bin_size = config.getdur(conf, "dag", "ci_bin_size")
    ci_bin_diff = config.getdur(conf, "dag", "ci_bin_diff")
    ci_func = conf.get("dag", "ci_func")
    binarize = is_binarize(ci_func)
    td = tsdb.TimeSeriesDB(conf)
    evmap = EventDefinitionMap(gid_name)

    d_input = {}
    kwargs = {"dts" : dt_range[0],
              "dte" : dt_range[1],
              "area" : area}
    for gid, host in td.whole_gid_host(**kwargs):
        _logger.debug("load event {0}".format((gid, host)))
        ev_kwargs = {"dts" : dt_range[0],
                     "dte" : dt_range[1],
                     "gid" : gid,
                     "host" : host}
        l_dt = [dt for dt in td.iter_ts(**ev_kwargs)]
        if len(l_dt) == 0:
            _logger.warning("empty event {0}".format((gid, host)))
            continue

        if method == "sequential":
            array = dtutil.discretize_sequential(l_dt, dt_range,
                                                 ci_bin_size, binarize)
        elif method == "slide":
            array = dtutil.discretize_slide(l_dt, dt_range, ci_bin_diff,
                                            ci_bin_size, binarize)
        elif method == "radius":
            ci_bin_radius = 0.5 * ci_bin_size
            array = dtutil.discretize_radius(l_dt, dt_range, ci_bin_diff,
                                             ci_bin_radius, binarize)

        eid = evmap.add_event(gid, host)
        d_input[eid] = array

    return d_input, evmap


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


#def event2input(evts, method, binsize, bin_slide, dt_range, binarize):
#    data = {}
#    for eid, l_dt in evts.items():
#        if len(l_dt) == 0:
#            continue
#        if method == "sequential":
#            array = dtutil.discretize_sequential(l_dt, dt_range,
#                                                 binsize, binarize)
#        elif method == "slide":
#            array = dtutil.discretize_slide(l_dt, dt_range,
#                                            bin_slide, binsize, binarize)
#        elif method == "radius":
#            bin_radius = 0.5 * binsize
#            array = dtutil.discretize_radius(l_dt, dt_range,
#                                             bin_slide, bin_radius, binarize)
#        data[eid] = array
#    return data
#
#
#def merge_events(l_args, conf, dt_range, area):
#    gid_name = conf.get("dag", "event_gid")
#    usefilter = conf.getboolean("dag", "usefilter")
#    evmap = EventDefinitionMap(gid_name)
#    evts = EventTimeSeries(dt_range)
#
#    for args in l_args:
#        _logger.debug("generating time-series data from db")
#        temp_evts, temp_evmap = get_event(args, load_org = True)
#        for temp_eid, l_dt in temp_evts.items():
#            evdef = temp_evmap.evdef(temp_eid)
#            if evmap.has_evdef(evdef):
#                eid = evmap.get_eid(evdef)
#                evts.add(eid, l_dt)
#            elif evmap.search_event(evdef.gid, evdef.host):
#                new_evdef = evmap.form_evdef(evdef.gid, evdef.host,
#                                             type_id = evmap.type_mixed,
#                                             note = None)
#                eid = evmap.add_evdef(new_evdef)
#                evts.add(eid, l_dt)
#            else:
#                eid = evmap.add_evdef(evdef)
#                evts.add(eid, l_dt)
#    evts.sort()
#    evts.dump(l_args[0], load_org = False)
#    evmap.dump(l_args[0], load_org = False)
#
#    return evts, evmap
#
#
#def extract_events(args, ext_dt_range):
#    gid_name = conf.get("dag", "event_gid")
#    usefilter = conf.getboolean("dag", "usefilter")
#    evmap = EventDefinitionMap(gid_name)
#    evts = EventTimeSeries(dt_range)
#
#    temp_evts, temp_evmap = get_event(args, load_org = True)
#    for temp_eid, l_dt in temp_evts.items():
#        ext_l_dt = [dt for dt in l_dt
#                    if ext_dt_range[0] <= dt < ext_dt_range[1]]
#        if len(ext_l_dt) > 0:
#            evdef = temp_evmap.evdef(temp_edi)
#            eid = evmap.add_evdef(evdef)
#            evts.add(eid, ext_l_dt)
#
#    evts.dump(l_args[0], load_org = False)
#    evmap.dump(l_args[0], load_org = False)
#
#    return evts, evmap



# visualize functions
# should be moved to tsdb

#def graph_filter(args, gid = None, host = None, binsize = None,
#                 conf_nofilter = None, dirname = "."):
#    conf, dt_range, area = args
#    if binsize is None:
#        binsize = config.getdur(conf, "dag", "ci_bin_size")
#    evts, evmap = get_event(args)
#
#    if conf_nofilter is None:
#        gid_name = conf.get("dag", "event_gid")
#        evmap2 = EventDefinitionMap(gid_name)
#        evts2 = EventTimeSeries(dt_range)
#
#        from amulog import log_db
#        ld = log_db.LogData(conf)
#        if gid_name == "ltid":
#            iterobj = ld.whole_host_lt(dt_range[0], dt_range[1], area)
#        elif gid_name == "ltgid":
#            iterobj = ld.whole_host_ltg(dt_range[0], dt_range[1], area)
#        else:
#            raise NotImplementedError
#        for temp_host, temp_gid in iterobj:
#            d = {gid_name: temp_gid,
#                 "host": temp_host,
#                 "top_dt": dt_range[0],
#                 "end_dt": dt_range[1]}
#            iterobj = ld.iter_lines(**d)
#            l_dt = [line.dt for line in iterobj]
#            eid = evmap2.add_event(temp_gid, temp_host,
#                                   EventDefinitionMap.type_normal, None)
#            evts2.add(eid, l_dt)
#    else:
#        args_nofilter = conf_nofilter, dt_range, area
#        evts2, evmap2 = get_event(args_nofilter)
#
#    for evdef in evmap.iter_evdef():
#        if (gid is None or evdef.gid == gid) and \
#                (host is None or evdef.host == host):
#            eid1 = evmap.get_eid(evdef)
#            data1 = dtutil.discretize_sequential(evts[eid1], dt_range,
#                                                 binsize, False)
#            eid2 = evmap2.search_event(evdef.gid, evdef.host)
#            data2 = dtutil.discretize_sequential(evts2[eid2], dt_range,
#                                                 binsize, False)
#
#            output = "{0}/{1}_{2}_{3}.pdf".format(dirname,
#                                                  arguments.args2name(args),
#                                                  evdef.gid, evdef.host)
#            plot_ts_diff(data2, data1, output)
#
#
#def graph_dis(args, gid = None, host = None, binsize = None, dirname = "."):
#    conf, dt_range, area = args
#    if binsize is None:
#        binsize = config.getdur(conf, "dag", "ci_bin_size")
#    evts, evmap = get_event(args)
#
#    ci_bin_method = conf.get("dag", "ci_bin_method")
#    ci_bin_size = config.getdur(conf, "dag", "ci_bin_size")
#    ci_bin_diff = config.getdur(conf, "dag", "ci_bin_diff")
#    ci_func = conf.get("dag", "ci_func")
#    from . import makedag
#    binarize = makedag.is_binarize(ci_func)
#    
#    data = {}
#    for eid, l_dt in evts.items():
#        data[eid] = dtutil.discretize_sequential(l_dt, dt_range,
#                                                 binsize, False)
#    data2 = event2input(evts, ci_bin_method, ci_bin_size,
#                        ci_bin_diff, dt_range, binarize)
#
#    for key in evts:
#        evdef = evmap.evdef(key)
#        if (gid is None or evdef.gid == gid) and \
#                (host is None or evdef.host == host):
#            output = "{0}/{1}_{2}_{3}.pdf".format(
#                dirname, arguments.args2name(args), evdef.gid, evdef.host)
#            plot_dis(data[key], data2[key], output)
#
#
#def plot_ts_diff(data1, data2, output):
#    import matplotlib
#    matplotlib.use('Agg')
#    import matplotlib.pyplot as plt
#    import matplotlib.dates
#
#    fig = plt.figure()
#    # a big subplot that is turned off axis lines and ticks
#    # only showing common labels
#    ax = fig.add_subplot(111)
#    ax.spines['top'].set_color('none')
#    ax.spines['bottom'].set_color('none')
#    ax.spines['left'].set_color('none')
#    ax.spines['right'].set_color('none')
#    ax.tick_params(labelcolor='w', top=None, bottom=None,
#                   left=None, right=None)
#    #ax.tick_params(labelcolor='w', top='off', bottom='off',
#    #               left='off', right='off')
#    ax.set_xlabel("Time")
#    ax.set_ylabel("Cumulative sum of time series")
#
#    ax1 = fig.add_subplot(211)
#    ax1.set_xlim(0, len(data1))
#    ax1.plot(range(len(data1)), np.cumsum(data1))
#    ax2 = fig.add_subplot(212)
#    ax2.set_xlim(0, len(data2))
#    ax2.plot(range(len(data2)), np.cumsum(data2))
#
#    plt.savefig(output)
#    plt.close()
#    print(output)
#
#
#def plot_dis(data1, data2, output):
#    import matplotlib
#    matplotlib.use('Agg')
#    import matplotlib.pyplot as plt
#    import matplotlib.dates
#
#    fig = plt.figure()
#    # a big subplot that is turned off axis lines and ticks
#    # only showing common labels
#    ax = fig.add_subplot(111)
#    ax.spines['top'].set_color('none')
#    ax.spines['bottom'].set_color('none')
#    ax.spines['left'].set_color('none')
#    ax.spines['right'].set_color('none')
#    ax.tick_params(labelcolor='w', top=None, bottom=None,
#                   left=None, right=None)
#    #ax.tick_params(labelcolor='w', top='off', bottom='off',
#    #               left='off', right='off')
#    ax.set_xlabel("Time")
#    ax.set_ylabel("Cumulative sum of time series")
#
#    ax1 = fig.add_subplot(211)
#    ax1.set_xlim(0, len(data1))
#    ax1.plot(range(len(data1)), np.cumsum(data1))
#    ax1.set_xlabel("Time")
#    ax1.set_ylabel("Cumulative sum of time series")
#    ax2 = fig.add_subplot(212)
#    ax2.set_xlim(0, len(data2))
#    ax2.plot(range(len(data2)), data2)
#    ax1.set_xlabel("Time")
#    ax1.set_ylabel("Value")
#
#    plt.savefig(output)
#    plt.close()
#    print(output)



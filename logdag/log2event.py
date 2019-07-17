#!/usr/bin/env python
# coding: utf-8

import logging
import pickle
from collections import namedtuple

from . import dtutil
from . import arguments
from amulog import config

_logger = logging.getLogger(__package__)
EvDef = namedtuple("EvDef", ["source", "host", "key"])


class EventDefinitionMap(object):
    """This class defines classified groups as "Event", and provide
    interconvirsion functions between Event IDs and their
    classifying criterions.

    The definition of Event is saved as a nametuple EvDef.
    Evdef has following attributes.
        source (str):
        key (int): 
        host (str):

    """
    l_attr = ["source", "host", "key"]

    def __init__(self):
        self._emap = {} # key : eid, val : evdef
        self._ermap = {} # key : evdef, val : eid

    def __len__(self):
        return len(self._emap)

    def eids(self):
        return self._emap.keys()

    def _next_eid(self):
        eid = len(self._emap)
        while eid in self._emap:
            eid += 1
        else:
            return eid

    @staticmethod
    def form_evdef(src, key, host):
        d = {"source": src,
             "host" : host,
             "key" : key}
        return EvDef(**d)

    def add_event(self, src, host, key):
        evdef = self.form_evdef(src, host, key)
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

    #def search_cond(self):
    #    return {key : getattr(self, key) for key in [self.gid_name, "host"]}
    #
    #def evdef_repr(self, ld, eid, dt_range, limit = 5):
    #    info = self._emap[eid]
    #    top_dt, end_dt = self.dt_range
    #    d = {"head" : limit, "foot" : limit,
    #         "top_dt" : top_dt, "end_dt" : end_dt}
    #    d[self.gid_name] = info.gid
    #    d["host"] = info.host
    #    return ld.show_log_repr(**d)

    def get_eid(self, info):
        return self._ermap[info]

    def iter_eid(self):
        return self._emap.keys()

    def iter_evdef(self):
        return self._ermap.keys()

    def dump(self, args):
        fp = arguments.ArgumentManager.evdef_filepath(args)
        obj = (self._emap, self._ermap)
        with open(fp, "wb") as f:
            pickle.dump(obj, f)

    def load(self, args):
        fp = arguments.ArgumentManager.evdef_filepath(args)
        with open(fp, "rb") as f:
            obj = pickle.load(f)
        self._emap, self._ermap = obj


class AreaTest():

    def __init__(self, conf):
        self._arearule = conf["dag"]["area"]
        self._areadict = config.GroupDef(conf["dag"]["area_def"])

        if self._arearule == "all":
            self._testfunc = self._test_all

    def _test_all(self, area, host):
        return True

    def _test_each(self, area, host):
        return area == host

    def _test_ingroup(self, area, host):
        return self._areadict.ingroup(area, host)

    def test(self, area, host):
        return self._testfunc(area, host)


def init_evgen(conf):
    if src == "log":
        from .source import evgen_log
        return evgen_log.LogEventLoader(conf)
    elif src == "snmp":
        from .source import evgen_snmp
        return evgen_snmp.SNMPEventLoader(conf)
    else:
        raise NotImplementedError


def _load_evgen_log(conf, dt_range, area, binarize):
    areatest = AreaTest(conf)
    method = conf.get("dag", "ci_bin_method")
    ci_bin_size = config.getdur(conf, "dag", "ci_bin_size")
    ci_bin_diff = config.getdur(conf, "dag", "ci_bin_diff")

    from .source import evgen_log
    evg = evgen_log.LogEventLoader(conf)
    for measure, host, gid in evg.all_condition(dt_range):
        if not areatest.test(area, host):
            continue

        if method == "sequential":
            df = evg.load(measure, host, gid, dt_range, ci_bin_size)
            if data is None:
                _logger.debug("{0} is empty".format((measure, host, gid)))
                continue
            if binarize:
                data[data > 0] = 1
        elif method == "slide":
            l_dt, l_array = zip(*evg.load_items(measure, host, gid, dt_range))
            data = dtutil.discretize_slide(l_dt, dt_range, ci_bin_diff,
                                           ci_bin_size, binarize,
                                           l_dt_values = l_array)
            df = pd.DataFrame(data, index = pd.to_datetime(l_dt))
        elif method == "radius":
            ci_bin_radius = 0.5 * ci_bin_size
            l_dt, l_array = zip(*evg.load_items(measure, host, gid, dt_range))
            data = dtutil.discretize_radius(l_dt, dt_range, ci_bin_diff,
                                            ci_bin_radius, binarize,
                                            l_dt_values = l_array)
        yield (host, gid, data)


def _load_evgen_snmp(conf, dt_range, area, binarize):
    areatest = AreaTest(conf)
    ci_bin_size = config.getdur(conf, "dag", "ci_bin_size")

    from .source import evgen_snmp
    evg = evgen_snmp.SNMPEventLoader(conf)
    for measure, host, key in evg.all_condition(dt_range):
        if not areatest.test(area, host):
            continue
        data = evg.load(measure, host, key, dt_range, ci_bin_size)
        if data is None:
            _logger.debug("{0} is empty".format((measure, host, key)))
            continue
        if binarize:
            data[data > 0] = 1
        name = "{0}_{1}".format(measure, key)
        yield (host, name, data)


def _load_evgen(src, conf, dt_range, area, binarize):
    if src == "log":
        return _load_evgen_log(conf, dt_range, area, binarize)
    elif src == "snmp":
        return _load_evgen_snmp(conf, dt_range, area, binarize)
    else:
        raise NotImplementedError


def makeinput(conf, dt_range, area, binarize):
    evmap = EventDefinitionMap()
    edict = {}
    sources = set(config.getlist(conf, "dag", "source"))
    for src in sources:
        for host, key, data in _load_evgen(src, conf, dt_range,
                                           area, binarize):
            eid = evmap.add_event(src, host, key)
            edict[eid] = data
            _logger.debug("loaded event {0} {1}".format(eid, evmap.evdef(eid)))
    return (edict, evmap)


#def ts2input(conf, dt_range, area, binarize):
#    from . import tsdb
#    gid_name = conf.get("dag", "event_gid")
#    method = conf.get("dag", "ci_bin_method")
#    ci_bin_size = config.getdur(conf, "dag", "ci_bin_size")
#    ci_bin_diff = config.getdur(conf, "dag", "ci_bin_diff")
#    td = tsdb.TimeSeriesDB(conf)
#    evmap = EventDefinitionMap(gid_name)
#
#    d_input = {}
#    kwargs = {"dts" : dt_range[0],
#              "dte" : dt_range[1],
#              "area" : area}
#    for gid, host in td.whole_gid_host(**kwargs):
#        _logger.debug("load event {0}".format((gid, host)))
#        ev_kwargs = {"dts" : dt_range[0],
#                     "dte" : dt_range[1],
#                     "gid" : gid,
#                     "host" : host}
#        l_dt = [dt for dt in td.iter_ts(**ev_kwargs)]
#        if len(l_dt) == 0:
#            _logger.warning("empty event {0}".format((gid, host)))
#            continue
#
#        if method == "sequential":
#            array = dtutil.discretize_sequential(l_dt, dt_range,
#                                                 ci_bin_size, binarize)
#        elif method == "slide":
#            array = dtutil.discretize_slide(l_dt, dt_range, ci_bin_diff,
#                                            ci_bin_size, binarize)
#        elif method == "radius":
#            ci_bin_radius = 0.5 * ci_bin_size
#            array = dtutil.discretize_radius(l_dt, dt_range, ci_bin_diff,
#                                             ci_bin_radius, binarize)
#
#        eid = evmap.add_event(gid, host)
#        d_input[eid] = array
#
#    return d_input, evmap


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



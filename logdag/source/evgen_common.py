#!/usr/bin/env python
# coding: utf-8


from amulog import config

source = ["log", "snmp"]


class EventLoader(object):
    fields = []

    def __init__(self, conf, dry=False):
        self.conf = conf
        self.dry = dry
        self.evdb = None

        raise NotImplementedError

    def all_condition(self, dt_range=None):
        for featurename in self.all_feature():
            measure = featurename
            if dt_range is None:
                l_d_tags = self.evdb.list_series(measure=measure)
            else:
                ut_range = tuple(dt.timestamp() for dt in dt_range)
                l_d_tags = self.evdb.list_series(measure=measure,
                                                 ut_range=ut_range)
            for d_tags in l_d_tags:
                yield (measure, d_tags["host"], int(d_tags["key"]))

    def load(self, measure, host, key, dt_range, binsize):
        d_tags = {"host": host, "key": key}
        ut_range = tuple(dt.timestamp() for dt in dt_range)
        str_bin = config.dur2str(binsize)
        return self.evdb.get_df(measure, d_tags, self.fields,
                                ut_range, str_bin, func="sum", fill=0)

    def load_items(self, measure, host, key, dt_range):
        d_tags = {"host": host, "key": key}
        ut_range = tuple(dt.timestamp() for dt in dt_range)
        return self.evdb.get_items(measure, d_tags, self.fields, ut_range)

    def load_all(self, dt_range, binsize):
        for measure, host, key in self.all_condition(dt_range):
            yield self.load(measure, host, key, dt_range, binsize)

    def drop_feature(self):
        for measure in self.all_feature():
            if not self.dry:
                self.evdb.drop_measure(measure)
            print(measure)

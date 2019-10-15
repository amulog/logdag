#!/usr/bin/env python
# coding: utf-8

import datetime
import pandas as pd

from amulog import config

source = ["log", "snmp"]


class EventLoader(object):
    fields = []

    def __init__(self, conf, dry=False):
        self.conf = conf
        self.dry = dry
        self.evdb = None

    #def all_condition(self):
    #    for featurename in self.all_feature():
    #        measure = featurename
    #        l_d_tags = self.evdb.list_series(measure=measure)
    #        for d_tags in l_d_tags:
    #            if "host" in d_tags and "key" in d_tags:
    #                yield (measure, d_tags["host"], d_tags["key"])

    def load(self, measure: str, tags: dict,
             dt_range: tuple, binsize: datetime.timedelta) -> pd.DataFrame:
        ut_range = tuple(dt.timestamp() for dt in dt_range)
        str_bin = config.dur2str(binsize)
        return self.evdb.get_df(measure, tags, self.fields,
                                ut_range, str_bin=str_bin, func="sum", fill=0)

    def load_orgdf(self, measure, tags, dt_range):
        ut_range = tuple(dt.timestamp() for dt in dt_range)
        return self.evdb.get_df(measure, tags, None, ut_range)

    def load_items(self, measure, tags, dt_range):
        ut_range = tuple(dt.timestamp() for dt in dt_range)
        return self.evdb.get_items(measure, tags, self.fields, ut_range)

    #def has_data(self, measure, host, key, dt_range):
    #    d_tags = {"host": host, "key": key}
    #    ut_range = tuple(dt.timestamp() for dt in dt_range)
    #    return self.evdb.has_data(measure, d_tags, self.fields, ut_range)

    def drop_features(self):
        for measure in self.all_feature():
            if not self.dry:
                self.evdb.drop_measure(measure)
            print(measure)

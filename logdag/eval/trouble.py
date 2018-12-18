#!/usr/bin/env python
# coding: utf-8

import json
from collections import defaultdict

from amulog import common

TR_ZEROS = 4
EMPTY_GROUP = "none"


class Trouble():

    def __init__(self, tid, **kwargs):
        self.tid = tid
        self.data = {}
        self.data["message"] = []
        if len(kwargs) > 0:
            self.set(**kwargs)

    def __str__(self):
        header = "Trouble {0}:".format(self.tid)
        for k in ("date", "group", "title"):
            if not k in self.data:
                mes = "empty"
                return " ".join((header, mes))
        length = len(self.data["message"])
        s = "{0[date]} ({0[group]}, {1}) {0[title]} ".format(self.data, length)
        return " ".join((header, s))

    def set(self, **kwargs):
        for k in kwargs:
            assert k in ("date", "group", "title", "message")
        self.data.update(kwargs)

    def add(self, lid):
        self.data["message"].append(lid)

    def get(self):
        return self.data["message"]

    def get_message(self, ld, show_lid = False):
        l_buf = []
        for lid in sorted(self.data["message"]):
            lm = ld.get_line(lid)
            if show_lid:
                buf = "{0} {1}".format(lm.lid, lm.restore_line())
            else:
                buf = lm.restore_line()
            l_buf.append(buf)
        return l_buf

    @staticmethod
    def _filepath(tid, dirname):
        return common.filepath(dirname, str(tid).zfill(TR_ZEROS))

    @classmethod
    def load(cls, tid, dirname):
        tr = Trouble(tid)
        fp = cls._filepath(tid, dirname)
        with open(fp, 'r', encoding='utf-8') as f:
            tr.data = json.load(f)
        return tr

    def dump(self, dirname):
        fp = self._filepath(self.tid, dirname)
        obj = self.data
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(obj, f, **common.json_args)


class TroubleManager():

    def __init__(self, dirname):
        self._dirname = dirname

    def __len__(self):
        return len(self._get_tids())

    def __iter__(self):
        return self._generator()

    def _generator(self):
        s = self._get_tids()
        for tid in s:
            yield Trouble.load(tid, self._dirname)

    def __getitem__(self, tid):
        assert isinstance(tid, int)
        try:
            return Trouble.load(tid, self._dirname)
        except IOError:
            raise KeyError

    def _get_tids(self):
        s = set()
        for fp in common.rep_dir(self._dirname):
            fn = common.filename(fp)
            tid = int(fn)
            s.add(tid)
        return s

    def next_tid(self):
        s_tid = self._get_tids()
        cnt = 0
        while cnt in s_tid:
            cnt += 1 
        else:
            return cnt

    def add(self, date, group, title, message = None):
        tid = self.next_tid()
        if not isinstance(date, str):
            from logdag import dtutil
            date = dtutil.shortstr(date)
        tr = Trouble(tid)
        tr.set(date = date, group = group, title = title)
        if message is not None:
            tr.set(message = message)
        tr.dump(self._dirname)
        return tr

    def add_lids(self, tid, l_lid):
        tr = Trouble.load(tid, self._dirname)
        for lid in l_lid:
            tr.add(lid)
        tr.dump(self._dirname)

    def update(self, tid, **kwargs):
        tr = Trouble.load(tid, self._dirname)
        tr.set(**kwargs)
        tr.dump(self._dirname)
        return tr


def event_stat(tr, ld, gid_name):
    d_gid = defaultdict(int)
    d_host = defaultdict(int)
    d_ev = defaultdict(int)
    for lid in tr.get():
        lm = ld.get_line(lid)
        gid = lm.lt.get(gid_name)
        host = lm.host
        d_gid[gid] += 1
        d_host[host] += 1
        d_ev[(gid, host)] += 1

    return d_ev, d_gid, d_host


def event_label(d_gid, ld, ll):
    d_group = defaultdict(list)
    for gid in d_gid.keys():
        label = ll.get_ltg_label(gid, ld.ltg_members(gid))
        group = ll.get_group(label)
        d_group[group].append(gid)
    return d_group



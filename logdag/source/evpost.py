#!/usr/bin/env python
# coding: utf-8


import numpy as np


def fillzero(sr):
    return sr.fillna(0)


def fillavg(sr):
    avg = np.nanmean(sr)
    return sr.fillna(avg)


def norm_fillavg(sr):
    avg = np.nanmean(sr)
    std = np.std(sr)
    if std == 0:
        ret = sr - avg
    else:
        ret = (sr - avg) / std
    return ret.fillna(0)


def root_square_diff(sr):
    ret = sr.diff()
    ret[0] = float(0)
    return ((ret ** 2) / sr) ** 0.5


def diff_abs(sr):
    ret = sr.diff()
    ret[0] = float(0)
    return np.abs(ret)


def getnan(sr):
    return sr.isnull() * 1


def outlier(sr, th=2.0):
    ret = sr * 0
    ret[sr > np.median(sr) + th] = 1
    return ret


def slice_index(sr, start, stop):
    return sr[start:stop]



#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd


def fillzero(sr, **kwargs):
    if len(sr.dropna()) == 0:
        return None
    else:
        return sr.fillna(0)


def fillavg(sr, **kwargs):
    if len(sr.dropna()) == 0:
        return None

    avg = np.nanmean(sr)
    return sr.fillna(avg)


def norm_fillavg(sr, **kwargs):
    if len(sr.dropna()) == 0:
        return None

    avg = np.nanmean(sr)
    std = np.std(sr)
    if std == 0:
        ret = sr - avg
    else:
        ret = (sr - avg) / std
    return ret.fillna(0)


def root_square_diff(sr, **kwargs):
    ret = sr.diff()
    ret[0] = float(0)
    return ((ret ** 2) / sr) ** 0.5


def diff_abs(sr, **kwargs):
    ret = sr.diff()
    ret[0] = float(0)
    return np.abs(ret)


def getnan(sr, **kwargs):
    return sr.isnull() * 1


def convolve(sr, convolve_radius=2, **kwargs):
    win = 2 * convolve_radius + 1
    a_win = np.ones(win) / win
    a_ret = np.convolve(sr, a_win, mode='same')
    return pd.Series(a_ret, index=sr.index, dtype=sr.dtype)


def outlier(sr, outlier_threshold=2.0, **kwargs):
    ret = sr * 0
    base = np.median(sr)
    ret[sr > base + outlier_threshold] = 1
    return ret


def outlier_median_absdev(sr, outlier_threshold=2.0, **kwargs):
    ret = sr * 0
    # median absolute deviation
    base = np.median(np.abs(sr - np.median(sr)))
    ret[sr > base + outlier_threshold] = 1
    return ret


def anomaly_lof(sr, **kwargs):
    from sklearn.neighbors import LocalOutlierFactor
    x = sr
    y = sr.diff()
    y[0] = float(0)
    data = pd.concat((x, y), axis=1)

    clf = LocalOutlierFactor(n_neighbors=20, contamination="auto")
    result = clf.fit_predict(data)
    anomaly = (result == -1) * 1.0
    return pd.Series(anomaly, index=sr.index)


def anomaly_if(sr, **kwargs):
    if len(sr[sr != float(0)]) == 0:
        # if all data is 0,
        # return 0 without processing Isolation Forest because
        # IsolationForest raise all data as anomaly in all zero case
        return sr

    from sklearn.ensemble import IsolationForest
    x = sr
    y = sr.diff()
    y[0] = float(0)
    data = pd.concat((x, y), axis=1)

    clf = IsolationForest(n_estimators=100, max_samples="auto",
                          contamination="auto", behaviour="new")
    result = clf.fit_predict(data)
    anomaly = (result == -1) * 1.0
    return pd.Series(anomaly, index=sr.index)


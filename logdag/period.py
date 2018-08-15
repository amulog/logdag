#!/usr/bin/env python
# coding: utf-8

import datetime
import logging
import numpy as np
import scipy.fftpack
import scipy.signal

round_int = lambda x: int(x + 0.5)
#_logger = logging.getLogger(__package__)


def fourier_remove(conf, array, binsize):
    th_spec = conf.getfloat("filter", "fourier_th_spec")
    th_std = conf.getfloat("filter", "fourier_th_eval")
    peak_order = conf.getfloat("filter", "fourier_peak_order")
    data = array
    #data = array[-power2(len(array)):]
    fdata = scipy.fftpack.fft(data)
    flag, interval = fourier_test_periodic(data, fdata, binsize,
                                           th_spec, th_std, peak_order)

    return flag, interval


def fourier_replace(conf, array, binsize):
    th_spec = conf.getfloat("filter", "fourier_th_spec")
    th_std = conf.getfloat("filter", "fourier_th_eval")
    th_restore = conf.getfloat("filter", "fourier_th_restore")
    peak_order = conf.getfloat("filter", "fourier_peak_order")
    #data = array[-power2(len(array)):]
    data = array
    fdata = scipy.fftpack.fft(data)
    flag, interval = fourier_test_periodic(data, fdata, binsize,
                                           th_spec, th_std, peak_order)
    if flag:
        data_filtered = part_filtered(data, fdata, binsize, th_spec)
        data_remain = restore_data(data, data_filtered, th_restore)
        return True, data_remain, interval
    else:
        return False, None, None


def fourier_test_periodic(data, fdata, binsize, th_spec, th_std, peak_order):
    #peak_order = 1
    peaks = 101

    dt = binsize.total_seconds()
    a_label = scipy.fftpack.fftfreq(len(data), d = dt)[1:int(0.5 * len(data))]
    a_spec = np.abs(fdata)[1:int(0.5 * len(data))]
    max_spec = max(a_spec)
    a_peak = scipy.signal.argrelmax(a_spec, order = peak_order)

    l_interval = []
    prev_freq = 0.0
    for freq, spec in (np.array([a_label, a_spec]).T)[a_peak]:
        if spec > th_spec * max_spec:
            interval = freq - prev_freq
            l_interval.append(interval)
            prev_freq = freq
        else:
            pass
    if len(l_interval) == 0:
        return False, None

    dist = np.array(l_interval[:(peaks - 1)])
    std = np.std(dist)
    mean = np.mean(dist)
    val = 1.0 * std / mean
    interval = round_int(1.0 / np.median(dist)) * datetime.timedelta(
            seconds = 1)
    return val < th_std, interval


def part_filtered(data, fdata, binsize, th_spec):
    dt = binsize.total_seconds()
    a_label = scipy.fftpack.fftfreq(len(data), d = dt)
    a_spec = np.abs(fdata)
    max_spec = max(a_spec)
    
    fdata[a_spec <= th_spec * max_spec] = np.complex(0)
    data_filtered = np.real(scipy.fftpack.ifft(fdata))
    return data_filtered


def restore_data(data, data_filtered, th_restore):
    thval = th_restore * max(data_filtered)

    periodic_time = (data > 0) & (data_filtered >= thval) # bool
    periodic_cnt = np.median(data[periodic_time])
    data_periodic = np.zeros(len(data))
    data_periodic[periodic_time] = periodic_cnt
    data_remain = data - data_periodic

    return data_remain


def power2(length):
    return 2 ** int(np.log2(length))


def power2ceil(length):
    return 2 ** math.ceil(np.log2(length))


def periodic_corr(conf, array, binsize):
    corr_th = conf.getfloat("filter", "corr_th")
    corr_diff = [config.str2dur(diffstr) for diffstr
            in conf.gettuple("filter", "corr_diff")]

    l_result = []
    for diff in corr_diff:
        c = self_corr(array, diff, binsize)
        l_result.append([c, diff])
    max_c, max_diff = max(l_result, key = lambda x: x[0])

    if max_c >= corr_th:
        return True, max_diff
    else:
        return False, None


def self_corr(data, diff, binsize):
    """
    Args:
        data (numpy.array)
        diff (datetime.timedelta)
        binsize (datetime.timedelta)

    Returns:
        float: Self-correlation coefficient with given lag.
    """
    diff_bin = int(diff.total_seconds() / binsize.total_seconds())
    if len(data) <= binnum * 2:
        return 0.0
    else:
        data1 = data[:len(data) - binnum]
        data2 = data[binnum:]
        assert len(data1) == len(data2)
        return np.corrcoef(data1, data2)[0, 1]



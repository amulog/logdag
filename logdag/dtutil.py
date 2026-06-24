#!/usr/bin/env python
# coding: utf-8

import datetime
import logging
import random
import numpy as np
import dateutil
from collections import defaultdict

# from itertools import chain

TIMEFMT = "%Y-%m-%d %H:%M:%S"
_logger = logging.getLogger(__package__)


def empty_timedelta():
    return datetime.timedelta(seconds=0)


def extdate(dt):
    """Return datetime representation of date component in given datetime."""
    ret = datetime.datetime.combine(dt.date(), datetime.time())
    return ret.replace(tzinfo=dt.tzinfo)


def is_intdate(arg):
    """Return True if the argument is an integer multiple of 1 day.
    Argument is acceptable for datetime.datetime and datetime.timedelta."""
    if isinstance(arg, datetime.datetime):
        return extdate(arg) == arg
    elif isinstance(arg, datetime.timedelta):
        return datetime.timedelta(days=arg.days) == arg
    else:
        raise NotImplementedError


def range_dt(dts, dte, interval):
    """
    Args:
        dts (datetime.datetime): start time
        dte (datetime.datetime): end time
        interval (datetime.timedelta): size of time bins

    Returns:
        list of datetime.datetime
    """

    tzinfo = dts.tzinfo
    tmp = np.arange(dts.timestamp(), dte.timestamp(), interval.total_seconds())
    # fromtimestamp(ut, tz=...) converts the epoch value to the given tz;
    # fromtimestamp(ut) alone interprets it in the local tz, then replace()
    # would mislabel the wall clock when tzinfo is not the local zone.
    return [datetime.datetime.fromtimestamp(ut, tz=tzinfo) for ut in tmp]

    #temp_dt = dt_range[0]
    #while temp_dt < dt_range[1] or (include_end is True and temp_dt == dt_range[1]):
    #    yield temp_dt
    #    temp_dt = temp_dt + interval


# def dtrange_term(top_dt, end_dt, duration):
#    temp_top_dt = top_dt
#    while temp_top_dt < end_dt:
#        temp_end_dt = temp_top_dt + duration
#        yield (temp_top_dt, temp_end_dt)
#        temp_top_dt = temp_end_dt
# -> iter_term


def discretize(l_dt, l_term, dt_range, binarize, l_dt_values=None):
    """Convert list of datetime into numpy array.
    This function use mapping algorithm: split datetime space by change points
    (i.e., all ends of datetime terms) and iterate l_dt only once
    with comparing to the change points.
    Args:
        l_dt (List[datetime.datetime]): An input datetime sequence.
        l_term (List[(datetime.datetime, datetime.datetime)]):
                A sequence of datetime ranges. Each ranges are
                corresponding to returning array index.
        dt_range(Tuple[datetime.datetime, datetime.datetime]):
                dt (in l_dt) out of dt_range is ignored even if
                it is included in any terms.
        binarize (bool): If True, return 0 or 1 for each bin. 1 means
                some datetime found in l_dt.
        l_dt_values (List[np.array], optional):
                Values to be aggregated, corresponding to l_dt.
                If None, this function returns timestamp count for each bins.

    Returns:
        np.array
    """

    if l_dt_values is None:
        l_dt_values = np.array([1] * len(l_dt))
        type_ret = int
    else:
        type_ret = type(l_dt_values[0])
    top_dt, end_dt = dt_range
    a_ret = np.zeros(len(l_term), dtype=type_ret)

    # extract change points
    d_cp = defaultdict(list)
    # tests top_dt
    for idx, term in enumerate(l_term):
        if term[0] <= top_dt < term[1]:
            d_cp[top_dt].append((idx, True))
    # tests both ends of terms
    for idx, term in enumerate(l_term):
        if term[0] > top_dt:
            d_cp[term[0]].append((idx, True))
        if end_dt >= term[1]:
            d_cp[term[1]].append((idx, False))
    # tests end_dt
    for idx, term in enumerate(l_term):
        if term[0] <= end_dt < term[1]:
            d_cp[end_dt].append((idx, False))

    # generate mapped change points
    l_cp = []
    tmp_idxs = set()
    for dt, changes in sorted(d_cp.items(), key=lambda x: x[0]):
        for idx, flag in changes:
            if flag:
                tmp_idxs.add(idx)
            else:
                tmp_idxs.remove(idx)
        l_cp.append((dt, np.array(tuple(tmp_idxs))))
    assert len(tmp_idxs) == 0

    # note: iteration does not use last component (uniquely used afterward)
    iterobj = zip(l_cp[:-1], l_cp[1:])
    try:
        (key, current_idxs), (next_key, next_idxs) = next(iterobj)
    except StopIteration:
        # changes are empty
        return a_ret

    for dt, v in zip(l_dt, l_dt_values):
        if not dt_range[0] <= dt < dt_range[1]:
            # out of given range, ignored
            continue
        # pass iteration to next matching bin
        assert dt >= key
        if next_key is not None:
            while dt >= next_key:
                try:
                    (key, current_idxs), (next_key, next_idxs) = next(iterobj)
                except StopIteration:
                    # not iterate after here and use last component
                    key, current_idxs = l_cp[-1]
                    next_key = None
                    break
        # following is processed only if key <= dt < next_key
        # (len, not sum: current_idxs holds bin indices, so sum() wrongly
        # skips the case where the only active bin is index 0)
        if len(current_idxs) > 0:
            if binarize:
                a_ret[current_idxs] = 1
            else:
                a_ret[current_idxs] += v

    return a_ret


def discretize_sequential(l_dt, dt_range, binsize,
                          binarize=False, l_dt_values=None):
    l_term = []
    top_dt, end_dt = dt_range
    temp_dt = top_dt
    while temp_dt < end_dt:
        l_term.append((temp_dt, temp_dt + binsize))
        temp_dt += binsize

    return discretize(l_dt, l_term, dt_range, binarize,
                      l_dt_values=l_dt_values)


def discretize_slide(l_dt, dt_range, bin_slide, binsize,
                     binarize=False, l_dt_values=None):
    l_term = []
    top_dt, end_dt = dt_range
    temp_dt = top_dt
    while temp_dt < end_dt:
        l_term.append((temp_dt, temp_dt + binsize))
        temp_dt += bin_slide

    return discretize(l_dt, l_term, dt_range, binarize,
                      l_dt_values=l_dt_values)


def discretize_radius(l_dt, dt_range, bin_slide, bin_radius,
                      binarize=False, l_dt_values=None):
    l_label = []
    top_dt, end_dt = dt_range
    temp_dt = top_dt + 0.5 * bin_slide
    while temp_dt < end_dt:
        l_label.append(temp_dt)
        temp_dt += bin_slide
    l_term = [(dt - bin_radius, dt + bin_radius) for dt in l_label]

    return discretize(l_dt, l_term, dt_range, binarize,
                      l_dt_values=l_dt_values)


# old
# def discretize(l_dt, l_label, method = "count", binarize = False):
#    """
#    Args:
#        l_dt (List[datetime.datetime]): An input datetime sequence.
#        l_label (List[datetime.datetime]): A sequence of separating times
#                of data bins. The number of labels is equal to
#                number of bins + 1. (Including the end of data term)
#        method (str): Returned data style. "count" returns the number of
#                object in each bin. "binary" returns 0 or 1 for each bin.
#                (1 means some object is in the bin.) "datetime" returns
#                the list of datetime object in each bin.
#        binarize (bool): If True, return 0 or 1 for each bin. 1 means
#                some datetime found in l_dt.
#                This is argument only for comparibility.
#                Use "method" instead of this argument.
#    """
#
#    def return_empty(size, method):
#        if method in ("count", "binary"):
#            return [0] * bin_num
#        elif method == "datetime":
#            return [[] for i in range(bin_num)]
#        else:
#            raise NotImplementedError(
#                "Invalid method name ({0})".format(method))
#
#    def init_tempobj(method):
#        if method == "count":
#            return 0
#        elif method == "binary":
#            return 0
#        elif method == "datetime":
#            return []
#        else:
#            raise NotImplementedError(
#                "Invalid method name ({0})".format(method))
#
#    def update_tempobj(temp, new_dt, method):
#        if method == "count":
#            return temp + 1
#        elif method == "binary":
#            return 1
#        elif method == "datetime":
#            temp.append(new_dt)
#            return temp
#        else:
#            raise NotImplementedError(
#                "Invalid method name ({0})".format(method))
#
#    if binarize:
#        method = "binary"
#
#    bin_num = len(l_label) - 1
#    l_dt_temp = sorted(l_dt)
#    if len(l_dt_temp) <= 0:
#        return_empty(bin_num, method)
#
#    iterobj = iter(l_dt_temp)
#    try:
#        new_dt = next(iterobj)
#    except StopIteration:
#        raise ValueError("Not empty list, but failed to get initial value")
#    while new_dt < l_label[0]:
#        try:
#            new_dt = next(iterobj)
#        except StopIteration:
#            return_empty(bin_num, method)
#
#    ret = []
#    stop = False
#    for label_dt in l_label[1:]:
#        temp = init_tempobj(method)
#        if stop:
#            ret.append(temp)
#            continue
#        while new_dt < label_dt:
#            temp = update_tempobj(temp, new_dt, method)
#            try:
#                new_dt = next(iterobj)
#            except StopIteration:
#                # "stop" make data after label term be ignored
#                stop = True
#                break
#        ret.append(temp)
#    return ret


# def auto_discretize(l_dt, binsize, dt_range = None, binarize = False):
#    """
#    Args:
#        l_dt (List[datetime.datetime])
#        binsize (datetime.timedelta)
#    """
#    if binsize == datetime.timedelta(seconds = 1):
#        return l_dt
#    else:
#        if dt_range is None:
#            top_dt = adj_sep(min(l_dt), binsize)
#            end_dt = radj_sep(max(l_dt), binsize)
#        else:
#            top_dt, end_dt = dt_range
#        l_label = label((top_dt, end_dt), binsize)
#        return discretize(l_dt, l_label, binarize = binarize)
#
#
# def auto_discretize_slide(l_dt, binsize, slide,
#                          dt_range = None, method = "count", binarize = False):
#    #assert slide <= binsize
#    if dt_range is None:
#        top_dt = adj_sep(min(l_dt), binsize)
#        end_dt = radj_sep(max(l_dt), binsize)
#    else:
#        top_dt, end_dt = dt_range
#    if binarize:
#        method = "binary"
#
#    if binsize < slide:
#        _logger.warning("binsize is smaller than slide, "
#                        "which means there is time-series sampling gap")
#    slide_width = max(int(binsize.total_seconds() / slide.total_seconds()), 1)
#    l_top = label((top_dt, end_dt), slide)[:-1]
#    l_end = [min(t + binsize, end_dt) for t in l_top]
#
#    ret = []
#    noslide = discretize(l_dt, l_top + [end_dt], method = "datetime")
#
#    for i, bin_end in enumerate(l_end):
#        #slide_area = chain.from_iterable(noslide[i:i+slide_width])
#        l_dt_temp = []
#        for b in noslide[i:i+slide_width]:
#            l_dt_temp.extend([dt for dt in b if dt <= bin_end])
#        
#        if method == "count":
#            ret.append(len(l_dt_temp))
#        elif method == "binary":
#            if len(l_dt_temp) > 0:
#                ret.append(1)
#            else:
#                ret.append(0)
#        else:
#            raise NotImplementedError(
#                "Invalid method name ({0})".format(method))
#
#    return ret


# def periodic(dt_range, interval):
#    top_dt, end_dt = dt_range
#    l_label = []
#    #temp_dt = top_dt + duration
#    temp_dt = top_dt
#    while temp_dt < end_dt:
#        l_label.append(temp_dt)
#        temp_dt += interval
#    l_label.append(end_dt)
#    return l_label


def shortstr(dt):
    date = datetime.datetime.combine(dt.date(), datetime.time(),
                                     tzinfo=dt.tzinfo)
    if date == dt:
        return dt.strftime("%Y%m%d")
    else:
        return dt.strftime("%Y%m%d_%H%M%S")


def shortstr2dt(dtstr):
    if "_" in dtstr:
        dt = datetime.datetime.strptime(dtstr, "%Y%m%d_%H%M%S")
    else:
        dt = datetime.datetime.strptime(dtstr, "%Y%m%d")
    return dt.replace(tzinfo=dateutil.tz.tzlocal())


def iter_term(whole_term, term_length, term_diff=None):
    # whole_term : tuple(datetime.datetime, datetime.datetime)
    # term_length : datetime.timedelta
    # term_diff : datetime.timedelta
    if term_diff is None:
        term_diff = term_length
    w_dts, w_dte = whole_term
    dts = w_dts
    while dts < w_dte:
        dte = dts + term_length
        yield (dts, dte)
        dts = dts + term_diff

#!/usr/bin/env python
# coding: utf-8

import numpy as np
from collections import defaultdict


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
    top_dt, end_dt = dt_range
    a_ret = np.zeros(len(l_term), dtype=int)

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
    temp_idxs = set()
    for dt, changes in sorted(d_cp.items(), key=lambda x: x[0]):
        for idx, flag in changes:
            if flag:
                temp_idxs.add(idx)
            else:
                temp_idxs.remove(idx)
        l_cp.append((dt, tuple(temp_idxs)))
    assert len(temp_idxs) == 0

    # iteration does not use last component (uniquely used afterward)
    iterobj = zip(l_cp[:-1], l_cp[1:])
    try:
        (key, l_rid), (next_key, next_l_rid) = next(iterobj)
    except StopIteration:
        return a_ret

    for dt, v in zip(l_dt, l_dt_values):
        if not dt_range[0] <= dt < dt_range[1]:
            # ignored
            continue
        assert dt >= key
        if next_key is not None:
            while dt >= next_key:
                try:
                    (key, l_rid), (next_key, next_l_rid) = next(iterobj)
                except StopIteration:
                    # not iterate after here and use last component
                    key, l_rid = l_cp[-1]
                    next_key = None
                    break
        # following processed only if key <= dt < next_key
        if binarize:
            a_ret[np.array(l_rid)] = 1
        else:
            a_ret[np.array(l_rid)] += v

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


import numpy as np
import pandas as pd
from dateutil import tz

from .. import dtutil


def pdtimestamp_naive(input_dt):
    """Generate pandas timestamp that is timezone-naive."""
    if isinstance(input_dt, pd.Timestamp):
        dt = input_dt
    else:
        dt = pd.Timestamp(input_dt)

    # If input_dt is timezone-aware,
    # convert it into naive(utc) for database input
    if dt.tz is not None:
        dt = dt.tz_convert(None)
        dt = dt.tz_localize(None)
    return dt


def pdtimestamp(input_dt):
    """Generate pandas timestamp that is timezone-aware."""
    if isinstance(input_dt, pd.Timestamp):
        dt = input_dt
    else:
        dt = pd.Timestamp(input_dt)

    # If input_dt is timezone-naive,
    # convert it into timezone-aware(local) for logdag use
    if dt.tz is None:
        dt = dt.tz_localize(tz.tzutc())
        dt = dt.tz_convert(tz.tzlocal())
    return dt


def pdtimestamps(input_dts):
    """Generate series of timestamp that is timezone-aware."""
    dtindex = pd.to_datetime(input_dts)

    # If input_dt is timezone-naive,
    # convert it into timezone-aware(local) for logdag use
    if dtindex.tz is None:
        dtindex = dtindex.tz_localize(tz.tzutc())
        dtindex = dtindex.tz_convert(tz.tzlocal())
    return dtindex


def timestamps2df(l_dt, l_values, fields, dt_range, binsize):
    dtindex = pdtimestamps(dtutil.range_dt(dt_range[0], dt_range[1], binsize))
    sortidx = np.argsort(l_dt)
    sorted_l_dt = [l_dt[idx] for idx in sortidx]
    sorted_l_values = [l_values[idx] for idx in sortidx]

    d_values = {}
    if len(l_dt) == 0:
        for field in fields:
            d_values[field] = [float(0)] * len(dtindex)
    else:
        for fid, series in enumerate(zip(*sorted_l_values)):
            a_cnt = dtutil.discretize_sequential(sorted_l_dt, dt_range,
                                                 binsize, l_dt_values=series)
            d_values[fields[fid]] = a_cnt

    return pd.DataFrame(d_values, index=dtindex)



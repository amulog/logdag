#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np

from amulog import config
from logdag import tsdb
from logdag import dtutil


def load_values(conf, gid, host, dt_range):
    method = conf.get("dag", "ci_bin_method")
    ci_bin_size = config.getdur(conf, "dag", "ci_bin_size")
    ci_bin_diff = config.getdur(conf, "dag", "ci_bin_diff")
    td = tsdb.TimeSeriesDB(conf)

    kwargs = {"dts": dt_range[0],
              "dte": dt_range[1],
              "gid": gid,
              "host": host}
    l_dt = [dt for dt in td.iter_ts(**kwargs)]

    if method == "sequential":
        array = dtutil.discretize_sequential(l_dt, dt_range,
                                             ci_bin_size, False)
    elif method == "slide":
        array = dtutil.discretize_slide(l_dt, dt_range, ci_bin_diff,
                                        ci_bin_size, False)
    elif method == "radius":
        ci_bin_radius = 0.5 * ci_bin_size
        array = dtutil.discretize_radius(l_dt, dt_range, ci_bin_diff,
                                         ci_bin_radius, False)
    return array


def show_dist_values(conf, gid, host, dt_range):
    array = load_values(conf, gid, host, dt_range)

    if sum(array) == 0:
        return "No log message (or all filtered) in given condition"

    bins = np.arange(0, max(array) + 1)
    hist = np.histogram(array, bins = bins, density = True)

    l_buf = []
    for cnt, v in zip(*hist):
        l_buf.append("{0} {1}".format(v, cnt))
    return "\n".join(l_buf)


def plot_dist_values(conf, gid, host, dt_range, output = "hoge.pdf"):
    array = load_values(conf, gid, host, dt_range)

    if sum(array) == 0:
        return "No log message (or all filtered) in given condition"

    bins = np.arange(0, max(array) + 1)
    hist = np.histogram(array, bins = bins, density = True)

    y, x = hist
    cum = np.cumsum(y)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Value")
    ax.set_ylabel("Appearance rate")
    ax.set_xticks(x)
    ax.bar(x[:-1], y)
    ax.plot(x[:-1], cum)

    plt.savefig(output)
    plt.close()
    print(output)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit("usage: {0} MODE CONF GID HOST DTS DTE").format(sys.argv[0])

    from logdag import arguments
    conf = config.open_config(sys.argv[1],
                              ex_defaults = [arguments.DEFAULT_CONFIG])
    mode = sys.argv[2]
    gid = int(sys.argv[3])
    host = sys.argv[4]
    dts = dtutil.shortstr2dt(sys.argv[5])
    dte = dtutil.shortstr2dt(sys.argv[6])

    if mode == "show":
        print(show_dist_values(conf, gid, host, (dts, dte)))
    elif mode == "plot":
        plot_dist_values(conf, gid, host, (dts, dte))
    else:
        raise NotImplementedError



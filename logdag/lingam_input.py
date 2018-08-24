#!/usr/bin/env python
# coding: utf-8

"""
LiNGAM option is under construction.
Also, lingam_fast library is officially not available in PyPI.
Use https://github.com/cpflat/LiNGAM-fast instead.
"""

import pandas as pd

from lingam_fast import lingam_fast


def estimate(data):
    # avoid recursion depth error
    import sys
    sys.setrecursionlimit(1000000)

    matrix = pd.DataFrame(data)
    lingam = lingam_fast.LiNGAM()
    ret = lingam.fit(matrix, use_sklearn = True, algorithm="fast",
                     reg_type = "lasso")
    graph = lingam.visualize(lib = "networkx")
    return graph


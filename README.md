# logdag

## Overview

This package generates causal DAGs among time-series events in syslog data.
This package works on python3.
The input log data is loaded with AMULOG (https://github.com/cpflat/amulog).
The output DAG is recorded in the format of NetworkX DiGraph.

This project was partially ported from repository LogCausalAnaysis.
(https://github.com/cpflat/LogCausalAnalysis)

## Package requirements

* amulog https://github.com/cpflat/amulog
* pcalg https://github.com/keiichishima/pcalg
* gsq https://github.com/keiichishima/gsq
* NetworkX https://networkx.github.io

## Related article

http://ieeexplore.ieee.org/document/8122062/

## Author

Satoru Kobayashi



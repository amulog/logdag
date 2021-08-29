######
logdag
######

Overview
========

This package generates causal DAGs among time-series events in syslog data.
This package works on python3.
The input log data is loaded with AMULOG (https://github.com/cpflat/amulog).
The output DAG is recorded in the format of NetworkX DiGraph.

This project was partially forked from repository LogCausalAnaysis.
(https://github.com/cpflat/LogCausalAnalysis)


Usage
======

All features are available from command line. First you should try following help command :code:`python -m logdag -h`.

short usage:

1. Generate amulog database (and its config) as the input log time-series source
2. Prepare logdag config file by referring :code:`logdag/data/config.conf.default`
3. Generate time-series db by :code:`logdag.source` features
4. Generate DAGs by subcommand :code:`makedag`
5. See results by commands such as :code:`show-edge-list`


Reference
=========

This project is evaluated in some papers `CNSM2019 <https://doi.org/10.23919/CNSM46954.2019.9012718>`_ and `TNSM2018 <https://doi.org/10.1109/TNSM.2017.2778096>`_.
If you use this code, please consider citing:

::

    @inproceedings{Kobayashi_CNSM2019,
      author = {Kobayashi, Satoru and Otomo, Kazuki and Fukuda, Kensuke},
      booktitle = {Proceedings of the 15th International Conference on Network and Service Management (CNSM'20)},
      title = {Causal analysis of network logs with layered protocols and topology knowledge},
      pages = {1-9},
      year = {2019}
    }


    @article{Kobayashi_TNSM2018,
      author = {Kobayashi, Satoru and Otomo, Kazuki and Fukuda, Kensuke and Esaki, Hiroshi},
      journal = {IEEE Transactions on Network and Service Management},
      volume = {15},
      number = {1},
      pages = {53-67},
      title = {Mining causes of network events in log data with causal inference},
      year = {2018}
    }


License
=======

3-Clause BSD license

Author
======

Satoru Kobayashi


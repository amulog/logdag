#################################
CNSM2019 reproduction with logdag
#################################

Overview
========

This document is for researchers who want to follow-up `CNSM2019 <https://doi.org/10.23919/CNSM46954.2019.9012718>`_ paper in their data.

::

    @inproceedings{Kobayashi_CNSM2019,
      author = {Kobayashi, Satoru and Otomo, Kazuki and Fukuda, Kensuke},
      booktitle = {Proceedings of the 15th International Conference on Network and Service Management (CNSM'20)},
      title = {Causal analysis of network logs with layered protocols and topology knowledge},
      pages = {1-9},
      year = {2019}
    }

- This document is written for :code:`amulog=v0.3.1`, :code:`log2seq=v0.2.3`, and :code:`logdag=v0.0.6`.

- Use sqlite instead of mysql, as mysql support is not stable currently (both in amulog and logdag).


Make amulog DB
==============

First, you need to prepare :code:`amulog.conf`, the amulog configuration file.
You may also need some additional files such as for log2seq if required.
For datails, see `amulog readme <https://github.com/amulog/amulog/blob/master/README.rst>`_ and `log2seq document <https://log2seq.readthedocs.io/en/latest/>`_.


Add classification labels for log templates
===========================================

You need to define classification labels (e.g., System and Rt-EGP in our paper) for the generated log templates in amulog DB.
Currently amulog generate labels based on regular-expression-based definition file.

There is `a sample file <https://github.com/amulog/amulog/blob/master/amulog/data/lt_label.conf.sample>`_ in amulog.
In this sample, 2 classes :code:`mgmt` and :code:`l7` are defined.
If a template matches one of following rules, the template is classified to the class.

After writing the definition file, add following lines to :code:`amulog.conf`.
In this case, templates that do not match any rule is classified into the :code:`system` class.

::

    [visual]
    tag_method = file
    tag_file = lt_label.txt
    tag_file_key = group
    tag_file_default_label = None
    tag_file_default_group = system

Then try :code:`python -m amulog db-tag -c amulog.conf`.

You can check the classification results by amulog commands like :code:`python -m amulog show-lt -c amulog.conf`.


Generate time-series data
=========================

In this directory, there are multiple configuration files for logdag.
:code:`logdag_base.conf` is the basic configuration.

Before causal analysis, you need to generate time-series DB from amulog DB.
Try :code:`python -m logdag.source make-evdb-log-all -c logdag_base.conf`.


Generate DAGs
=============

Try :code:`python -m logdag make-dag -c logdag_base.conf`.
This will generate DAGs with no prior knowledge.
If you change the :code:`logdag_base.conf` in the command to others,
you will obtain other results.

The proposed method in CNSM2019 corresponds to :code:`logdag_proposed.conf`.
In this method, you also need network topology definition file (:code:`l2.conf` and :code:`l3.conf` in the config file).
The file is networkx Graph object in json format (use :code:`networkx.node_link_graph()` function).

For :code:`logdag_area.conf`, you also need area definition file.
There is `a sample file <https://github.com/amulog/logdag/blob/master/logdag/data/area_def.txt.sample>`_ in logdag repository.

After DAG generation, you can check the results with some logdag commands.
See logdag help by :code:`python -m logdag -h`.

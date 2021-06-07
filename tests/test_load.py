#!/usr/bin/env python
# coding: utf-8

import os
import unittest
import tempfile

from amulog import config
from amulog import testutil

from logdag import arguments


class TestLoader(unittest.TestCase):

    _path_testlog = None
    _path_amulogdb = None
    _path_ltgendump = None
    _path_testdb = None
    _path_amulogconf = None

    @classmethod
    def setUpClass(cls):
        fd_testlog, cls._path_testlog = tempfile.mkstemp()
        os.close(fd_testlog)
        fd_amulogdb, cls._path_amulogdb = tempfile.mkstemp()
        os.close(fd_amulogdb)
        fd_ltgendump, cls._path_ltgendump = tempfile.mkstemp()
        os.close(fd_ltgendump)
        fd_testdb, cls._path_testdb = tempfile.mkstemp()
        os.close(fd_testdb)

        cls._amulog_conf = config.open_config()
        cls._amulog_conf['general']['src_path'] = cls._path_testlog
        cls._amulog_conf['database']['sqlite3_filename'] = cls._path_amulogdb
        cls._amulog_conf['manager']['indata_filename'] = cls._path_ltgendump
        fd_amulogconf, cls._path_amulogconf = tempfile.mkstemp()
        f = os.fdopen(fd_amulogconf, "w")
        cls._amulog_conf.write(f)
        f.close()

        tlg = testutil.TestLogGenerator(testutil.DEFAULT_CONFIG, seed=3)
        tlg.dump_log(cls._path_testlog)
        cls._whole_term = tlg.term

        from amulog import __main__ as amulog_main
        from amulog import manager
        targets = amulog_main.get_targets_conf(cls._amulog_conf)
        manager.process_files_online(cls._amulog_conf, targets, reset_db=True)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls._path_testlog)
        os.remove(cls._path_amulogdb)
        os.remove(cls._path_ltgendump)
        os.remove(cls._path_testdb)
        os.remove(cls._path_amulogconf)

    def test_load_asis(self):
        conf = config.open_config(arguments.DEFAULT_CONFIG,
                                  base_default=False)
        conf["general"]["evdb"] = "sql"
        conf["database_sql"]["database"] = "sqlite3"
        conf["database_amulog"]["source_conf"] = self._path_amulogconf
        conf["database_sql"]["sqlite3_filename"] = self._path_testdb

        conf["filter"]["rules"] = ""

        from logdag import dtutil
        from logdag.source import evgen_log
        w_term = self._whole_term
        size = config.str2dur("1d")
        el = evgen_log.LogEventLoader(conf)
        for dt_range in dtutil.iter_term(w_term, size):
            el.read(dt_range, dump_org=False)

        am = arguments.ArgumentManager(conf)
        am.generate(arguments.all_args)

        from logdag import makedag
        edge_cnt = 0
        for args in am:
            ldag = makedag.makedag_main(args, do_dump=False)
            edge_cnt += ldag.number_of_edges()
        assert edge_cnt > 0

    def test_load_filter(self):
        conf = config.open_config(arguments.DEFAULT_CONFIG,
                                  base_default=False)
        conf["general"]["evdb"] = "sql"
        conf["database_sql"]["database"] = "sqlite3"
        conf["database_amulog"]["source_conf"] = self._path_amulogconf
        conf["database_sql"]["sqlite3_filename"] = self._path_testdb

        from logdag import dtutil
        from logdag.source import evgen_log
        w_term = self._whole_term
        size = config.str2dur("1d")
        el = evgen_log.LogEventLoader(conf)
        for dt_range in dtutil.iter_term(w_term, size):
            el.read(dt_range, dump_org=False)

        am = arguments.ArgumentManager(conf)
        am.generate(arguments.all_args)

        from logdag import makedag
        edge_cnt = 0
        for args in am:
            ldag = makedag.makedag_main(args, do_dump=False)
            edge_cnt += ldag.number_of_edges()
        assert edge_cnt > 0

    def test_load_slide(self):
        conf = config.open_config(arguments.DEFAULT_CONFIG,
                                  base_default=False)
        conf["general"]["evdb"] = "sql"
        conf["database_sql"]["database"] = "sqlite3"
        conf["database_amulog"]["source_conf"] = self._path_amulogconf
        conf["database_sql"]["sqlite3_filename"] = self._path_testdb
        conf["dag"]["ci_bin_method"] = "slide"
        conf["dag"]["ci_bin_size"] = "1m"
        conf["dag"]["ci_bin_diff"] = "5m"

        from logdag import dtutil
        from logdag.source import evgen_log
        w_term = self._whole_term
        size = config.str2dur("1d")
        el = evgen_log.LogEventLoader(conf)
        for dt_range in dtutil.iter_term(w_term, size):
            el.read(dt_range, dump_org=False)

        am = arguments.ArgumentManager(conf)
        am.generate(arguments.all_args)

        from logdag import makedag
        edge_cnt = 0
        for args in am:
            ldag = makedag.makedag_main(args, do_dump=False)
            edge_cnt += ldag.number_of_edges()
        assert edge_cnt > 0


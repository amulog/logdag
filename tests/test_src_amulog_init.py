#!/usr/bin/env python
# coding: utf-8

"""Regression for src_amulog.init_amulogloader argument wiring.

The old code passed dt_range and the source_conf *path* in the wrong
constructor positions (conf <- dt_range, dt_range <- path string), so the
loader built by eval/match_edge.py was misconfigured. This pins the fix:
source_conf is opened via config.open_config and arguments land in the right
slots. Stubs avoid touching a real amulog DB.
"""

import configparser
import unittest

from amulog import config
from logdag.source import src_amulog


class TestInitAmulogLoader(unittest.TestCase):

    def test_argument_wiring(self):
        conf = configparser.ConfigParser()
        conf["database_amulog"] = {
            "source_conf": "amulog.conf",
            "event_gid": "ltgid",
            "use_anonymize_mapping": "false",
            "host_tier": "",
        }

        captured = {}

        class _FakeLoader:
            def __init__(self, conf, dt_range=None, gid_name="ltid",
                         use_mapping=False, ld=None, host_tier=None):
                captured.update(conf=conf, dt_range=dt_range,
                                gid_name=gid_name, use_mapping=use_mapping,
                                host_tier=host_tier)

        orig_loader = src_amulog.AmulogLoader
        orig_open = config.open_config
        src_amulog.AmulogLoader = _FakeLoader
        config.open_config = lambda path: ("AMULOG_CONF", path)
        try:
            src_amulog.init_amulogloader(conf, ("DT0", "DT1"))
        finally:
            src_amulog.AmulogLoader = orig_loader
            config.open_config = orig_open

        # source_conf path is opened, not passed raw; dt_range in its own slot
        self.assertEqual(captured["conf"], ("AMULOG_CONF", "amulog.conf"))
        self.assertEqual(captured["dt_range"], ("DT0", "DT1"))
        self.assertEqual(captured["gid_name"], "ltgid")
        self.assertFalse(captured["use_mapping"])
        self.assertEqual(captured["host_tier"], "")


if __name__ == "__main__":
    unittest.main()

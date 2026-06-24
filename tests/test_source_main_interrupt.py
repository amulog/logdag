#!/usr/bin/env python
# coding: utf-8

"""Tests for KeyboardInterrupt handling in source.__main__ make-evdb handlers.

A Ctrl-C during a long store should: clean up via ``finally: el.terminate()``,
log a warning (so the interruption is not a silent success), and not propagate /
crash. Previously the handlers did ``except KeyboardInterrupt: pass`` -- the
interruption looked like a normal completion. This pins the new behaviour.
"""

import types
import unittest
from unittest import mock

from logdag.source import __main__ as smain


class TestMakeEvdbInterrupt(unittest.TestCase):

    def _ns(self):
        return types.SimpleNamespace(org=False, dry=False, parallel=False)

    def test_interrupt_logs_warning_and_terminates(self):
        conf = mock.Mock()
        conf.getboolean.return_value = False
        el = mock.Mock()
        el.store_all.side_effect = KeyboardInterrupt

        with mock.patch.object(smain, "open_logdag_config", return_value=conf), \
                mock.patch.object(smain, "_whole_term", return_value=("a", "b")), \
                mock.patch("logdag.source.evgen_snmp.SNMPEventLoader",
                           return_value=el):
            with self.assertLogs(smain._logger, level="WARNING") as cm:
                # must not raise despite the KeyboardInterrupt inside store_all
                smain.make_evdb_snmp_all(self._ns())

        el.store_all.assert_called_once()
        el.terminate.assert_called_once()  # cleanup still runs
        self.assertTrue(any("interrupted" in line for line in cm.output))


if __name__ == "__main__":
    unittest.main()

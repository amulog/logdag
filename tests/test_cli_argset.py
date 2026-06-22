#!/usr/bin/env python
# coding: utf-8

"""Smoke tests for the command-line dispatch tables (DICT_ARGSET).

`logdag` and its sub-libraries (source / eval / visual) drive amulog.cli with a
``DICT_ARGSET`` mapping each subcommand to ``[description, [arg-specs], handler]``
where every arg-spec is ``[[flags...], {add_argument kwargs}]``. These tests
exercise the tables without running any handler: each entry is well formed, the
handler is callable, and every arg-spec actually builds an argparse argument
(this catches a malformed spec or a duplicate-flag clash within a subcommand).
"""

import argparse
import importlib
import unittest


_MODULES = [
    "logdag.__main__",
    "logdag.source.__main__",
    "logdag.eval.__main__",
    "logdag.visual.__main__",
]


def _iter_argsets():
    for modname in _MODULES:
        mod = importlib.import_module(modname)
        for cmd, entry in mod.DICT_ARGSET.items():
            yield modname, cmd, entry


class TestCliArgset(unittest.TestCase):

    def test_entries_well_formed(self):
        for modname, cmd, entry in _iter_argsets():
            with self.subTest(module=modname, command=cmd):
                self.assertEqual(len(entry), 3)
                description, args, handler = entry
                self.assertIsInstance(description, str)
                self.assertIsInstance(args, list)
                self.assertTrue(callable(handler),
                                "{0} handler not callable".format(cmd))

    def test_arg_specs_build_a_parser(self):
        # the strongest check: each subcommand's specs must produce a working
        # argparse parser (malformed spec / clashing flags would raise here)
        for modname, cmd, entry in _iter_argsets():
            _, args, _ = entry
            with self.subTest(module=modname, command=cmd):
                parser = argparse.ArgumentParser(prog=cmd, add_help=False)
                for spec in args:
                    flags, kwargs = spec[0], spec[1]
                    self.assertIsInstance(flags, list)
                    self.assertIsInstance(kwargs, dict)
                    parser.add_argument(*flags, **kwargs)

    def test_known_commands_wired(self):
        from logdag import __main__ as m
        self.assertIs(m.DICT_ARGSET["make-dag"][2], m.make_dag)
        self.assertIs(m.DICT_ARGSET["show-edge"][2], m.show_edge)


if __name__ == "__main__":
    unittest.main()

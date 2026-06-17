#!/usr/bin/env python
# coding: utf-8

"""Static source-hygiene checks for the logdag package.

These guard against debugging residue being committed. Originally added
after a code review found live ``pdb.set_trace()`` calls left in the main
pipeline (``log2event.load_event_log_all``) and in
``source.evgen_log.LogEventLoader.details``, which would halt any run that
reached them.
"""

import os
import re
import unittest

import logdag

# package root (the directory that contains logdag/__init__.py)
_PKG_DIR = os.path.dirname(os.path.abspath(logdag.__file__))

# matches a live (non-commented) set_trace() call, e.g.
#   import pdb; pdb.set_trace()
#   pdb.set_trace()
#   breakpoint()
_TRACE_RE = re.compile(r"(?:\bpdb\s*\.\s*set_trace\s*\(|\bbreakpoint\s*\()")


def _iter_py_files(root):
    for dirpath, _, filenames in os.walk(root):
        if "__pycache__" in dirpath:
            continue
        for name in filenames:
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


class TestNoDebuggerResidue(unittest.TestCase):

    def test_no_live_set_trace(self):
        offenders = []
        for path in _iter_py_files(_PKG_DIR):
            with open(path, encoding="utf-8") as f:
                for lineno, line in enumerate(f, start=1):
                    # ignore commented-out lines
                    if line.lstrip().startswith("#"):
                        continue
                    code = line.split("#", 1)[0]
                    if _TRACE_RE.search(code):
                        rel = os.path.relpath(path, _PKG_DIR)
                        offenders.append("{0}:{1}: {2}".format(
                            rel, lineno, line.strip()))
        self.assertEqual(
            offenders, [],
            "live debugger calls found in logdag package:\n" +
            "\n".join(offenders))


if __name__ == "__main__":
    unittest.main()

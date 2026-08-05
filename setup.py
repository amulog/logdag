#!/usr/bin/env python

import sys
import os
import re
from setuptools import setup, find_packages


def load_readme():
    with open('README.rst', 'r') as fd:
        return fd.read()


def load_requirements():
    """Parse requirements.txt"""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as fd:
        requirements = [line.rstrip() for line in fd]
    return requirements


sys.path.append("./tests")
package_name = 'logdag'

with open(os.path.join(os.path.dirname(__file__), package_name, '__init__.py')) as f:
    version = re.search("__version__ = '([^']+)'", f.read()).group(1)

setup(name=package_name,
      version=version,
      description='A tool to generate causal DAGs from syslog time-series.',
      long_description=load_readme(),
      author='Satoru Kobayashi',
      author_email='sat@3at.work',
      url='https://github.com/amulog/logdag/',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          "Intended Audience :: Developers",
          'License :: OSI Approved :: BSD License',
          "Operating System :: OS Independent",
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
          'Programming Language :: Python :: 3.13',
          'Programming Language :: Python :: 3.14',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Software Development :: Libraries :: Python Modules'],
      license='The 3-Clause BSD License',

      packages=find_packages(exclude=['tests', 'tests.*']),
      install_requires=load_requirements(),
      extras_require={
          # optional InfluxDB v1 backend (logdag.source.influx) and the tests
          # that exercise it; install with `pip install -e .[influx]`.
          # The v3 backend (logdag.source.influx3) needs no extra package -- it
          # talks to the v3 HTTP API (SQL / Line Protocol) over stdlib urllib.
          'influx': ['influxdb'],
          # optional LiNGAM methods (cause_algorithm = lingam / lingam-corr);
          # install with `pip install -e .[lingam]`. Kept out of the core
          # requirements because lingam depends on semopy, whose polycorr module
          # calls scipy.stats.mvn.mvnun -- an attribute the scipy shim stopped
          # exposing in 1.14. lingam therefore pins scipy<=1.13.1, and that scipy
          # predates Python 3.13, so on 3.13/3.14 pip falls back to the scipy
          # sdist and the source build fails for want of OpenBLAS. logdag itself
          # only uses ICALiNGAM / DirectLiNGAM / make_prior_knowledge, none of
          # which touch the semopy path, so a newer scipy is fine for us -- it is
          # lingam's own dependency pin that cannot be satisfied there.
          # The python_version marker mirrors that limit, so asking for the extra
          # on 3.13/3.14 installs nothing instead of failing the scipy build.
          'lingam': ['lingam; python_version < "3.13"'],
          # vendored logdag.causaltestdata: only HawkesEventVariable needs the
          # Hawkes package (imported lazily). The other variable types rely on
          # numpy/scipy/pandas/networkx, which are already core requirements.
          # Hawkes ships no wheel for Python >= 3.13 (C++ extension), so it is
          # constrained to <3.13; the rest of the package supports 3.13/3.14.
          'testdata': ['Hawkes; python_version < "3.13"'],
      },
      package_data={'logdag': ['data/*']},
      entry_points={
          'console_scripts': [
              'logdag = logdag.__main__:main',
              'logdag.source = logdag.source.__main__:main',
              'logdag.eval = logdag.eval.__main__:main',
              'logdag.visual = logdag.visual.__main__:main',
          ],
      },
      test_suite="tests"
      )

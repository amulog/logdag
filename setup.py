#!/usr/bin/env python

import sys
import os
import re
from setuptools import setup


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
data_dir = "/".join((package_name, "data"))
data_files = ["/".join((data_dir, fn)) for fn in os.listdir(data_dir)]

package_name = 'logdag'

with open(os.path.join(os.path.dirname(__file__), package_name, '__init__.py')) as f:
    version = re.search("__version__ = '([^']+)'", f.read()).group(1)

setup(name=package_name,
      version=version,
      description='A tool to generate causal DAGs from syslog time-series.',
      long_description=load_readme(),
      author='Satoru Kobayashi',
      author_email='sat@nii.ac.jp',
      url='https://github.com/cpflat/logdag/',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          "Intended Audience :: Developers",
          'License :: OSI Approved :: BSD License',
          "Operating System :: OS Independent",
          'Programming Language :: Python :: 3.8',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Software Development :: Libraries :: Python Modules'],
      license='The 3-Clause BSD License',

      packages=['logdag'],
      install_requires=load_requirements(),
      package_data={'logdag': data_files},
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

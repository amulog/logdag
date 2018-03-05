#!/usr/bin/env python

import sys
import os
from setuptools import setup

try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print('pandoc is not installed.')
    read_md = lambda f: open(f, 'r').read()

sys.path.append("./tests")
package_name = 'logdag'
data_dir = "/".join((package_name, "data"))
data_files = ["/".join((data_dir, fn)) for fn in os.listdir(data_dir)]

setup(name='logdag',
    version='0.0.1',
    description='',
    long_description=read_md('README.md'),
    author='Satoru Kobayashi',
    author_email='sat@hongo.wide.ad.jp',
    url='https://github.com/cpflat/logdag/',
    install_requires=['numpy>=1.12.1', 'scipy>=1.0.0', 'pcalg>=0.1.6',
                      'gsq>=0.1.6', 'citestfz>=0.0.1', 'networkx>=2.1'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.4.3',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules'],
    license='The 3-Clause BSD License',
    
    packages=['logdag'],
    package_data={'logdag' : data_files},
    #test_suite = "suite.suite"
    )

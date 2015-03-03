#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup

# Version number
version = '0.0.1'

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name = 'SymDiffPy',
      version = version,
      description = ("SymDiffPy"),
      license = "BSD 3-clause",
      keywords = "machine-learning gaussian-processes kernels",
      packages = ["SymDiffPy.models",
                  "SymDiffPy.kernel",
                  ],
      package_dir={'SymDiffPy': 'SymDiffPy'},
      install_requires=['numpy>=1.7', 'scipy>=0.12', 'theano>=0.6'],
      )

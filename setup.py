# -*- coding: utf-8; -*-
#
# sparsegrad - automatic calculation of sparse gradient
# Copyright (C) 2016-2018 Marek Zdzislaw Szymanski (marek@marekszymanski.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License, version 3,
# as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

from setuptools import setup

with open('sparsegrad/_version.py') as version_file:
    exec(version_file.read())

setup(name='sparsegrad',
      version=version,
      packages=['sparsegrad',
                'sparsegrad.base',
                'sparsegrad.forward',
                'sparsegrad.impl',
                'sparsegrad.impl.sparse',
                'sparsegrad.impl.sparsevec',
                'sparsegrad.sparsevec',
                'sparsegrad.tests'],
      url='http://www.marekszymanski.com/software/sparsegrad',
      license='GNU Affero General Public License v3',
      author='Marek Zdzislaw Szymanski',
      install_requires=[
          'numpy>=1.10.0',
          'scipy>=0.14.0',
          'packaging>=14.0',
          'nose'],
      author_email='marek@marekszymanski.com',
      description='pure Python automatic calculation of sparse Jacobian',
      long_description=open('README.rst').read(),
      include_package_data=True,
      platforms='any',
      classifiers=[
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
          'Development Status :: 3 - Alpha',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU Affero General Public License v3',
          'Topic :: Scientific/Engineering :: Mathematics'
      ]
      )

# -*- coding: utf-8; -*-
#
# sparsegrad - automatic calculation of sparse gradient
# Copyright (C) 2016 Marek Zdzislaw Szymanski
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
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
      packages=['sparsegrad','sparsegrad.tests'],
      url='http://www.marekszymanski.com/software/sparsegrad',
      license='GNU Affero General Public License v3',
      author='Marek Zdzislaw Szymanski',
      install_requires=['numpy>=1.10.0','scipy>=0.14.0','packaging>=14.0'],
      author_email='marek@marekszymanski.com',
      description='automatic calculation of sparse gradient',
      include_package_data=True,
      platforms='any',
      classifiers = [
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

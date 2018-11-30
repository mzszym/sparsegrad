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

import numpy as np
from parameterized import parameterized
from sparsegrad.testing.utils import verify_sparsity

test_functions = ['x', 'x[::-1]', 'stack(x**2, x**3)',
                  'stack(x**2, (x**3)[::-1])',
                  'where(False, x, x[::-1])',
                  'where(x >= 0, x, x[::-1])',
                  'where(x > 0, x, x[::-1])',
                  'sum(x)',
                  'stack(x, sum(x))',
                  'where(x > 0, x, sum(x))']

test_vectors = [np.linspace(0, 1, 0),
                np.linspace(0, 1, 1),
                np.linspace(0, 1, 2),
                np.linspace(1., 3., 3)]


@parameterized((x, f) for x in test_vectors for f in test_functions)
def test_sparsity(x, f):
    verify_sparsity(dict(ns='sg'), x, f)

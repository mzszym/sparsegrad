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
from scipy import sparse
from sparsegrad import forward
from test_basic import check_general, check_vector
from sparsegrad.base import stack
from scipy.sparse import csr_matrix


def f(x): return x**2


def g(x): return x**3


def df(x): return 2 * x


def dg(x): return 3 * x**2


def test_simple():
    for x in [-5, 5]:
        yield check_general, None, x, lambda x: stack(f(x), g(x)), csr_matrix([[df(x)],
                                                                               [dg(x)]])
        yield check_general, None, x, lambda x: stack(f(x), 0), csr_matrix([[df(x)],
                                                                            [0]])
        yield check_general, None, x, lambda x: stack(0, f(x)), csr_matrix([[0],
                                                                            [df(x)]])


def test_vector():
    for n in [0, 1, 2, 3, 7]:
        x = np.linspace(0., 4., n)
        n = int(len(x) / 2)

        def f(x): return stack(x[:n], x[n:])
        yield check_vector, None, x, f, lambda x: 1.


def test_vector_scalar():
    for x in [-3, 3]:
        for y in [np.ones(0), np.ones(1), np.ones(2), np.linspace(0, 1, 3)]:
            df2d = np.atleast_2d(df(x))
            dv = np.atleast_2d(np.zeros_like(y)).transpose()
            dgxy2d = g(forward.seed(x) * y).dvalue.todense()
            yield check_general, None, x, lambda x: stack(f(x), y), csr_matrix(np.vstack((df2d, dv)))
            yield check_general, None, x, lambda x: stack(y, f(x)), csr_matrix(np.vstack((dv, df2d)))
            yield check_general, None, x, lambda x: stack(f(x), g(x * y)), csr_matrix(np.vstack((df2d, dgxy2d)))
            yield check_general, None, x, lambda x: stack(g(x * y), f(x)), csr_matrix(np.vstack((dgxy2d, df2d)))

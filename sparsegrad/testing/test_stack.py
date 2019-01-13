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
from scipy.sparse import csr_matrix
from sparsegrad import forward
from sparsegrad.testing.utils import check_general, check_vector
from sparsegrad.testing.namespaces import sg

stack = sg.stack


def f(x): return x**2


def g(x): return x**3


def df(x): return 2 * x


def dg(x): return 3 * x**2


@parameterized((x,) for x in [-5, 2])
def test_simple(x):
    check_general(x, lambda x: stack(f(x), g(x)),
                  csr_matrix([[df(x)], [dg(x)]]))
    check_general(x, lambda x: stack(f(x), 0), csr_matrix([[df(x)], [0]]))
    check_general(x, lambda x: stack(0, f(x)), csr_matrix([[0], [df(x)]]))


@parameterized((x,) for x in [0, 1, 2, 3, 7])
def test_split_stack(n):
    x = np.linspace(0., 4., n)
    n = int(len(x) / 2)

    def f(x): return stack(x[:n], x[n:])
    check_vector(x, f, lambda x: 1)


@parameterized((x, v) for x in [-3, 3] for v in [np.ones(0),
                                                 np.ones(1), np.ones(2), np.linspace(0, 1, 3)])
def test_vector_scalar(x, y):
    df2d = np.atleast_2d(df(x))
    dv = np.atleast_2d(np.zeros_like(y)).transpose()
    dgxy2d = g(forward.seed(x) * y).dvalue.toarray()
    check_general(
        x, lambda x: stack(
            f(x), y), csr_matrix(
            np.vstack(
                (df2d, dv))))
    check_general(
        x, lambda x: stack(
            y, f(x)), csr_matrix(
            np.vstack(
                (dv, df2d))))
    check_general(x, lambda x: stack(f(x), g(x * y)),
                  csr_matrix(np.vstack((df2d, dgxy2d))))
    check_general(x, lambda x: stack(g(x * y), f(x)),
                  csr_matrix(np.vstack((dgxy2d, df2d))))

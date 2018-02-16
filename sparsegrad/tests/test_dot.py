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
import scipy.sparse
from sparsegrad import forward
from sparsegrad.base import dot
from test_basic import check_general


def check_dot1(x, M):
    def f(x): return dot(M, x)
    check_general(None, x, f, M)


def check_dot2(x, M):
    def g(x): return 5 * x**3 + 3 * x
    dg = g(forward.seed(x)).dvalue

    def f(x): return dot(M, g(x))
    check_general(None, x, f, dot(M, dg))


def check_dot3(x, M):
    def g(x): return 5 * x**3 + 3 * x

    def f(x): return g(dot(M, x))
    dg = g(forward.seed(dot(M, x))).dvalue
    check_general(None, x, f, dot(dg, M))


def test_dot():
    np.random.seed(0)
    for n in [0, 1, 3, 5]:
        x = np.random.rand(n)
        M = scipy.sparse.csr_matrix(np.matrix(np.random.rand(n, n)))
        yield check_dot1, x, M
        yield check_dot2, x, M
        yield check_dot3, x, M

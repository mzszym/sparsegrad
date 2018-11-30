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
from scipy.sparse import csr_matrix
from parameterized import parameterized
from sparsegrad.testing.utils import check_general


def f(x): return x**2


def _test_simple():
    x = np.linspace(0., 4., 5)
    yield x, 'f(x)[0]', 0.
    yield x, 'f(x)[1]', csr_matrix([[0, 2, 0, 0, 0]])
    yield x, 'f(x)[3]', csr_matrix([[0, 0, 0, 6, 0]])
    yield x, 'f(x)[-1]', csr_matrix([[0, 0, 0, 0, 8]])
    # slice
    yield x, 'f(x)[1:-2]', csr_matrix([[0, 2, 0, 0, 0],
                                       [0, 0, 4, 0, 0]])
    # fancy indexing
    yield x, 'f(x)[np.asarray([1, 1])]', csr_matrix([[0, 2, 0, 0, 0],
                                                     [0, 2, 0, 0, 0]])
    yield x, 'x * x[2]', csr_matrix([[2, 0, 0, 0, 0],
                                     [0, 2, 1, 0, 0],
                                     [0, 0, 4, 0, 0],
                                     [0, 0, 3, 2, 0],
                                     [0, 0, 4, 0, 2]])


@parameterized(_test_simple)
def test_simple(x, func, result):
    def f(x): return eval(func, globals(), dict(x=x))
    return check_general(x, f, result)

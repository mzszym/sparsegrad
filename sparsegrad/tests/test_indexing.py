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
from sparsegrad.forward import *
from test_basic import check_general

from scipy.sparse import csr_matrix


def test_simple():
    def f(x): return x**2
    x = np.linspace(0., 4., 5)

    yield check_general, None, x, lambda x: f(x)[0], 0.
    yield check_general, None, x, lambda x: f(x)[1], csr_matrix([[0, 2, 0, 0, 0]])
    yield check_general, None, x, lambda x: f(x)[3], csr_matrix([[0, 0, 0, 6, 0]])
    yield check_general, None, x, lambda x: f(x)[-1], csr_matrix([[0, 0, 0, 0, 8]])
    # slice
    yield check_general, None, x, lambda x: f(x)[1:-2], csr_matrix([[0, 2, 0, 0, 0],
                                                                    [0, 0, 4, 0, 0]])
    # fancy indexing
    yield check_general, None, x, lambda x: f(x)[np.asarray([1, 1])], csr_matrix([[0, 2, 0, 0, 0],
                                                                                  [0, 2, 0, 0, 0]])
    yield check_general, None, x, lambda x: x * x[2], csr_matrix([[2, 0, 0, 0, 0],
                                                                  [0, 2, 1, 0, 0],
                                                                  [0, 0, 4, 0, 0],
                                                                  [0, 0, 3, 2, 0],
                                                                  [0, 0, 4, 0, 2]])

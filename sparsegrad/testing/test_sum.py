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
from sparsegrad.testing.utils import verify_scalar, check_general, lambdify

scalar_tests = [
    (1, 'sum(x)', '1')
]


@parameterized(scalar_tests)
def test_simple(x, f, df):
    verify_scalar(dict(ns='sg'), x, f, df)


vector_tests = [
    (np.ones(3), 'sum(x)', csr_matrix([[1, 1, 1]])),
    (np.asarray([1, 2, 3]), 'sum(x**2)', csr_matrix([[2, 4, 6]])),
    (np.asarray([3, 5, 7]), 'sum(x)**2', csr_matrix([[30, 30, 30]]))
]


@parameterized(vector_tests)
def test_vector(x, func, mat):
    f = lambdify(func, dict(ns='sg'))
    check_general(x, f, mat)

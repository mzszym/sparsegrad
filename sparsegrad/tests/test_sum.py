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
from test_basic import check_scalar, check_general
from sparsegrad.base import sum
from scipy.sparse import csr_matrix


def test_simple():
    yield check_scalar, None, 1, sum, lambda x: 1.


def test_vector():
    yield check_general, None, np.ones(3), sum, csr_matrix([[1, 1, 1]])
    yield check_general, None, np.asarray([1, 2, 3]), lambda x: sum(x**2), csr_matrix([[2, 4, 6]])
    yield check_general, None, np.asarray([3, 5, 7]), lambda x: sum(x)**2, csr_matrix([[30, 30, 30]])

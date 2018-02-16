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
from sparsegrad.base import where
from test_basic import check_scalar, check_vector, generate

test_values = [-1e3, -1e2, 1e-1, -1., 0., 1., 1e1, 1e2, 1e3]

_funcs = [
    (lambda x:where(x < 0., x, 0.), lambda x:where(x < 0., 1., 0.)),
    (lambda x:where(x < 0., 0., x), lambda x:where(x < 0., 0., 1.)),
    (lambda x:where(x > 0., x, 0.), lambda x:where(x > 0., 1., 0.)),
    (lambda x:where(x > 0., 0., x), lambda x:where(x > 0., 0., 1.)),
    (lambda x:where(True, x, 0.), lambda x:1.),
    (lambda x:where(False, x, 0.), lambda x:0.),
    (lambda x:where(True, 0., x), lambda x:0.),
    (lambda x:where(False, 0., x), lambda x:1.)
]


def test_scalar():
    for t in generate(_funcs, test_values, check_scalar):
        yield t


def test_vector():
    for t in generate(_funcs, [np.ones(0), np.ones(
            1), np.asarray(test_values)], check_vector):
        yield t

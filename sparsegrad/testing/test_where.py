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
from sparsegrad.testing.utils import verify_scalar, verify_vector, product

test_scalars = [-1e3, -1e2, 1e-1, -1., 0., 1., 1e1, 1e2, 1e3]
test_vectors = [np.ones(0), np.ones(1), np.asarray(test_scalars)]

functions_with_where = [
    ('where(x < 0., x, 0.)', 'where(x < 0., 1., 0.)'),
    ('where(x < 0., 0., x)', 'where(x < 0., 0., 1.)'),
    ('where(x > 0., x, 0.)', 'where(x > 0., 1., 0.)'),
    ('where(x > 0., 0., x)', 'where(x > 0., 0., 1.)'),
    ('where(True, x, 0.)', '1.'),
    ('where(False, x, 0.)', '0.'),
    ('where(True, 0., x)', '0.'),
    ('where(False, 0., x)', '1.')
]


@parameterized(product(functions_with_where, test_scalars, namespaces=['sg']))
def test_scalar(*args):
    verify_scalar(*args)


@parameterized(product(functions_with_where, test_vectors, namespaces=['sg']))
def test_vector(*args):
    verify_vector(*args)

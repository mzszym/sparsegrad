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

from parameterized import parameterized
import numpy as np
from sparsegrad.testing.utils import verify_vector, verify_scalar, check_vector_scalar, product
from sparsegrad import forward

polynominals = [
    # + - *
    ('+x', '1'),
    ('-x', '-1'),
    ('0 + x', '1'),
    ('x + 0', '1'),
    ('1 + x', '1'),
    ('x + 1', '1'),
    ('0 * x', '0'),
    ('x * 0', '0'),
    ('x * 5', '5'),
    ('x**2', '2 * x'),
    ('x + 2 * x', '3.'),
    ('5 * x + 1', '5.'),
    ('x - 3 * x', '-2.'),
    ('x * x - 2 * x + 3', '2 * x - 2'),
    ('x * (0.5 * x + 3.)', 'x + 3.'),
    ('(2 + x) * (3 - x) * (5 + x) * (7 - x)',
     '4 * x**3 - 9 * x**2 - 78 * x + 47'),
    ('(x + 2) * (x - 3) * (x + 5) * (7 - x)',
     '-(4 * x**3 - 9 * x**2 - 78 * x + 47)'),
]

basic_functions = [
    ('2. / x', '-2. / x**2'),
    ('x / 3', '1. / 3.'),
    ('reciprocal(x)', '-1. / x**2'),
    ('(11 + x) * (13 - x) / ((7 + x) * (5 - x))',
     '4 * (x**2 + 54 * x + 89) / (x**4 + 4 * x**3 - 66 * x**2 - 140 * x + 1225)'),
    ('5 * x * reciprocal(2 + x)',
     '-5 * x / (x + 2)**2 + 5 / (x + 2)'),
    ('x**5', '5 * x**4'),
    ('1.01**x', '0.00995033085316809 * 1.01**x'),
    ('abs(x)', 'sign(x)'),
    ('square(2 * x)', '8 * x'),
    ('sqrt(x + 1000.00001)', '1. / (2 * sqrt(x + 1000.00001))'),
    ('sqrt(x * x)', 'sign(x)'),
    ('exp(x / 1e2)', 'exp(x / 1e2) / 1e2'),
    ('log(x + 1000.00001)', '1. / (x + 1000.00001)'),
    ('log(exp(x / 1e2))', '1e-2')
]

trigonometric = [
    ('sin(x)', 'cos(x)'),
    ('cos(x)', '-sin(x)'),
    ('sin(x)**2 + cos(x)**2', '0.'),
    ('sin(x) / cos(x) - tan(x)', '0.'),
    ('arcsin(sin(x)) - x', '0.'),
    ('arccos(cos(x)) - abs(x)', '0.'),
    ('arctan(tan(x)) - x', '0.'),
]

hyperbolic = [
    ('sinh(x / 1e2)', 'cosh(x / 1e2) / 1e2'),
    ('cosh(x / 1e2)', 'sinh(x / 1e2) / 1e2'),
    ('sinh(x / 1e2) / cosh(x / 1e2) - tanh(x / 1e2)', '0.'),
    ('arcsinh(sinh(x / 1e2)) - x / 1e2', '0.'),
    ('arccosh(cosh(x / 1e2)) - abs(x / 1e2)', '0.'),
    ('arctanh(tanh(x / 1e2)) - x / 1e2', '0.')
]

all_functions = polynominals + basic_functions + trigonometric + hyperbolic
real_dtypes = [np.float64, np.float128]
complex_dtypes = [np.complex128, np.complex256]
test_dtypes = real_dtypes + complex_dtypes
test_scalars_py = [-1e3, -1., -1e-3, 1e-3, 1., 1e3]
test_scalars_np = [np.asarray(s, dtype=dtype)
                   for s in test_scalars_py for dtype in test_dtypes]
test_vectors_ = [np.ones(0), np.ones(1), np.asarray(test_scalars_py)]
test_vectors = [np.asarray(v, dtype=dtype)
                for v in test_vectors_ for dtype in test_dtypes]

test_scalars = test_scalars_py + test_scalars_np


@parameterized(product(all_functions, test_scalars, namespaces=['sg', 'np']))
def test_scalar(*args):
    verify_scalar(*args)


@parameterized(product(all_functions, test_vectors, namespaces=['sg', 'np']))
def test_vector(*args):
    verify_vector(*args)


@parameterized((x, v) for x in test_scalars for v in test_vectors)
def test_vector_scalar_add(x, v):
    check_vector_scalar(x, v, lambda x, v: x+v, lambda x, v: 1.)


@parameterized((x, v) for x in test_scalars for v in test_vectors)
def test_vector_scalar_multiplication(x, v):
    check_vector_scalar(x, v, lambda x, v: x*v, lambda x, v: v)


def test_array_priority():
    x = np.zeros(5)
    g = np.zeros(5)
    assert isinstance(forward.seed(x) * g, forward.value)
    assert isinstance(g * forward.seed(x), forward.value)

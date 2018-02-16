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

from numpy.testing import *
from sparsegrad import forward
import numpy as np
from sparsegrad.utils import *
import inspect
import scipy.sparse


def check_scalar(msg, x, f, df):
    y = f(forward.seed(x))
    assert_almost_equal(y.value, f(x))
    assert_almost_equal(y.dvalue, df(x))


def _check_grad(a, b):
    err = a - b
    if isinstance(err, scipy.sparse.csr_matrix):
        assert_almost_equal(err.data, 0.)
    else:
        assert_almost_equal(err, 0.)


def check_vector(msg, x, f, df):
    y = f(forward.seed(x))
    d = np.atleast_1d(df(x) * np.ones_like(y.value))
    dy = scipy.sparse.spdiags(d, 0, len(d), len(d), format='csr')
    assert_almost_equal(y.value, f(x))
    _check_grad(y.dvalue, dy)


def check_vector_scalar(msg, x, v, f, df):
    y = f(forward.seed(x), v)
    assert_almost_equal(y.value, f(x, v))
    assert y.dvalue.shape == (len(y.value), 1)
    assert_almost_equal(np.asarray(y.dvalue.todense()).flatten(), df(x, v))


def check_general(msg, x, f, grad):
    y = f(forward.seed(x))
    dy = grad
    assert_almost_equal(y.value, f(x))
    _check_grad(y.dvalue, dy)


_funcs = [
    # + - *
    (lambda x:+x, lambda x:1),
    (lambda x:-x, lambda x:-1),
    (lambda x:0 + x, lambda x:1),
    (lambda x:x + 0, lambda x:1),
    (lambda x:1 + x, lambda x:1),
    (lambda x:x + 1, lambda x:1),
    (lambda x:0 * x, lambda x:0),
    (lambda x:x * 0, lambda x:0),
    (lambda x:x * 5, lambda x:5),
    (lambda x:x**2, lambda x:2 * x),
    (lambda x:x + 2 * x, lambda x:3.),
    (lambda x:5 * x + 1, lambda x:5.),
    (lambda x:x - 3 * x, lambda x:-2.),
    (lambda x:x * x - 2 * x + 3, lambda x:2 * x - 2),
    (lambda x:x * (0.5 * x + 3.), lambda x:x + 3.),
    (lambda x:(2 + x) * (3 - x) * (5 + x) * (7 - x),
     lambda x:4 * x**3 - 9 * x**2 - 78 * x + 47),
    (lambda x:(x + 2) * (x - 3) * (x + 5) * (7 - x),
     lambda x:-(4 * x**3 - 9 * x**2 - 78 * x + 47)),

    # / reciprocal
    (lambda x:2. / x, lambda x:-2. / x**2),
    (lambda x:x / 3, lambda x:1. / 3.),
    (lambda x:np.reciprocal(x), lambda x:-1. / x**2),
    (lambda x:(11 + x) * (13 - x) / ((7 + x) * (5 - x)), lambda x:4 * \
     (x**2 + 54 * x + 89) / (x**4 + 4 * x**3 - 66 * x**2 - 140 * x + 1225)),
    (lambda x:5 * x * np.reciprocal(2 + x),
     lambda x:-5 * x / (x + 2)**2 + 5 / (x + 2)),

    # power
    (lambda x:x**5, lambda x:5 * x**4),
    (lambda x:1.01**x, lambda x:0.00995033085316809 * 1.01**x),

    # abs
    (lambda x:np.abs(x), lambda x:np.sign(x)),

    # sqrt, square
    (lambda x:np.square(2 * x), lambda x:8 * x),
    (lambda x:np.sqrt(x + 1000.00001), lambda x:1. / (2 * np.sqrt(x + 1000.00001))),
    (lambda x:np.sqrt(x * x), lambda x:np.sign(x)),

    # exp, log
    (lambda x:np.exp(x / 1e2), lambda x:np.exp(x / 1e2) / 1e2),
    (lambda x:np.log(x + 1000.00001), lambda x:1. / (x + 1000.00001)),
    (lambda x:np.log(np.exp(x / 1e2)), lambda x:1e-2),

    # sin, cos, tan and inverses
    (lambda x:np.sin(x), lambda x:np.cos(x)),
    (lambda x:np.cos(x), lambda x:-np.sin(x)),
    (lambda x:np.sin(x)**2 + np.cos(x)**2, lambda x:0.),
    (lambda x:np.sin(x) / np.cos(x) - np.tan(x), lambda x:0.),
    (lambda x:np.arcsin(np.sin(x)) - x, lambda x:0.),
    (lambda x:np.arccos(np.cos(x)) - abs(x), lambda x:0.),
    (lambda x:np.arctan(np.tan(x)) - x, lambda x:0.),

    # sinh, cosh, tanh and inverses
    (lambda x:np.sinh(x / 1e2), lambda x:np.cosh(x / 1e2) / 1e2),
    (lambda x:np.cosh(x / 1e2), lambda x:np.sinh(x / 1e2) / 1e2),
    (lambda x:np.sinh(x / 1e2) / np.cosh(x / 1e2) - np.tanh(x / 1e2), lambda x:0.),
    (lambda x:np.arcsinh(np.sinh(x / 1e2)) - x / 1e2, lambda x:0.),
    (lambda x:np.arccosh(np.cosh(x / 1e2)) - abs(x / 1e2), lambda x:0.),
    (lambda x:np.arctanh(np.tanh(x / 1e2)) - x / 1e2, lambda x:0.)
]

_vsfuncs = [
    (lambda x, v:x + v, lambda x, v:1.),
    (lambda x, v:x * v, lambda x, v:v)
]

test_values = [-1e3, -1., -1e-3, 1e-3, 1., 1e3]


def generate(funcs, values, check):
    for f, df in funcs:
        for x in values:
            yield check, inspect.getsource(f), x, f, df


def scalar_tests():
    for t in generate(_funcs, test_values, check_scalar):
        yield t


def vector_tests():
    for t in generate(_funcs, [np.ones(0), np.ones(
            1), np.asarray(test_values)], check_vector):
        yield t


def vector_scalar_tests():
    for v in [np.ones(0), np.ones(1), np.asarray([1, 2, 3])]:
        for f, df in _vsfuncs:
            for x in test_values:
                yield check_vector_scalar, inspect.getsource(f), x, v, f, df


def test_array_priority():
    x = np.zeros(5)
    g = np.zeros(5)
    assert isinstance(forward.seed(x) * g, forward.value)
    assert isinstance(g * forward.seed(x), forward.value)

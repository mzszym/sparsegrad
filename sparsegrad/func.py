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

"Differentiation formulas for arithmetic operations and elementary functions"

import numpy as np

known_funcs = {}
known_ufuncs = known_funcs


class ufunc_deriv(object):
    def __init__(self, func, deriv):
        self.evaluate = func
        self.deriv = deriv
        self.nin = func.nin


class custom_func(object):
    def __init__(self, func, deriv):
        self.evaluate = func
        self.deriv = deriv


def uderiv(func):
    def apply(deriv):
        name = func.__name__
        assert name not in known_funcs
        obj = ufunc_deriv(func, deriv)
        known_funcs[name] = obj
        return obj
    return apply


@uderiv(np.add)
def add(args, value):
    yield lambda: 1.
    yield lambda: 1.


@uderiv(np.subtract)
def subtract(args, value):
    yield lambda: 1.
    yield lambda: -1.


@uderiv(np.multiply)
def multiply(args, value):
    yield lambda: args[1]
    yield lambda: args[0]


def _reciprocal(x):
    # Problem with numpy reciprocal: np.reciprocal(2)==0
    return 1. / x


@uderiv(np.divide)
def divide(args, value):
    a, b = args
    t = _reciprocal(b)
    yield lambda: t
    yield lambda: -a * t**2


@uderiv(np.power)
def power(args, value):
    a, b = args
    yield lambda: b * a**(b - 1.)
    yield lambda: value * np.log(a)


true_divide = divide


@uderiv(np.negative)
def negative(args, value):
    yield lambda: -1.


@uderiv(np.abs)
def abs(args, value):
    yield lambda: np.sign(args[0])


absolute = abs


@uderiv(np.reciprocal)
def reciprocal(args, value):
    yield lambda: -value**2


@uderiv(np.exp)
def exp(args, value):
    yield lambda: value


@uderiv(np.log)
def log(args, value):
    yield lambda: _reciprocal(args[0])


@uderiv(np.sqrt)
def sqrt(args, value):
    yield lambda: 0.5 / value


@uderiv(np.square)
def square(args, value):
    yield lambda: 2. * value


@uderiv(np.sin)
def sin(args, value):
    yield lambda: np.cos(args[0])


@uderiv(np.cos)
def cos(args, value):
    yield lambda: -np.sin(args[0])


@uderiv(np.tan)
def tan(args, value):
    yield lambda: value**2 + 1.


@uderiv(np.arcsin)
def arcsin(args, value):
    yield lambda: _reciprocal(np.sqrt(1. - args[0]**2))


@uderiv(np.arccos)
def arccos(args, value):
    yield lambda: -_reciprocal(np.sqrt(1. - args[0]**2))


@uderiv(np.arctan)
def arctan(args, value):
    yield lambda: _reciprocal(1. + np.square(args[0]))


@uderiv(np.sinh)
def sinh(args, value):
    yield lambda: np.cosh(args[0])


@uderiv(np.cosh)
def cosh(args, value):
    yield lambda: np.sinh(args[0])


@uderiv(np.tanh)
def tanh(args, value):
    yield lambda: -np.square(value) + 1.


@uderiv(np.arcsinh)
def arcsinh(args, value):
    yield lambda: _reciprocal(np.sqrt(np.square(args[0]) + 1.))


@uderiv(np.arccosh)
def arccosh(args, value):
    yield lambda: _reciprocal(np.sqrt(np.square(args[0]) - 1.))


@uderiv(np.arctanh)
def arctanh(args, value):
    yield lambda: _reciprocal(-np.square(args[0]) + 1.)


@uderiv(np.expm1)
def expm1(args, value):
    x, = args
    yield lambda: np.exp(x)


@uderiv(np.log1p)
def log1p(args, value):
    x, = args
    yield lambda: _reciprocal(1. + x)

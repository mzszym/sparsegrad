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

from sparsegrad.impl.multipledispatch import Dispatcher
from .import ufunc_routing

known_funcs = {}
known_ufuncs = known_funcs


class DifferentiableFunction(object):
    pass


class SplitElementwiseDifferentiableFunction(DifferentiableFunction):
    def __init__(self, func, deriv):
        self.func = func
        self.deriv = deriv

    def f_df(self, nargs):
        y = self.func(*nargs)
        return y, self.deriv(nargs, y)


class OneCallElementwiseDifferentiableFunction(DifferentiableFunction):
    def __init__(self, f_df):
        self.f_df = f_df

    def func(self, *nargs):
        raise NotImplementedError()

    def deriv(self, nargs, y):
        raise NotImplementedError()


class FunctionDispatcher(Dispatcher):
    def __init__(self, name, func, **kwargs):
        super(FunctionDispatcher, self).__init__(name=name, **kwargs)
        self.nin = func.nin
        self.func = func
        self.add((object,)*self.nin, self.func.func)

    def addHandlers(self, value_type, func):
        for i in range(self.nin):
            types = [object]*self.nin
            types[i] = value_type
            self.add(types, func)


class UFuncWrapper(SplitElementwiseDifferentiableFunction):
    def __init__(self, func, deriv):
        super(UFuncWrapper, self).__init__(func, deriv)
        self.nin = func.nin


def uderiv(func):
    def apply(deriv):
        name = func.__name__
        assert name not in known_funcs
        obj = UFuncWrapper(func, deriv)
        known_funcs[name] = obj
        assert not hasattr(ufunc_routing, name)
        setattr(ufunc_routing, name, FunctionDispatcher(name, obj))
        getattr(ufunc_routing, '__all__').append(name)
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


@uderiv(np.sign)
def sign(args, value):
    return lambda: np.where(args[0] != 0, 0., np.nan)


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
    yield lambda: 2. * args[0]


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


__all__ = ['SplitElementwiseDifferentiableFunction',
           'OneCallElementwiseDifferentiableFunction', 'asdifferentiable']


def asdifferentiable(f=None, deriv=None, f_df=None):
    if f_df is not None:
        assert f is None and deriv is None
        return OneCallElementwiseDifferentiableFunction(f_df)
    else:
        assert f is not None and deriv is not None
        return SplitElementwiseDifferentiableFunction(f, deriv)

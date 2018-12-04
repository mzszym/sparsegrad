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

from sparsegrad.functions import ufunc, ufunc_routing, routing, utils


class expr_base(object):
    """
    Base class for numpy-compatible operator overloading

    It provides default overloads of arithmetic operators and methods for mathematical functions.
    The default overloads call abstract apply method to calculate the result of operation.
    """

    __array_priority__ = 100
    __array_wrap__ = None

    @classmethod
    def apply(cls, func, args):
        """
        Apply DifferentiableFunction to args
        """
        raise NotImplementedError()

    def apply1(self, func):
        """
        Apply single argument DifferentiableFunction to value
        """
        raise NotImplementedError()

    def __add__(self, other):
        return self.__class__.apply(ufunc.add, (self, other))

    def __radd__(self, other):
        return self.__class__.apply(ufunc.add, (other, self))

    def __sub__(self, other):
        return self.__class__.apply(ufunc.subtract, (self, other))

    def __rsub__(self, other):
        return self.__class__.apply(ufunc.subtract, (other, self))

    def __mul__(self, other):
        return self.__class__.apply(ufunc.multiply, (self, other))

    def __rmul__(self, other):
        return self.__class__.apply(ufunc.multiply, (other, self))

    def __div__(self, other):
        return self.__class__.apply(ufunc.divide, (self, other))

    def __rdiv__(self, other):
        return self.__class__.apply(ufunc.divide, (other, self))

    def __truediv__(self, other):
        return self.__class__.apply(ufunc.divide, (self, other))

    def __rtruediv__(self, other):
        return self.__class__.apply(ufunc.divide, (other, self))

    def __pow__(self, other):
        return self.__class__.apply(ufunc.power, (self, other))

    def __rpow__(self, other):
        return self.__class__.apply(ufunc.power, (other, self))

    def __pos__(self):
        return self

    def __neg__(self):
        return self.apply1(ufunc.negative)

    def __getitem__(self, idx):
        return self.apply(ufunc.getitem, (self, idx))

    def __abs__(self):
        return self.apply1(ufunc.abs)

    def compare(self, operator, other):
        raise NotImplementedError()


class _comparison_proxy(object):
    def __init__(self, operator):
        self.operator = operator

    def __get__(self, instance, cls):
        return lambda other: instance.compare(self.operator, other)


class _function_proxy1(object):
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        return lambda *args, **kwargs: instance.apply1(self.func)


class _wrapper(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        impl = routing.find_implementation(args, default=expr_base)
        return impl.__class__.apply(self.func, args)


class _wrapper1(_wrapper):
    def __call__(self, arg):
        return arg.apply1(self.func)


def _register():
    for operator in ['__lt__', '__le__', '__eq__', '__ne__', '__ge__', '__gt__']:
        setattr(expr_base, operator, _comparison_proxy(operator))
    for name, func in ufunc.known_funcs.items():
        if func.nin == 1:
            setattr(expr_base, name, _function_proxy1(func))
            getattr(ufunc_routing, name).add((expr_base,), _wrapper1(func))
        else:
            wrapper = _wrapper(func)
            getattr(ufunc_routing, name).addHandlers(expr_base, wrapper)


_register()


def _apply(impl, func, args):
    return impl.__class__.apply(func, args)


routing.apply.add((expr_base, object, object), _apply)


def _is_expr_numeric(x):
    return False


utils.isnvalue.add((expr_base,), _is_expr_numeric)

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
from sparsegrad.functions import routing
from sparsegrad.impl.multipledispatch import GenericFunction
from sparsegrad.functions import func

class function_proxy(object):
    def __init__(self, target):
        self.target = target

    def __get__(self, instance, cls):
        return lambda *args, **kwargs: instance.apply(
            self.target, instance, *args, **kwargs)

class comparison_proxy(object):
    def __init__(self, operator):
        self.operator = operator

    def __get__(self, instance, cls):
        return lambda other: getattr(instance.value, self.operator)(other)

class expr_base(object):
    """
    Base class for numpy-compatible operator overloading

    It provides default overloads of arithmetic operators and methods for mathematical functions.
    The default overloads call abstract apply method to calculate the result of operation.
    """

    __array_priority__ = 100
    __array_wrap__ = None

    def apply(self, func, *args):
        """
        Evaluate and return func(*args)

        Subclasses do not need to call this for all functions.
        """
        raise NotImplementedError()

    def __add__(self, other):
        return self.apply(func.add, self, other)

    def __radd__(self, other):
        return self.apply(func.add, other, self)

    def __sub__(self, other):
        return self.apply(func.subtract, self, other)

    def __rsub__(self, other):
        return self.apply(func.subtract, other, self)

    def __mul__(self, other):
        return self.apply(func.multiply, self, other)

    def __rmul__(self, other):
        return self.apply(func.multiply, other, self)

    def __div__(self, other):
        return self.apply(func.divide, self, other)


    def __rdiv__(self, other):
        return self.apply(func.divide, other, self)

    def __truediv__(self, other):
        return self.apply(func.divide, self, other)

    def __rtruediv__(self, other):
        return self.apply(func.divide, other, self)

    def __pow__(self, other):
        return self.apply(func.power, self, other)

    def __rpow__(self, other):
        return self.apply(func.power, other, self)

    def __pos__(self):
        return self

    def __neg__(self):
        return self.apply(func.negative, self)

    def __getitem__(self, idx):
        return self.apply(func.getitem, self, idx)

    def __abs__(self):
        return self.apply(func.abs, self)


for operator in ['__lt__', '__le__', '__eq__', '__ne__', '__ge__', '__gt__']:
    setattr(expr_base, operator, comparison_proxy(operator))

class wrapped_func():
    "Wrap function for compatibility with expr_base"

    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        impl = routing.find_implementation(args, default=self)
        return impl.apply(self.func, *args)

    def apply(self, f, *args):
        return f.evaluate(*args)


for name, f in func.known_funcs.items():
    dispatcher = GenericFunction(name)
    dispatcher.add((object,)*f.nin, getattr(np, name))
    if f.nin == 1:
        setattr(expr_base, name, function_proxy(getattr(func, name)))
    globals()[name] = dispatcher

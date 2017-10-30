# -*- coding: utf-8; -*-
#
# sparsegrad - automatic calculation of sparse gradient
# Copyright (C) 2016, 2017 Marek Zdzislaw Szymanski (marek@marekszymanski.com)
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

from . import func
import numpy as np


def _genu():
    def _():
        for name, f in func.known_ufuncs.items():
            if f.nin == 1:
                yield "def %s(self): return self.apply(func.%s,self)" % (name, name)
    return "\n".join(_())


def _geng():
    def _():
        for name in func.known_funcs.keys():
            yield "%s=wrapped_func(func.%s)" % (name, name)
    return "\n".join(_())


class expr_base(object):
    __array_priority__ = 100
    __array_wrap__ = None

    def apply(self, func, *args):
        raise NotImplementedError()

    def __add__(self, other): return self.apply(func.add, self, other)

    def __radd__(self, other): return self.apply(func.add, other, self)

    def __sub__(self, other): return self.apply(func.subtract, self, other)

    def __rsub__(self, other): return self.apply(func.subtract, other, self)

    def __mul__(self, other): return self.apply(func.multiply, self, other)

    def __rmul__(self, other): return self.apply(func.multiply, other, self)

    def __div__(self, other): return self.apply(func.divide, self, other)

    def __rdiv__(self, other): return self.apply(func.divide, other, self)

    def __truediv__(self, other): return self.apply(func.divide, self, other)

    def __rtruediv__(self, other): return self.apply(func.divide, other, self)

    def __pow__(self, other): return self.apply(func.power, self, other)

    def __rpow__(self, other): return self.apply(func.power, other, self)

    def __pos__(self): return self

    def __neg__(self): return self.apply(func.negative, self)

    def __getitem__(self, idx):
        return self.apply(func.getitem, self, idx)

    def __abs__(self):
        return self.apply(func.abs, self)

    # ufuncs
    exec(_genu())


class wrapped_func():
    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        impl = _find_arr(args, 'apply', default=self)
        return impl.apply(self.func, *args)

    def apply(self, f, *args):
        return f.evaluate(*args)


# non ufuncs
exec(_geng())


def dot(a, b):
    if hasattr(b, 'rdot'):
        return b.rdot(a)
    else:
        return a.dot(b)


def _find_arr(arrays, attr, default=None, default_priority=0.):
    highest = default
    current = default_priority
    for a in arrays:
        if hasattr(a, attr):
            priority = getattr(a, '__array_priority__', 0.)
            if highest is None or priority > current:
                highest, current = a, priority
    return highest


def where(cond, a, b):
    impl = _find_arr((cond, a, b), 'where', default=np)
    return impl.where(cond, a, b)


def hstack(arrays):
    impl = _find_arr(arrays, 'hstack', default=np)
    return impl.hstack(arrays)


def sum(a):
    if isinstance(a, expr_base):
        return a.sum()
    else:
        return np.sum(a)


def stack(*arrays):
    return hstack(arrays)

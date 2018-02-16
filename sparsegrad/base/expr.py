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

from sparsegrad import func
from sparsegrad import impl
import sparsegrad.impl.sparsevec as impl_sparsevec
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


class bool_expr(object):
    "Abstract base class for boolean expressions"
    pass


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
    "Wrap function for compatibility with expr_base"

    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        impl = _find_arr(args, 'apply', default=self)
        return impl.apply(self.func, *args)

    def apply(self, f, *args):
        return f.evaluate(*args)


# non ufuncs
exec(_geng())


def _find_arr(arrays, attr, default=None, default_priority=0.):
    highest = default
    current = default_priority
    for a in arrays:
        if hasattr(a, attr):
            priority = getattr(a, '__array_priority__', 0.)
            if highest is None or priority > current:
                highest, current = a, priority
    return highest


def dot(a, b):
    "Equivalent of scipy.sparse.dot function aware of expr_base"
    impl_ = _find_arr((a, b), 'dot_', default=impl)
    return impl_.dot_(a, b)


def where(cond, a, b):
    "Equivalent of numpy.where function aware of expr_base"
    impl = _find_arr((cond, a, b), 'where', default=np)
    return impl.where(cond, a, b)


def hstack(arrays):
    "Equivalent of numpy.hstack function aware of expr_base"
    impl = _find_arr(arrays, 'hstack', default=np)
    return impl.hstack(arrays)


def sum(a):
    "Equivalent of numpy.sum function aware of expr_base"
    if isinstance(a, expr_base):
        return a.sum()
    else:
        return np.sum(a)


def stack(*arrays):
    "Alias for hstack, taking arrays as separate arguments"
    return hstack(arrays)


def sparsesum(terms, **kwargs):
    "Sparse summing function aware of expr_base"
    impl_ = _find_arr(
        (a.v for a in terms),
        'sparsesum',
        default=impl_sparsevec)
    return impl_.sparsesum(terms, **kwargs)


def as_condition_value(a):
    "Return value as concrete boolean value"
    return np.asarray(a, dtype=np.bool)


def broadcast_to(arr, shape):
    "Equivalent of numpy.broadcast_to aware of expr_base"
    impl = _find_arr([arr], 'broadcast_to', default=np)
    return impl.broadcast_to(arr, shape)


def branch(cond, iftrue, iffalse):
    """
    Branch execution

    Note that, in some cases (propagation of sparsity pattern), both branches can executed
    more than once.

    Parameters:
    -----------
    cond : bool vector
        Condition
    iftrue : callable(idx)
        Function called to evaluate elements with indices idx, where cond is True
    iffalse : callable(idx)
        Function called to evaluate elements with indices idx, where cond is False

    """
    if isinstance(cond, bool_expr) and cond.hasattr('branch'):
        return cond.branch(iftrue, iffalse)

    def _branch(cond, iftrue, iffalse):
        if not cond.shape:
            if cond:
                return iftrue(None)
            else:
                return iffalse(None)
        n = len(cond)
        r = np.arange(len(cond))
        ixtrue = r[cond]
        ixfalse = r[np.logical_not(cond)]
        vtrue = impl_sparsevec.sparsevec(
            n, ixtrue, broadcast_to(
                iftrue(ixtrue), ixtrue.shape))
        vfalse = impl_sparsevec.sparsevec(
            n, ixfalse, broadcast_to(
                iffalse(ixfalse), ixfalse.shape))
        return sparsesum([vtrue, vfalse])
    value = _branch(as_condition_value(cond), iftrue, iffalse)
    if hasattr(value, 'branch_join'):
        return value.branch_join(cond, iftrue, iffalse)
    else:
        return value

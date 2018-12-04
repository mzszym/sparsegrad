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
from sparsegrad.impl import sparse
from sparsegrad.impl import sparsevec as sparsevec_impl
from sparsegrad.base import expr_base
from sparsegrad import functions

__all__ = ['value', 'seed', 'seed_sparse_gradient', 'seed_sparsity', 'nvalue']


def nvalue(x):
    "return numeric value of x, x of type (forward_value, numeric types)"
    if isinstance(x, forward_value):
        return x.value
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.asarray(x)


class forward_value(expr_base):
    def __new__(cls, *args, **kwargs):
        value = kwargs.pop('value')
        deriv = kwargs.pop('deriv')
        assert hasattr(value, 'shape')
        assert isinstance(deriv, sparse.sdcsr)
        if not value.shape:
            assert deriv.mshape[0] is None
        else:
            assert deriv.mshape[0] == len(value)
        obj = super(forward_value, cls).__new__(cls, *args, **kwargs)
        obj.value = value
        obj.deriv = deriv
        return obj

    # getting values
    @property
    def gradient(self):
        return self.deriv.tovalue()

    dvalue = gradient
    sparsity = gradient

    # basic arithmetic: + - * /
    def __add__(self, other):
        if isinstance(other, forward_value):
            y = self.value + other.value
            dy = self.deriv.fma2(y, (1., self.deriv), (1., other.deriv))
        else:
            y = self.value + nvalue(other)
            dy = self.deriv.broadcast(y)
        return self.__class__(value=y, deriv=dy)
    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, forward_value):
            y = self.value * other.value
            dy = self.deriv.fma2(
                y, (other.value, self.deriv), (self.value, other.deriv))
        else:
            x = nvalue(other)
            y = self.value * x
            dy = self.deriv.chain(y, x)
        return self.__class__(value=y, deriv=dy)
    __rmul__ = __mul__

    def __sub__(self, other):
        if isinstance(other, forward_value):
            y = self.value - other.value
            dy = self.deriv.fma2(y, (1., self.deriv), (-1., other.deriv))
        else:
            y = self.value - nvalue(other)
            dy = self.deriv.broadcast(y)
        return self.__class__(value=y, deriv=dy)

    def __rsub__(self, other):
        if isinstance(other, forward_value):
            y = other.value - self.value
            dy = self.deriv.fma2(nvalue, (-1., self.deriv), (1., other.deriv))
        else:
            y = nvalue(other) - self.value
            dy = self.deriv.chain(y, -1.)
        return self.__class__(value=y, deriv=dy)

    def __div__(self, other):
        x = self.value
        if isinstance(other, forward_value):
            z = other.value
            #t = np.reciprocal(np.asarray(z,dtype=np.result_type(x,z)))
            #y = x * t
            y = x / z
            t = np.reciprocal(np.asarray(z, dtype=y.dtype))
            dy = self.deriv.fma2(y, (t, self.deriv), (-y * t, other.deriv))
        else:
            z = nvalue(other)
            y = x / z
            t = np.reciprocal(np.asarray(z, dtype=y.dtype))
            dy = self.deriv.chain(y, t)
        return self.__class__(value=y, deriv=dy)

    def __rdiv__(self, other):
        #t = 1. / self.value
        z = self.value
        if isinstance(other, forward_value):
            x = other.value
            y = x / z
            t = np.reciprocal(np.asarray(z, dtype=y.dtype))
            dy = self.deriv.fma2(y, (-y * t, self.deriv), (t, other.deriv))
        else:
            x = nvalue(other)
            y = x / z
            t = np.reciprocal(np.asarray(z, dtype=y.dtype))
            dy = self.deriv.chain(y, -y * t)
        return self.__class__(value=y, deriv=dy)
    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    def _onearg(self, y, dy):
        return self.__class__(value=y, deriv=self.deriv.chain(y, dy))

    def __pos__(self):
        return self

    def __neg__(self):
        y = -self.value
        return self._onearg(y, -1.)

    def apply1(self, func):
        y, (dy_,) = func.f_df((self.value,))
        return self.__class__(value=y, deriv=self.deriv.chain(y, dy_()))

    @classmethod
    def apply(cls, func, args):
        nargs = tuple(map(nvalue, args))
        y, df = func.f_df(nargs)
        terms = tuple((f(), a.deriv)
                      for f, a in zip(df, args) if isinstance(a, forward_value))
        return cls(value=y, deriv=terms[0][1].fma(y, *terms))

    # indexing
    def getitem_array(self, idx):
        x = self.value
        if idx.dtype == np.bool:
            idx = np.arange(len(idx))[idx]
        y = np.take(x, idx)
        n = len(idx)
        if n and np.amin(idx) < 0:
            idx = (idx + n) % n
        return self.__class__(value=y, deriv=self.deriv.getitem_arrayp(y, idx))

    def getitem_slice(self, idx):
        y = np.asarray(self.value[idx])
        return self.__class__(
            value=y, deriv=self.deriv.getitem_general(y, idx))
    getitem_scalar = getitem_slice

    def __getitem__(self, idx):
        if isinstance(idx, np.ndarray):
            if idx.shape:
                return self.getitem_array(idx)
            else:
                return self.getitem_scalar(idx)
        elif isinstance(idx, slice):
            return self.getitem_slice(idx)
        else:
            return self.getitem_scalar(idx)

    # Extended functions
    @classmethod
    def dot_(cls, A, x):
        if isinstance(A, value) or not isinstance(x, value):
            raise NotImplementedError('only supported dot(const,value)')
        A = sparse.csr_matrix.fromcsr(A)
        y = A.dot(x.value)
        dy = x.deriv.rdot(y, A)
        return cls(value=y, deriv=dy)

    @classmethod
    def where(cls, cond, a, b):
        # could be improved and has problems with propagation of NaN
        return np.where(cond, 1., 0.) * a + \
            np.where(np.logical_not(cond), 1., 0.) * b

    def sparsesum(self, terms, **kwargs):
        def wrap(idx, v, y):
            n = len(y)
            M = v.deriv.tovalue().tocsc()
            rows = np.take(idx, M.indices)
            M = sparse.csc_matrix(
                (M.data, rows, M.indptr), shape=(
                    n, M.shape[1]))
            M.sort_indices()
            M = M.tocsr()
            return forward_value(
                value=y, deriv=self.deriv.__class__(mshape=M.shape, M=M))
        return sparsevec_impl.sparsesum(
            terms, hstack=self.hstack, nvalue=nvalue, wrap=wrap, **kwargs)

    def sum(self):
        y = np.sum(self.value)
        dy = self.deriv.sum()
        return self.__class__(value=y, deriv=dy)

    def hstack(self, arrays):
        y = np.hstack(nvalue(a) for a in arrays)

        def deriv(arr):
            if isinstance(arr, forward_value):
                return arr.deriv
            return self.deriv.zero(nvalue(arr))
        dy = self.deriv.vstack(y, (deriv(a) for a in arrays))
        return self.__class__(value=y, deriv=dy)

    @classmethod
    def broadcast_to(cls, self, shape):
        if self.value.shape == shape:
            return self
        return np.ones(shape) * self

    def compare(self, operator, other):
        return getattr(self.value, operator)(other)


def forward_value_isscalar(x):
    return not x.value.shape


def forward_value_nvalue(x):
    return x.value


functions.where.add((object, forward_value, object), forward_value.where)
functions.where.add((object, object, forward_value), forward_value.where)
functions.dot.add((object, forward_value), forward_value.dot_)
functions.sum.add((forward_value,), forward_value.sum)
functions.broadcast_to.add((forward_value, object), forward_value.broadcast_to)
functions.nvalue.add((forward_value, ), forward_value_nvalue)
functions.isscalar.add((forward_value,), forward_value_isscalar)


class forward_value_sparsity(forward_value):
    # inherited where happens to conserve sparsity
    def branch_join(self, cond, iftrue, iffalse):
        t = np.ones_like(cond)
        return self.where(cond, iftrue(t), iffalse(t))


def seed(x, T=forward_value):
    x = np.asarray(x)
    if x.shape:
        return T(value=x, deriv=sparse.sdcsr(mshape=(x.shape[0], x.shape[0])))
    else:
        return T(value=x, deriv=sparse.sdcsr(mshape=(None, None)))


def seed_sparsity(x, T=forward_value_sparsity):
    x = np.asarray(x)
    if x.shape:
        return T(value=x, deriv=sparse.sparsity_csr(
            mshape=(x.shape[0], x.shape[0])))
    else:
        return T(value=x, deriv=sparse.sparsity_csr(mshape=(None, None)))


seed_sparse_gradient = seed

value = forward_value

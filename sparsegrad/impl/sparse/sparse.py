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

"""
This module contains implementation details sparse matrix operations
"""

from packaging.version import Version
import numpy as np
from sparsegrad import impl
__all__ = [
    'sdcsr',
    'sparsity_csr',
    'sample_csr_rows',
    'csr_matrix',
    'csc_matrix']

scipy_sparse = impl.scipy.sparse

index_dtype = scipy_sparse.csr_matrix((0, 0)).indptr.dtype


def sample_csr_rows(csr, rows):
    "return (indptr,ix) such that csr[rows]=csr_matrix((csr.data[ix],csr.indices[ix],indptr))"
    start = np.take(csr.indptr, rows)
    count = np.take(csr.indptr, rows + 1) - start
    indptr = np.empty(len(rows) + 1, dtype=csr.indptr.dtype)
    indptr[0] = 0
    np.cumsum(count, out=indptr[1:])
    ix = np.repeat(start - indptr[:-1], count) + np.arange(indptr[-1])
    return indptr, ix


class csr_matrix_nochecking(scipy_sparse.csr_matrix):
    """

    Subclass of scipy.sparse.csr_matrix which does not perform checking matrix format checking.

    When possible, it avoids going through original csr_matrix constructor which is very slow.
    """

    def __init__(self, *args, **kwargs):
        if not args and not kwargs:
            scipy_sparse.spmatrix.__init__(self)
        elif len(args) == 1 and 'shape' in kwargs and not kwargs.get('copy', False):
            scipy_sparse.spmatrix.__init__(self)
            data, indices, indptr = args[0]
            self.data = np.asarray(data, dtype=kwargs.get('dtype', data.dtype))
            self.indices = np.asarray(indices, dtype=index_dtype)
            self.indptr = np.asarray(indptr, dtype=index_dtype)
            self._shape = kwargs['shape']
        else:
            super(csr_matrix_nochecking, self).__init__(*args, **kwargs)

    @classmethod
    def fromarrays(cls, data, indices, indptr, shape):
        "Optimized matrix constructor from individual CSR arrays, returns csr_matrix((data,indices,indptr),shape=shape)"
        self = cls()
        self.data = data
        self.indices = np.asarray(indices, dtype=index_dtype)
        self.indptr = np.asarray(indptr, dtype=index_dtype)
        self._shape = shape
        return self

    @classmethod
    def fromcsr(cls, csr):
        "Optimized matrix construction from CSR matrix, returns csr_matrix(csr)"
        self = cls()
        if not isinstance(csr, scipy_sparse.csr_matrix):
            csr = csr.tocsr()
        self.data = csr.data
        self.indices = csr.indices
        self.indptr = csr.indptr
        self._shape = csr.shape
        return self

    @classmethod
    def getrows(cls, csr, rows):
        "Optimize row extractor, returns csr[rows]"
        indptr, ix = sample_csr_rows(csr, rows)
        return cls.fromarrays(np.take(csr.data, ix), np.take(
            csr.indices, ix), indptr, (len(rows), csr.shape[1]))

    def check_format(self, full_check=True):
        pass


csr_matrix = csr_matrix_nochecking


class csc_matrix_unchecked(impl.scipy.sparse.csc_matrix):
    """
    Subclass of scipy.sparse.csc_matrix which does not perform checking matrix format checking.
    """

    def check(self, *args):
        pass


csc_matrix = csc_matrix_unchecked


def _stackconv(mat):
    return mat


# Workaround for Scipy versions older than 0.18.0:
# vstack has problems with empty matrices, unless all are converted to csr
if Version(impl.scipy.version.version) < Version('0.18.0'):
    def _stackconv(mat):
        if not mat.shape:
            return csr_matrix((np.atleast_1d(mat), np.zeros(
                1, dtype=index_dtype), np.arange(2, dtype=index_dtype)), shape=(1, 1))
        if not mat.shape[0]:
            return mat.tocsr()
        return mat


def diagonal(x, n):
    "Return n x n matrix diag(x)"
    return csr_matrix.fromarrays(x, np.arange(n), np.arange(n + 1), (n, n))


class sdcsr(object):
    r"""
    Scaled matrix, which is stored as

    .. math::

       s \cdot diag( \mathbf{diag} ) \cdot \mathbf{M}

    where s is scalar, diag is row scaling vector (scalar and vector allowed), and
    M is general part (None is allowed to indicate diagonal matrix).

    mshape stores matrix shape. None for mshape[0] denotes differentation of scalar.
    None for mshape[1] denotes differentiation with repsect to scalar.

    No copies of M, diag are made, therefore they must be constant objects.
    """

    def __init__(self, mshape, s=np.asarray(1), diag=np.asarray(1), M=None):
        self.mshape = mshape
        self.s = s
        self.diag = diag
        #assert M is None or isinstance(M,scipy.sparse.csr_matrix), 'invalid matrix type %r'%M.__class__
        self.M = M
        self._value = None

    def _evaluate(self):
        p = self.s * self.diag
        if self.M is None:
            if self.mshape == (None, None):
                return p
            else:
                n = self.mshape[0]
                # self.mshape[1] must be n
                if not p.shape:
                    v = np.empty(n, dtype=p.dtype)
                    v.fill(p)
                    p = v
                return diagonal(p, n)
        else:
            if p.shape:
                return csr_matrix.fromarrays(self.M.data[:self.M.indptr[-1]] * np.repeat(
                    p, np.diff(self.M.indptr)), self.M.indices, self.M.indptr, self.M.shape)
            else:
                if p != 1.:
                    return csr_matrix.fromarrays(
                        self.M.data * p, self.M.indices, self.M.indptr, self.M.shape)
                else:
                    return self.M

    def tovalue(self):
        "Return this matrix as standard CSR matrix. The result is cached."
        if self._value is None:
            self._value = self._evaluate()
        return self._value

    def getitem_general(self, output, idx):
        "Generate Jacobian matrix for operation output=x[idx], this matrix being Jacobian of x. General version."
        mshape = self._mshape(output)
        if self.M is None:
            v = self.tovalue()[idx]
            return self.__class__(mshape=mshape, M=v)
        else:
            M = self.M[idx]
            if self.diag.shape:
                diag = np.asarray(self.diag[idx])
            else:
                diag = self.diag
            return self.__class__(mshape=mshape, s=self.s, diag=diag, M=M)

    def getitem_arrayp(self, output, idx):
        "Generate Jacobian matrix for operation output=x[idx], this matrix being Jacobian of x. idx is array with all entries positive."
        if self.diag.shape:
            p = self.s * np.take(self.diag, idx)
        else:
            p = self.s * self.diag
        n = len(idx)
        mshape = self._mshape(output)
        if self.M is None:
            P = csr_matrix.fromarrays(
                np.ones(
                    n, dtype=p.dtype), idx, np.arange(
                    len(idx) + 1), mshape)
            return self.new(mshape, p, P)
        else:
            return self.new(mshape, p, csr_matrix.getrows(self.M, idx))
        #
        #P = csr_matrix.fromarrays(np.ones(n,dtype=dtype),idx,np.arange(len(idx)+1),mshape)
        # if self.M is not None:
        #    return self.new(mshape,p,P * self.M)
        # else:
        #    return self.new(mshape,p,P)

    @classmethod
    def new(cls, mshape, diag=np.asarray(1), M=None):
        "Alternative constructor, which checks dimension of diag and assigns to scalar/vector part properly"
        if diag.shape:
            return cls(mshape, diag=diag, M=M)
        else:
            return cls(mshape, s=diag, M=M)

    def zero(self, output):
        "Return empty Jacobian, which would result from output=0*x, this matrix being Jacobian of x."
        mshape = self._mshape(output)
        if mshape == (None, None):
            return self.__class__(mshape, s=np.asarray(0.))
        else:
            n, m = mshape
            if n is None:
                n = 1
            if m is None:
                m = 1
            return self.__class__(mshape, M=csr_matrix((n, m)))

    def _mshape(self, output):
        if output.shape:
            return (output.shape[0], self.mshape[1])
        else:
            return (None, self.mshape[1])

    def _broadcast(self, n):
        # Workaround for numpy bug. Sometimes scalar + [scalar] is returned as
        # scalar.
        if n is None:
            n = 1

        B = csr_matrix.fromarrays(
            np.ones(n), np.zeros(n), np.arange(
                n + 1), (n, 1))
        if self.M is None:
            return B
        else:
            return B * self.M

    def broadcast(self, output):
        "Return broadcast matrix :math:`\mathbf{B_{output}}` for broadcasting x to output, this matrix being Jacobian of x"
        mshape = self._mshape(output)
        if mshape[0] == self.mshape[0]:
            return self
        return self.__class__(
            mshape, s=self.s, diag=self.diag, M=self._broadcast(mshape[0]))

    def chain(self, output, x):
        r"""Apply chain rule for elementwise operation

        Jacobian of elementwise operation is :math:`diag(\mathbf{x})`. Return :math:`diag(\mathbf{B_{output}} \cdot \mathbf{x})\cdot\mathbf{B_{output}}\cdot\mathbf{self}`
        """
        diag = (self.s * x) * self.diag
        mshape = self._mshape(output)
        if mshape[0] != self.mshape[0]:
            M = self._broadcast(mshape[0])
        else:
            M = self.M
        return self.new(mshape, diag, M)

    @classmethod
    def fma(cls, output, *terms):
        r"""
        Apply chain rule to elementwise functions

        Returns sum(d.chain(output,x) for x,d in terms)
        """
        xfirst, dfirst = terms[0]
        if output.shape:
            mshape = (output.shape[0], dfirst.mshape[1])
        else:
            mshape = (None, dfirst.mshape[1])
        M = dfirst.M
        if all(d.M is M for x, d in terms[1:]):
            diag = sum((d.s * x) * d.diag for x, d in terms)
            if mshape[0] != dfirst.mshape[0]:
                M = dfirst._broadcast(mshape[0])
            return cls.new(mshape, diag, M)
        v = dfirst.chain(output, xfirst).tovalue()
        for x, d in terms[1:]:
            v = v + d.chain(output, x).tovalue()
        return cls(mshape, M=v)

    fma2 = fma

    def __add__(self, other):
        if other.M is self.M:
            return self.new(self.mshape, self.s * self.diag +
                            other.s * other.diag, self.M)
        else:
            return self.__class__(
                self.mshape, M=self.tovalue() + other.tovalue())

    def __repr__(self):
        return '<sdcsr mshape=%r s=%r diag=%r M=%r>' % (
            self.mshape, self.s, self.diag, self.M)

    def rdot(self, y, other):
        r"Return Jacobian of :math:`\mathbf{y} = \mathbf{other} \cdot \mathbf{self}`, with :math:`\cdot` denoting matrix multiplication."
        d = csr_matrix.fromcsr(other) * self.tovalue()
        if d.shape:
            return self.__class__(d.shape, M=d)
        else:
            return self.__class__((None, self.mshape[1]), s=d)

    def sum(self):
        "Return Jacobian of y=sum(x), this matrix being Jacobian of x"
        v = self.tovalue()
        mshape = (None, self.mshape[1])
        if v.shape:
            return self.__class__(
                mshape, M=csr_matrix(self.tovalue().sum(axis=0)))
        else:
            return self.__class__(mshape, s=v)

    def vstack(self, output, parts):
        "Return Jacobian of output=hstack(parts)"
        mshape = self._mshape(output)
        M = scipy_sparse.vstack(_stackconv(p.tovalue()) for p in parts).tocsr()
        return self.new(mshape, M=M)


class sparsity_csr(sdcsr):
    "This is a variant of matrix only propagating sparsity information"

    def __init__(self, mshape, s=None, diag=None, M=None):
        if M is not None:
            if not isinstance(M, scipy_sparse.csr_matrix):
                M = csr_matrix(M)
            M.sort_indices()
            M.data.fill(1)
        super(sparsity_csr, self).__init__(mshape, M=M)

    def chain(self, output, x):
        return self.broadcast(output)

    @classmethod
    def fma(cls, output, *terms):
        xfirst, dfirst = terms[0]
        if output.shape:
            mshape = (output.shape[0], dfirst.mshape[1])
        else:
            mshape = (None, dfirst.mshape[1])
        M = dfirst.M
        if all(d.M is M for x, d in terms[1:]):
            if mshape[0] != dfirst.mshape[0]:
                M = dfirst._broadcast(mshape[0])
            return cls.new(mshape, M=M)
        v = dfirst.chain(output, xfirst).tovalue()
        for x, d in terms[1:]:
            v = v + d.chain(output, x).tovalue()
        return cls(mshape, M=v)

    fma2 = fma

    def rdot(self, y, other):
        # must convert other to sparsity pattern, otherwise cancellation could
        # occur
        x = csr_matrix(other).sorted_indices()
        x.data.fill(1.)
        d = x.dot(self.tovalue())
        if d.shape:
            return self.__class__(d.shape, M=d)
        else:
            return self.__class__((None, self.mshape[1]), s=d)

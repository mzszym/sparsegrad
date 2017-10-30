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

from packaging.version import Version
import numpy as np
import scipy.sparse
__all__ = ['sdcsr', 'sparsity_csr', 'sample_csr_rows']

index_dtype = scipy.sparse.csr_matrix((0, 0)).indptr.dtype


def sample_csr_rows(csr, rows):
    "return (indptr,ix) such that csr[rows]=csr_matrix((csr.data[ix],csr.indices[ix],indptr))"
    start = np.take(csr.indptr, rows)
    count = np.take(csr.indptr, rows + 1) - start
    indptr = np.empty(len(rows) + 1, dtype=csr.indptr.dtype)
    indptr[0] = 0
    np.cumsum(count, out=indptr[1:])
    ix = np.repeat(start - indptr[:-1], count) + np.arange(indptr[-1])
    return indptr, ix


class csr_matrix_nochecking(scipy.sparse.csr_matrix):
    def __init__(self, *args, **kwargs):
        if not args and not kwargs:
            scipy.sparse.spmatrix.__init__(self)
        elif len(args) == 1 and 'shape' in kwargs and not kwargs.get('copy', False):
            scipy.sparse.spmatrix.__init__(self)
            data, indices, indptr = args[0]
            self.data = np.asarray(data, dtype=kwargs.get('dtype', data.dtype))
            self.indices = np.asarray(indices, dtype=index_dtype)
            self.indptr = np.asarray(indptr, dtype=index_dtype)
            self.shape = kwargs['shape']
        else:
            super(csr_matrix_nochecking, self).__init__(*args, **kwargs)

    @classmethod
    def fromarrays(cls, data, indices, indptr, shape):
        self = cls()
        self.data = data
        self.indices = np.asarray(indices, dtype=index_dtype)
        self.indptr = np.asarray(indptr, dtype=index_dtype)
        self.shape = shape
        return self

    @classmethod
    def fromcsr(cls, csr):
        self = cls()
        if not isinstance(csr, scipy.sparse.csr_matrix):
            csr = csr.tocsr()
        self.data = csr.data
        self.indices = csr.indices
        self.indptr = csr.indptr
        self.shape = csr.shape
        return self

    @classmethod
    def getrows(cls, csr, rows):
        indptr, ix = sample_csr_rows(csr, rows)
        return cls.fromarrays(np.take(csr.data, ix), np.take(
            csr.indices, ix), indptr, (len(rows), csr.shape[1]))

        # n=len(rows)
        # m=csr.shape[1]
        # if not n:
        #    return cls.fromarrays(csr.data[:0].copy(),csr.indices[:0].copy(),csr.indptr[:1].copy(),(0,m))
        # if n == 1:
        #    row,=rows
        #    i0,i1=csr.indptr[row],csr.indptr[row+1]
        # return
        # cls.fromarrays(csr.data[i0:i1].copy(),csr.indices[i0:i1].copy(),np.asarray([0,i1-i0],dtype=csr.indptr.dtype),(1,m))

    def check_format(self, full_check=True):
        pass


csr_matrix = csr_matrix_nochecking


def _stackconv(mat):
    return mat


# Workaround for Scipy versions older than 0.18.0:
# vstack has problems with empty matrices, unless all are converted to csr
if Version(scipy.version.version) < Version('0.18.0'):
    def _stackconv(mat):
        if not mat.shape:
            return csr_matrix((np.atleast_1d(mat), np.zeros(
                1, dtype=index_dtype), np.arange(2, dtype=index_dtype)), shape=(1, 1))
        if not mat.shape[0]:
            return mat.tocsr()
        return mat


def diagonal(x, n):
    return csr_matrix.fromarrays(x, np.arange(n), np.arange(n + 1), (n, n))


class sdcsr(object):
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
        if self._value is None:
            self._value = self._evaluate()
        return self._value

    def getitem_general(self, output, idx):
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
        "alternative constructor, which checks diagonal part and assigns to scalar/vector part properly"
        if diag.shape:
            return cls(mshape, diag=diag, M=M)
        else:
            return cls(mshape, s=diag, M=M)

    def zero(self, output):
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
        B = csr_matrix.fromarrays(
            np.ones(n), np.zeros(n), np.arange(
                n + 1), (n, 1))
        if self.M is None:
            return B
        else:
            return B * self.M

    def broadcast(self, output):
        mshape = self._mshape(output)
        if mshape[0] == self.mshape[0]:
            return self
        return self.__class__(
            mshape, s=self.s, diag=self.diag, M=self._broadcast(mshape[0]))

    def chain(self, output, x):
        "return DIAG(B_output(x)).B_output'.self"
        diag = (self.s * x) * self.diag
        mshape = self._mshape(output)
        if mshape[0] != self.mshape[0]:
            M = self._broadcast(mshape[0])
        else:
            M = self.M
        return self.new(mshape, diag, M)

    @classmethod
    def fma(cls, output, *terms):
        "return sum(d.chain(output,x) for x,d in terms)"
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
        d = csr_matrix.fromcsr(other) * self.tovalue()
        if d.shape:
            return self.__class__(d.shape, M=d)
        else:
            return self.__class__((None, self.mshape[1]), s=d)

    def sum(self):
        v = self.tovalue()
        mshape = (None, self.mshape[1])
        if v.shape:
            return self.__class__(
                mshape, M=csr_matrix(self.tovalue().sum(axis=0)))
        else:
            return self.__class__(mshape, s=v)

    def vstack(self, output, parts):
        mshape = self._mshape(output)
        M = scipy.sparse.vstack(_stackconv(p.tovalue()) for p in parts).tocsr()
        return self.new(mshape, M=M)


class sparsity_csr(sdcsr):
    def __init__(self, mshape, s=None, diag=None, M=None):
        if M is not None:
            if not isinstance(M, scipy.sparse.csr_matrix):
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

# -*- coding: utf-8; -*-
#
# sparsegrad - automatic calculation of sparse gradient
# Copyright (C) 2016 Marek Zdzislaw Szymanski
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
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
import scipy.sparse

from .utils import *

class MatrixSum:
    def __init__(self,factory):
        self.s=None
        self.M=factory
    def __call__(self,u,copy=False):
        if self.s is None:
            if copy:
                self.s = u.copy()
            else:
                self.s = u
        else:
            self.s += u
    def value(self):
        if isscalar(self.s):
            return self.s
        else:
            return scipy.sparse.csr_matrix(self.s)

class MatrixFactory(object):
    def identityMatrix(self,n,dtype):
        "Create square identity matrix nxn"
        raise NotImplementedError()
    def diagonalMatrix(self,n,xf):
        "Create diagonal matrix nxn, with diagonal xf()"
        raise NotImplementedError()
    def zeroMatrix(self,shape,dtype):
        "Create empty matrix with given shape"
        raise NotImplementedError()
    def chain(self,a,b):
        "Derivative chain"
        raise NotImplementedError()
    def dot(self,a,b):
        "Matrix multiplication"
        raise NotImplementedError()
    def vstack(self,*args):
        "Vertically stack matrices"
        raise NotImplementedError()
    def accumulator(self):
        "Return accumulator"
        return MatrixSum(self)
    def eforward(self,value,a,da,dff):
        "Forward propagate elementwise operation. dff() returns dvalue/da as scalar or 1-d vector, which is diagonal"
        raise NotImplementedError()

class StandardFactory(MatrixFactory):
    def identityMatrix(self,n,dtype):
        return scipy.sparse.csr_matrix((np.ones(n,dtype=dtype),np.arange(n,dtype='i4'),np.arange(n+1,dtype='i4')),shape=(n,n))
    def diagonalMatrix(self,n,xf):
        x=xf()
        if isscalar(x):
            xa=np.empty(n,dtype=np.asarray(x).dtype)
            xa.fill(x)
        else:
            xa=x
        return scipy.sparse.csr_matrix((xa,np.arange(n,dtype='i4'),np.arange(n+1,dtype='i4')),shape=(n,n))
    def zeroMatrix(self,shape,dtype):
        return scipy.sparse.csr_matrix(shape,dtype=dtype)
    def chain(self,a,b):
        if isscalar(a):
            return a*b
        return self.dot(a,b)
    def dot(self,a,b):
        return a.dot(b)
    def vstack(self,*args):
        return scipy.sparse.vstack(*args)
    def broadcastMatrix01d(self,n,dtype):
        return scipy.sparse.csr_matrix((np.ones(n,dtype=dtype),np.zeros(n,dtype='i4'),np.arange(n+1,dtype='i4')),shape=(n,1))
    def eforward(self,value,a,da,dff):
        if isscalar(a) and not isscalar(value):
            da=self.dot(self.broadcastMatrix01d(len1d(value),value.dtype),da)
        if isscalar(value):
            df=dff()
        else:
            df=self.diagonalMatrix(len1d(value),dff)
        return self.chain(df,da)

class sparsematrix(object):
    def __init__(self,shape):
        self.T=socsr
        self.shape=shape
        self.ndim=2       
    def __getitem__(self,idx):
        return self.T(self.tocsr().__getitem__(idx))
    def sum(self,*args,**kwargs):
        return self.T(scipy.sparse.csr_matrix(self.tocsr().sum(*args,**kwargs)))
    
class socsr(sparsematrix):
    "Sparsity-only CSR"
    def __init__(self,m):
        "Initialize with matrix m"
        sparsematrix.__init__(self,m.shape)
        if not hasattr(m,'indices') or not hasattr(m,'indptr'):
            if hasattr(m,'tocsr'):
                m=m.tocsr()
            else:
                m=scipy.sparse.csr_matrix(m)
        self.indices=np.asarray(m.indices,dtype='i4')
        self.indptr=np.asarray(m.indptr,dtype='i4')
        
    def tocsr(self):
        return scipy.sparse.csr_matrix((np.ones(len(self.indices)),self.indices,self.indptr),shape=self.shape)

class soidentity(sparsematrix):
    def __init__(self,n):
        sparsematrix.__init__(self,(n,n))
    def tocsr(self):
        return standard.identityMatrix(self.shape[0],np.double)

class sozero(sparsematrix):
    def __init__(self,shape):
        sparsematrix.__init__(self,shape)
    def tocsr(self):
        return standard.zeroMatrix(self.shape,np.double)

class sosum(object):
    def __init__(self,factory):
        self.s=None
        self.M=factory
    def __call__(self,u,copy=False):
        u=u.tocsr()
        if self.s is None:
            if copy:
                self.s = u.copy()
            else:
                self.s = u
        else:
            self.s += u
    def value(self):
        return socsr(self.s)

class SparsityFactory(MatrixFactory):
    def identityMatrix(self,n,dtype):
        return soidentity(n)
    def diagonalMatrix(self,n,xf):
        return soidentity(n)
    def zeroMatrix(self,shape,dtype):
        return sozero(shape)
    def chain(self,a,b):
        if isscalar(a):
            return b
        if isscalar(b):
            return a
        return self.dot(a,b)
    def dot(self,a,b):
        assert a.shape[1]==b.shape[0], "incompatible matrix dimensions"
        shape=(a.shape[0],b.shape[1])
        if isinstance(a,sozero) or isinstance(b,sozero):
            return self.zeroMatrix(shape,None)
        if isinstance(a,soidentity):
            return b
        if isinstance(b,soidentity):
            return a
        if not isinstance(a,socsr):
            a=socsr(a)
        return socsr(a.tocsr().dot(b.tocsr()))
    def vstack(self,*args):
        def csrarg(x):
            return x.tocsr()
        l,=args
        return socsr(scipy.sparse.vstack(map(csrarg,l)))
    def accumulator(self):
        return sosum(self)
    def broadcastMatrix01d(self,n,dtype):
        return scipy.sparse.csr_matrix((np.ones(n,dtype=dtype),np.zeros(n,dtype='i4'),np.arange(n+1,dtype='i4')),shape=(n,1))
    def eforward(self,value,a,da,dff):
        if isscalar(a) and not isscalar(value):
            return self.chain(standard.broadcastMatrix01d(len1d(value),np.double),da)
        else:
            return da

standard=StandardFactory()
sparsity=SparsityFactory()



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
import scipy.version
from packaging.version import Version

from .utils import *

class MatrixSum(object):
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

    # Workaround for Scipy versions older than 0.15.0:
    # += is not implemented for sparse matrices
    if Version(scipy.version.version)<Version('0.15.0'):
        def __call__(self,u,copy=False):
            if self.s is None:
                if copy:
                    self.s = u.copy()
                else:
                    self.s = u
            else:
                self.s = self.s + u

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
    def vstack(self,args):
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
    def vstack(self,args):
        return scipy.sparse.vstack(args)
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

    # Workaround for Scipy versions older than 0.18.0:
    # vstack has problems with empty matrices, unless all are converted to csr
    if Version(scipy.version.version)<Version('0.18.0'):
        def vstack(self,args):
            return scipy.sparse.vstack([a.tocsr() for a in args])
    
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

class sosum(MatrixSum):
    def __call__(self,u,**kwargs):
        MatrixSum.__call__(self,u.tocsr(),**kwargs)
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
    def vstack(self,args):
        def csrarg(x):
            return x.tocsr()
        return socsr(scipy.sparse.vstack(map(csrarg,args)))
    def accumulator(self):
        return sosum(self)
    def broadcastMatrix01d(self,n,dtype):
        return scipy.sparse.csr_matrix((np.ones(n,dtype=dtype),np.zeros(n,dtype='i4'),np.arange(n+1,dtype='i4')),shape=(n,1))
    def eforward(self,value,a,da,dff):
        if isscalar(a) and not isscalar(value):
            return self.chain(standard.broadcastMatrix01d(len1d(value),np.double),da)
        else:
            return da

class csr_nocheck(scipy.sparse.csr_matrix):
    "Scipy CSR matrix where check_format does nothing"
    def check_format(self,full_check=False):
        pass

class sdcsr(scipy.sparse.spmatrix):
    """
    Scaled CSR matrix, which can be written as product s diag(d) M,
    where s is scalar, d is scalar or vector, and M is a 
    general matrix (None denotes identity matrix).
    
    This allows to optimize typical operations:
    
    - multiplication by scalar: 
    k (s diag(d) M) = (ks) diag(d)
    
    - multiplication by diagonal matrix: 
    diag(u) (s diag(d) M) = k diag(d u) M
    
    - construction of diagonal matrix
    
    - summing, if general parts are the same:
    (s_1 diag(d_1) M) + (s_2 diag(d_2) M) = diag(s_1*d_1+s_2*d_2) M
    
    @TODO Think how to support matrices which are not CSR.
    """
    def __init__(self,shape,scalar=1.,diagonal=1.,general=None):
        scipy.sparse.spmatrix.__init__(self)
        self.shape=shape
        self.ndim=2
        self.format='und'
        
        self.scalar=scalar
        self.diagonal=diagonal
        self.general=general
        
    @property
    def dtype(self):
        # Note: This is pretty slow
        t=np.common_type(np.asarray(self.scalar),np.asarray(self.diagonal))
        if not self.general:
            return t
        else:
            return np.find_common_type([t,self.general.dtype])
    
    def dot(self,other):
        "Matrix multiplication"
        assert self.shape[1]==other.shape[0]
        if isinstance(other,sdcsr):
            scalar=self.scalar*other.scalar
            diagonal=self.diagonal*other.diagonal
            gother=other.general
        else:
            scalar=self.scalar
            diagonal=self.diagonal
            gother=other
        if self.general is None:
            general=gother
        elif gother is None:
            general=self.general
        else:
            general=self.general.dot(gother)
        return sdcsr((self.shape[0],other.shape[1]),scalar=scalar,diagonal=diagonal,general=general)
    
    def __mul__(self,a):
        "Multiplication by scalar"
        assert np.ndim(a)==0
        return sdcsr(self.shape,scalar=self.scalar*a,diagonal=self.diagonal,general=self.general)
    
    def tocsr(self,copy=False):
        "Conversion to CSR format"
        k=self.scalar*self.diagonal
        if np.ndim(k)==0:
            k=np.repeat(k,self.shape[0])
        if self.general is None:
            indptr=np.arange(self.shape[0]+1,dtype='i4')
            if self.shape[1]<self.shape[0]:
                indptr[self.shape[1]:]=self.shape[1]
                indices=np.arange(self.shape[1],dtype='i4')
                k=k[:self.shape[1]]
            else:
                indices=np.arange(self.shape[0],dtype='i4')
            return csr_nocheck((k,indices,indptr),shape=self.shape)
        else:
            s=np.repeat(k,np.diff(self.general.indptr))
            return csr_nocheck((self.general.data[:len(s)]*s,
                                self.general.indices[:len(s)],self.general.indptr),
                               shape=self.shape)

    def tocoo(self,copy=False):
        return self.tocsr().tocoo()
        
    def __iadd__(self,other):
        "Inplace addition, supports only sdcsr with same general part"
        assert isinstance(other,sdcsr) and other.shape==self.shape
        assert other.general is self.general
        self.diagonal=self.scalar*self.diagonal+other.scalar*other.diagonal
        self.scalar=1.
        return self
        
    def __add__(self,other):
        "Addition, general purpose"
        assert self.shape==other.shape
        if isinstance(other,sdcsr) and other.general is self.general:
            return sdcsr(self.shape,scalar=1.,diagonal=self.scalar*self.diagonal+other.scalar*other.diagonal,general=self.general)
        else:
            return self.tocsr()+other
        
    def __getitem__(self,idx):
        return self.tocsr().__getitem__(idx)
    
    def sum(self,**kwargs):
        return csr_nocheck(self.tocsr().sum(**kwargs))
                
    __radd__=__add__
    __rmul__=__mul__
        
    def __repr__(self):
        return 'sdcsr'
        
    def __str__(self):
        return 'sdcsr'
    
    def copy(self):
        return sdcsr(self.shape,scalar=self.scalar,diagonal=self.diagonal,general=self.general)

class sdsum:
    """
    Accumulator accelerating operations involving sdcsr, taking advantage of
    
    (s_1 diag(d_1) M) + (s_2 diag(d_2) M) = diag(s_1*d_1+s_2*d_2) M
    
    All sdcsr matrices are accumulated by general part M (which is assumed
    immutable), and general matrices are summed directly. 
    
    At the end, if there was only one general matrix M involved, result is
    returned as sdcsr. Otherwise, sdcsr terms are converted to general matrices
    and summed.
    
    Summing order is preserved as follows: firstly, general matrices in order
    of appearance. Then, sdcsr in order of appearance of general parts.
    """
    def __init__(self,factory):
        self.acc={}
        self.general=MatrixSum(factory)
        self.order=[]
        self.M=factory
        
    def __call__(self,u,copy=False):
        if isinstance(u,sdcsr):
            k=id(u.general)
            if k in self.acc:
                self.acc[k]+=u
            else:
                if copy:
                    self.acc[k]=u.copy()
                else:
                    self.acc[k]=u
                self.order.append(k)
        else:
            self.general(u,copy=copy)
    
    def value(self):
        # No need to make CSR?
        if self.general.s is None and len(self.order)==1:
            return self.acc[self.order[0]]        
        # Accumulate final CSR
        for k in self.order:
            self.general(self.acc[k].tocsr(),copy=False)
        self.acc={}
        self.order=[]
        return self.general.value()

class sdcsrFactory(StandardFactory):
    "MatrixFactory involving sdcsr in computation"
    def identityMatrix(self, n, dtype):
        return sdcsr((n,n),scalar=np.ones((),dtype=dtype))
    def diagonalMatrix(self, n, xf):
        return sdcsr((n,n),diagonal=xf())
    def zeroMatrix(self,shape,dtype):
        return sdcsr(shape,scalar=np.zeros((),dtype=dtype))
    def accumulator(self):
        return sdsum(self)
        
#standard=StandardFactory()
standard=sdcsrFactory()
sparsity=SparsityFactory()


run_csr_checker=False        

def disable_checking_csr():
    global csr_checker
    csr_checker=scipy.sparse.csr_matrix.check_format
    def check_format(obj,full_check=False):
        if run_csr_checker:
            csr_checker(obj,full_check=full_check)
    scipy.sparse.csr_matrix.check_format=check_format

disable_checking_csr()

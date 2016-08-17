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

from .utils import *

class function(object):
    "Basic function"
    def __init__(self,evaluate=None,**kwargs):
        if evaluate != None:
            self.evaluate=evaluate
        for k,f in kwargs.items():
            setattr(self,k,f)

    def evaluate(self,*args,**kwargs):
        """
        Evaluate function
        """
        raise NotImplementedError()

    def forward(self,args,value,dargs,accumulate,**kwargs):
        """
        Forward propagate 1st order derivative.
        value = f(*args,**kwargs)
        
        dargs[i] is args[i]' or None for zero
        
        For each argument
        if dargs[i] is not None: accumulate(f'arg.args(i))
        """
        raise NotImplementedError()

class ufunc(function):
    """
    Numpy ufunc. Currently with the following restrictions:
    - only scalars and one dimensional vectors supported as arguments
    - keyword arguments are not supported
    - only one output is supported
    """
    def __init__(self,func,deriv):
        self.func=func
        self.deriv=deriv
        if hasattr(func,'nin'):
            self.nin=func.nin
        else:
            self.nin=None
        if hasattr(func,'nout'):
            assert func.nout == 1, "only one output supported"
        
    def evaluate(self,*args,**kwargs):
        assert not kwargs
        return self.func(*args)
    
    def forward(self,args,value,dargs,accumulate,**kwargs):
        for a,da,dff in zip(args,dargs,self.deriv(args,value)):
            if da is None or dff is None:
                continue
            accumulate(accumulate.M.eforward(value,a,da,dff))

    def __repr__(self):
        return self.func.__repr__()

from functools import partial
import numpy as np

known_ufuncs={}
known_funcs={}

def instance(c):
    u=c()
    assert c.__name__ not in known_funcs
    known_funcs[c.__name__]=u
    return u

def uderiv(func):
    def apply(deriv):
        name=func.__name__
        assert name not in known_funcs
        f=ufunc(func,deriv)
        known_ufuncs[name]=f
        known_funcs[name]=f
        return f
    return apply

@uderiv(np.add)
def add(args,value):
    yield lambda:1.
    yield lambda:1.

@uderiv(np.subtract)
def subtract(args,value):
    yield lambda:1.
    yield lambda:-1.

@uderiv(np.multiply)
def multiply(args,value):
    yield lambda:args[1]
    yield lambda:args[0]

def _reciprocal(x):
    # Problem with numpy reciprocal: np.reciprocal(2)==0
    return 1./x
    
@uderiv(np.divide)
def divide(args,value):
    a,b=args
    t=_reciprocal(b)
    yield lambda:t
    yield lambda:-a*t**2

@uderiv(np.power)
def power(args,value):
    a,b=args
    yield lambda:b*a**(b-1.)
    yield lambda:value*np.log(a)
true_divide=divide

@uderiv(np.negative)
def negative(args,value):
    yield lambda:-1.

@uderiv(np.abs)
def abs(args,value):
    yield lambda:np.sign(args[0])
absolute=abs

@uderiv(np.reciprocal)
def reciprocal(args,value):
    yield lambda:-value**2

@uderiv(np.exp)
def exp(args,value):
    yield lambda:value

@uderiv(np.log)
def log(args,value):
    yield lambda:_reciprocal(args[0])

@uderiv(np.sqrt)
def sqrt(args,value):
    yield lambda:0.5/value

@uderiv(np.square)
def square(args,value):
    yield lambda:2.*value

@uderiv(np.sin)
def sin(args,value):
    yield lambda:np.cos(args[0])

@uderiv(np.cos)
def cos(args,value):
    yield lambda:-np.sin(args[0])

@uderiv(np.tan)
def tan(args,value):
    yield lambda:value**2+1.

@uderiv(np.arcsin)
def arcsin(args,value):
    yield lambda:_reciprocal(np.sqrt(1.-args[0]**2))

@uderiv(np.arccos)
def arccos(args,value):
    yield lambda:-_reciprocal(np.sqrt(1.-args[0]**2))

@uderiv(np.arctan)
def arctan(args,value):
    yield lambda:_reciprocal(1.+np.square(args[0]))

@uderiv(np.sinh)
def sinh(args,value):
    yield lambda:np.cosh(args[0])

@uderiv(np.cosh)
def cosh(args,value):
    yield lambda:np.sinh(args[0])

@uderiv(np.tanh)
def tanh(args,value):
    yield lambda:-np.square(value)+1.

@uderiv(np.arcsinh)
def arcsinh(args,value):
    yield lambda:_reciprocal(np.sqrt(np.square(args[0])+1.))

@uderiv(np.arccosh)
def arccosh(args,value):
    yield lambda:_reciprocal(np.sqrt(np.square(args[0])-1.))

@uderiv(np.arctanh)
def arctanh(args,value):
    yield lambda:_reciprocal(-np.square(args[0])+1.)    

@instance
class getitem(function):
    def evaluate(self,*args,**kwargs):
        assert not kwargs
        x,idx=args
        return x[idx]

    def forward(self,args,value,dargs,accumulate,**kwargs):
        x,idx=args
        dx,didx=dargs
        assert didx is None
        accumulate(dx[idx])

@instance
class stack(function):
    def evaluate(self,*args,**kwargs):
        assert not kwargs
        return np.hstack(args)

    def forward(self,args,value,dargs,accumulate,**kwargs):
        sparse=accumulate.M
        assert not kwargs
        if not any(a is not None for a in dargs):
            return None
        def width(d):
            if np.ndim(d) == 2:
                return np.shape(d)[1]
            elif isscalar(d):
                return 1
            else:
                raise RuntimeError('unsupported shape')
        m=max([width(d) for d in dargs if d is not None])
        def marg(d):
            a,da=d
            if isscalar(a):
                n=1
                dtype=np.asarray(a).dtype
            else:
                n=len1d(a)
                dtype=a.dtype
            if da is None:
                return sparse.zeroMatrix((n,m),dtype)
            if isscalar(da):
                return sparse.diagonalMatrix(n,lambda:da)
            return da
        accumulate(sparse.vstack(list(map(marg,zip(args,dargs)))))

@instance
class sum(function):
    def evaluate(self,*args,**kwargs):
        assert not kwargs
        return np.sum(args)

    def forward(self,args,value,dargs,accumulate,**kwargs):
        assert not kwargs
        d,=dargs
        if d is not None:
            if isscalar(d):
                accumulate(d)
            else:
                accumulate(d.sum(axis=0))

@instance
class dot(function):
    def evaluate(self,*args,**kwargs):
        assert not kwargs
        a,b=args
        return a.dot(b)

    def forward(self,args,value,dargs,accumulate,**kwargs):
        assert not kwargs
        a,b=args
        da,db=dargs
        assert da is None
        accumulate(accumulate.M.dot(a,db))

@instance
class where(ufunc):
    def __init__(self):
        ufunc.__init__(self,np.where,self._deriv)

    def _deriv(self,args,value):
        cond,trueval,falseval=args
        yield None
        yield lambda:np.where(cond,1.,0.)
        yield lambda:np.where(np.logical_not(cond),1.,0.)


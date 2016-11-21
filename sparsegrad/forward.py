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

from .expr import expr_base
import numpy as np
from . import sparse

from .utils import *

def _operators():
    def _():
        for op in [ '__lt__','__le__','__eq__','__ne__','__ge__','__gt__' ]:
            yield 'def %s(self,other): return self.value.%s(other)'%(op,op)
    return '\n'.join(_())

class value(expr_base):
    def __init__(self,value,dvalue=None,M=None):
        expr_base.__init__(self,expr)
        self.value=value
        self.dvalue=dvalue
        self.M=M

    exec(_operators())

    @property
    def gradient(self):
        return self.dvalue

    @property
    def sparsity(self):
        return sparse.socsr(self.dvalue)

class expr(value):
    def __init__(self,f,*args,**kwargs):
        argsv=[]
        dargsv=[]
        M=None
        has_deriv=False
        for a in args:
            if isinstance(a,value):
                assert M is None or M==a.M, "inconsistent matrix factories"
                M=a.M
                argsv.append(a.value)
                dargsv.append(a.dvalue)
                has_deriv=True
            else:
                argsv.append(a)
                dargsv.append(None)
        y=f.evaluate(*argsv,**kwargs)
        dy=None
        if has_deriv:
            s=M.accumulator()
            f.forward(argsv,y,dargsv,s,**kwargs)
            dy=s.value()
        value.__init__(self,y,dvalue=dy,M=M)

def seed(x,M=sparse.standard):
    "Return seed for x"
    if isscalar(x):
        return value(x,1.,M=M)
    else:
        return value(x,M.identityMatrix(len1d(x),dtype=x.dtype),M=M)

def seed_sparsity(x):
    "Return seed for sparsity pattern calculation"
    return seed(x,M=sparse.sparsity)

def seed_sparse_gradient(x):
    "Return seed for sparse gradient calculation"
    return seed(x,M=sparse.standard)

def nvalue(x):
    "Return numerical value of x"
    if isinstance(x,value):
        return x.value
    else:
        return x

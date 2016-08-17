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

from . import func

def _genu():
    def _():
        for name,f in func.known_ufuncs.items():
            if f.nin == 1:
                yield "def %s(self): return self.T(func.%s,self)"%(name,name)
    return "\n".join(_())
def _geng():
    def _():
        for name in func.known_funcs.keys():
            yield "%s=wrapped_func(func.%s)"%(name,name)
    return "\n".join(_())       

class expr_base(object):
    __array_priority__ = 100
    
    def __init__(self,T):
        self.T=T

    def __add__(self,other): return self.T(func.add,self,other)
    def __radd__(self,other): return self.T(func.add,other,self)
    def __sub__(self,other): return self.T(func.subtract,self,other)
    def __rsub__(self,other): return self.T(func.subtract,other,self)
    def __mul__(self,other): return self.T(func.multiply,self,other)
    def __rmul__(self,other): return self.T(func.multiply,other,self)
    def __div__(self,other): return self.T(func.divide,self,other)
    def __rdiv__(self,other): return self.T(func.divide,other,self)
    def __truediv__(self,other): return self.T(func.divide,self,other)
    def __rtruediv__(self,other): return self.T(func.divide,other,self)
    def __pow__(self,other): return self.T(func.power,self,other)
    def __rpow__(self,other): return self.T(func.power,other,self)
    def __pos__(self): return self
    def __neg__(self): return self.T(func.negative,self)

    def __getitem__(self,idx):
        return self.T(func.getitem,self,idx)

    def __abs__(self):
        return self.T(func.abs,self)

    # ufuncs
    exec(_genu())

class wrapped_func:
    def __init__(self,f):
        self.f=f

    def __call__(self,*args,**kwargs):
        for a in args:
            if isinstance(a,expr_base):
                return a.T(self.f,*args,**kwargs)
        return self.f.evaluate(*args,**kwargs)

# non ufuncs
exec(_geng())

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
from sparsegrad import forward
from sparsegrad.expr import dot
from test_basic import check_general

def check_dot1(x,M):
    f=lambda x:dot(M,x)
    check_general(None,x,f,M)

def check_dot2(x,M):
    g=lambda x:5*x**3+3*x
    dg=g(forward.seed(x)).dvalue
    f=lambda x:dot(M,g(x))
    check_general(None,x,f,dot(M,dg))

def check_dot3(x,M):
    g=lambda x:5*x**3+3*x
    f=lambda x:g(dot(M,x))
    dg=g(forward.seed(dot(M,x))).dvalue
    check_general(None,x,f,dot(dg,M))

def test_dot():
    np.random.seed(0)
    for n in [ 0, 1, 3, 5 ]:
        x=np.random.rand(n)
        M=scipy.sparse.csr_matrix(np.matrix(np.random.rand(n,n)))
        yield check_dot1,x,M
        yield check_dot2,x,M
        yield check_dot3,x,M

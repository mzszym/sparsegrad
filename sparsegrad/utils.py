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

def isscalar(x):
    if isinstance(x,float):
        return True
    return np.ndim(x) == 0

def len1d(x):
    if np.ndim(x) != 1:
        raise RuntimeError('only 1-d vector supported')
    n,=np.shape(x)
    return n

def shape2d(x):
    if np.ndim(x) != 2:
        raise RuntimeError('only 2-d object supported')
    n,m=np.shape(x)
    return (n,m)



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

"Utility functions"

import numpy as np


def isscalar(x):
    "Return if x is numeric scalar (Python or numpy)"
    if isinstance(x, float):
        return True
    return np.ndim(x) == 0


def len1d(x):
    "Return length of 1-d array x, raise ValueError if argument is not 1-d array"
    if np.ndim(x) != 1:
        raise ValueError('only 1-d vector supported')
    n, = np.shape(x)
    return n


def shape2d(x):
    "Return n,m shape of 2-d array x"
    if np.ndim(x) != 2:
        raise ValueError(
            'only 2-d object supported, raise ValueError is argument is not 2-d array')
    n, m = np.shape(x)
    return (n, m)

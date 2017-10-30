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

import numpy as np
import scipy.sparse


def isscalar(x):
    if isinstance(x, float):
        return True
    return np.ndim(x) == 0


def len1d(x):
    if np.ndim(x) != 1:
        raise RuntimeError('only 1-d vector supported')
    n, = np.shape(x)
    return n


def shape2d(x):
    if np.ndim(x) != 2:
        raise RuntimeError('only 2-d object supported')
    n, m = np.shape(x)
    return (n, m)

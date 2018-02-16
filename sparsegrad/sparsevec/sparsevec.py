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

import numpy as np
from sparsegrad.impl.sparsevec import sparsevec
from sparsegrad.base import sparsesum

__all__ = ['sparsevec', 'sparsesum', 'sparsesum_bare']


def sparsesum_bare(n, terms, return_sparse=False, **kwargs):
    terms = list(terms)
    if not terms:
        return np.zeros(n)
    result = sparsesum(list(sparsevec(n, idx, v)
                            for idx, v in terms), return_sparse=return_sparse, **kwargs)
    if return_sparse:
        return result.idx, result.v
    else:
        return result

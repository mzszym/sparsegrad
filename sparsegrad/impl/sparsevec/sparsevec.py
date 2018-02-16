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

"This module contains implementation details of summing sparse vectors."

import numpy as np
from sparsegrad.impl.sparse import csr_matrix, csc_matrix


class sparsevec(object):
    """
    Sparse vector of length n, with nonzero entries in arrays (idx,v) where
    idx contains the indices of entries, and v contains the corresponding
    values
    """

    def __init__(self, n, idx, v):
        self.idx = idx
        self.v = v
        self.shape = (n,)


def sparsesum(terms, hstack=np.hstack, nvalue=lambda x: x,
              wrap=lambda idx, v, y: y, check_unique=False, return_sparse=False):
    """
    Sum sparse vectors

    This is a general function, which propagates index information and numerical
    values. Suitable functions must be supplied as arguments to propagate other
    information.

    Parameters
    ----------
    terms : list of sparsevec
        terms
    hstack : callable(vectors)
        function to use for concatenating vectors
    nvalue : callable(vector)
        function to use for extracting numerical value
    wrap : callable(idx, v, result)
        function to use for wrapping the result, with idx, v being concatenated inputs
    check_unique : bool
        whether to perform test for double assignments (useful when this function is
        used to replace item assignment)
    return_sparse : bool
        whether to calculate sparse results
    """
    terms = list(terms)
    n, = terms[0].shape
    if not all(t.shape == (n,) for t in terms[1:]):
        raise ValueError('different shapes of terms')
    idx, v = zip(*((t.idx, t.v) for t in terms))
    idx = np.hstack(idx)
    v = hstack(v)
    if check_unique:
        if len(np.unique(idx)) < len(idx):
            raise ValueError('indices not unique')

    def process_dense(n, idx, v):
        y = csr_matrix((nvalue(v), idx, np.asarray(
            [0, len(idx)])), shape=(1, n)).todense().A1
        return wrap(idx, v, y)

    def process_compressed(n, idx, v):
        cidx = np.unique(idx)
        idx = np.searchsorted(cidx, idx)
        v = process_dense(len(cidx), idx, v)
        return sparsevec(n, cidx, v)
    if return_sparse:
        return process_compressed(n, idx, v)
    else:
        return process_dense(n, idx, v)

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
from .expr import hstack
from .sparse import csr_matrix, sdcsr
from .forward import forward_value, nvalue
import scipy.sparse

__all__ = ['sparsevec', 'sparsesum', 'sparsesum_bare']


class sparsevec(object):
    def __init__(self, n, idx, v):
        self.idx = idx
        self.v = v
        self.shape = (n,)


class csc_matrix_unchecked(scipy.sparse.csc_matrix):
    def check(self, *args):
        pass


def sparsesum(terms, compress=False):
    terms = list(terms)
    n, = terms[0].shape
    if not all(t.shape == (n,) for t in terms[1:]):
        raise ValueError('different shapes of terms')
    idx, v = zip(*((t.idx, t.v) for t in terms))
    idx = hstack(idx)
    v = hstack(v)

    def process_dense(n, idx, v):
        y = csr_matrix((nvalue(v), idx, np.asarray(
            [0, len(idx)])), shape=(1, n)).todense().A1
        if isinstance(v, forward_value):
            M = v.deriv.tovalue()
            M = M.tocsc()
            rows = idx[M.indices]
            M = csc_matrix_unchecked(
                (M.data, rows, M.indptr), shape=(
                    n, M.shape[1]))
            M.sort_indices()
            M = M.tocsr()
            return forward_value(value=y, deriv=sdcsr(mshape=M.shape, M=M))
        else:
            return y

    def process_compressed(n, idx, v):
        cidx = np.unique(idx)
        idx = np.searchsorted(cidx, idx)
        v = process_dense(len(cidx), idx, v)
        return sparsevec(n, cidx, v)
    if compress:
        return process_compressed(n, idx, v)
    else:
        return process_dense(n, idx, v)


def sparsesum_bare(n, terms, unique=False, compress=False):
    if not terms:
        return np.zeros(n)
    result = sparsesum((sparsevec(n, idx, v)
                        for idx, v in terms), compress=compress)
    if compress:
        return result.idx, result.v
    else:
        return result

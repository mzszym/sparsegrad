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
from sparsegrad.impl.sparse import csr_matrix, csc_matrix


class sparsevec(object):
    def __init__(self, n, idx, v):
        self.idx = idx
        self.v = v
        self.shape = (n,)


def sparsesum(terms, hstack=np.hstack, nvalue=lambda x: x,
              wrap=lambda idx, v, y: y, check_unique=False, return_sparse=False):
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

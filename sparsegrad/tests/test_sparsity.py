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
from numpy.testing import *
from sparsegrad import forward
from sparsegrad.base import *
import inspect

_funcs = [lambda x:x,
          lambda x:x[::-1],
          lambda x:stack(x**2, x**3),
          lambda x:stack(x**2, (x**3)[::-1]),
          lambda x:where(False, x, x[::-1]),
          lambda x:where(x >= 0, x, x[::-1]),
          lambda x:where(x > 0, x, x[::-1]),
          lambda x:sum(x),
          lambda x:stack(x, sum(x)),
          lambda x:where(x > 0, x, sum(x))
          ]

_vecs = [
    np.linspace(
        0, 1, 0), np.linspace(
            0, 1, 1), np.linspace(
                0, 1, 2), np.linspace(
                    1., 3., 3)]


def check_sparsity(msg, f, x):
    df = f(forward.seed_sparse_gradient(x)**2).gradient
    sdf = f(forward.seed_sparsity(x)**2).sparsity
    assert sdf.shape
    assert_almost_equal((sdf.tocsr().multiply(df) - df).data, 0.)


def test_sparsity():
    for f in _funcs:
        for x_ in _vecs:
            for x in [x_, -x_]:
                yield check_sparsity, inspect.getsource(f), f, x

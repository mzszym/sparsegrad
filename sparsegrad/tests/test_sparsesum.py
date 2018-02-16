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

from sparsegrad import *
from sparsegrad.sparsevec import *
import numpy as np
sparsesum = sparsesum_bare


def test_sparsesum():
    idx, v = sparsesum(10, [(0, 1), (0, 1), (3, -1)], return_sparse=True)
    assert (idx == np.asarray([0, 3])).all()
    assert (v == np.asarray([2., -1.])).all()

    x = forward.seed(np.linspace(0, 1, 11))
    idx, d = sparsesum(
        20, [(np.arange(11), x), (np.arange(11), x)], return_sparse=True)
    assert (idx == np.arange(11)).all()
    assert (d.gradient.diagonal() == 2.).all()

    d = sparsesum(1, [(np.zeros(2, dtype=int), np.ones(2)),
                      (np.zeros(1, dtype=int), np.ones(1))])
    assert d == 3

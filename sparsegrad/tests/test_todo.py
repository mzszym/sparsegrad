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

# File for things not working as expected, to be fixed

import numpy as np
from sparsegrad import forward,where

def test_index_fail():
    return
    # Not working?
    u=np.asarray([1,2,3])
    f=lambda x:x[[0]]*where(x[[0]]<0,x[1:],-x[1:])
    f(forward.seed(u))

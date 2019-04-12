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
from sparsegrad.impl.multipledispatch import dispatch, GenericFunction
from .ufunc import DifferentiableFunction


@dispatch(object)
def get_implementation(obj):
    return getattr(obj, '__array_priority__', 0), obj

def find_implementation(arrays, default=np, default_priority=0):
    best_impl = default
    best_priority = default_priority
    for a in arrays:
        priority, impl = get_implementation(a)
        if best_impl is None or priority > best_priority:
            best_priority, best_impl = priority, impl
    return best_impl

@dispatch(object, object)
def hstack(impl, arrays):
    return impl.hstack(arrays)


apply = GenericFunction('apply')


def apply_numeric(_, func, args):
    return func.func(*args)


apply.add((object, object, object), apply_numeric)
apply.add((np.ndarray, object, object), apply_numeric)

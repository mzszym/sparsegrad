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

__all__ = ['dot', 'where', 'sum', 'broadcast_to', 'hstack', 'stack',
           'sparsesum', 'branch', 'isscalar', 'nvalue', 'apply', 'isnvalue']

import numbers
import numpy as np
from sparsegrad import impl
import sparsegrad.impl.sparsevec as impl_sparsevec
from sparsegrad.impl.multipledispatch import dispatch, GenericFunction
from . import routing

dot = GenericFunction('dot')
dot.add((object, object), impl.dot_)

where = GenericFunction('where')
where.add((object, object, object), np.where)

sum = GenericFunction('sum')
sum.add((object,), np.sum)

broadcast_to = GenericFunction('broadcast_to')
broadcast_to.add((object, object), np.broadcast_to)


def hstack(arrays):
    "Generalized version of numpy.hstack"
    return routing.hstack(routing.find_implementation(arrays), arrays)


def stack(*arrays):
    "Alias for hstack, taking arrays as separate arguments"
    return hstack(arrays)


def sparsesum(terms, **kwargs):
    "Generalized version of sparsesum"
    impl = routing.find_implementation(
        (a.v for a in terms), default=impl_sparsevec)
    return routing.sparsesum(impl, terms, **kwargs)


@dispatch(object, object, object)
def branch(cond, iftrue, iffalse):
    """
    Branch execution

    Note that, in some cases (propagation of sparsity pattern), both branches can executed
    more than once.

    Parameters:
    -----------
    cond : bool vector
        Condition
    iftrue : callable(idx)
        Function called to evaluate elements with indices idx, where cond is True
    iffalse : callable(idx)
        Function called to evaluate elements with indices idx, where cond is False

    """

    def _branch(cond, iftrue, iffalse):
        if not cond.shape:
            if cond:
                return iftrue(None)
            return iffalse(None)
        n = len(cond)
        r = np.arange(len(cond))
        ixtrue = r[cond]
        ixfalse = r[np.logical_not(cond)]
        vtrue = impl_sparsevec.sparsevec(
            n, ixtrue, broadcast_to(
                iftrue(ixtrue), ixtrue.shape))
        vfalse = impl_sparsevec.sparsevec(
            n, ixfalse, broadcast_to(
                iffalse(ixfalse), ixfalse.shape))
        return sparsesum([vtrue, vfalse])
    value = _branch(cond, iftrue, iffalse)
    return value


isscalar = GenericFunction('isscalar')


def _is_number_scalar(x):
    return True


def _is_array_scalar(x):
    return not x.shape


isscalar.add((numbers.Number,), _is_number_scalar)
isscalar.add((np.ndarray,), _is_array_scalar)

nvalue = GenericFunction('nvalue')


def _py_number_nvalue(x):
    return x


def _ndarray_nvalue(x):
    return x


nvalue.add((numbers.Number,), _py_number_nvalue)
nvalue.add((np.ndarray,), _ndarray_nvalue)


def apply(function, args):
    impl = routing.find_implementation(args, default=None)
    return routing.apply(impl, function, args)


isnvalue = GenericFunction('isnvalue')


def _is_pynumber_numeric(x):
    return True


def _is_ndarray_numeric(x):
    return True


isnvalue.add((numbers.Number,), _is_pynumber_numeric)
isnvalue.add((np.ndarray,), _is_ndarray_numeric)

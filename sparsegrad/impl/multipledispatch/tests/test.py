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

import unittest
from sparsegrad.impl.multipledispatch import GenericFunction, GenericMethod, DispatchError


class TestDispatch(unittest.TestCase):
    def testAmbigous(self):
        f = GenericFunction('f')

        def wrong_fun():
            raise NotImplementedError()

        def wrong_fun2():
            raise NotImplementedError()
        f.add((object, object), wrong_fun)
        f.add((object, int), wrong_fun)
        f.add((int, object), wrong_fun2)
        with self.assertRaises(DispatchError):
            f(-1, 0)
        f.add((int, int), lambda x, y: x+y)
        self.assertEqual(f(-1, 0), -1)

    def testMethods(self):
        class A(object):
            fun = GenericMethod('fun')

        A.fun.add((object,), lambda self, x: x)
        a = A()
        self.assertEqual(a.fun(-1), -1)

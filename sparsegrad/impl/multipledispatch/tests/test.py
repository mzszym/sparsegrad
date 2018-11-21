from sparsegrad.impl.multipledispatch import GenericFunction, GenericMethod, DispatchError
import unittest

class TestDispatch(unittest.TestCase):
    def testAmbigous(self):
        f = GenericFunction('f')
        def wrong_fun():
            raise NotImplementedError()
        f.add((object, object), wrong_fun)
        f.add((object, int), wrong_fun)
        f.add((int, object), wrong_fun)
        with self.assertRaises(DispatchError):
            f(-1, 0)
        f.add((int, int), lambda x,y:x+y)
        self.assertEqual(f(-1,0),-1)

    def testMethods(self):
        class A(object):
            fun = GenericMethod('fun')

        A.fun.add((object,), lambda self, x:x)
        a = A()
        self.assertEqual(a.fun(-1), -1)

__all__ = [ 'check_scalar', 'check_vector', 'check_vecto_scalar', 'check_general', 'product', 'lambdify','check_general']

from numpy.testing import assert_almost_equal
from sparsegrad.testing import namespaces
import numpy as np
from sparsegrad import forward
import scipy.sparse

def check_scalar(x, f, df):
    y = f(forward.seed(x))
    assert_almost_equal(y.value, f(x))
    assert_almost_equal(y.dvalue, df(x))

def verify_scalar(settings, x, f, df):
    f = lambdify(f, settings)
    df = lambdify(df, settings)
    return check_scalar(x, f, df)

def _check_grad(a, b):
    err = a - b
    if isinstance(err, scipy.sparse.csr_matrix):
        assert_almost_equal(err.data, 0.)
    else:
        assert_almost_equal(err, 0.)

def check_vector(x, f, df):
    y = f(forward.seed(x))
    d = np.atleast_1d(df(x) * np.ones_like(y.value))
    dy = scipy.sparse.spdiags(d, 0, len(d), len(d), format='csr')
    assert_almost_equal(y.value, f(x))
    _check_grad(y.dvalue, dy)

def verify_vector(settings, x, f, df):
    f = lambdify(f, settings)
    df = lambdify(df, settings)
    return check_vector(x, f, df)

def check_vector_scalar(x, v, f, df):
    y = f(forward.seed(x), v)
    assert_almost_equal(y.value, f(x, v))
    assert y.dvalue.shape == (len(y.value), 1)
    assert_almost_equal(np.asarray(y.dvalue.todense()).flatten(), df(x, v))

def check_general(x, f, grad):
    y = f(forward.seed(x))
    dy = grad
    assert_almost_equal(y.value, f(x))
    _check_grad(y.dvalue, dy)

def product(functions, values, namespaces = None ):
    if namespaces is None:
        namespaces = [ 'np' ] 
    return [(dict(ns=ns), x, f, df) for x in values for (f,df) in functions for ns in namespaces]

def lambdify(func, settings=None):
    if settings is None:
        settings = dict(ns='default')
    return lambda x:eval(func, getattr(namespaces, settings['ns']).__dict__, dict(x=x))

def check_sparsity(x, f):
    df = f(forward.seed_sparse_gradient(x)**2).gradient
    sdf = f(forward.seed_sparsity(x)**2).sparsity
    assert sdf.shape
    assert_almost_equal((sdf.tocsr().multiply(df) - df).data, 0.)

def verify_sparsity(gvars, x, func):
    f = lambdify(func, gvars)
    check_sparsity(x, f)

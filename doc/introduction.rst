Introduction
============

sparsegrad is an automatic differentiation library specialized for functions using numpy, for which sparse Jacobian must be calculated. sparsegrad is particularly useful for problems on irregular grids and graphs, for which stencil cannot be fined. The calculated sparse Jacobian matrix is usually used in Newton's method for solving non-linear system of equations or for sensitivity analysis.

sparsegrad is applied for unmodified functions, if they are written in supported subset of numpy. Assume a function f is defined:

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> import numpy as np
   >>> x = np.linspace(0,1,5)
   >>> def f(x):
   ...    return np.sqrt(x**2+1)
   >>> print(f(x))
   [ 1.          1.03077641  1.11803399  1.25        1.41421356]

Sparse Jacobian is calculated as follows:

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> from sparsegrad import forward
   >>> y = f(forward.seed(x))
   >>> # Print value and derivative
   >>> print(y.value)
   [ 1.          1.03077641  1.11803399  1.25        1.41421356]
   >>> print(y.dvalue)
   (0, 0)	0.0
   (1, 1)	0.242535625036
   (2, 2)	0.4472135955
   (3, 3)	0.6
   (4, 4)	0.707106781187

sparsegrad is highly optimized and can deliver Jacobian matrices with millions of entries in reasonable time.


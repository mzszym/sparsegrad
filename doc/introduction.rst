Introduction
============

``sparsegrad`` is a Python library for automatic calculation of sparse Jacobian matrices. It is applicable to unmodified Python calculations of arbitrary complexity expressed using :ref:`supported operations<features>`. 

Assume that a Python function ``f`` is defined, performing some calculations with ``numpy``

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> import numpy as np
   >>> x = np.linspace(0,1,5)
   >>> def f(x):
   ...    return np.sqrt(x**2+1)
   >>> print(f(x))
   [ 1.          1.03077641  1.11803399  1.25        1.41421356]

Sparse Jacobian is calculated by evaluating function ``f`` on a suitable `seed` object:

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> from sparsegrad import forward
   >>> y = f(forward.seed(x))

Result ``y`` now contains sparse Jacobian information:

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> print(y.dvalue)
   (0, 0)	0.0
   (1, 1)	0.242535625036
   (2, 2)	0.4472135955
   (3, 3)	0.6
   (4, 4)	0.707106781187

``sparsegrad`` is primarily intended to be used in Newton's method for solving systems of nonlinear equations. Systems with more than one million degrees of freedom per node are feasible. The algorithm of differentiation is selected and optimized to limit the runtime and the memory requirements. In some :ref:`cases<timing>`, ``sparsegrad`` outperforms other libraries.

``sparsegrad`` is implemented in pure Python without extension modules. This simplifies both the installation and the deployment. The implementation depends only on ``numpy`` and ``scipy`` packages.

``sparsegrad`` is distributed under GNU Affero General Public License version 3. The full text of the license is provided in file ``LICENSE`` in the root directory of distribution. 


sparsegrad
==========

|Travis-CI-badge| |Readthedocs-badge|


``sparsegrad`` automatically and efficiently calculates analytical Jacobian of ``numpy`` vector valued functions. It is designed to be useful for solving large systems of non-linear equations. ``sparsegrad`` is memory efficient because it does not use the graph of computation. Arbitrary computations are supported through indexing, matrix multiplication, branching, and custom functions. 

Taking Jacobian with respect to variable `x` is done by replacing numerical value of `x` with sparsegrad `seed`

..  code:: python

    >>> import numpy as np
    >>> import sparsegrad.forward as ad
    >>> def f(x):
    ...       return x-x[::-1]
    >>> x=np.linspace(0,1,3)
    >>> print(f(ad.seed(x)).dvalue)
    (0, 0)	1.0
    (0, 2)	-1.0
    (2, 0)	-1.0
    (2, 2)	1.0

``sparsegrad`` is written in pure Python. For easy installation and best portability, it does not contain extension modules. In realistic problems, it can provide similar or better performance than ADOL-C best case of `repeated calculation`. This is possible thanks to algorithmic optimizations and optimizations to avoid slow parts of ``scipy.sparse``. 

``sparsegrad`` relies on ``numpy`` and ``scipy`` for computations. It is compatible with both Python 2.7 and 3.x.

Installation
------------

.. code:: bash

   pip install sparsegrad

It is recommended to run test suite after installing

.. code:: bash

   python -c "import sparsegrad; sparsegrad.test()"

.. |Travis-CI-badge| image:: https://travis-ci.org/mzszym/sparsegrad.svg?branch=master
   :target: https://travis-ci.org/mzszym/sparsegrad

.. |Readthedocs-badge| image:: https://readthedocs.org/projects/sparsegrad/badge/?version=latest
   :target: http://sparsegrad.readthedocs.io/en/latest/?badge=latest
      

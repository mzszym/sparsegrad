
Tutorial
========

.. code:: python

    import sparsegrad

Testing installation
--------------------

.. code:: python

    sparsegrad.test()


.. parsed-literal::

    Running unit tests for sparsegrad
    NumPy version 1.13.3
    NumPy relaxed strides checking option: True
    NumPy is installed in /usr/lib/python3.6/site-packages/numpy
    Python version 3.6.4 (default, Dec 23 2017, 19:07:07) [GCC 7.2.1 20171128]
    nose version 1.3.7


.. parsed-literal::

    ....................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................
    ----------------------------------------------------------------------
    Ran 676 tests in 1.340s
    
    OK




.. parsed-literal::

    <nose.result.TextTestResult run=676 errors=0 failures=0>



Calculation of Jacobian
-----------------------

.. code:: python

    import numpy as np
    import sparsegrad as ad

.. code:: python

    def function(x):
        return np.exp(-x**2)

.. code:: python

    x0=np.linspace(-1,1,5)

Calculate value and function gradient by forward mode automatic
differentiation:

.. code:: python

    y=function(ad.forward.seed_sparse_gradient(x0))

Access function value:

.. code:: python

    y.value




.. parsed-literal::

    array([ 0.36787944,  0.77880078,  1.        ,  0.77880078,  0.36787944])



Access gradient as sparse matrix:

.. code:: python

    y.dvalue.tocsr()




.. parsed-literal::

    <5x5 sparse matrix of type '<class 'numpy.float64'>'
    	with 5 stored elements in Compressed Sparse Row format>



.. code:: python

    print(y.dvalue.todense())


.. parsed-literal::

    [[ 0.73575888  0.          0.          0.          0.        ]
     [ 0.          0.77880078  0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.         -0.77880078  0.        ]
     [ 0.          0.          0.          0.         -0.73575888]]


Calculation of sparsity pattern
-------------------------------

.. code:: python

    y=function(ad.forward.seed_sparsity(np.zeros_like(x0)))

Access positions of possible nonzeros in AIJ format:

.. code:: python

    y.sparsity.indices




.. parsed-literal::

    array([0, 1, 2, 3, 4], dtype=int32)



.. code:: python

    y.sparsity.indptr




.. parsed-literal::

    array([0, 1, 2, 3, 4, 5], dtype=int32)



Access positions of possible nonzeros as scipy CSR matrix:

.. code:: python

    y.sparsity.tocsr()




.. parsed-literal::

    <5x5 sparse matrix of type '<class 'numpy.int64'>'
    	with 5 stored elements in Compressed Sparse Row format>



.. code:: python

    print(y.sparsity.tocsr().todense())


.. parsed-literal::

    [[1 0 0 0 0]
     [0 1 0 0 0]
     [0 0 1 0 0]
     [0 0 0 1 0]
     [0 0 0 0 1]]


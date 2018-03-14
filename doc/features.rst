.. _features:

Supported operations
====================

Arithmetics
-----------

- Python scalars

- numpy ``ndarray`` with dimensionality <2

- broadcasting

- mathematical operators (+, -, ...)

- numpy elementwise mathematical functions (``sin``, ``exp``, ...)

Indexing
--------

``sparsegrad`` has full support for indexing for reading arrays:

- indexing by scalars, for example ``x[0]`` and ``x[-1]``

- indexing by slice, for example ``x[::-1]``

- indexing by arrays, for example ``x[np.arange(10)]``

Setting individual elements in arrays should be replaced with summing sparse vectors.

dtype promotion
---------------

``sparsegrad`` does not assume a specific ``dtype``. It follows ``numpy`` dtype coercion rules.

Branching and control flow
--------------------------

Since ``sparsegrad`` does not reuse data between evaluations, arbitrary branching of execution is allowed through Python control flow statements such as ``if``. ``sparsegrad`` objects implements all the comparison operators.

Branching at vector element level is supported through functions ``where`` and ``branch``.

``where`` is an equivalent of the standard ``numpy`` function, but it supports correctly ``sparsegrad`` objects. As the standard version, it has a disadvantage that both possible values must be evaluated for each element. In the case of expensive calculations, this is avoided by using ``branch`` function, which only evaluates used values.

Sparse vectors
--------------

``sparsegrad`` provides functions for summing sparse vectors with derivative information.

Irregular memory access
-----------------------

Collecting values from non-sequential locations in memory, with optional summing, is supported through multiplication by sparse matrix (``dot``).

Writing values to non-sequential locations in memory, with optional summing, is supported through summing sparse vectors (``sparsesum``).

Calculation of sparsity pattern
-------------------------------

Sparsity pattern is calculating using ``seed_sparsity``. 

Other functions
---------------

sparsegrad provides variants of standard functions that work for both numpy and sparsegrad values:

- ``dot(A,x)`` : matrix - vector multiplication, where matrix `A` is constant

- ``sum(x)`` : sum of elements of a vector `x`

- ``hstack(vecs)``, ``stack(*vecs)`` : concatenation of vectors `vecs`


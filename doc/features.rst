Supported features
==================

Arithmetics
-----------

- Python scalars

- numpy ndarray with dimensionality 0 or 1

- broadcasting

- mathematical operators (+, -, ...)

- numpy elementwise mathematical functions (`np.sin`, `np.exp`, ...)

Indexing
--------

sparsegrad has full support for indexing when reading arrays:

- indexing by scalars, for example `x[0]` and `x[-1]`

- indexing by slice, for example `x[::-1]`

- indexing and by arrays, for example `x[np.arange(10)]`

Setting individual elements in arrays should be replaced with operation of summing sparse vectors.

dtype promotion
---------------

sparsegrad does not assume a specific dtype. It follows numpy/scipy dtype coercion rules.

Branching and control flow
--------------------------

Since sparsegrad does not reuse any structures, arbitrary branching of execution is allowed at level of Python code. sparsegrad objects implement all comparison operators.

Branching at element level is supported through functions `where` and `branch`. `where` is equivalent of numpy `where`, but supports correctly both numpy and sparsegrad objects.

`where` has disadvantage that, for each element, both possible values are evaluated. This can be avoided by using `branch` function when calculation is expensive.

Sparse vectors
--------------

`sparsesum` is provided for summing sparse vectors with derivative information.

Irregular vector access
-----------------------

All irregular access, both reading and writing, is supported through multiplication by sparsematrix matrix (`dot`).

Calculation of sparsity pattern
-------------------------------

Sparsity pattern is calculating using `seed_sparsity`. 

Other functions
---------------

sparsegrad provides variants of functions which work for both numpy and sparsegrad values:

- `dot(A,x)` : matrix - vector multiplication, where matrix is constant

- `sum(x)` : sum of elements of a vector x

- `hstack(vecs)`, `stack(*vecs)` : concatenation of vectors vecs

Large problems
--------------

sparsegrad avoids building and storing graph of data flow of whole computation. Derivative data is reachable if and only if value is reachable.

Python only
-----------

sparsegrad is easy to install because it does not contain extension modules.

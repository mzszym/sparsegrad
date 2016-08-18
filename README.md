# sparsegrad - automatic computation of sparse Jacobian matrices from ```numpy``` expressions

```sparsegrad``` performs automatic differentiation of vector valued functions in Python. A significant subset of ```numpy``` operation is supported on Python scalars and ```ndarrays``` with dimensionality less than 2:

- all arithmetic operators
- all elementary functions
- simple and fancy indexing
- matrix-vector product ```dot```, restricted to constant matrix
- concatenation of vectors ```stack```
- vectorized selection ```where```
- sum reduction ```sum```

Depending on use, ```sparsegrad``` can provide Jacobian matrix or sparsity pattern.

The primary use of ```sparsegrad``` is to automatically evaluate Jacobian matrices when solving non-linear systems of equations. ```sparsegrad``` uses forward mode automatic differentiation. In contrast to backward mode automatic differentiation, this allows to better control the memory usage of calculation.

```sparsegrad``` is Python-only and requires only ```numpy``` and ```scipy```. It works both in Python 2.7 and 3.x. In contrast to other pure Python automatic differentiation modules, ```sparsegrad``` attempts to be better suited for calculating moderately large sparse matrices. It has been used for solving problems with >1M equations and >20M nonzeros without causing bottleneck in terms of running time or memory usage.

For basic usage, see tutorial ```doc/tutorial.ipynb```.

```sparsegrad``` is not yet tested on many combinations of ```numpy``` and ```scipy``` versions. After installing, it is highly recommended to check if all tests pass by running:

```
import sparsegrad
sparsegrad.test()
```

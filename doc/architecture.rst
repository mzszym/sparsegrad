Architecture
============

sparsegrad currently support forward model automatic differentiation on vector valued function.

sparsegrad value is a pair (x,x') where x is value and x' is the Jacobian.

As a major optimization, Jacobian is stored as a product of scalar, diagonal matrix and a general sparse matrix. The general parts are considered constant and are shared between values. This allows to avoid some rebuilding the sparse matrix when evaluating operations not creating new sparsity patterns.



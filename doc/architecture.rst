Architecture
============

``sparsegrad`` performs forward mode automatic differentiation on vector valued function.

sparsegrad forward value is a pair :math:`\left( \mathbf{y}, \mathbf{\partial \mathbf{y} / \partial \mathbf{x}} \right)` where `y` is referred to as ``value`` and the Jacobian :math:`\partial \mathbf{y} / \partial \mathbf{x}` is referred to as ``dvalue``.

During the evaluation, ``sparsegrad`` values are propagated in place of standard ``numpy`` arrays. This simple approach gives good result when only first order derivative is required. Most importantly, it does not involve solving graph coloring problem on the whole computation graph. For large problems, storing complete computation graph is very expensive. 

When a mathematical function :math:`\mathbf{f} = \mathbf{f} \left( \mathbf{y_1}, \mathbf{y_2},  \ldots, \mathbf{y_n} \right)` is evaluated, the derivatives are propagated using the chain rule

.. math::

   \frac { \partial \mathbf{f} } { \partial \mathbf { x } } = 
   \sum_i \frac { \partial \mathbf{f} } { \partial \mathbf { y_i } }
   \frac { \partial \mathbf{y_i} } { \partial \mathbf { x } }

Support for ``numpy``
---------------------

To support ``numpy`` functions, broadcasting must be included. In the discussion below, scalars are treated as one element vectors.

Application of a numpy function involves implicit broadcasting from vector :math:`\mathbf{y_i}`, with its proper shape, to vector :math:`\mathbf{\bar{y_i}}`, with shape of :math:`\mathbf{f}` if both shapes are not the same. This can be denoted by multiplication by matrix :math:`\mathbf{B_i}`. The result of function evaluation is then

.. math::

   \mathbf{f} =
   \mathbf{f} \left(
   \mathbf{B_1} \mathbf{y_1},
   \mathbf{B_2} \mathbf{y_2}, \ldots
   \mathbf{B_n} \mathbf{y_n}
   \right)

and the derivative is

.. math::

   \frac { \partial \mathbf{f} } { \partial \mathbf { x } } = 
   \sum_i \frac { \partial \mathbf{f} } { \partial \mathbf { \bar{y_i} } }
   \mathbf{B_i}
   \frac { \partial \mathbf{y_i} } { \partial \mathbf { x } }

The Jacobian of ``numpy`` function application is a diagonal matrix. If :math:`\mathbf{g_i}` is a ``numpy`` elementwise derivative of :math:`\mathbf{f}` with respect of to :math:`\mathbf{y_i}`, then

.. math::

   \frac { \partial \mathbf{f} } { \partial \mathbf { \bar{y_i} } } =
   \mathrm{diag} \left( \mathbf{g_i}\left(
   \mathbf{B_1} \mathbf{y_1},
   \mathbf{B_2} \mathbf{y_2}, \ldots
   \mathbf{B_n} \mathbf{y_n}
   \right) \right)

Optimizations
-------------

As a major optimization, the Jacobian matrices are stored as a product of scalar :math:`s`, diagonal matrix :math:`\mathrm{diag} \left( \mathbf{d} \right)` and a general sparse matrix :math:`\mathbf{M}`:

.. math::

   \frac { \partial \mathbf{f} } { \partial \mathbf { x } } =
   s \cdot \mathrm{diag} \left( \mathbf{d} \right) \mathbf{M}

The general parts :math:`\mathbf{M}` are constant shared objects. If not specified, the diagonal parts and the general parts are assumed to be identity.

The factored matrix is referred to as ``sdcsr`` in ``sparsegrad`` code. It allows to peform the most common operations with rebuilding the general matrix part:

.. math::

   \left[ s_1 \cdot \mathrm{diag} \left( \mathbf{d_1} \right) \mathbf{M} \right] + 
   \left[ s_2 \cdot \mathrm{diag} \left( \mathbf{d_2} \right) \mathbf{M} \right] =
   \mathrm{diag} \left ( s_1 \cdot \mathbf{d_1} + s_2 \cdot \mathbf{d_2} \right ) \mathbf{M}

.. math::

   \alpha \cdot \mathrm{diag} \left( \mathbf{x} \right)
   \left[ s \cdot \mathrm{diag} \left( \mathbf{d} \right) \mathbf{M} \right] =
   \left( \alpha s \right) \cdot
   \mathrm{diag} \left( \mathbf{x} \circ \mathbf{d} \right)
   \mathbf{M}

with :math:`\circ` denoting elementwise multiplication.

Backward mode
-------------

Backward mode is currently not implemented because of prohibitive memory requirements for large calculations. In backward mode, each intermediate value has to accessed twice: during the forward evaluation of function value, and during backward evaluation of derivative. The memory requirements to store intermediate values are prohibitive for functions with large number of outputs, and grows linearly with the number of steps in computation.


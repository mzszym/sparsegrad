.. _timing:


Runtime comparison
==================

``sparsegrad`` comes with an example of a fully implicit solver for shallow water equations. This serves as an example of a problem resulting from discretisation of partial differential equations, with realistic complexity and size. The code resides in ``examples/shallow-water.ipynb``.

Runtime comparison of ``sparsegrad`` with ADOL-C is below. The test was run on a single core of Xeon E5-2620v4:

==================== =====
Calculation          ms
==================== =====
numpy                2.3
sparsegrad           70
ADOL-C repeated      142
ADOL-C full          2130
==================== =====

``numpy`` only calculates function value. ``sparsegrad`` calculation calculates function value and derivative.

``ADOL-C repeated`` calculates the function value and the derivative using computation graph and graph coloring previously stored in memory. Whole calculation is run by C code. ``ADOL-C`` repeated is only available when there is no change of control flow in the calculation.

``ADOL-C full`` builds the computation graph, solves the graph coloring problem in addition to computing the actual output. It must be used when control flow changes in the computation leading to change in sparsity structure.

On this particular example, ``sparsegrad`` is from 2 to 30 times faster than ``ADOL-C``.

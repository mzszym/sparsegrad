Installing
==========

Requirements
------------

- Python 2.7 or Python 3.4+
- numpy >= 1.10
- scipy >= 0.14.0
- packaging >= 14.0

Installation
------------

sparsegrad is best installed from PyPI by running command

.. code-block:: bash

   $ pip install sparsegrad

It is advised to append --user option to avoid system-wide installation which requires administrator's rights.

Current development version of sparsegrad can be obtained from github, from https://github.com/mzszym/sparsegrad .

After installing, it is advised to run test suite to check if sparsegrad works correctly on your system:

.. doctest::
   :options: +SKIP

   >>> import sparsegrad
   >>> sparsegrad.test()
   Running unit tests for sparsegrad...
   OK
   <nose.result.TextTestResult run=676 errors=0 failures=0>

If errors are found, sparsegrad is not compatible with your system. Either your numpy scientific stack is too old, or there is a bug in sparsegrad. 

sparsegrad is evolving, and guarantees of backward compatibility are offered. It is recommended to check which version you are using:

.. doctest::

   >>> import sparsegrad
   >>> sparsegrad.version
   '0.0.6'

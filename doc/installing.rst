Installation
============

Requirements
------------

- Python 2.7 or Python 3.4+
- numpy >= 1.10
- scipy >= 0.14.0
- packaging >= 14.0

Installation from PyPI
----------------------

This is the preferred way of installing ``sparsegrad``.

Two variants of the installation are possible:

- system wide installation:

.. code-block:: bash

   $ pip install sparsegrad

- local installation not requiring administrator's rights:

.. code-block:: bash

   $ pip install sparsegrad --user

In the case of local installation, ``sparsegrad`` is installed inside user's home directory. In Linux, this defaults to ``$HOME/.local``.

After installing, it is advised to run the test suite to ensure that ``sparsegrad`` works correctly on your system:

.. doctest::
   :options: +SKIP

   >>> import sparsegrad
   >>> sparsegrad.test()
   Running unit tests for sparsegrad...
   OK
   <nose.result.TextTestResult run=676 errors=0 failures=0>

If any errors are found, ``sparsegrad`` is not compatible with your system. Either your Python scientific stack is too old, or there is a bug. 

sparsegrad is evolving, and backward compatibility is not yet offered. It is recommended to check which version you are using:

.. doctest::

   >>> import sparsegrad
   >>> sparsegrad.version
   '0.0.6'

Development installation (advanced)
-----------------------------------

Current development version of sparsegrad can be installed from the development repository by running

.. code-block:: bash

   $ git clone https://github.com/mzszym/sparsegrad.git
   $ cd sparsegrad
   $ pip install -e .

The option ``-e`` tells that ``sparsegrad`` code should be loaded from ``git`` controlled directory, instead of being copied to the Python libraries directory. As with the regular installation, ``--user`` option should be appended for local installation.

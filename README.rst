======================================
Destructive Deep Learning (ddl) README
======================================

Destructive deep learning estimators and functions.
Estimators are compatible with scikit-learn.
Source code is distributed under the BSD 3-clause license.

Reproduce experiments from ICML 2018 paper
------------------------------------------

Please cite the following paper if you use this code:

    Deep Density Destructors David I. Inouye, Pradeep Ravikumar To
    appear in *International Conference on Machine Learning* (ICML),
    2018.

*Prerequisites to reproduce experiments*: MLPACK (mlpack.org) I have a singularity and docker image
that comes preinstalled with MLPACK and will add instructions soon.

To reproduce the 2D experiment in the paper open and run the notebook
``notebooks/demo_toy_experiment.ipynb``.

To reproduce the MNIST and CIFAR-10 experiments run:

.. code:: bash

    $ python scripts/icml_2018_experiment.py --model_names=MODEL_NAME --data_names=DATA_NAME

To run all the models run

.. code:: bash

    $ python scripts/icml_2018_experiment.py --model_names=deep-copula,image-pairs-copula,image-pairs-tree --data_names=mnist,cifar10

Coding guidelines
-----------------

Please read through the following high-level guidelines:

1. Zen of Python - https://www.python.org/dev/peps/pep-0020/
2. Python style guidelines - https://www.python.org/dev/peps/pep-0008/
3. ``scikit-learn`` coding guidelines -
   http://scikit-learn.org/stable/developers/contributing.html#coding-guidelines

Project-specific guidelies
^^^^^^^^^^^^^^^^^^^^^^^^^^

For this particular project, please follow these additional guidelines:

-  Use lower case with underscores for variable names and functions.
-  Please use longer names with full spellings especially for public
   interfaces to allow for super lightweight documentation. The variable
   names should be descriptive of its function. For example, a
   constructor name should be ``fitted_canonical_destructor`` rather
   than ``fitted_destructor`` or ``destructor`` or ``fit_canon_destr``
   or ``fcd``. Another example, ``univariate_estimators`` rather than
   ``univ_est`` or ``univariate_est`` or ``uest``. It is much easier to
   change a long variable name to short one than the other way around.
-  Methods should generally be private designated by underscore prefix
   unless sure the method should be exposed publicly.
-  For non-negative integer count variables prefix with ``n_`` rather
   than ``num_`` or ``number_of_``
-  Use variable names ``n_samples``, ``n_features``, and
   ``n_components`` (number of mixture components, number of PCA
   vectors, etc) and ``n_layers`` instead of ambiguous single letter
   variable names like ``n``, ``p`` or ``k``.

-  In the library and tests, please use the logging API instead of print
   statements. In particular, create a logger for each module and call
   the appropriate logging function (usually ``logger.debug(message)``)

   .. code:: python

       import logging
       logger = logging.getLogger(__name__)
       def foo():
       logger.debug('Checking inside foo')

-  To avoid the module from outputing anything unless requested, the
   root module file ``__init__.py`` redirects the logging output to
   ``None`` as follows:

   .. code:: python

       import logging
       from logging import NullHandler
       logging.getLogger(__name__).addHandler(NullHandler())

-  Thus, to view these logs when executing a program and capture
   warnings as logs for a particular module you must setup logging to
   output to standard out (and/or a local file). For example, you could
   write:

   .. code:: python

       logging.basicConfig(stream=sys.stdout)  # Push towards stdout instead of null handler
       logging.captureWarnings(True)  # Capture warnings in loggers
       logging.getLogger('ddl').setLevel(logging.DEBUG)  # Show everything above DEBUG level for the root ddl module

TODOs
-----

-  Change most functions to use log probabilities for numerical accuracy
   whenever possible. We could even operate in the log space all the
   time for canonical destructors (everything would be strictly
   negative). For example, node.value and node.threshold in tree
   densities/destructors. These should be log values if possible.
-  Change all ``n_dim`` and ``n_dim_`` to ``n_features`` and
   ``n_features_`` to conform with scikit-learn style. Likely this would
   only take a global replace all but would probably want to test this.
-  Reformat atomic density destructors (i.e. non-composite that inherit
   from ``BaseDensityDestructor``) to take a density as the main (or
   only) parameter rather than replicating the density parameters.
-  Add mutability test for ``transform``, ``inverse_transform`` and
   ``score_samples`` (maybe others) to ``check_destructor``.
-  Setup a test suite to check all common destructors (ideally with
   continuous testing/integration).
-  Add more documentation.

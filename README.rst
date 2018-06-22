.. -*- mode: rst -*-

|CircleCI|_ |ReadtheDocs|_

.. |CircleCI| image:: https://circleci.com/gh/davidinouye/destructive-deep-learning/tree/master.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/davidinouye/destructive-deep-learning
.. |ReadtheDocs| image:: https://readthedocs.org/projects/destructive-deep-learning/badge/?version=latest
.. _ReadtheDocs: https://destructive-deep-learning.readthedocs.io/en/latest/?badge=latest

======================================
Destructive Deep Learning (ddl) README
======================================

.. default-role:: any

Destructive deep learning estimators and functions.
Estimators are compatible with scikit-learn.
Source code is distributed under the BSD 3-clause license.

Please cite the following paper if you use this code:

    | `Deep Density Destructors`_ (`pdf`_)
    | David I. Inouye, Pradeep Ravikumar
    | To appear in *International Conference on Machine Learning* (ICML), 2018.

.. _`Deep Density Destructors`: http://www.cs.cmu.edu/~dinouye/papers/inouye2018-deep-density-destructors-icml2018.pdf
.. _`pdf`: http://www.cs.cmu.edu/~dinouye/papers/inouye2018-deep-density-destructors-icml2018.pdf

Documentation
-------------

Please see the `API reference`_ for basic documentation.

.. _`API reference`: https://destructive-deep-learning.readthedocs.io/en/latest/

Environment Setup
-----------------

Environment setup instructions for:
1. Docker or Singularity containers (recommended)
2. Linux Setup (Ubuntu)
3. Mac OSX (unsupported)

1. Docker or Singularity Setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Because `MLPACK`_ is required for the tree density destructors used in the experiments,
the suggested installation method is to download and start a shell in a `Docker <https://www.docker.com/>`_
or `Singularity <http://singularity.lbl.gov/>`_ container as below.  
(If you are using `Docker for Mac`_ or `Docker for Windows`_, you will probably have 
to increase the available memory to Docker for these experiments. See Docker documentation.)
For Docker (recommended if available):

.. _`MLPACK`: http://mlpack.org/

.. _`Docker for Mac`: https://docs.docker.com/docker-for-mac/

.. _`Docker for Windows`: https://docs.docker.com/docker-for-windows/

.. code:: console

    docker run -it davidinouye/destructive-deep-learning:icml2018 /bin/bash


Or, for Singularity:

.. code:: console

    singularity shell -s /bin/bash shub://davidinouye/destructive-deep-learning:icml2018

2. Linux Setup (Ubuntu)
^^^^^^^^^^^^^^^^^^^^^^^
Install build essentials and cmake (needed for building mlpack destructors), laplack and blas (for fast linear operations), boost and armadillo libraries (required to build mlpack).

.. code:: console

    apt-get update && apt-get install \
        build-essential \
        cmake \
        liblapack-dev \
        libblas-dev \
        libboost-math-dev \
        libboost-program-options-dev \
        libboost-test-dev \
        libboost-serialization-dev \
        libarmadillo-dev

3. Mac OSX (unsupported)
^^^^^^^^^^^^^^^^^^^^^^^^
Install ``homebrew`` as per `homebrew documentation <https://brew.sh/>`_ (note that this also install the required xcode tools).
Then, install cmake, armadillo and boost (required to build mlpack) and llvm (required for openmp support for mlpack):

.. code:: console
    brew update && brew install \
        cmake \
        armadillo \
        boost \
        llvm


Installation
------------

Once your environment is setup via one of the methods described above, download and compile the code to link to `MLPACK`_.

.. code:: console

    git clone https://github.com/davidinouye/destructive-deep-learning.git
    cd destructive-deep-learning
    make

To run tests (which uses `pytest <https://docs.pytest.org/en/latest/>`_), execute:

.. code:: console

    make test

Reproduce experiments from ICML 2018 paper
------------------------------------------

NOTE: `MLPACK`_ is required to reproduce experiments, please
see installation instructions. 

To reproduce the 2D experiment in the paper and generate the paper figures
open and run the notebook `notebooks/demo_toy_experiment.ipynb <notebooks/demo_toy_experiment.ipynb>`_ 
or run the notebook from the command line.
Note that this notebook may take a while to run.
Also, if the command below is interrupted with Ctrl+C, the underlying python process
may need to be killed manually.

.. code:: console

    jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute notebooks/demo_toy_experiment.ipynb

To reproduce the MNIST and CIFAR-10 experiments execute the command below.
Note that this script will download the MNIST and CIFAR-10 datasets into 
``data/download_cache`` if not downloaded already.
The results are stored in ``data/results`` both the log files and pickle files
that include the fitted models.
Note that the log files will always append to the previous log file rather
than overwriting the existing log file.

.. code:: console

    # Download data cache directly since mldata.org is sometimes down
    wget http://www.cs.cmu.edu/~dinouye/data/data-icml2018.tar.gz && tar -xzvf data-icml2018.tar.gz && rm data-icml2018.tar.gz

    # Example command for deep copula model and MNIST data
    python scripts/icml_2018_experiment.py --model_names=deep-copula --data_names=mnist

    # View tail of output log files
    tail data/results/data-mnist_model-deep-copula_n_jobs-1.log 

    # Command for all models and datasets (using commas to separate)
    python scripts/icml_2018_experiment.py --model_names=deep-copula,image-pairs-copula,image-pairs-tree --data_names=mnist,cifar10

    # Command to run all experiments in parallel using subprocesses
    python scripts/icml_2018_experiment.py --model_names=deep-copula,image-pairs-copula,image-pairs-tree --data_names=mnist,cifar10 --parallel_subprocesses=True 


============
Contributing
============

General coding guidelines
-------------------------

Please read through the following high-level guidelines:

1. Zen of Python - https://www.python.org/dev/peps/pep-0020/
2. Python style guidelines - https://www.python.org/dev/peps/pep-0008/
3. ``scikit-learn`` coding guidelines -
   http://scikit-learn.org/stable/developers/contributing.html#coding-guidelines

Project-specific guidelies
--------------------------

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

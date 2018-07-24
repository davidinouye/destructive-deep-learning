Changelog
==========

All notable changes to this project will be documented in this file.

The format is based on `Keep a
Changelog <http://keepachangelog.com/en/1.0.0/>`__ and this project
adheres to `Semantic Versioning <http://semver.org/spec/v2.0.0.html>`__.

[Unreleased]
------------

Added
^^^^^

-  Placeholder @davidinouye.

   -  Subitem placeholder @davidinouye

Changed
^^^^^^^

- Changed ``n_dim`` to ``n_features`` everywhere in the source to match with scikit-learn convention.
- Moved ``ddl.deep.CompositeDestructor`` to ``ddl.base.CompositeDestructor`` since it seemed more like
  a fundamental building block rather than a deep component.
- Removed explicit dependency on pre-built mlpack and updated docker/singularity images.
- Updated build system especially for circleci to make it simpler to build mlpack extension.

Removed
^^^^^^^

-  Placeholder @davidinouye.


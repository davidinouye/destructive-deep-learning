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

- Added ``create_fitted`` class method to many densities and destructors to make manual creation
  of fitted estimators simpler and more uniform.
- Added ``ddl.base.create_inverse_transformer`` to create inverse transformations that may not be
  destructors (e.g., an inverse of a ``ddl.linear.LinearProjector``).
- Added bias/shift term to ``ddl.linear.LinearProjector``

Changed
^^^^^^^

- Fixed bug in ``ddl.univariate.HistogramUnivariateDensity`` when bin widths were not uniform.
  Scipy histogram do not normalize for bin widths so that is needed before calling Scipy histogram.
- Updated implementation of ``ddl.base.create_implicit_destructor`` 
  and ``ddl.base.create_inverse_canonical_destructor``
- Renamed ``ddl.base.get_implicit_density`` to ``ddl.base.create_implicit_destructor``
  and renamed ``ddl.base.get_inverse_canonical_destructor`` to 
  ``ddl.base.create_inverse_canonical_destructor``
  (with previous names kept for backwards compatability but will issue a ``DeprecationWarning``).
- Fixed various small bugs.


[0.0.2] - 2018-08-21
---------

Added
^^^^^

- Significant documentation has been added including:
   - New demo notebooks
   - Docstrings for many classes and functions

Changed
^^^^^^^

- Changed ``n_dim`` to ``n_features`` everywhere in the source to match with scikit-learn convention.
- Moved ``ddl.deep.CompositeDestructor`` to ``ddl.base.CompositeDestructor`` since it seemed more like
  a fundamental building block rather than a deep component.
- Fixed functionality of ``ddl.base.CompositeDestructor``, ``ddl.deep.DeepDestructor`` and
  ``ddl.deep.DeepDestructorCV`` to use the ``random_state`` parameter to set the global random state
  so that the random state for each sub destructor does not need to be set manually.
- Made several univariate densities private since not used outside of module.
- Merged ``ddl.deep.DeepCVMixin`` into ``ddl.deep.DeepDestructorCV`` since no useful case for mixin.
- Removed explicit dependency on pre-built mlpack and updated docker/singularity images.
- Updated build system especially for circleci to make it simpler to build mlpack extension.
- Simplified :class:`ddl.univariate.HistogramUnivariateDensity` implementation by merely using scipy's :func:`scipy.stats.rv_histogram`

Removed
^^^^^^^

- Removed :class:`ddl.base.AutoregressiveMixin` because it was not needed.
- Removed (by making them private) the validation functions :func:`ddl.validation.check_destructor_interface`, :func:`ddl.validation.check_uniformability`, :func:`ddl.validation.check_invertibility`, :func:`ddl.validation.check_canonical_domain`, :func:`ddl.validation.check_identity_element`, because these are already called in :func:`ddl.validation.check_destructor` and :param:`properties_to_skip` can be used to check only one property if needed.

[0.0.1] - 2018-06-11
--------------------

Added
^^^^^
- Initial code release but not cleaned up.
- Note this was only released on pypi mainly just to reserve the name

"""Base destructors and destructor mixins."""
from __future__ import division, print_function

import logging
import warnings
from abc import abstractmethod
from builtins import super
from copy import deepcopy
from functools import wraps

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

# noinspection PyProtectedMember
from .utils import (_INF_SPACE, _UNIT_SPACE, check_X_in_interval, get_domain_or_default,
                    get_support_or_default)

logger = logging.getLogger(__name__)


class ScoreMixin(object):
    """Mixin for :func:`score` that returns mean of :func:`score_samples`."""

    def score(self, X, y=None):
        """Return the mean log likelihood (or log(det(Jacobian))).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples and n_features
            is the number of features.

        y : None, default=None
            Not used but kept for compatibility.

        Returns
        -------
        log_likelihood : float
            Mean log likelihood data points in X.

        """
        return np.mean(self.score_samples(X, y))


class DestructorMixin(ScoreMixin, TransformerMixin):
    """Mixin helper class to add universal destructor methods.

    Adds ``sample``, ``get_domain``, and ``score`` *if* the destructor
    defines the ``density_`` attribute after fitting. (Also, if the
    destructor defines the attribute ``n_features_``, no sampling is
    required to determine the number of features, see note below.)

    Note that this finds the data dimension by looking sequentally for
    the fitted ``n_features_`` attribute, the ``density_.n_features_``
    attribute, and finally attempting to call `self.density_.sample(1)`
    and determine the dimension from the density sample.
    """

    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from this density/destructor.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate. Defaults to 1.

        random_state : int, RandomState instance or None, optional (default=None)
            If int, `random_state` is the seed used by the random number
            generator; If :class:`~numpy.random.RandomState` instance,
            `random_state` is the random number generator; If None, the random
            number generator is the :class:`~numpy.random.RandomState` instance
            used by :mod:`numpy.random`.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample.

        """
        rng = check_random_state(random_state)
        U = rng.rand(n_samples, self._get_n_features())
        X = self.inverse_transform(U)
        return X

    # Utility method to attempt to automatically determine the number of dimensions.
    def _get_n_features(self):
        return get_n_features(self)


def get_n_features(destructor, try_destructor_sample=False):
    """Get the number of features for a fitted destructor.

    Attempt to find ``n_features`` either from
    ``destructor.n_features_``, ``destructor.density_.n_features_``,
    or via density sampling ``destructor.density_.sample(1,
    random_state=0).shape[1]``.


    Parameters
    ----------
    destructor : estimator
        The (fitted) destructor from which to extract the number of features.
    try_destructor_sample : bool, optional, default=False
        If ``True``, additionally attempt ``destructor.sample(1,
        random_state=0).shape[ 1]``. This option could cause infinite
        recursion since :class:`~ddl.base.DestructorMixin` uses
        :func:`get_n_features` in order to sample but this can be avoided if
        the destructor reimplements sample without :func:`get_n_features`
        such as in the :class:`ddl.base.CompositeDestructor`.

    """
    n_features = np.nan
    if hasattr(destructor, 'n_features_'):
        n_features = destructor.n_features_
    elif hasattr(destructor, 'density_') and hasattr(destructor.density_, 'n_features_'):
        n_features = destructor.density_.n_features_
    elif hasattr(destructor, 'density_') and hasattr(destructor.density_, 'sample'):
        warnings.warn('Because `destructor.n_features_` does not exist and'
                      ' `destructor.density_.n_features_` does not exist'
                      ' we attempt to determine the dimension by sampling'
                      ' from destructor.density_, which may be computationally'
                      ' demanding.  Add destructor.n_features_ to reduce time if necessary.',
                      _NumDimWarning)
        n_features = np.array(destructor.density_.sample(n_samples=1, random_state=0)).shape[1]
    else:
        if try_destructor_sample:
            # Attempt to sample from destructor
            if hasattr(destructor, 'sample'):
                try:
                    n_features = np.array(
                        destructor.sample(n_samples=1, random_state=0)
                    ).shape[1]
                except RuntimeError:
                    err = True
                else:
                    err = False
            else:
                err = True
            if err:
                raise RuntimeError(
                    'Could not find n_features in destructor.n_features_, '
                    'destructor.density_.n_features_, '
                    'destructor.density_.sample(1).shape[1], or destructor.sample('
                    '1).shape[1]. '
                )
        else:
            raise RuntimeError('Could not find n_features in destructor or density.'
                               'Checked destructor.n_features_, destructor.density_.n_features_, '
                               'and '
                               ' attempted to sample from destructor.density_ to determine'
                               ' n_features but failed in all cases.')
    return n_features


class BoundaryWarning(DataConversionWarning):
    """Warning that data is on the boundary of the required set.

    Warning when data is on the boundary of the domain or range and
    is converted to data that lies inside the boundary. For example, if
    the domain is (0,inf) rather than [0,inf), values of 0 will be made
    a small epsilon above 0.
    """


class _NumDimWarning(UserWarning):
    """Warning about the number of dimensions.

    Warning that we have to use 1 sample in order to determine the
    number of dimensions. (Because `trans.n_features_` does not exist and
    ``trans.density_.n_features_` does not exist we attempt to determine the
    dimension by sampling from self.density_, which may be
    computationally demanding.  Add self.n_features_ to reduce time if
    necessary.)
    """


class BaseDensityDestructor(BaseEstimator, DestructorMixin):
    """Abstract destructor derived from an explicit underlying density.

    This should be used if the destructor is based on an *explicit*
    underlying density such as a ``TreeDestructor`` or
    ``IndepedentDestructor``.

    The only methods that need to be implemented in this case are
    ``get_density_estimator``, ``transform`` and ``inverse_transform``.

    Attributes
    ----------
    density_ : estimator
        Fitted underlying density.

    """

    @abstractmethod
    def _get_density_estimator(self):
        """(Abstract) Get density estimator."""
        raise NotImplementedError()

    @abstractmethod
    def transform(self, X, y=None):
        """[Placeholder].

        Parameters
        ----------
        X :
        y :

        """
        raise NotImplementedError()

    @abstractmethod
    def inverse_transform(self, X, y=None):
        """[Placeholder].

        Parameters
        ----------
        X :
        y :

        """
        raise NotImplementedError()

    def fit(self, X, y=None, density_fit_params=None):
        """[Placeholder].

        Parameters
        ----------
        X :
        y :
        density_fit_params :

        Returns
        -------
        obj : object

        """
        if density_fit_params is None:
            density_fit_params = {}
        density = clone(self._get_density_estimator()).fit(X, y, **density_fit_params)
        self.fit_from_density(density)
        return self

    def fit_from_density(self, density):
        """[Placeholder].

        Parameters
        ----------
        density :

        Returns
        -------
        obj : object

        """
        self.density_ = density
        return self

    def score_samples(self, X, y=None):
        """Compute log-likelihood (or log(det(Jacobian))) for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples and n_features
            is the number of features.

        y : None, default=None
            Not used but kept for compatibility.

        Returns
        -------
        log_likelihood : array, shape (n_samples,)
            Log likelihood of each data point in X.

        """
        self._check_is_fitted()
        X = check_array(X, ensure_min_samples=0)
        X = check_X_in_interval(X, get_domain_or_default(self))
        return self.density_.score_samples(X)

    def get_domain(self):
        """Get the domain of this destructor.

        Returns
        -------
        domain : array-like, shape (2,) or shape (n_features, 2)
            If shape is (2, ), then ``domain[0]`` is the minimum and
            ``domain[1]`` is the maximum for all features. If shape is
            (`n_features`, 2), then each feature's domain (which could
            be different for each feature) is given similar to the first
            case.

        """
        # Either get from the density estimator parameter
        #  or fitted density attribute
        try:
            self._check_is_fitted()
        except NotFittedError:
            return get_support_or_default(self._get_density_estimator())
        else:
            return get_support_or_default(self.density_)

    def _check_is_fitted(self):
        check_is_fitted(self, ['density_'])


class IdentityDestructor(BaseDensityDestructor):
    """Identity destructor/transform.

    This assumes a canonical uniform density on the unit hypercube and
    has a domain of [0, 1].

    Attributes
    ----------
    density_ : estimator
        Fitted underlying density.

    See Also
    --------
    UniformDensity

    """

    @classmethod
    def create_fitted(cls, n_features):
        destructor = cls()
        destructor.density_ = UniformDensity.create_fitted(n_features)
        return destructor

    def _get_density_estimator(self):
        """Get the *unfitted* density associated with this destructor.

        NOTE: The returned estimator is NOT fitted but is a clone or new
        instantiation of the underlying density estimator. This is just
        a helper function that needs to be overridden by subclasses of
        :class:`~ddl.base.BaseDensityDestructor`.

        Returns
        -------
        density : estimator
            The *unfitted* density estimator associated wih this
            destructor.

        """
        return UniformDensity()

    def transform(self, X, y=None, copy=True):
        """[Placeholder].

        Parameters
        ----------
        X :
        y :
        copy :

        Returns
        -------
        obj : object

        """
        self._check_is_fitted()
        X = check_array(X, ensure_min_samples=0)
        X = check_X_in_interval(X, get_domain_or_default(self))
        self._check_dim(X)
        if copy:
            X = X.copy()
        return X

    def inverse_transform(self, X, y=None, copy=True):
        """[Placeholder].

        Parameters
        ----------
        X :
        y :
        copy :

        Returns
        -------
        obj : object

        """
        self._check_is_fitted()
        X = check_array(X, ensure_min_samples=0)
        X = check_X_in_interval(X, np.array([0, 1]))
        self._check_dim(X)
        if copy:
            X = X.copy()
        return X

    def get_domain(self):
        """Get the domain of this destructor.

        Returns
        -------
        domain : array-like, shape (2,) or shape (n_features, 2)
            If shape is (2, ), then ``domain[0]`` is the minimum and
            ``domain[1]`` is the maximum for all features. If shape is
            (`n_features`, 2), then each feature's domain (which could
            be different for each feature) is given similar to the first
            case.

        """
        return np.array([0, 1])

    def _check_dim(self, X):
        if X.shape[1] != self.density_.n_features_:
            raise ValueError('Dimension of input does not match dimension of the original '
                             'training data.')


class UniformDensity(BaseEstimator, ScoreMixin):
    """Uniform density estimator.

    Only the ``n_features_`` attribute needs fitting. This nearly
    trivial density is used as the underlying density for the
    ``IdentityDestructor``.

    Attributes
    ----------
    n_features_ : int
        Number of features of the training data.

    See Also
    --------
    IdentityDestructor

    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Fit estimator to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : None, default=None
            Not used in the fitting process but kept for compatibility.

        Returns
        -------
        self : estimator
            Returns the instance itself.

        """
        X = check_array(X)
        X = check_X_in_interval(X, get_support_or_default(self))
        self.n_features_ = X.shape[1]
        return self

    @classmethod
    def create_fitted(cls, n_features):
        density = cls()
        density.n_features_ = n_features
        return density

    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from this density/destructor.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate. Defaults to 1.

        random_state : int, RandomState instance or None, optional (default=None)
            If int, `random_state` is the seed used by the random number
            generator; If :class:`~numpy.random.RandomState` instance,
            `random_state` is the random number generator; If None, the random
            number generator is the :class:`~numpy.random.RandomState` instance
            used by :mod:`numpy.random`.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample.

        """
        self._check_is_fitted()
        generator = check_random_state(random_state)
        return generator.rand(n_samples, self.n_features_)

    def score_samples(self, X, y=None):
        """Compute log-likelihood (or log(det(Jacobian))) for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples and n_features
            is the number of features.

        y : None, default=None
            Not used but kept for compatibility.

        Returns
        -------
        log_likelihood : array, shape (n_samples,)
            Log likelihood of each data point in X.

        """
        self._check_is_fitted()
        X = check_array(X, ensure_min_samples=0)
        return np.zeros(X.shape[0])  # Log-likelihood so log(1) = 0

    # noinspection PyMethodMayBeStatic
    def get_support(self):
        """Get the support of this density (i.e. the positive density region).

        Returns
        -------
        support : array-like, shape (2,) or shape (n_features, 2)
            If shape is (2, ), then ``support[0]`` is the minimum and
            ``support[1]`` is the maximum for all features. If shape is
            (`n_features`, 2), then each feature's support (which could
            be different for each feature) is given similar to the first
            case.

        """
        return np.array([0, 1])

    def _check_is_fitted(self):
        check_is_fitted(self, ['n_features_'])


def create_implicit_density(fitted_destructor, copy=False):
    """Create the implicit density associated with a fitted destructor.

    Extracting the *implicit* density associated with an already-fitted
    destructor must be handled carefully to enable proper ``sklearn``
    cloning and ``check_destructor`` tests that require the
    ``n_features_`` attribute to be available. Thus we have implemented
    this method instead of explicitly exposing an implicit density class.

    Parameters
    ----------
    fitted_destructor : estimator
        A fitted destructor estimator from which to construct the implicit
        density.

    copy : bool
        If ``copy=True``, the new destructor will create a deep copy of the
        fitted destructor rather than just copying a reference to it.

    Returns
    -------
    density : _ImplicitDensity

    """
    return _ImplicitDensity(
        destructor=fitted_destructor
    ).fit(None, y=None, copy=copy, transformer_already_fitted=True)


def get_implicit_density(*args, **kwargs):
    warnings.warn(DeprecationWarning(
        'Should use `create_implicit_density` instead'
    ))
    return create_implicit_density(*args, **kwargs)


def create_inverse_transformer(fitted_transformer, copy=False):
    """Create inverse transformer from fitted transformer.

    Note that only a canonical transformer has an inverse which is also a
    transformer. See ``get_inverse_canonical_transformer``.

    Parameters
    ----------
    fitted_transformer : estimator
        A fitted transformer from which to construct the implicit
        inverse transformer.

    copy : bool
        If ``copy=True``, the new transformer will create a deep copy of the
        fitted transformer rather than just copying a reference to it.


    Returns
    -------
    transformer : _InverseDestructor

    """
    return _InverseTransformer(
        transformer=fitted_transformer
    ).fit(None, y=None, copy=copy, transformer_already_fitted=True)


def create_inverse_canonical_destructor(fitted_canonical_destructor, copy=False):
    """Create inverse destructor of a fitted *canonical* destructor.

    Note that only a canonical destructor has an inverse which is also a
    destructor.

    Extracting the inverse destructor associated with an already-fitted
    destructor must be handled carefully to enable proper ``sklearn``
    cloning and ``check_destructor`` tests that require the
    ``n_features_`` attribute to be available. Thus we have implemented
    this method instead of explicitly exposing an implicit density class.

    Parameters
    ----------
    fitted_canonical_destructor : estimator
        A fitted *canonical* destructor from which to construct the implicit
        inverse destructor.

    copy : bool
        If ``copy=True``, the new destructor will create a deep copy of the
        fitted destructor rather than just copying a reference to it.


    Returns
    -------
    destructor : _InverseCanonicalDestructor

    """
    return _InverseCanonicalDestructor(
        transformer=fitted_canonical_destructor, output_space=_UNIT_SPACE
    ).fit(None, y=None, copy=copy, transformer_already_fitted=True)


def get_inverse_canonical_destructor(*args, **kwargs):
    warnings.warn(DeprecationWarning(
        'Should use `create_inverse_canonical_destructor` instead'
    ))
    return create_inverse_canonical_destructor(*args, **kwargs)


class _InverseTransformer(BaseEstimator, ScoreMixin, TransformerMixin):
    """An inverse of a transformer (might not be transformer)."""

    def __init__(self, transformer=None, output_space=None):
        self.transformer = transformer
        self.output_space = output_space

    def _get_transformer(self):
        check_is_fitted(self, ['fitted_transformer_'])
        return self.fitted_transformer_

    def fit(self, X, y=None, copy=False, transformer_already_fitted=False):
        """[Placeholder].

        Parameters
        ----------
        X :
        y :
        copy :
        transformer_already_fitted :

        Returns
        -------
        obj : object

        """
        if transformer_already_fitted:
            self.fitted_transformer_ = self.transformer
            if copy:
                self.fitted_transformer_ = deepcopy(self.fitted_transformer_)
        else:
            self.fitted_transformer_ = clone(self.transformer).fit(X, y)

        if self.output_space is not None:
            self.domain_ = self.output_space
        else:
            self.domain_ = _INF_SPACE

        self.n_features_ = get_n_features(self.fitted_transformer_)
        return self

    def transform(self, X, y=None):
        """Apply destructive transformation to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : None, default=None
            Not used in the transformation but kept for compatibility.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_features)
            Transformed data.

        """
        return self._get_transformer().inverse_transform(X, y)

    def inverse_transform(self, X, y=None):
        """Apply inverse destructive transformation to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : None, default=None
            Not used in the transformation but kept for compatibility.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_features)
            Transformed data.

        """
        return self._get_transformer().transform(X, y)

    def score_samples(self, X, y=None):
        """Compute log-likelihood (or log(det(Jacobian))) for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples and n_features
            is the number of features.

        y : None, default=None
            Not used but kept for compatibility.

        Returns
        -------
        log_likelihood : array, shape (n_samples,)
            Log likelihood of each data point in X.

        """
        d = self._get_transformer()
        return -d.score_samples(d.inverse_transform(X, y))

    def get_domain(self):
        """Get the domain of this destructor.

        Returns
        -------
        domain : array-like, shape (2,) or shape (n_features, 2)
            If shape is (2, ), then ``domain[0]`` is the minimum and
            ``domain[1]`` is the maximum for all features. If shape is
            (`n_features`, 2), then each feature's domain (which could
            be different for each feature) is given similar to the first
            case.

        """
        if hasattr(self, 'domain_'):
            return self.domain_
        else:
            return _INF_SPACE


class _InverseCanonicalDestructor(_InverseTransformer, DestructorMixin):
    """An inverse canonical destructor, which is also a destructor.

    There is a slight technical condition that the canonical destructor
    must uniquely map every point of the unit hypercube (or similarly
    that the associated density has support everywhere in the hypercube).

    """

    def fit(self, X, y=None, **kwargs):
        super().fit(X, y=y, **kwargs)
        self.density_ = create_implicit_density(
            self, copy=False)  # Copy has already occurred above if needed
        self.domain_ = _UNIT_SPACE
        return self

    def get_domain(self):
        """Get the domain of this destructor.

        Returns
        -------
        domain : array-like, shape (2,) or shape (n_features, 2)
            If shape is (2, ), then ``domain[0]`` is the minimum and
            ``domain[1]`` is the maximum for all features. If shape is
            (`n_features`, 2), then each feature's domain (which could
            be different for each feature) is given similar to the first
            case.

        """
        return _UNIT_SPACE


class _ImplicitDensity(BaseEstimator, ScoreMixin):
    """The density implied by a destructor which can already be fitted."""

    def __init__(self, destructor=None):
        self.destructor = destructor

    def _get_destructor(self):
        check_is_fitted(self, ['fitted_destructor_'])
        return self.fitted_destructor_

    def fit(self, X, y=None, copy=False, transformer_already_fitted=False):
        """[Placeholder].

        Parameters
        ----------
        X :
        y :
        copy :
        transformer_already_fitted :

        Returns
        -------
        obj : object

        """
        if transformer_already_fitted:
            self.fitted_destructor_ = self.destructor
            if copy:
                self.fitted_destructor_ = deepcopy(self.fitted_destructor_)
        else:
            self.fitted_destructor_ = clone(self.destructor).fit(X, y)
        return self

    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from this density/destructor.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate. Defaults to 1.

        random_state : int, RandomState instance or None, optional (default=None)
            If int, `random_state` is the seed used by the random number
            generator; If :class:`~numpy.random.RandomState` instance,
            `random_state` is the random number generator; If None, the random
            number generator is the :class:`~numpy.random.RandomState` instance
            used by :mod:`numpy.random`.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample.

        """
        return self._get_destructor().sample(
            n_samples=n_samples, random_state=random_state)

    def score_samples(self, X, y=None):
        """Compute log-likelihood (or log(det(Jacobian))) for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples and n_features
            is the number of features.

        y : None, default=None
            Not used but kept for compatibility.

        Returns
        -------
        log_likelihood : array, shape (n_samples,)
            Log likelihood of each data point in X.

        """
        return self._get_destructor().score_samples(X, y)

    def get_support(self):
        """Get the support of this density (i.e. the positive density region).

        Returns
        -------
        support : array-like, shape (2,) or shape (n_features, 2)
            If shape is (2, ), then ``support[0]`` is the minimum and
            ``support[1]`` is the maximum for all features. If shape is
            (`n_features`, 2), then each feature's support (which could
            be different for each feature) is given similar to the first
            case.

        """
        return get_domain_or_default(self.destructor)


def _check_global_random_state(f):
    """Decorate function to save, set and reset the global random state.

    This is useful for composite or deep destructors where one does not
    want to set the random_state for each individual destructor but
    wants exact reproducibility.
    """
    @wraps(f)
    def decorated(self, *args, **kwargs):
        """[Placeholder].

        Parameters
        ----------
        self :
        args :
        kwargs :

        Returns
        -------
        obj : object

        """
        # If random_state is None then Just call function directly
        if self.random_state is None:
            return f(self, *args, **kwargs)

        # Save original global random state
        #  and seed global random state
        saved_random_state = np.random.get_state()
        rng = check_random_state(self.random_state)
        np.random.set_state(rng.get_state())

        # Call function and then reset global random state
        ret_val = f(self, *args, **kwargs)
        np.random.set_state(saved_random_state)
        return ret_val
    return decorated


class CompositeDestructor(BaseEstimator, DestructorMixin):
    """Meta destructor composed of multiple destructors.

    This meta destructor composes multiple destructors or other
    transformations (e.g. relative destructors like LinearProjector)
    into a single composite destructor. This is a fundamental building
    block for creating more complex destructors from simple atomic
    destructors.

    Parameters
    ----------
    destructors : list
        List of destructor estimators to use as subdestructors.

    random_state : int, RandomState instance or None, optional (default=None)
        Global random state used if any of the subdestructors are
        random-based. By seeding the global :mod:`numpy.random`` via
        `random_state` and then resetting to its previous state,
        we can avoid having to carefully pass around random states for
        random-based sub destructors.

        If int, `random_state` is the seed used by the random number
        generator; If :class:`~numpy.random.RandomState` instance,
        `random_state` is the random number generator; If None, the random
        number generator is the :class:`~numpy.random.RandomState` instance
        used by :mod:`numpy.random`.

    Attributes
    ----------
    fitted_destructors_ : list
        List of fitted (sub)destructors. (Note that these objects are cloned
        via ``sklearn.base.clone`` from the ``destructors`` parameter so as
        to avoid mutating the ``destructors`` parameter.)

    density_ : estimator
        *Implicit* density of composite destructor.

    """

    def __init__(self, destructors=None, random_state=None):
        self.destructors = destructors
        self.random_state = random_state

    def fit(self, X, y=None, **fit_params):
        """Fit estimator to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : None, default=None
            Not used in the fitting process but kept for compatibility.

        fit_params : dict, optional
            Optional extra fit parameters.

        Returns
        -------
        self : estimator
            Returns the instance itself.

        """
        self.fit_transform(X, y, **fit_params)
        return self

    @_check_global_random_state
    def fit_transform(self, X, y=None, **fit_params):
        """Fit estimator to X and then transform X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : None, default=None
            Not used in the fitting process but kept for compatibility.

        fit_params : dict, optional
            Parameters to pass to the fit method.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_features)
            Transformed data.

        """
        Z = check_array(X, copy=True)

        # Fit and transform all destructors
        destructors = []
        for d in self._get_destructor_iterable():
            Z = self._single_fit_transform(d, Z, y)
            destructors.append(d)
            if np.any(np.isnan(Z)):
                raise RuntimeError('Need to check')

        self.fitted_destructors_ = np.array(destructors)
        self.density_ = create_implicit_density(self)
        return Z

    @classmethod
    def create_fitted(cls, fitted_destructors, **kwargs):
        """Create fitted destructor.

        Parameters
        ----------
        fitted_destructors : array-like of Destructor
            Fitted destructors.

        **kwargs
            Other parameters to pass to constructor.

        Returns
        -------
        fitted_transformer : Transformer
            Fitted transformer.

        """
        destructor = cls(**kwargs)
        destructor.fitted_destructors_ = np.array(fitted_destructors)
        destructor.density_ = create_implicit_density(destructor)
        return destructor

    def _single_fit_transform(self, d, Z, y):
        if y is not None:
            pass
            # warnings.warn('y is not None but this is not an adversarial composite/deep'
            #               'destructor. '
            #               'Did you mean to use an adversarial version of this destructor?')
        return d.fit(Z, y).transform(Z, y)

    def transform(self, X, y=None, partial_idx=None):
        """Apply destructive transformation to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : None, default=None
            Not used in the transformation but kept for compatibility.

        partial_idx : list or None, default=None
            List of indices of the fitted destructor to use in
            the transformation. The default of None uses all
            the fitted destructors. Mainly used for visualization
            or debugging.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_features)
            Transformed data (possibly only partial transformation).

        """
        self._check_is_fitted()
        Z = check_array(X, copy=True)

        fitted_destructors = self._get_partial_destructors(partial_idx)
        for d in fitted_destructors:
            Z = d.transform(Z, y)
        return Z

    def inverse_transform(self, X, y=None, partial_idx=None):
        """Apply inverse destructive transformation to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : None, default=None
            Not used in the transformation but kept for compatibility.

        partial_idx : list or None, default=None
            List of indices of the fitted destructor to use in
            the transformation. The default of None uses all
            the fitted destructors. Mainly used for visualization
            or debugging.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_features)
            Transformed data (possibly only partial transformation).

        """
        self._check_is_fitted()
        Z = check_array(X, copy=True)

        fitted_destructors = self._get_partial_destructors(partial_idx)
        for d in reversed(fitted_destructors):
            Z = d.inverse_transform(Z, y)
        return Z

    def sample(self, n_samples=1, y=None, random_state=None):
        """Sample from composite destructor.

        Nearly the same as ``DestructorMixin.sample`` but the number of
        features is found from first fitted destructor to avoid recursion.
        """
        self._check_is_fitted()
        rng = check_random_state(random_state)
        n_features = get_n_features(self.fitted_destructors_[-1])
        U = rng.rand(n_samples, n_features)
        X = self.inverse_transform(U, y)
        return X

    def score_samples(self, X, y=None, partial_idx=None):
        """Compute log-likelihood (or log(det(Jacobian))) for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples and n_features
            is the number of features.

        y : None, default=None
            Not used but kept for compatibility.

        partial_idx : list or None, default=None
            List of indices of the fitted destructor to use in
            the computing the log likelihood. The default of None uses all
            the fitted destructors. Mainly used for visualization
            or debugging.

        Returns
        -------
        log_likelihood : array, shape (n_samples,)
            Log likelihood of each data point in X.

        """
        return np.sum(self.score_samples_layers(X, y, partial_idx=partial_idx), axis=1)

    def score_samples_layers(self, X, y=None, partial_idx=None):
        """[Placeholder].

        Parameters
        ----------
        X :
        y :
        partial_idx :

        Returns
        -------
        obj : object

        """
        self._check_is_fitted()
        X = check_array(X, copy=True)

        fitted_destructors = self._get_partial_destructors(partial_idx)
        log_likelihood_layers = np.zeros((X.shape[0], len(fitted_destructors)))
        for i, d in enumerate(fitted_destructors):
            log_likelihood_layers[:, i] = d.score_samples(X, y)
            # Don't transform for the last destructor
            if i < len(fitted_destructors) - 1:
                X = d.transform(X, y)
        return log_likelihood_layers

    def score(self, X, y=None, partial_idx=None):
        """Override super class to allow for partial_idx."""
        return np.mean(self.score_samples(X, y, partial_idx=partial_idx))

    def score_layers(self, X, y=None, partial_idx=None):
        """Override super class to allow for partial_idx."""
        return np.mean(self.score_samples_layers(X, y, partial_idx=partial_idx), axis=0)

    def get_domain(self):
        """Get the domain of this destructor.

        Returns
        -------
        domain : array-like, shape (2,) or shape (n_features, 2)
            If shape is (2, ), then ``domain[0]`` is the minimum and
            ``domain[1]`` is the maximum for all features. If shape is
            (`n_features`, 2), then each feature's domain (which could
            be different for each feature) is given similar to the first
            case.

        """
        # Get the domain of the first destructor (or relative destructor like LinearProjector)
        return next(iter(self._get_destructor_iterable())).get_domain()

    def _get_partial_destructors(self, partial_idx):
        if partial_idx is not None:
            return np.array(self.fitted_destructors_)[partial_idx]
        else:
            return self.fitted_destructors_

    def _get_destructor_iterable(self):
        if self.destructors is None:
            return [IdentityDestructor()]
        elif isinstance(self.destructors, (list, tuple, np.array)):
            return [clone(d) for d in self.destructors]
        else:
            raise ValueError('`destructors` must be a list, tuple or numpy array. Sets are not '
                             'allowed because order is important and general iterators/generators '
                             'are not allowed because we need the estimator parameters to stay '
                             'constant after inspecting.')

    def _check_is_fitted(self):
        check_is_fitted(self, ['fitted_destructors_'])

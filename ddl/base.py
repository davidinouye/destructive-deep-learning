from __future__ import division, print_function

from copy import deepcopy
import logging
import warnings
from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

# noinspection PyProtectedMember
from .utils import _UNIT_SPACE, check_X_in_interval, get_domain_or_default, get_support_or_default

logger = logging.getLogger(__name__)


class AutoregressiveMixin(object):
    """Core methods that autoregressive densities need to handle."""

    @abstractmethod
    def conditional_densities(self, X, cond_idx_arr, not_cond_idx_arr):
        raise NotImplementedError()

    @abstractmethod
    def marginal_cdf(self, x, target_idx):
        raise NotImplementedError()

    @abstractmethod
    def marginal_inverse_cdf(self, x, target_idx):
        raise NotImplementedError()


class ScoreMixin(object):
    """Simple returns mean of score_samples(), which should return log-likelihood."""

    def score(self, X, y=None):
        return np.mean(self.score_samples(X, y))


class DestructorMixin(ScoreMixin, TransformerMixin):
    """
    Adds `sample`, `get_domain`, and score *if* the destructor defines
    the `density_` attribute after fitting. (Also supplying `self.n_features_` can reduce
    some computation, see note below.)

    Note that this finds the data dimension by looking for the `self.n_features_`
    attribute, the `self.density_.n_features_` attribute and finally attempting
    to call `self.density_.sample(1)` and determine the dimension from the density
    sample.
    """

    def sample(self, n_samples=1, random_state=None):
        rng = check_random_state(random_state)
        U = rng.rand(n_samples, self._get_n_features())
        X = self.inverse_transform(U)
        return X

    # Utility method to attempt to automatically determine the number of dimensions.
    def _get_n_features(self):
        return get_n_features(self)


def get_n_features(destructor, try_destructor_sample=False):
    """Attempt to find n_features either from `destructor.n_features_`,
    `destructor.density_.n_features_`, or
    via density sampling `destructor.density_.sample(1, random_state=0).shape[1]`.
    If `try_destructor_sample=True`, additionally attempt 
    `destructor.sample(1, random_state=0).shape[1]`. This option could cause infinite recursion
    since `DestructorMixin` uses `get_n_features(destructor)` in order to sample but this can be
    avoided if the destructor reimplements sample without `get_n_features()` such as in the
    `CompositeDestructor`.
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
                      ' demanding.  Add destructor.n_features_ to reduce time if necessary.'
                      , _NumDimWarning)
        n_features = np.array(destructor.density_.sample(n_samples=1, random_state=0)).shape[1]
    else:
        if try_destructor_sample:
            # Attempt to sample from destructor
            if hasattr(destructor, 'sample'):
                try:
                    n_features = np.array(
                        fitted_destructor.sample(n_samples=1, random_state=0)
                    ).shape[1]
                except RuntimeError:
                    err = True
                else:
                    err = False
            else:
                err = True
            if err:
                raise RuntimeError(
                    'Could not find n_features in fitted_destructor.n_features_, '
                    'fitted_destructor.density_.n_features_, '
                    'fitted_destructor.density_.sample(1).shape[1], or fitted_destructor.sample('
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
    """Warning when data is on the boundary of the domain or range and
    is converted to data that lies inside the boundary. For example, if
    the domain is (0,inf) rather than [0,inf), values of 0 will be made
    a small epsilon above 0.
    """


class _NumDimWarning(UserWarning):
    """Warning that we have to use 1 sample in order to determine the
    number of dimensions. (Because `trans.n_features_` does not exist and
    ``trans.density_.n_features_` does not exist we attempt to determine the
    dimension by sampling from self.density_, which may be
    computationally demanding.  Add self.n_features_ to reduce time if
    necessary.)
    """


class BaseDensityDestructor(BaseEstimator, DestructorMixin):
    @abstractmethod
    def get_density_estimator(self):
        raise NotImplementedError()

    @abstractmethod
    def transform(self, X, y=None):
        raise NotImplementedError()

    @abstractmethod
    def inverse_transform(self, X, y=None):
        raise NotImplementedError()

    def fit(self, X, y=None, density_fit_params=None):
        if density_fit_params is None:
            density_fit_params = {}
        density = clone(self.get_density_estimator()).fit(X, y, **density_fit_params)
        self.fit_from_density(density)
        return self

    def fit_from_density(self, density):
        self.density_ = density
        return self

    def score_samples(self, X, y=None):
        self._check_is_fitted()
        X = check_array(X, ensure_min_samples=0)
        X = check_X_in_interval(X, get_domain_or_default(self))
        return self.density_.score_samples(X)

    def get_domain(self):
        # Either get from the density estimator parameter
        #  or fitted density attribute
        try:
            self._check_is_fitted()
        except NotFittedError:
            return get_support_or_default(self.get_density_estimator())
        else:
            return get_support_or_default(self.density_)

    def _check_is_fitted(self):
        check_is_fitted(self, ['density_'])


class IdentityDestructor(BaseDensityDestructor):
    def get_density_estimator(self):
        return UniformDensity()

    def transform(self, X, y=None, copy=True):
        self._check_is_fitted()
        X = check_array(X, ensure_min_samples=0)
        X = check_X_in_interval(X, get_domain_or_default(self))
        self._check_dim(X)
        if copy:
            X = X.copy()
        return X

    def inverse_transform(self, X, y=None, copy=True):
        self._check_is_fitted()
        X = check_array(X, ensure_min_samples=0)
        X = check_X_in_interval(X, np.array([0, 1]))
        self._check_dim(X)
        if copy:
            X = X.copy()
        return X

    def get_domain(self):
        return np.array([0, 1])

    def _check_dim(self, X):
        if X.shape[1] != self.density_.n_features_:
            raise ValueError('Dimension of input does not match dimension of the original '
                             'training data.')


class UniformDensity(BaseEstimator, ScoreMixin):
    """Uniform density estimator (no estimation necessary except number of dimensions)."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = check_array(X)
        X = check_X_in_interval(X, get_support_or_default(self))
        self.n_features_ = X.shape[1]
        return self

    def sample(self, n_samples=1, random_state=None):
        self._check_is_fitted()
        generator = check_random_state(random_state)
        return generator.rand(n_samples, self.n_features_)

    def score_samples(self, X, y=None):
        self._check_is_fitted()
        X = check_array(X, ensure_min_samples=0)
        return np.zeros(X.shape[0])  # Log-likelihood so log(1) = 0

    # noinspection PyMethodMayBeStatic
    def get_support(self):
        return np.array([0, 1])

    def _check_is_fitted(self):
        check_is_fitted(self, ['n_features_'])


def get_implicit_density(fitted_destructor, copy=False):
    """Returns an implicit density based on a fitted destructor.
    This must be handled carefully to enable proper sklearn cloning and check_destructor() tests
    that require n_features to be available.
    If copy=True, the new destructor will create a deep copy of the fitted destructor rather than
    just copying a reference to it.
    """
    return _ImplicitDensity(
        destructor=fitted_destructor
    ).fit(None, y=None, copy=copy, destructor_already_fitted=True)


def get_inverse_canonical_destructor(fitted_canonical_destructor, copy=False):
    """Returns the inverse of a fitted canonical destructor.
    This must be handled carefully to enable proper sklearn cloning and check_destructor() tests
    that require n_features to be available.
    If copy=True, the new destructor will create a deep copy of the fitted destructor rather than
    just copying a reference to it.
    """
    return _InverseCanonicalDestructor(
        canonical_destructor=fitted_canonical_destructor
    ).fit(None, y=None, copy=copy, destructor_already_fitted=True)


class ShouldOnlyBeInTestWarning(UserWarning):
    pass


class _InverseCanonicalDestructor(BaseEstimator, DestructorMixin):
    """Defines an inverse canonical destructor which is also a canonical destructor.
    There is a slight technical condition that the canonical destructor must uniquely
    map every point of the unit hypercube (or similarly that the associated density
    has support everywhere in the hypercube).
    """

    def __init__(self, canonical_destructor=None):
        self.canonical_destructor = canonical_destructor

    def _get_destructor(self):
        check_is_fitted(self, ['fitted_canonical_destructor_'])
        return self.fitted_canonical_destructor_

    def fit(self, X, y=None, copy=False, destructor_already_fitted=False):
        if destructor_already_fitted:
            self.fitted_canonical_destructor_ = self.canonical_destructor
            if copy:
                self.fitted_canonical_destructor_ = deepcopy(self.fitted_canonical_destructor_)
        else:
            self.fitted_canonical_destructor_ = clone(self.canonical_destructor).fit(X, y)

        self.n_features_ = get_n_features(self.fitted_canonical_destructor_)
        self.density_ = get_implicit_density(
            self, copy=False)  # Copy has already occurred above if needed
        return self

    def get_domain(self):
        return _UNIT_SPACE

    def transform(self, X, y=None):
        return self._get_destructor().inverse_transform(X, y)

    def inverse_transform(self, X, y=None):
        return self._get_destructor().transform(X, y)

    def score_samples(self, X, y=None):
        d = self._get_destructor()
        return -d.score_samples(d.inverse_transform(X, y))


class _ImplicitDensity(BaseEstimator, ScoreMixin):
    """The density implied by a destructor which can already be fitted."""

    def __init__(self, destructor=None):
        self.destructor = destructor

    def _get_destructor(self):
        check_is_fitted(self, ['fitted_destructor_'])
        return self.fitted_destructor_

    def fit(self, X, y=None, copy=False, destructor_already_fitted=False):
        if destructor_already_fitted:
            self.fitted_destructor_ = self.destructor
            if copy:
                self.fitted_destructor_ = deepcopy(self.fitted_destructor_)
        else:
            self.fitted_destructor_ = clone(self.destructor).fit(X, y)
        return self

    def sample(self, n_samples=1, random_state=None):
        return self._get_destructor().sample(
            n_samples=n_samples, random_state=random_state)

    def score_samples(self, X, y=None):
        return self._get_destructor().score_samples(X, y)

    def get_support(self):
        return get_domain_or_default(self.destructor)


class CompositeDestructor(BaseEstimator, DestructorMixin):
    """Joins multiple destructors (or relative destructors like
    LinearProjector) into a composite destructor.
    """

    def __init__(self, destructors=None, random_state=None):
        """`random_state` is needed if any of the atomic destructors are random-based.
        By seeding the global np.random via random_state and then resetting to its previous
        state, we can avoid having to carefully pass around random_states for random atomic
        destructors.
        """
        self.destructors = destructors
        self.random_state = random_state

    def fit(self, X, y=None, **fit_params):
        self.fit_transform(X, y, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        rng = check_random_state(self.random_state)
        # Save old random state and seed via internal random number generator
        # saved_random_state = np.random.get_state()
        # np.random.seed(rng.randint(2 ** 32, dtype=np.uint32))

        Z = check_array(X, copy=True)

        # Fit and transform all destructors
        destructors = []
        for d in self._get_destructor_iterable():
            Z = self._single_fit_transform(d, Z, y)
            destructors.append(d)
            if np.any(np.isnan(Z)):
                raise RuntimeError('Need to check')

        self.fitted_destructors_ = np.array(destructors)
        self.density_ = get_implicit_density(self)

        # Reset random state
        # np.random.set_state(saved_random_state)
        return Z

    def _single_fit_transform(self, d, Z, y):
        if y is not None:
            warnings.warn('y is not None but this is not an adversarial composite/deep destructor. '
                          'Did you mean to use an adversarial version of this destructor?')
        return d.fit_transform(Z, y)

    def transform(self, X, y=None, partial_idx=None):
        self._check_is_fitted()
        Z = check_array(X, copy=True)

        fitted_destructors = self._get_partial_destructors(partial_idx)
        for d in fitted_destructors:
            Z = d.transform(Z, y)
        return Z

    def inverse_transform(self, X, y=None, partial_idx=None):
        self._check_is_fitted()
        Z = check_array(X, copy=True)

        fitted_destructors = self._get_partial_destructors(partial_idx)
        for d in reversed(fitted_destructors):
            Z = d.inverse_transform(Z, y)
        return Z

    def sample(self, n_samples=1, random_state=None):
        """Nearly the same as `DestructorMixin.sample` but n_features is found from first 
        fitted destructor to avoid recursion.
        """
        self._check_is_fitted()
        rng = check_random_state(random_state)
        n_features = get_n_features(self.fitted_destructors_[-1])
        U = rng.rand(n_samples, n_features)
        X = self.inverse_transform(U)
        return X

    def score_samples(self, X, y=None, partial_idx=None):
        return np.sum(self.score_samples_layers(X, y, partial_idx=partial_idx), axis=1)

    def score_samples_layers(self, X, y=None, partial_idx=None):
        self._check_is_fitted()
        X = check_array(X, copy=True)

        fitted_destructors = self._get_partial_destructors(partial_idx)
        log_likelihood_layers = np.zeros((X.shape[0], len(fitted_destructors)))
        for i, d in enumerate(fitted_destructors):
            log_likelihood_layers[:, i] = d.score_samples(X)
            # Don't transform for the last destructor
            if i < len(fitted_destructors) - 1:
                X = d.transform(X, y)
        return log_likelihood_layers

    def score(self, X, y=None, partial_idx=None):
        """Overrides super class to allow for partial_idx"""
        return np.mean(self.score_samples(X, y, partial_idx=partial_idx))

    def score_layers(self, X, y=None, partial_idx=None):
        """Overrides super class to allow for partial_idx"""
        return np.mean(self.score_samples_layers(X, y, partial_idx=partial_idx), axis=0)

    def get_domain(self):
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

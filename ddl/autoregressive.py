from __future__ import division, print_function

from abc import abstractmethod

import numpy as np
import scipy.stats
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, DensityMixin, TransformerMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, column_or_1d

from .base import BaseDensityDestructor, ScoreMixin
from .gaussian import GaussianDensity
from .independent import IndependentDensity
# noinspection PyProtectedMember
from .utils import (_UNIT_SPACE, check_X_in_interval, get_domain_or_default,
                    make_interior_probability)


class AutoregressiveDestructor(BaseDensityDestructor):
    def __init__(self, density_estimator=None, order=None, random_state=None):
        self.density_estimator = density_estimator
        self.order = order
        self.random_state = random_state

    def get_density_estimator(self):
        if self.density_estimator is None:
            return GaussianDensity()
        else:
            return clone(self.density_estimator)

    def transform(self, X, y=None):
        return self._autoregress(X, y, inverse=False)

    def inverse_transform(self, X, y=None):
        return self._autoregress(X, y, inverse=True)

    def _autoregress(self, X, y=None, inverse=False):
        X = check_array(X, copy=True)
        try:
            self._check_is_fitted()
        except NotFittedError:
            pass
        else:
            n_features = self._get_n_features()
            if X.shape[1] != n_features:
                raise ValueError('Incorrect number of dimensions.')
        if inverse:
            X = check_X_in_interval(X, _UNIT_SPACE)
            X = make_interior_probability(X)
        else:
            X = check_X_in_interval(X, get_domain_or_default(self))

        order = self._get_order_or_default(X.shape[1])
        Z = np.zeros(X.shape)
        # Could be parallel for non-inverse
        for i in range(len(order)):
            target_idx = order[i]
            cond_idx_arr = order[:i]
            not_cond_idx_arr = order[i:]
            # Target index is always 0 *after* conditioning because of the construction of
            # cond_idx_arr and not_cond_idx_arr
            cond_target_idx = 0

            # Get conditional densities
            if inverse:
                A = Z
            else:
                A = X
            conditionals = self.density_.conditional_densities(
                A, cond_idx_arr, not_cond_idx_arr)

            if not hasattr(conditionals, '__len__'):
                # Handle case where conditional are all the same (i.e. independent dimensions)
                z = (conditionals.marginal_inverse_cdf(X[:, target_idx], cond_target_idx)
                     if inverse else conditionals.marginal_cdf(X[:, target_idx], cond_target_idx))
            else:
                # Handle dependent conditional_densities
                z = np.array([
                    (cond.marginal_inverse_cdf(x, cond_target_idx)
                     if inverse else cond.marginal_cdf(x, cond_target_idx))
                    for x, cond in zip(X[:, target_idx], conditionals)
                ])
            Z[:, i] = z

        if not inverse:
            # Clean up probabilities from numerical errors
            Z = np.minimum(Z, 1)
            Z = np.maximum(Z, 0)
        return Z

    def _check_X(self, X):
        return X

    def _get_order_or_default(self, n_features):
        if self.order is None:
            return np.array(list(range(n_features)))
        elif self.order == 'random':
            rng = check_random_state(self.random_state)
            return rng.permutation(n_features)
        elif len(self.order) == n_features:
            return np.array(self.order)
        else:
            raise ValueError('`order` should be either None, \'random\', or something that has '
                             'length = n_features')

from __future__ import division, print_function

import itertools
import logging
import pdb
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.stats
from scipy.interpolate import interp1d
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, DensityMixin, TransformerMixin, clone
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state, column_or_1d

from .base import BaseDensityDestructor, BoundaryWarning, ScoreMixin
from .univariate import STANDARD_NORMAL_DENSITY, ScipyUnivariateDensity
# noinspection PyProtectedMember
from .utils import (_DEFAULT_SUPPORT, _INF_SPACE, _UNIT_SPACE, check_X_in_interval,
                    get_domain_or_default, get_support_or_default, make_finite,
                    make_interior_probability, make_positive)

logger = logging.getLogger(__name__)


class IndependentDestructor(BaseDensityDestructor):
    def __init__(self, independent_density=None):
        self.independent_density = independent_density

    def get_density_estimator(self):
        if self.independent_density is None:
            return IndependentDensity()
        else:
            return clone(self.independent_density)

    def transform(self, X, y=None):
        # Standard checks
        self._check_is_fitted()
        X = check_array(X)
        if X.shape[0] == 0:
            return X
        self._check_dim(X)
        X = check_X_in_interval(X, get_domain_or_default(self))

        # Use cdf of univariate densities
        Z = np.array([
            u_dens.cdf(np.reshape(x_col, (-1, 1))).ravel()
            for u_dens, x_col in zip(self.density_.univariate_densities_, X.transpose())
        ]).transpose()
        return Z

    def inverse_transform(self, X, y=None):
        # Standard checks
        self._check_is_fitted()
        X = check_array(X, ensure_min_samples=0)
        if X.shape[0] == 0:
            return X
        self._check_dim(X)
        X = check_X_in_interval(X, np.array([0, 1]))

        # Use cdf of univariate densities
        Z = np.array([
            u_dens.inverse_cdf(np.reshape(x_col, (-1, 1))).ravel()
            for u_dens, x_col in zip(self.density_.univariate_densities_, X.transpose())
        ]).transpose()
        return Z

    def _check_dim(self, X):
        if X.shape[1] != len(self.density_.univariate_densities_):
            raise ValueError('Dimension of input does not match dimension of the original '
                             'training data.')


class IndependentDensity(BaseEstimator, ScoreMixin):
    def __init__(self, univariate_estimators=None):
        """Default assumes that univariate_estimators are Gaussian.
        `univariate_estimators` should be:
        
            #. None (defaults to `ScipyUnivariateDensity()`),
            #. univariate density estimator,
            #. array-like of univariate density estimators.
        """
        self.univariate_estimators = univariate_estimators

    def fit(self, X, y=None, **fit_params):
        def _check_univariate(estimators, n_features):
            if estimators is None:
                return [IndependentDensity._get_default_univariate() for _ in range(n_features)]
            elif len(np.array(estimators).shape) == 0:
                return [estimators for _ in range(n_features)]
            elif len(estimators) == n_features:
                return estimators
            else:
                try:
                    temp = itertools.cycle(estimators)
                except TypeError:
                    raise ValueError('`univariate_estimators` should be either None, a single '
                                     'estimator, or an array-like of estimators.')
                else:
                    return list(itertools.islice(temp, n_features))

        X = check_array(X)
        est_arr = _check_univariate(self.univariate_estimators, X.shape[1])

        # Fit univariate densities for each column
        self.univariate_densities_ = np.array([
            clone(est).fit(np.reshape(x_col, (-1, 1)))
            for est, x_col in zip(est_arr, X.transpose())
        ])
        self.n_features_ = len(self.univariate_densities_)
        return self

    def sample(self, n_samples=1, random_state=None):
        self._check_is_fitted()
        rng = check_random_state(random_state)
        X = np.array([
            np.ravel(u_dens.sample(n_samples=n_samples, random_state=rng))
            for u_dens in self.univariate_densities_
        ]).transpose()
        return X

    def score_samples(self, X, y=None):
        self._check_is_fitted()
        X = check_array(X)
        # Extract log-likelihood for all dimensions
        independent_scores = np.array([
            u_dens.score_samples(np.reshape(x_col, (-1, 1))).ravel()
            for u_dens, x_col in zip(self.univariate_densities_, X.transpose())
        ]).transpose()
        # Sum of log-likelihood is product of likelihoods because independent variables
        return independent_scores.sum(axis=1)

    def conditional_densities(self, X, cond_idx, not_cond_idx):
        # Since independent, the conditional is equal to the marginal
        return self.marginal_density(not_cond_idx)

    def marginal_density(self, marginal_idx):
        marginal_density = clone(self)
        marginal_density.univariate_densities_ = self.univariate_densities_[marginal_idx]
        marginal_density.n_features_ = len(marginal_idx)
        # noinspection PyProtectedMember
        marginal_density._check_is_fitted()
        return marginal_density

    def marginal_cdf(self, x, target_idx):
        return self.univariate_densities_[target_idx].cdf(np.array(x).reshape(-1, 1)).reshape(
            np.array(x).shape)

    def marginal_inverse_cdf(self, x, target_idx):
        return self.univariate_densities_[target_idx].inverse_cdf(
            np.array(x).reshape(-1, 1)).reshape(np.array(x).shape)

    def get_support(self):
        def _unwrap_support(est):
            # Univariate density estimators should return [[a,b]] because there is only one
            # dimension, thus this unwraps this even if the default is returned of [a,b]
            return np.array(get_support_or_default(est)).ravel()

        # Check if fitted first
        try:
            self._check_is_fitted()
        except NotFittedError:
            # Use defaults from parameters
            estimators = self.univariate_estimators
            if estimators is None:
                return _unwrap_support(IndependentDensity._get_default_univariate())
            elif len(np.array(estimators).shape) == 0:
                return _unwrap_support(estimators)
            else:
                return np.array([_unwrap_support(est) for est in estimators])
        else:
            # Use fitted support
            return np.array([_unwrap_support(dens) for dens in self.univariate_densities_])

    def _check_is_fitted(self):
        check_is_fitted(self, ['univariate_densities_', 'n_features_'])

    @staticmethod
    def _get_default_univariate():
        return ScipyUnivariateDensity()


class IndependentInverseCdf(BaseEstimator, ScoreMixin, TransformerMixin):
    """A transformer (or *relative* destructor) that performs the
    inverse CDF transform independently for fitted univariate densities.
    The default is the inverse CDF of the standard normal; this default
    is useful to make linear projection destructors canonical (i.e. unit
    domain and correspondingly the identity element property).
    """

    def fit(self, X, y=None, fitted_densities=None, **fit_params):
        """
        X is only used to get the number of features.
        Default assumes that `fitted_densities` are standard Gaussian.
        `fitted_densities` should be fitted versions of the following
        similar to the `univariate_estimators` parameter of `IndependentDensity`:

            #. None (defaults to fitted `ScipyUnivariateDensity()`),
            #. univariate density estimator,
            #. array-like of univariate density estimators.
        """
        X = check_array(X)

        # Mainly just get default and make array of densities if needed
        dens_arr = self._get_densities_or_default(fitted_densities, X.shape[1])
        self.fitted_densities_ = dens_arr
        return self

    def score_samples(self, X, y=None):
        self._check_is_fitted()
        X = check_array(X)
        self._check_dim(X)
        X = check_X_in_interval(X, get_domain_or_default(self))
        X = make_interior_probability(X)

        self.transpose = np.array([
            # Derivative of inversecdf = 1/pdf(inversecdf(X)) -> -logpdf(inversecdf(X)), which is
            # the log(J^{-1}), because Jacobian is diagonal
            -d.score_samples(d.inverse_cdf(x_col.reshape(-1, 1))).ravel()
            for d, x_col in zip(self.fitted_densities_, X.transpose())
        ]).transpose()
        independent_scores = self.transpose
        # Sum of log-likelihood is product of likelihoods because independent variables
        return independent_scores.sum(axis=1)

    def transform(self, X, y=None):
        self._check_is_fitted()
        X = check_array(X)
        self._check_dim(X)
        X = check_X_in_interval(X, get_domain_or_default(self))
        X = make_interior_probability(X)
        X = np.array([
            d.inverse_cdf(x_col.reshape(-1, 1)).ravel()
            for d, x_col in zip(self.fitted_densities_, X.transpose())
        ]).transpose()
        return X

    def inverse_transform(self, X, y=None):
        self._check_is_fitted()
        X = check_array(X)
        self._check_dim(X)
        X = check_X_in_interval(X, self._get_density_support())

        X = np.array([
            d.cdf(x_col.reshape(-1, 1)).ravel()
            for d, x_col in zip(self.fitted_densities_, X.transpose())
        ]).transpose()
        return X

    def get_domain(self):
        return _UNIT_SPACE

    def _get_density_support(self):
        # Get the density support which is the same as the range of this transformer (or the
        # domain of the inverse transformation
        def _check_univariate_support(support):
            shape = np.array(support).shape
            if len(shape) != 2 or shape[0] != 1 or shape[1] != 2:
                raise RuntimeError('Should be univariate support with shape (1,2), i.e. the '
                                   'number of dimensions is fixed at 1 but the following shape '
                                   'was given: %s.' % str(shape))
            return support.ravel()

        self._check_is_fitted()
        return np.array([
            _check_univariate_support(get_support_or_default(d))
            for d in self.fitted_densities_
        ])

    def _get_densities_or_default(self, fitted_densities, n_features):
        if fitted_densities is None:
            return np.array([STANDARD_NORMAL_DENSITY for _ in range(n_features)])
        elif len(np.array(fitted_densities).shape) == 0:
            return np.array([fitted_densities for _ in range(n_features)])
        else:
            return np.array(fitted_densities)

    def _check_dim(self, X):
        if X.shape[1] != len(self.fitted_densities_):
            raise ValueError('Dimension of input does not match dimension of the original '
                             'training data.')

    def _check_is_fitted(self):
        check_is_fitted(self, ['fitted_densities_'])

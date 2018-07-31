"""Module for mixture densities and destructors."""
from __future__ import division, print_function

import logging
import warnings

import numpy as np
from scipy.optimize import brentq as scipy_brentq
from scipy.special import logsumexp as scipy_logsumexp
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state, column_or_1d

from .base import ScoreMixin
from .gaussian import GaussianDensity
from .independent import IndependentDensity
# noinspection PyProtectedMember
from .univariate import _check_univariate_X
# noinspection PyProtectedMember
from .utils import _DEFAULT_SUPPORT, make_interior_probability

logger = logging.getLogger(__name__)


class _MixtureMixin(ScoreMixin):
    def _get_component_densities(self):
        """Should return the density components."""
        return self.component_densities_

    def _check_is_fitted(self):
        check_is_fitted(self, ['weights_', 'component_densities_'])

    def _process_conditionals(self, conditionals, n_samples):
        if not hasattr(conditionals, '__len__'):
            return np.array([conditionals for _ in range(n_samples)])
        else:
            return conditionals

    def _set_fit_params(self, mixture, w, c_arr):
        mixture.weights_ = w
        mixture.component_densities_ = c_arr
        return mixture

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
        components = self._get_component_densities()
        component_log_likelihood = np.array([
            comp.score_samples(X)
            for comp in components
        ])  # (n_components, n_samples)
        weighted_log_likelihood = np.log(self.weights_).reshape(-1, 1) + component_log_likelihood
        return scipy_logsumexp(weighted_log_likelihood, axis=0)

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

        rng = check_random_state(random_state)
        n_samples_per_comp = rng.multinomial(n_samples, self.weights_)

        X = np.vstack([
            comp.sample(comp_n_samples, random_state=rng)
            for comp, comp_n_samples in zip(self._get_component_densities(), n_samples_per_comp)
        ])

        # Not used but if clusters are needed, see here
        # y = np.concatenate([j * np.ones(comp_n_samples, dtype=int)
        #                    for j, comp_n_samples in enumerate(n_samples_per_comp)])
        return X

    def conditional_densities(self, X, cond_idx, not_cond_idx):
        """Should return a either a single density if all the same or a list of GaussianMixtureDensity
        densities with modified parameters."""
        if len(cond_idx) + len(not_cond_idx) != X.shape[1]:
            raise ValueError('`cond_idx_arr` and `not_cond_idx_arr` should be complements that '
                             'have the a total of X.shape[1] number of values.')
        # Handle trivial case
        if len(cond_idx) == 0:
            return self
        self._check_is_fitted()

        # Retrieve components
        components = self._get_component_densities()  # (n_components,)

        n_samples, n_features = X.shape

        # Get new weights
        X_cond = X[:, cond_idx]
        marginal_log_likelihood = np.array([
            component.marginal_density(cond_idx).score_samples(X_cond)
            for component in components
        ]).transpose()  # (n_samples, n_components)

        # Numerically safe way to get new weights if marginal_log_likelihood is very low e.g.
        # -1000, which would mean np.exp(-1000) = 0
        log_cond_weights = np.log(self.weights_) + marginal_log_likelihood
        log_cond_weights -= scipy_logsumexp(log_cond_weights, axis=1).reshape(-1, 1)
        cond_weights = np.exp(log_cond_weights)
        cond_weights = np.maximum(1e-100, cond_weights)  # Just to avoid complete zeros

        # Condition each Gaussian
        cond_components = np.array([
            self._process_conditionals(component.conditional_densities(X, cond_idx, not_cond_idx),
                                       n_samples)
            for component in components
        ]).transpose()  # (n_samples, n_components)

        # Create final conditional densities
        conditional_densities = np.array([
            self._set_fit_params(clone(self), cond_w, cond_c_arr)
            for cond_w, cond_c_arr in zip(cond_weights, cond_components)
        ])
        return conditional_densities

    def marginal_cdf(self, x, target_idx):
        """Should return the marginal cdf of `x` at the dimension given by `target_idx`."""
        cdf_components = np.array([
            np.reshape(comp.marginal_cdf(x, target_idx), (-1,))
            for j, comp in enumerate(self._get_component_densities())
        ]).transpose()  # (n_samples, n_components)
        if not hasattr(x, '__len__'):
            return np.dot(cdf_components.ravel(), self.weights_)
        else:
            return np.dot(cdf_components, self.weights_)  # (n_samples,)

    def marginal_inverse_cdf(self, x, target_idx):
        """Should return the marginal inverse cdf of `x` at the dimension given by `target_idx`."""
        # Get bounds on left and right from min and max of components
        # Note that these are global bounds for given x so they only have to be computed once
        components = self._get_component_densities()
        bound_left = np.min([comp.marginal_inverse_cdf(np.min(x), target_idx)
                             for comp in components])
        bound_right = np.max([comp.marginal_inverse_cdf(np.max(x), target_idx)
                              for comp in components])
        # Handle trivial case where bounds are equal
        if bound_left == bound_right:
            return bound_left * np.ones(np.array(x).shape)

        # Scale bounds by 10% to ensure they will produce positive and negative values
        # because there may be numerical error
        bound_center = (bound_right + bound_left) / 2
        bound_left, bound_right = (1.1 * (np.array([bound_left, bound_right]) - bound_center)
                                   + bound_center)
        weights = np.array(self.weights_, copy=False)

        # Setup function to get inverse cdf for a scalar value
        def _get_inverse_cdf(_x_scalar):
            def _bound_marginal_cdf(a):
                # Rewrote for efficiency
                # u - self.marginal_cdf(_x_scalar, target_idx)
                marginal_cdf = np.dot(
                    weights,
                    [comp.marginal_cdf(a, target_idx) for comp in components]
                )
                return marginal_cdf - _x_scalar

            # If requesting an inverse_cdf for 1 - eps, just return inverse_cdf
            if (1 - _x_scalar) < 2 * np.finfo(float).eps:
                return _bound_marginal_cdf(bound_right)
            return scipy_brentq(_bound_marginal_cdf, bound_left, bound_right)

        if not hasattr(x, '__len__'):
            return _get_inverse_cdf(x)
        else:
            return np.array([_get_inverse_cdf(x_scalar) for x_scalar in x])


class _MixtureDensity(BaseEstimator, _MixtureMixin):
    """Mixture of independent types."""

    def __init__(self, cluster_estimator=None, component_density_estimator=None):
        self.cluster_estimator = cluster_estimator
        self.component_density_estimator = component_density_estimator

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
        cluster_estimator = self._get_cluster_or_default()
        density_estimator = self._get_density_estimator_or_default()

        # Fit y if not given using cluster_estimator
        if y is None:
            try:
                y = cluster_estimator.fit_predict(X)
            except AttributeError:
                y = cluster_estimator.fit(X).predict(X)
        else:
            warnings.warn('y was given so not clustering.  If you want to cluster the data, '
                          'then set y=None when fitting.')
        y = column_or_1d(y)
        self.weights_ = np.array([np.sum(y == label) for label in np.unique(y)]) / len(y)

        # Fit component densities using cluster labels
        self.component_densities_ = np.array([
            clone(density_estimator).fit(X[y == label, :])
            for label in np.unique(y)
        ])
        self.n_features_ = X.shape[1]
        return self

    def _get_density_estimator_or_default(self):
        if self.component_density_estimator is None:
            return IndependentDensity()
        else:
            return self.component_density_estimator

    def _get_cluster_or_default(self):
        if self.cluster_estimator is None:
            return KMeans(n_clusters=2, random_state=0)
        else:
            return self.cluster_estimator


class _GaussianMixtureMixin(object):
    """Overrides several methods in GaussianMixture to comply with density specifications and
    adds a few methods. """

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
        super(_GaussianMixtureMixin, self).fit(X, y)
        self.n_features_ = X.shape[1]
        return self

    def sample(self, n_samples=1, random_state=None):
        """Modification from GaussianMixture to just return X instead of
        (X,y) tuple.
        """
        # Set random state of this object before calling sample
        old_random_state = self.random_state
        self.random_state = random_state
        X, y = super(_GaussianMixtureMixin, self).sample(n_samples)
        self.random_state = old_random_state
        return X

    def _get_component_densities(self):
        n_components, _ = self.means_.shape
        # noinspection PyProtectedMember
        return np.array([
            GaussianDensity(covariance_type=self.covariance_type)._fit_direct(
                mean=self.means_[j],
                covariance=self._get_covariance(j),
                precision=self._get_precision(j),
                precision_cholesky=self._get_precision_cholesky(j),
                copy=False)
            for j in range(n_components)
        ])  # (n_components,)

    # noinspection PyProtectedMember
    def _process_conditionals(self, conditionals, n_samples):
        if not hasattr(conditionals, '__len__'):
            conditionals._fit_auxiliary()
            return np.array([conditionals for _ in range(n_samples)])
        else:
            return np.array([cond._fit_auxiliary() for cond in conditionals])

    def _set_fit_params(self, mixture, w, c_arr):
        def _check_tied(attr):
            if mixture.covariance_type == 'tied':
                return getattr(c_arr[0], attr)
            else:
                return np.array([getattr(gaussian, attr) for gaussian in c_arr])

        mixture.weights_ = w
        mixture.means_ = np.array([gaussian.mean_ for gaussian in c_arr], copy=False)
        mixture.covariances_ = _check_tied('covariance_')
        mixture.precisions_ = _check_tied('precision_')
        mixture.precisions_cholesky_ = _check_tied('precision_cholesky_')
        mixture.converged_ = False
        mixture.n_iter_ = 0
        mixture.lower_bound_ = np.NaN
        return mixture

    def _get_covariance(self, component_idx):
        return _get_component_array(self.covariances_, component_idx, self.covariance_type)

    def _get_precision(self, component_idx):
        return _get_component_array(self.precisions_, component_idx, self.covariance_type)

    def _get_precision_cholesky(self, component_idx):
        return _get_component_array(self.precisions_cholesky_, component_idx, self.covariance_type)

    def _check_X(self, X):
        """Taken mostly from sklearn.mixture.base._check_X"""
        X = check_array(X, dtype=[np.float64, np.float32])

        # Try to get shape from fitted means
        try:
            self._check_is_fitted()
        except NotFittedError:
            n_components = None
            n_features = None
        else:
            n_components, n_features = self.means_.shape

        # Check that X conform
        if n_components is not None and X.shape[0] < n_components:
            raise ValueError('Expected n_samples >= n_components '
                             'but got n_components = %d, n_samples = %d'
                             % (n_components, X.shape[0]))
        if n_features is not None and X.shape[1] != n_features:
            raise ValueError("Expected the input data X have %d features, "
                             "but got %d features"
                             % (n_features, X.shape[1]))
        return X


class GaussianMixtureDensity(_GaussianMixtureMixin, GaussianMixture, _MixtureMixin):
    """Simple class for Gaussian mixtures.
    Note that _GaussianMixtureMixin must override some things in GaussianMixture
    but _MixtureMixin should not override GaussianMixture.  Thus, the order of multiple
    inheritance should remain _GaussianMixtureMixin, GaussianMixture, _MixtureMixin.
    """


class _BayesianGaussianMixtureDensity(_GaussianMixtureMixin, BayesianGaussianMixture,
                                      _MixtureMixin):
    """Simple class for Bayesian Gaussian mixtures.
    See note for GaussianMixtureDensity.
    """


class _MixtureUnivariateMixin(object):
    def fit(self, X, y=None):
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
        X = self._check_X(X)
        return super(_MixtureUnivariateMixin, self).fit(X, y, **fit_params)

    def cdf(self, X, y=None):
        """[Placeholder].

        Parameters
        ----------
        X :
        y :

        Returns
        -------
        obj : object

        """
        self._check_is_fitted()
        X = self._check_X(X)
        u = self.marginal_cdf(X.ravel(), 0)
        u = make_interior_probability(u, eps=0)  # Just correct for numerical errors around 0 and 1
        return u.reshape(-1, 1)

    def inverse_cdf(self, X, y=None):
        """[Placeholder].

        Parameters
        ----------
        X :
        y :

        Returns
        -------
        obj : object

        """
        self._check_is_fitted()
        X = self._check_X(X, inverse=True)
        return self.marginal_inverse_cdf(X.ravel(), 0).reshape(-1, 1)

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
        return np.array([_DEFAULT_SUPPORT])

    def _check_X(self, X, inverse=False):
        return _check_univariate_X(X, self.get_support(), inverse=inverse)


class _GaussianMixtureUnivariateDensity(_MixtureUnivariateMixin, GaussianMixtureDensity):
    """Just a quick class for Gaussian mixture as a univariate density."""


class _BayesianGaussianMixtureUnivariateDensity(_MixtureUnivariateMixin,
                                                _BayesianGaussianMixtureDensity):
    """Just a quick class for Gaussian mixture as a univariate density."""


class _RandomGaussianMixtureDensity(GaussianMixtureDensity):
    def __init__(self, n_components=1, covariance_type='full', reg_covar=1e-06, alpha=1,
                 random_state=None):
        # Set two specific parameters to ensure warm start and only one em iteration
        max_iter = 1
        warm_start = True
        super(_RandomGaussianMixtureDensity, self).__init__(
            n_components=n_components, covariance_type=covariance_type, reg_covar=reg_covar,
            max_iter=max_iter, random_state=random_state, warm_start=warm_start)
        self.alpha = alpha

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
        rng = check_random_state(self.random_state)
        n_samples, n_features = X.shape
        n_components = self.n_components
        covariance_type = self.covariance_type

        # Randomly initialize centers by picking from X (n_components, n_features)
        self.means_ = X[rng.permutation(n_samples)[:n_components], :].copy()
        # Randomly initialize weights via dirichlet
        self.weights_ = rng.dirichlet(self.alpha * np.ones(n_components))
        # self.weights_ = np.ones(n_components)/n_components

        # Set covariances to unity
        covariances = rng.gamma(shape=self.alpha, size=n_components)
        precisions = 1 / covariances
        logger.debug(precisions)
        if covariance_type == 'full':
            self.precisions_ = np.array([
                prec * np.eye(n_features, n_features)
                for prec in precisions
            ])
            self.covariances_ = np.array([
                1 / prec * np.eye(n_features, n_features)
                for prec in precisions
            ])
        elif covariance_type == 'tied':
            self.precisions_ = np.eye(n_features, n_features)
        elif covariance_type == 'diag':
            self.precisions_ = (np.ones((n_components, n_features)).T * precisions).T
            self.covariances_ = (np.ones((n_components, n_features)).T * (1 / precisions)).T
        elif covariance_type == 'spherical':
            self.precisions_ = precisions
            self.covariances_ = 1 / precisions
        else:
            raise RuntimeError('Incorrect covariance type of %s' % covariance_type)
        self.precisions_cholesky_ = np.sqrt(self.precisions_)

        # Run one iteration of em using the warm start parameters
        if self.max_iter != 1 or not self.warm_start:
            raise RuntimeError('The following two properties should be set max_iter=1 and '
                               'warm_start=True')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            super(_RandomGaussianMixtureDensity, self).fit(X, y)
        logger.debug(self.weights_)

        return self


class FirstFixedGaussianMixtureDensity(GaussianMixtureDensity):
    """Mixture density where one component is fixed as the standard normal.

    """

    def __init__(self, fixed_weight=0.5, n_components=1, covariance_type='full'):
        self.fixed_weight = fixed_weight
        super(FirstFixedGaussianMixtureDensity, self).__init__(
            n_components=n_components, covariance_type=covariance_type,
        )

    def _m_step(self, X, log_resp):
        super(FirstFixedGaussianMixtureDensity, self)._m_step(X, log_resp)
        _, n_features = X.shape
        # Fix first component during second stage of fitting
        if self._second_stage:
            self.means_[0, :] = 0
            if self.covariance_type == 'full':
                self._set_first_standard_covariance(
                    np.eye(n_features, n_features).reshape(n_features, n_features, 1)
                )
            elif self.covariance_type == 'tied':
                raise RuntimeError('FirstFixed cannot support covariance_type="tied"')
            elif self.covariance_type == 'diag':
                self._set_first_standard_covariance(np.ones(n_features).reshape(1, -1))
            elif self.covariance_type == 'spherical':
                self._set_first_standard_covariance(np.array([1]))
            else:
                raise RuntimeError('Invalid covariance_type "%s"' % self.covariance_type)

            # Reset mixture weights
            self.weights_[0] = self.fixed_weight
            self.weights_[1:] *= (1 - self.fixed_weight) / np.sum(self.weights_[1:])
            # logger.debug('Second stage n_components=%d, weights[0]=%g, means_[0]=%s'
            #              % (self.weights_.shape[0], self.weights_[0], str(self.means_[0])))
        else:
            # logger.debug('First stage n_components=%d, weights[0]=%g, means_[0]=%s'
            #              % (self.weights_.shape[0], self.weights_[0], str(self.means_[0])))
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
        # Fit with one less component
        self.n_components -= 1
        self._second_stage = False
        super(FirstFixedGaussianMixtureDensity, self).fit(X, y)

        # Update means, etc. with single Gaussian component
        self.n_components += 1
        n_samples, n_features = X.shape
        self.means_ = np.append(self.means_, np.zeros(n_features).reshape(1, -1), axis=0)
        if self.covariance_type == 'full':
            self._append_standard_covariance(
                np.eye(n_features, n_features).reshape(n_features, n_features, 1)
            )
        elif self.covariance_type == 'tied':
            raise RuntimeError('FirstFixed cannot support covariance_type="tied"')
        elif self.covariance_type == 'diag':
            self._append_standard_covariance(np.ones(n_features).reshape(1, -1))
        elif self.covariance_type == 'spherical':
            self._append_standard_covariance(np.array([1]))
        else:
            raise RuntimeError('Invalid covariance_type "%s"' % self.covariance_type)

        # Rescale weights
        if self.fixed_weight < 0 or self.fixed_weight > 1:
            raise ValueError('weight should be between 0 and 1')
        self.weights_ *= (1 - self.fixed_weight)
        self.weights_ = np.append(self.weights_, np.array([self.fixed_weight]))

        # Reset parameters for second fitting
        old_warm_start = self.warm_start
        self.warm_start = True
        self.converged_ = False
        self.lower_bound_ = -np.infty

        # Refit with second stage set (i.e. fixed standard normal)
        self._second_stage = True
        super(FirstFixedGaussianMixtureDensity, self).fit(X, y)
        self.warm_start = old_warm_start

        return self

    def _append_standard_covariance(self, cov):
        """Only works for standard covariance (i.e. identity covariance)"""
        self.covariances_ = np.append(self.covariances_, cov, axis=0)
        self.precisions_ = np.append(self.precisions_, cov, axis=0)
        self.precisions_cholesky_ = np.append(self.precisions_cholesky_, cov, axis=0)

    def _set_first_standard_covariance(self, cov):
        """Only works for standard covariance (i.e. identity covariance)"""
        self.covariances_[0] = cov
        self.precisions_[0] = cov
        self.precisions_cholesky_[0] = cov


class _RegularizedGaussianMixtureDensity(GaussianMixtureDensity):
    def __init__(self, main_weight=0.5, n_components=1, covariance_type='full'):
        self.main_weight = main_weight
        super(_RegularizedGaussianMixtureDensity, self).__init__(
            n_components=n_components, covariance_type=covariance_type,
        )

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
        super(_RegularizedGaussianMixtureDensity, self).fit(X, y)

        # Update means, etc. with single Gaussian component
        self.n_components += 1
        n_samples, n_features = X.shape
        self.means_ = np.append(self.means_, np.zeros(n_features).reshape(1, -1), axis=0)
        if self.covariance_type == 'full':
            self._append_standard_covariance(
                np.eye(n_features, n_features).reshape(n_features, n_features, 1)
            )
        elif self.covariance_type == 'tied':
            raise RuntimeError('Regularized cannot support covariance_type="tied"')
        elif self.covariance_type == 'diag':
            self._append_standard_covariance(np.ones(n_features).reshape(1, -1))
        elif self.covariance_type == 'spherical':
            self._append_standard_covariance(np.array([1]))
        else:
            raise RuntimeError('Invalid covariance_type "%s"' % self.covariance_type)

        # Rescale weights
        if self.main_weight < 0 or self.main_weight > 1:
            raise ValueError('main_weight should be between 0 and 1')
        self.weights_ *= (1 - self.main_weight)
        self.weights_ = np.append(self.weights_, np.array([self.main_weight]))

        return self

    def _append_standard_covariance(self, cov):
        self.covariances_ = np.append(self.covariances_, cov, axis=0)
        self.precisions_ = np.append(self.precisions_, cov, axis=0)
        self.precisions_cholesky_ = np.append(self.precisions_cholesky_, cov, axis=0)


class _AugmentedGaussianDensity(GaussianMixtureDensity):
    def __init__(self, reg_covar=1e-06, random_state=None):
        n_components = 2
        covariance_type = 'diag'
        super(_AugmentedGaussianDensity, self).__init__(
            n_components=n_components, covariance_type=covariance_type,
            reg_covar=reg_covar, random_state=random_state)

    def _set_weights(self, w):
        self.weights_ = np.array([1.0 - w, w])

    def _set_variance(self, v, idx=1):
        self.covariances_[idx, :] = v
        self.precisions_[idx, :] = 1.0 / v
        self.precisions_cholesky_[idx, :] = np.sqrt(1.0 / v)

    def _set_mean(self, m):
        self.means_[1, :] = m

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
        n_samples, n_features = X.shape
        rng = check_random_state(self.random_state)
        n_samples, n_features = X.shape

        # Main mean and variance
        # main_mean = np.mean(X, axis=0)
        # main_variance = np.var(X, axis=0)
        main_mean = np.zeros(n_features)
        main_variance = np.ones(n_features)

        # Set parameters
        if self.covariance_type != 'diag':
            raise RuntimeError('covariance_type must be diag')
        self.weights_ = np.array([0.5, 0.5])
        self.means_ = np.array([main_mean, main_mean])
        self.covariances_ = np.array([main_variance, main_variance])
        self.precisions_ = 1.0 / self.covariances_
        self.precisions_cholesky_ = np.sqrt(self.precisions_)
        score_orig = self.score(X)

        # Grid search weight and mean
        # Setup test parameters
        test_weights = np.logspace(-3, -1, 20)
        # test_var = np.array([0.05])  # np.logspace(-1, 0, 20)
        test_var = np.linspace(0.01, 0.1, 10)
        if np.max(test_weights) > 1 or np.min(test_weights) < 0:
            raise RuntimeError('Weight should be between 0 and 1')
        test_idx = rng.permutation(n_samples)[:np.minimum(100, n_samples)]

        # Grid search for best parameters
        scores = np.empty((len(test_idx), len(test_weights), len(test_var)))
        self.n_iter_ = 0
        best_log_likelihood = -np.inf
        best_weight = np.nan
        best_var = np.nan
        best_mean = np.nan
        for ii, bump_idx in enumerate(test_idx):
            # Now add bump
            bump_sel = np.zeros(n_samples, dtype=np.bool)
            bump_sel[bump_idx] = True
            bump_mean = X[bump_idx, :]
            X_not_bump = X[~bump_sel, :]
            self._set_mean(bump_mean)

            for wi, weight in enumerate(test_weights):
                self._set_weights(weight)
                for vi, var in enumerate(test_var):
                    self._set_variance(var)

                    log_likelihood = self.score(X_not_bump)
                    # logger.debug('log_likelihood=%g, weight=%g, var=%g'
                    #              % (log_likelihood, weight, var))
                    scores[ii, wi, vi] = log_likelihood

                    if log_likelihood > best_log_likelihood:
                        best_weight = weight
                        best_var = var
                        best_log_likelihood = log_likelihood
                        best_mean = bump_mean
                    self.n_iter_ += 1

        # Plot stuff for debugging
        # import matplotlib.pyplot as plt
        # plt.semilogx(test_var, scores[ii, :, :].T)
        # plt.xlabel('Mixture var')
        # plt.legend(test_weights)
        # plt.show()

        # Reset weight and variance based on best
        if best_log_likelihood != np.max(scores):
            logger.debug('%g (best) vs %g (max)' % (best_log_likelihood, np.max(scores)))
            raise RuntimeError('best log likelihood is not correct')
        self._set_weights(best_weight)
        self._set_variance(best_var)
        self._set_mean(best_mean)

        score_after = self.score(X)
        if score_after < score_orig:
            self.n_components = 1
            self.means_ = np.array([main_mean])
            self.weights_ = np.array([1])
            self.covariances_ = np.array([main_variance])
            self.precisions_ = 1 / self.covariances_
            self.precisions_cholesky_ = np.sqrt(self.precisions_)
            # logger.debug('Just using single Gaussian fit %g (after) and %g (before)'
            #              % (score_after, score_orig))
        else:
            logger.debug('best log_likelihood=%g, weight=%g, var=%g'
                         % (best_log_likelihood, best_weight, best_var))
            pass

        # Save some variables
        self.test_weights_ = test_weights
        self.test_var_ = test_var
        self.test_scores_ = scores
        self.best_weight_ = best_weight
        self.best_var_ = best_var
        self.best_log_likelihood_ = best_log_likelihood
        self.lower_bound_ = np.nan
        self.converged_ = True
        self.n_features_ = n_features
        return self


def _get_component_array(array, component_idx, covariance_type):
    if covariance_type == 'full':
        return array[component_idx, :, :]
    elif covariance_type == 'tied':
        return array
    elif covariance_type == 'diag':
        return array[component_idx, :]
    elif covariance_type == 'spherical':
        return array[component_idx]
    else:
        raise ValueError('Covariance type is invalid.')

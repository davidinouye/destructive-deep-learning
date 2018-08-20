"""Module for mixture densities and destructors."""
from __future__ import division, print_function

import logging
import warnings

import numpy as np
from scipy.optimize import brentq as scipy_brentq
from scipy.special import logsumexp as scipy_logsumexp
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state, column_or_1d

from .base import ScoreMixin
from .gaussian import GaussianDensity
from .independent import IndependentDensity

logger = logging.getLogger(__name__)


class _MixtureMixin(ScoreMixin):
    def _get_component_densities(self):
        """Return the density components."""
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
        """Compute conditional densities.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to condition on based on `cond_idx`.

        cond_idx : array-like of int
            Indices to condition on.

        not_cond_idx :
            Indices not to condition on.

        Returns
        -------
        conditional_densities : array-like of estimators
            Either a single density if all the same or a list of Gaussian
            densities with conditional variances and means.

        """
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
        """Return the marginal cdf of `x` at feature `target_idx`."""
        cdf_components = np.array([
            np.reshape(comp.marginal_cdf(x, target_idx), (-1,))
            for j, comp in enumerate(self._get_component_densities())
        ]).transpose()  # (n_samples, n_components)
        if not hasattr(x, '__len__'):
            return np.dot(cdf_components.ravel(), self.weights_)
        else:
            return np.dot(cdf_components, self.weights_)  # (n_samples,)

    def marginal_inverse_cdf(self, x, target_idx):
        """Return the marginal inverse cdf of `x` at feature `target_idx`."""
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
    """Overrides several methods in GaussianMixture.

    Needed to comply with density specifications and adds a few methods.
    """

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
        X = self._check_X(X)
        self.n_features_ = X.shape[1]
        return self

    def sample(self, n_samples=1, random_state=None):
        """Sample from GaussianMixture and return only X instead of (X, y)."""
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
        """Taken mostly from `sklearn.mixture.base._check_X`."""
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
    """Gaussian mixture density that can be used with AutoregressiveDestructor.

    This subclasses off of :class:`sklearn.mixture.GaussianMixture`. It
    overrides several methods such as :func:`sample` and :func:`score` to
    ensure that the interface conforms to the other density estimators. In
    addition, the necessary conditional and marginal distribution methods
    needed for :class:`ddl.autoregressive.AutoregressiveDestructor` were
    added. See :class:`sklearn.mixture.GaussianMixture` for parameters and
    attributes.

    Note that _GaussianMixtureMixin must override some things in
    GaussianMixture but _MixtureMixin should not override GaussianMixture.
    Thus, the order of multiple inheritance should remain
    _GaussianMixtureMixin, GaussianMixture, _MixtureMixin.

    See Also
    --------
    sklearn.mixture.GaussianMixture
    ddl.autoregressive.AutoregressiveDestructor
    BayesianGaussianMixtureDensity

    """


class _BayesianGaussianMixtureDensity(_GaussianMixtureMixin, BayesianGaussianMixture,
                                      _MixtureMixin):
    """Bayesian Gaussian mixture, useful for AutoregressiveDestructor.

    This subclasses off of :class:`sklearn.mixture.BayesianGaussianMixture`.
    It overrides several methods such as :func:`sample` and :func:`score` to
    ensure that the interface conforms to the other density estimators. In
    addition, the necessary conditional and marginal distribution methods
    needed for :class:`ddl.autoregressive.AutoregressiveDestructor` were
    added. See :class:`sklearn.mixture.BayesianGaussianMixture` for
    parameters and attributes.

    Note that _GaussianMixtureMixin must override some things in
    GaussianMixture but _MixtureMixin should not override GaussianMixture.
    Thus, the order of multiple inheritance should remain
    _GaussianMixtureMixin, GaussianMixture, _MixtureMixin.

    See Also
    --------
    sklearn.mixture.BayesianGaussianMixture
    ddl.autoregressive.AutoregressiveDestructor
    GaussianMixtureDensity

    """


class FirstFixedGaussianMixtureDensity(GaussianMixtureDensity):
    """Mixture density where one component is fixed as the standard normal.

    This is useful for creating a regularized Gaussian mixture destructor.
    In particular, if this is paired with an inverse Gaussian cdf (i.e.
    :class:`~ddl.independent.IndependentInverseCdf`), and the weight of the
    fixed standard normal approaches 1, then the composite destructor
    approaches an identity. Thus, the `fixed_weight` parameter can be used
    to control the amount of regularization.

    Note this is implemented by overriding the :func:`_m_step` private
    method of :class:`sklearn.mixture.GaussianMixture` so it may not be
    compatible with future releases of sklearn.

    More specifically first n_components-1 Gaussian components are fit using
    the standard Gaussian mixture estimator. Then, we manually add a fixed
    standard normal component with the desired fixed weight. Then, we refit
    but override the M step so that the fixed weight component does not
    change.

    Parameters
    ----------
    fixed_weight : float, default=0.5
        The fixed weight between 0 and 1 that is given to the first Gaussian
        component. As this weight approaches 1, there is full regularization
        and no learning from the data. If the weight approaches 0,
        then there is no regularization and the fitting is determined
        entirely from the data.

    n_components : int, default=1
        The number of mixture components to fit.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        String describing the type of covariance parameters to use.
        Must be one of::

            'full' (each component has its own general covariance matrix),
            'tied' (all components share the same general covariance matrix),
            'diag' (each component has its own diagonal covariance matrix),
            'spherical' (each component has its own single variance).

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
        """Append standard covariance matrix.

        Only works for standard covariance (i.e. identity covariance).
        """
        self.covariances_ = np.append(self.covariances_, cov, axis=0)
        self.precisions_ = np.append(self.precisions_, cov, axis=0)
        self.precisions_cholesky_ = np.append(self.precisions_cholesky_, cov, axis=0)

    def _set_first_standard_covariance(self, cov):
        """Set first to standard covariance.

        Only works for standard covariance (i.e. identity covariance).
        """
        self.covariances_[0] = cov
        self.precisions_[0] = cov
        self.precisions_cholesky_[0] = cov


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

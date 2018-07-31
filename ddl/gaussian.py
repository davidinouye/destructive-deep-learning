"""Module for Gaussian density."""
from __future__ import division, print_function

import numpy as np
import scipy.stats
from scipy import linalg
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import check_array, column_or_1d

from .base import AutoregressiveMixin, ScoreMixin


class GaussianDensity(BaseEstimator, AutoregressiveMixin, ScoreMixin):
    """Allow for conditioning that will return a new proxy density.
    Also allows for marginal density computation.
    """

    def __init__(self, covariance_type='full', reg_covar=1e-06):
        self.reg_covar = reg_covar
        self.covariance_type = covariance_type

    def fit(self, X, y=None):
        """Should do a simple regularized fit."""
        X = check_array(X)
        self.mean_ = np.mean(X, axis=0)
        _, n_features = X.shape

        if self.covariance_type == 'full' or self.covariance_type == 'tied':
            if X.shape[0] == 1:
                self.covariance_ = np.zeros((n_features, n_features))
                if self.reg_covar <= 0:
                    raise ValueError('reg_covar <= 0 but only 1 sample given so variance '
                                     'impossible to estimate.')
            else:
                self.covariance_ = np.cov(X, rowvar=False).reshape((n_features, n_features))
            self.covariance_.flat[::len(self.covariance_) + 1] += self.reg_covar
        elif self.covariance_type == 'diag':
            self.covariance_ = np.var(X, axis=0)
        elif self.covariance_type == 'spherical':
            self.covariance_ = np.var(X)
        else:
            raise ValueError('covariance_type of %s not recognized' % str(self.covariance_type))

        self.precision_ = None
        self.precision_cholesky_ = None
        self._fit_auxiliary()
        return self

    def score_samples(self, X, y=None):
        """

        Parameters
        ----------
        X :
        y :

        Returns
        -------

        """
        self._fit_auxiliary()
        X = check_array(X)
        return _estimate_log_gaussian_prob(
            X,
            self.mean_.reshape(1, -1),
            self.precision_cholesky_,
            self.covariance_type
        ).ravel()

    def sample(self, n_samples=1, random_state=None):
        """

        Parameters
        ----------
        n_samples :
        random_state :

        Returns
        -------

        """
        rng = check_random_state(random_state)
        if self.covariance_type == 'full' or self.covariance_type == 'tied':
            X = rng.multivariate_normal(self.mean_, self.covariance_, n_samples)
        elif self.covariance_type == 'diag' or self.covariance_type == 'spherical':
            # Covariance will broadcast if scalar or 1D array
            X = self.mean_ + rng.randn(n_samples, len(self.mean_)) * np.sqrt(self.covariance_)
        else:
            raise ValueError('Invalid covariance_type.')
        return X

    def marginal_density(self, marginal_idx):
        """Return a single marginal density based on `marginal_idx`."""

        def _get_covariance():
            if self.covariance_type == 'full' or self.covariance_type == 'tied':
                return self.covariance_[np.ix_(marginal_idx, marginal_idx)]
            elif self.covariance_type == 'diag':
                return self.covariance_[marginal_idx]
            elif self.covariance_type == 'spherical':
                return self.covariance_
            else:
                raise ValueError('Invalid `covariance_type`')

        self._fit_auxiliary()
        dens = GaussianDensity(covariance_type=self.covariance_type)
        dens._fit_direct(
            mean=self.mean_[marginal_idx],
            covariance=_get_covariance(),
            copy=False,
        )
        return dens

    def conditional_densities(self, X, cond_idx, not_cond_idx):
        """Should return a either a single density if all the same or a list of Gaussian
        densities with modified variance and means. """
        # Get new precision which is constant w.r.t. X
        self._fit_auxiliary()
        if len(cond_idx) + len(not_cond_idx) != X.shape[1]:
            raise ValueError('`cond_idx_arr` and `not_cond_idx_arr` should be complements that '
                             'have the a total of X.shape[1] number of values.')
        # Handle trivial case
        if len(cond_idx) == 0:
            return self

        if self.covariance_type == 'full' or self.covariance_type == 'tied':
            cond_precision = self.precision_[np.ix_(not_cond_idx, not_cond_idx)]
            # Compute necessary matrices based on precision matrix
            before_mean = GaussianDensity(covariance_type=self.covariance_type)._fit_direct(
                precision=cond_precision, mean=self.mean_[not_cond_idx], copy=False)
            before_mean._fit_auxiliary()

            # Compute conditional means
            cov_NC = self.covariance_[np.ix_(not_cond_idx, cond_idx)]
            chol = _compute_precision_cholesky(self.covariance_[np.ix_(cond_idx, cond_idx)], 'full')
            inv_cov_CC = _cholesky_to_full(chol, self.covariance_type)
            proj_mat = cov_NC.dot(inv_cov_CC)  # Sigma_{12} Sigma_{22}^{-1}
            cond_means = (
                    self.mean_[not_cond_idx]
                    - np.dot(proj_mat, self.mean_[cond_idx]).ravel()
                    + np.dot(proj_mat, X[:, cond_idx].transpose()).transpose()
            )

            # Create Gaussian densities
            conditionals = np.array([
                GaussianDensity(covariance_type=self.covariance_type)._fit_direct(
                    mean=c_mean,
                    covariance=before_mean.covariance_,
                    precision=before_mean.precision_,
                    precision_cholesky=before_mean.precision_cholesky_,
                    copy=False)
                for c_mean in cond_means
            ])
        else:  # diagonal or spherical
            if self.covariance_type == 'diag':
                cond_precision = self.precision_[not_cond_idx]
                cond_covariance = self.covariance_[not_cond_idx]
                cond_precision_cholesky = self.precision_cholesky_[not_cond_idx]
            elif self.covariance_type == 'spherical':
                cond_precision = self.precision_
                cond_covariance = self.covariance_
                cond_precision_cholesky = self.precision_cholesky_
            else:
                raise ValueError('Invalid `covariance_type`.')

            # Also constant w.r.t. X
            cond_mean = self.mean_[not_cond_idx]
            cond_density = GaussianDensity(covariance_type=self.covariance_type)._fit_direct(
                mean=cond_mean, covariance=cond_covariance, precision=cond_precision,
                precision_cholesky=cond_precision_cholesky, copy=False)
            conditionals = cond_density
        return conditionals

    def marginal_cdf(self, x, target_idx):
        """

        Parameters
        ----------
        x :
        target_idx :

        Returns
        -------

        """
        self._fit_auxiliary()
        if len(np.array(x).shape) != 0:
            x = column_or_1d(x)
        params = self._get_marginal_params(target_idx)
        return scipy.stats.norm.cdf(x, **params)

    def marginal_pdf(self, x, target_idx):
        """Should return marginal log-likelihood of a each dimension or
        a particular dimension specified by target_idx.
        x can be either a scalar or vector.
        """
        self._fit_auxiliary()
        if len(np.array(x).shape) != 0:
            x = column_or_1d(x)
        params = self._get_marginal_params(target_idx)
        return scipy.stats.norm.pdf(x, **params)

    def marginal_inverse_cdf(self, x, target_idx):
        """

        Parameters
        ----------
        x :
        target_idx :

        Returns
        -------

        """
        self._fit_auxiliary()
        if len(np.array(x).shape) != 0:
            x = column_or_1d(x)
        params = self._get_marginal_params(target_idx)
        return scipy.stats.norm.ppf(x, **params)

    def _fit_direct(self, mean=None, covariance=None,
                    precision=None, precision_cholesky=None, copy=True):
        """Should directly fit the estimator with the given parameters.
        Note that some parameters do not need to be set.
        """

        def _copy(X):
            if X is not None:
                return X.copy()
            else:
                return X

        def _no_copy(X):
            return X

        # Only allow tied, diagonal or spherical (full doesn't make sense because we only have one)
        if copy:
            maybecopy = _copy
        else:
            maybecopy = _no_copy

        self.mean_ = maybecopy(mean)
        self.covariance_ = maybecopy(covariance)
        self.precision_ = maybecopy(precision)
        self.precision_cholesky_ = maybecopy(precision_cholesky)
        return self

    def _fit_auxiliary(self):
        """Compute precision or covariance if necessary."""
        # Compute precision variables from other precision variables
        if self.precision_ is None and self.precision_cholesky_ is not None:
            self.precision_ = _cholesky_to_full(self.precision_cholesky_, self.covariance_type)
        elif self.precision_ is not None and self.precision_cholesky_ is None:
            if self.covariance_type == 'full' or self.covariance_type == 'tied':
                self.precision_cholesky_ = linalg.cholesky(self.precision_, lower=True)
            else:
                self.precision_cholesky_ = np.sqrt(self.precision_)

        # Compute covariance variables from precision if needed
        if self.precision_ is None and self.covariance_ is None:
            raise RuntimeError('Either precision_ or covariance_ must be set.')
        elif self.covariance_ is None:
            # Get from precision
            covariance_chol = _compute_covariance_cholesky(self.precision_,
                                                           self.covariance_type)
            self.covariance_ = _cholesky_to_full(covariance_chol, self.covariance_type)
        elif self.precision_ is None:
            self.precision_cholesky_ = _compute_precision_cholesky(self.covariance_,
                                                                   self.covariance_type)
            self.precision_ = _cholesky_to_full(self.precision_cholesky_, self.covariance_type)
        self.n_features_ = len(self.mean_)
        return self

    def _get_marginal_params(self, target_idx):
        # Get variance based on covariance_type
        if self.covariance_type == 'full' or self.covariance_type == 'tied':
            var = self.covariance_[::len(self.covariance_) + 1][0, target_idx]
        elif self.covariance_type == 'diag':
            var = self.covariance_[target_idx]
        elif self.covariance_type == 'spherical':
            var = self.covariance_
        else:
            raise ValueError('incorrect covariance_type')

        if len(self.mean_.shape) > 1:
            raise RuntimeError('DEBUG')
        loc = self.mean_[target_idx]
        scale = np.sqrt(var)
        return dict(loc=loc, scale=scale)


def _cholesky_to_full(X_chol, covariance_type):
    # Idea from sklearn.mixture.gaussian_mixture._set_parameters
    if covariance_type == 'full' or covariance_type == 'tied':
        X = np.dot(X_chol, X_chol.T)
    else:
        X = X_chol ** 2
    return X


def _compute_covariance_cholesky(precisions, precision_type):
    return _compute_precision_cholesky(precisions, precision_type)


def _compute_precision_cholesky(covariances, covariance_type):
    """
    (Edited from sklearn.mixture.gaussian_mixture.py v. 0.19.1)

    Compute the Cholesky decomposition of the precisions.
    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.
    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    """
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar.")

    if covariance_type == 'tied' or covariance_type == 'full':
        _, n_features = covariances.shape
        try:
            cov_chol = linalg.cholesky(covariances, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol = linalg.solve_triangular(cov_chol, np.eye(n_features),
                                                  lower=True).T
    else:
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1. / np.sqrt(covariances)
    return precisions_chol


def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type):
    """
    (Edited from sklearn.mixture.gaussian_mixture.py v. 0.19.1)

    Estimate the log Gaussian probability.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    means : array-like, shape (n_components, n_features)
    precisions_chol : array-like,
        Cholesky decompositions of the precision matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    if covariance_type == 'full':
        covariance_type = 'tied'
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    # det(precision_cholesky) is half of det(precision)
    log_det = _compute_log_det_cholesky(
        precisions_chol, covariance_type, n_features)

    if covariance_type == 'full':
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'tied':
        log_prob = np.empty((n_samples, n_components))
        for k, mu in enumerate(means):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'diag':
        precisions = precisions_chol ** 2
        log_prob = (np.sum(means.ravel() ** 2 * precisions) -
                    2. * np.dot(X, (means.ravel() * precisions)) +
                    np.dot(X ** 2, precisions))

    elif covariance_type == 'spherical':
        precisions = precisions_chol ** 2
        log_prob = (np.sum(means ** 2, 1) * precisions -
                    2 * np.dot(X, means.T * precisions) +
                    np.outer(row_norms(X, squared=True), precisions))
    else:
        raise ValueError('covariance_type invalid')
    return -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det


def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    """
    (Edited from sklearn.mixture.gaussian_mixture.py v. 0.19.1)

    Compute the log-det of the cholesky decomposition of matrices.
    Parameters
    ----------
    matrix_chol : array-like,
        Cholesky decompositions of the matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
    n_features : int
        Number of features.
    Returns
    -------
    log_det_precision_chol : array-like, shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    if covariance_type == 'full':
        covariance_type = 'tied'

    if covariance_type == 'full':
        n_components, _, _ = matrix_chol.shape
        log_det_chol = (np.sum(np.log(
            matrix_chol.reshape(
                n_components, -1)[:, ::n_features + 1]), 1))

    elif covariance_type == 'tied':
        log_det_chol = (np.sum(np.log(np.diag(matrix_chol))))

    elif covariance_type == 'diag':
        log_det_chol = (np.sum(np.log(matrix_chol)))

    else:
        log_det_chol = n_features * (np.log(matrix_chol))
    return log_det_chol

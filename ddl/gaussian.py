from __future__ import division
from __future__ import print_function
import warnings

import numpy as np
from scipy import linalg
import scipy.stats
from sklearn.base import clone, BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d
from sklearn.utils.extmath import row_norms
from sklearn.exceptions import NotFittedError

from .base import ScoreMixin
# noinspection PyProtectedMember
from .utils import _UNIT_SPACE, _INF_SPACE
from .base import AutoregressiveMixin
from .independent import IndependentDestructor, IndependentDensity
from .univariate import UnivariateDensity
from .utils import make_interior_probability


class JointGaussianCopulaDensity(BaseEstimator, AutoregressiveMixin, ScoreMixin):
    """Defines a joint copula model with marginal density and the copula density."""
    def __init__(self, univariate_estimators=None):
        self.univariate_estimators = univariate_estimators

    # noinspection PyProtectedMember
    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = X.shape
        if n_samples < n_features:
            warnings.warn('Currently, we have not implemented high-dimensional estimation so just '
                          'using identity correlation matrix.')
            R_spearman = np.eye(n_features, n_features)
        else:
            # Non-paranormal SKEPTIC estimation of Gaussian covariance/precision matrix
            # Though not using high-dimensional estimator yet
            if n_features == 1:
                rho = np.array([[1]])
            else:
                rho, _ = scipy.stats.spearmanr(X)
                if n_features == 2:
                    rho = np.array([[1, rho], [rho, 1]])  # Fix for only two features
            if np.any(rho.shape != np.array([n_features, n_features])):
                raise RuntimeError('Rho should have shape (n_features, n_features)')
            R_spearman = 2*np.sin((np.pi/6)*rho)

        gaussian_density = GaussianDensity()._fit_direct(
            mean=np.zeros(n_features),
            covariance=R_spearman,
        )
        gaussian_density._fit_auxiliary()

        # Fit marginal density
        independent_destructor = IndependentDestructor(
            independent_density=IndependentDensity(
                univariate_estimators=self.univariate_estimators
            )
        )
        independent_destructor.fit(X, y)

        self.independent_destructor_ = independent_destructor
        self.gaussian_density_ = gaussian_density
        self.n_dim_ = n_features
        return self

    def score_samples(self, X, y=None):
        self._check_is_fitted()
        X = check_array(X)

        # Get independent log density estimates
        independent_density = self.independent_destructor_.density_
        independent_score_samples = independent_density.score_samples(X)

        # Define transformation
        U = self.independent_destructor_.transform(X)
        U = make_interior_probability(U)
        Z = scipy.stats.norm.ppf(U)

        # Get density of copula (density of fitted gaussian vs density of standard gaussian)
        copula_score_samples = (self.gaussian_density_.score_samples(Z)
                                - np.sum(scipy.stats.norm.logpdf(Z), axis=1))

        return independent_score_samples + copula_score_samples

    def sample(self, n_samples=1, random_state=None):
        self._check_is_fitted()
        Z = self.gaussian_density_.sample(n_samples, random_state=random_state)
        U = scipy.stats.norm.cdf(Z)
        U = make_interior_probability(U)
        X = self.independent_destructor_.inverse_transform(U)
        return X

    def conditional_densities(self, X, cond_idx, not_cond_idx):
        X = check_array(X)
        if len(cond_idx) == 0:  # Handle no conditioning
            return self

        # Extract conditional marginal destructor
        independent_density = self.independent_destructor_.density_
        cond_independent_destructor = clone(self.independent_destructor_)
        cond_independent_destructor.density_ = independent_density.marginal_density(cond_idx)

        # Get appropriate Z to condition on
        U_cond = cond_independent_destructor.transform(X[:, cond_idx])
        U_cond = make_interior_probability(U_cond)
        Z_cond = scipy.stats.norm.ppf(U_cond)
        # Note Z[:, not_cond_idx_arr] not used when conditioning
        Z = np.zeros(X.shape)
        Z[:, cond_idx] = Z_cond

        # Condition Gaussian densities
        cond_gaussians = self.gaussian_density_.conditional_densities(
            Z, cond_idx, not_cond_idx)

        # Map each conditional Gaussian to a new JointGaussianCopulaDensity
        cond_univ_densities = self.independent_destructor_.density_.univariate_densities_[not_cond_idx]
        conditional_densities = np.array([
            self._get_conditional_copula(cg, cond_univ_densities)
            for cg in cond_gaussians
        ])
        return conditional_densities

    def marginal_density(self, marginal_idx):
        marginal_density = clone(self)
        marginal_density.gaussian_density_ = self.gaussian_density_.marginal_density(marginal_idx)
        marginal_density.independent_destructor_ = self._make_destructor(
            self.independent_destructor_.density_.univariate_densities_[marginal_idx]
        )
        return marginal_density

    def marginal_cdf(self, x, target_idx):
        x = np.array(x)
        orig_shape = x.shape
        univariate_densities = self.independent_destructor_.density_.univariate_densities_
        u = univariate_densities[target_idx].cdf(x.ravel().reshape(-1, 1))
        return np.reshape(u, orig_shape)

    def marginal_inverse_cdf(self, x, target_idx):
        u = np.array(x)
        orig_shape = u.shape
        univariate_densities = self.independent_destructor_.density_.univariate_densities_
        x = univariate_densities[target_idx].inverse_cdf(u.ravel().reshape(-1, 1))
        return np.reshape(x, orig_shape)

    def get_support(self):
        try:
            self._check_is_fitted()
        except NotFittedError:
            return IndependentDensity(
                univariate_estimators=self.univariate_estimators).get_support()
        else:
            return self.independent_destructor_.density_.get_support()

    def _get_conditional_copula(self, gaussian, univ_densities):
        # noinspection PyProtectedMember
        gaussian._fit_auxiliary()

        # Extract mean and variance change...
        mean = gaussian.mean_
        cov = gaussian.covariance_
        var = np.diag(cov)
        std = np.sqrt(var)
        R_cond = (cov / std).transpose() / std
        R_cond[::R_cond.shape[0] + 1] = 1  # Ensure diagonal is 1

        # Setup new shifted Gaussian
        # noinspection PyProtectedMember
        cond_gaussian = GaussianDensity()._fit_direct(
            mean=np.zeros(mean.shape),
            covariance=R_cond,
        )

        # Create Univariate composite marginals with adjusted statistics
        def _fit_marginal_dens(dens, _mean, _std):
            marginal_dens = _CopulaConditionalUnivariateDensity()
            marginal_dens.univariate_density_ = dens
            marginal_dens.mean_ = _mean
            marginal_dens.std_ = _std
            return marginal_dens

        if len(univ_densities) != len(mean):
            raise RuntimeError('univ_densities should have same length as mean')

        cond_univ_densities = np.array([
            _fit_marginal_dens(dens, m, s)
            for dens, m, s in zip(univ_densities, mean, std)
        ])

        # Setup fitted copula model
        cond_copula = JointGaussianCopulaDensity(
            univariate_estimators=_CopulaConditionalUnivariateDensity())
        cond_copula.independent_destructor_ = self._make_destructor(cond_univ_densities)
        cond_copula.gaussian_density_ = cond_gaussian
        return cond_copula

    def _make_destructor(self, univariate_densities, same_density_estimator=True):
        if same_density_estimator:
            estimators = clone(univariate_densities[0])
        else:
            estimators = np.array([clone(est) for est in univariate_densities])

        # Setup fitted density
        independent_density = IndependentDensity(univariate_estimators=estimators)
        independent_density.univariate_densities_ = univariate_densities
        independent_density.n_dim_ = len(univariate_densities)

        # Setup fitted destructor
        independent_destructor = IndependentDestructor(
            independent_density=IndependentDensity(
                univariate_estimators=estimators
            )
        )
        independent_destructor.density_ = independent_density
        return independent_destructor

    def _check_is_fitted(self):
        check_is_fitted(self, ['independent_destructor_', 'gaussian_density_'])


class _CopulaConditionalUnivariateDensity(UnivariateDensity):
    def fit(self, X, y=None, **fit_params):
        raise NotImplementedError('Copula conditional densities should not be fit directly.')

    def sample(self, n_samples=1, random_state=None):
        self._check_is_fitted()
        rng = check_random_state(random_state)
        z = scipy.stats.norm.rvs(size=n_samples, loc=self.mean_, scale=self.std_)
        u = scipy.stats.norm.cdf(z)
        u = make_interior_probability(u)
        X = self.univariate_density_.inverse_cdf(u.reshape(-1, 1))
        return X

    def score_samples(self, X, y=None):
        self._check_is_fitted()
        X = self._check_X(X)

        marginal_log_likelihood = self.univariate_density_.score_samples(X)

        u = self.univariate_density_.cdf(X).ravel()
        z = scipy.stats.norm.ppf(u)
        copula_log_likelihood = (
            scipy.stats.norm.logpdf(z, loc=self.mean_, scale=self.std_)
            - scipy.stats.norm.logpdf(z)
            - np.log(self.std_)
        )
        return marginal_log_likelihood + copula_log_likelihood

    def cdf(self, X, y=None):
        self._check_is_fitted()
        X = self._check_X(X)

        U_X = self.univariate_density_.cdf(X)
        Z_X = scipy.stats.norm.ppf(U_X)
        Z = (Z_X - self.mean_)/self.std_
        return scipy.stats.norm.cdf(Z)

    def inverse_cdf(self, X, y=None):
        self._check_is_fitted()
        U = self._check_X(X, inverse=True)

        Z = scipy.stats.norm.ppf(U)
        Z_shift = self.std_ * Z + self.mean_
        U_Z = scipy.stats.norm.cdf(Z_shift)
        return self.univariate_density_.inverse_cdf(U_Z)

    def get_support(self):
        self._check_is_fitted()
        return self.univariate_density_.get_support()

    def _check_is_fitted(self):
        check_is_fitted(self, ['univariate_density_', 'mean_', 'std_'])


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
        _, n_dim = X.shape

        if self.covariance_type == 'full' or self.covariance_type == 'tied':
            if X.shape[0] == 1:
                self.covariance_ = np.zeros((n_dim, n_dim))
                if self.reg_covar <= 0:
                    raise ValueError('reg_covar <= 0 but only 1 sample given so variance '
                                     'impossible to estimate.')
            else:
                self.covariance_ = np.cov(X, rowvar=False).reshape((n_dim, n_dim))
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
        self._fit_auxiliary()
        X = check_array(X)
        return _estimate_log_gaussian_prob(
            X,
            self.mean_.reshape(1, -1),
            self.precision_cholesky_,
            self.covariance_type
        ).ravel()

    def sample(self, n_samples=1, random_state=None):
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
        self.n_dim_ = len(self.mean_)
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

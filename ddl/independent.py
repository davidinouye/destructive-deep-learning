"""Module for independent densities and destructors."""
from __future__ import division, print_function

import itertools
import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state

from .base import BaseDensityDestructor, ScoreMixin
from .univariate import STANDARD_NORMAL_DENSITY, ScipyUnivariateDensity
# noinspection PyProtectedMember
from .utils import (_UNIT_SPACE, check_X_in_interval, get_domain_or_default, get_support_or_default,
                    make_interior_probability)

logger = logging.getLogger(__name__)


class IndependentDestructor(BaseDensityDestructor):
    """Coordinate-wise destructor based on underlying independent density.

    This destructor assumes that the underlying density is independent (i.e.
    :class:`~ddl.independent.IndependentDensity`) and thus the
    transformation merely applys a univariate CDF to each feature
    independently of other features. The user can specify the univariate
    densities for each feature using the random variables defined in
    :mod:`scipy.stats`.  The fit method merely fits an independent density.
    For transform and inverse transform, this destrcutor mereley applies the
    corresponding CDFs and inverse CDFs to transform each feature
    independently.

    Parameters
    ----------
    independent_density : IndependentDensity
        The independent density estimator for this destructor.

    Attributes
    ----------
    density_ : IndependentDensity
        Fitted underlying independent density.

    See Also
    --------
    IndependentDensity

    """

    def __init__(self, independent_density=None):
        self.independent_density = independent_density

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
        if self.independent_density is None:
            return IndependentDensity()
        else:
            return clone(self.independent_density)

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
    """Independent density estimator.

    This density assumes that the underlying density is independent. The
    user can specify the univariate densities for each feature.

    Parameters
    ----------
    univariate_estimators : estimator or array-like of shape (n_features,)
        Univariate estimator(s) for this independent density. Default
        assumes univariate Gaussian densities for all features. Should be
        one of the following:

        1. None (default, assumes independent Gaussian density).
        2. Univariate density estimator (assumes all features have \
        the same density class, but the fitted parameters can be \
        different, e.g. the means of features 1 and 2 could be \
        different even though they are both Gaussian estimators.).
        3. Array-like of univariate density estimators for each feature.

    Attributes
    ----------
    univariate_densities_ : array, shape (n_features, )
        *Fitted* univariate estimators for each feature.

    n_features_ : int
        Number of features.

    See Also
    --------
    TreeDestructor
    ddl.univariate
    ddl.univariate.ScipyUnivariateDensity
    ddl.univariate.HistogramUnivariateDensity

    """

    def __init__(self, univariate_estimators=None):
        self.univariate_estimators = univariate_estimators

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
        X = np.array([
            np.ravel(u_dens.sample(n_samples=n_samples, random_state=rng))
            for u_dens in self.univariate_densities_
        ]).transpose()
        return X

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
        X = check_array(X)
        # Extract log-likelihood for all dimensions
        independent_scores = np.array([
            u_dens.score_samples(np.reshape(x_col, (-1, 1))).ravel()
            for u_dens, x_col in zip(self.univariate_densities_, X.transpose())
        ]).transpose()
        # Sum of log-likelihood is product of likelihoods because independent variables
        return independent_scores.sum(axis=1)

    def conditional_densities(self, X, cond_idx, not_cond_idx):
        """[Placeholder].

        Parameters
        ----------
        X :
        cond_idx :
        not_cond_idx :

        Returns
        -------
        obj : object

        """
        # Since independent, the conditional is equal to the marginal
        return self.marginal_density(not_cond_idx)

    def marginal_density(self, marginal_idx):
        """[Placeholder].

        Parameters
        ----------
        marginal_idx :

        Returns
        -------
        obj : object

        """
        marginal_density = clone(self)
        marginal_density.univariate_densities_ = self.univariate_densities_[marginal_idx]
        marginal_density.n_features_ = len(marginal_idx)
        # noinspection PyProtectedMember
        marginal_density._check_is_fitted()
        return marginal_density

    def marginal_cdf(self, x, target_idx):
        """[Placeholder].

        Parameters
        ----------
        x :
        target_idx :

        Returns
        -------
        obj : object

        """
        return self.univariate_densities_[target_idx].cdf(np.array(x).reshape(-1, 1)).reshape(
            np.array(x).shape)

    def marginal_inverse_cdf(self, x, target_idx):
        """[Placeholder].

        Parameters
        ----------
        x :
        target_idx :

        Returns
        -------
        obj : object

        """
        return self.univariate_densities_[target_idx].inverse_cdf(
            np.array(x).reshape(-1, 1)).reshape(np.array(x).shape)

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
    """Independent inverse CDF transformer applied coordinate-wise.

    A transformer (or *relative* destructor) that performs the inverse CDF
    transform independently for the fitted univariate densities
    corresponding to each feature. The default is the inverse CDF of the
    standard normal; this default is useful to make linear projection
    destructors canonical by prepending this as a preprocessing step so that
    the domain of the destructor is the unit hypercube (i.e. canonical
    domain).

    See :func:`fit` function documentation for more information.

    Attributes
    ----------
    fitted_densities_ : array, shape (n_features,)

        Fitted univariate densities for each feature. Note that these must
        be passed in as parameters to the :func:`fit` function. All needed
        transformation and scoring information is built into the univariate
        densities.  For example, the :func:`transform` function merely uses
        the :func:`inverse_cdf` function.

    See Also
    --------
    ddl.univariate
    IndependentDestructor

    """

    def fit(self, X, y=None, fitted_densities=None, **fit_params):
        """Fit estimator to X.

        X is only used to get the number of features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : None, default=None
            Not used in the fitting process but kept for compatibility.

        fitted_densities : array-like of estimators
            Default assumes that `fitted_densities` are standard Gaussian.
            `fitted_densities` should be fitted versions of the following
            similar to the `univariate_estimators` parameter of
            `IndependentDensity`:

                #. None (defaults to fitted `ScipyUnivariateDensity()`),
                #. univariate density estimator,
                #. array-like of univariate density estimators.

        Returns
        -------
        self : estimator
            Returns the instance itself.

        """
        X = check_array(X)

        # Mainly just get default and make array of densities if needed
        dens_arr = self._get_densities_or_default(fitted_densities, X.shape[1])
        self.fitted_densities_ = dens_arr
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

from __future__ import division, print_function

import logging
import warnings
from abc import abstractmethod

import numpy as np
import scipy.stats
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, DensityMixin, TransformerMixin, clone
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state, column_or_1d

from .base import BoundaryWarning, ScoreMixin
# noinspection PyProtectedMember
from .utils import (_DEFAULT_SUPPORT, check_X_in_interval, make_finite, make_interior_probability,
                    make_positive)

logger = logging.getLogger(__name__)

SCIPY_RV_NON_NEGATIVE = ['expon', 'chi']
SCIPY_RV_STRICLTY_POSITIVE = ['gamma', 'invgamma', 'chi2', 'lognorm']
SCIPY_RV_UNIT_SUPPORT = ['uniform', 'beta']


def _check_univariate_X(X, support, inverse=False):
    X = check_array(X, ensure_2d=True)  # ensure_2d=True is default but just making explicit
    # Check that X is a column vector but first ravel because check_estimator passes
    #  a matrix to fit
    if X.shape[1] > 1:
        warnings.warn(DataConversionWarning(
            'Input should be column vector with shape (n, 1) but found matrix. Converting to '
            'column vector via `array.ravel().reshape((-1, 1))`. '
            'Ideally, this would raise an error but in order to pass the checks in '
            '`sklearn.utils.check_estimator`, we convert the data rather than raise an error. '
        ))
        X = np.ravel(X).reshape((-1, 1))

    # Check that values are within support or range(inverse)
    if inverse:
        X = check_X_in_interval(X, np.array([0, 1]))
    else:
        X = check_X_in_interval(X, support)
    return np.array(X)


class UnivariateDensity(BaseEstimator, ScoreMixin):
    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        pass

    @abstractmethod
    def sample(self, n_samples=1, random_state=None):
        pass

    @abstractmethod
    def score_samples(self, X, y=None):
        pass

    @abstractmethod
    def cdf(self, X, y=None):
        pass

    @abstractmethod
    def inverse_cdf(self, X, y=None):
        pass

    def get_support(self):
        return np.array([_DEFAULT_SUPPORT])

    def _check_X(self, X, inverse=False):
        return _check_univariate_X(X, self.get_support(), inverse=inverse)


class ScipyUnivariateDensity(UnivariateDensity):
    """Density estimator based on random variables defined by
    `scipy.stats`.
    """
    def __init__(self, scipy_rv=None, scipy_fit_kwargs=None):
        """Default random variable is a Gaussian (i.e.
        `scipy.stats.norm`) if `scipy_rv=None`.

        scipy_fit_kwargs defaults differently depending on `scipy_rv`.
        For example for the `scipy.stats.beta` we set `floc=0` and
        `fscale=1`, i.e. fix the location and scale of the beta
        distribution.
        """
        self.scipy_rv = scipy_rv
        self.scipy_fit_kwargs = scipy_fit_kwargs

    def fit(self, X, y=None, **fit_params):
        def _check_scipy_kwargs(kwargs, _scipy_rv):
            if kwargs is None:
                if self._is_special(SCIPY_RV_UNIT_SUPPORT):
                    # logger.debug('Fixing floc=0, fscale=1')
                    return dict(floc=0, fscale=1)
                elif self._is_special(SCIPY_RV_NON_NEGATIVE + SCIPY_RV_STRICLTY_POSITIVE):
                    # logger.debug('Fixing floc=0')
                    return dict(floc=0)
                else:
                    return {}
            elif isinstance(kwargs, dict):
                return kwargs
            else:
                raise ValueError('`scipy_fit_kwargs` should be either None or a `dict` object.')

        # Input validation
        scipy_rv = self._get_scipy_rv_or_default()
        scipy_fit_kwargs = _check_scipy_kwargs(self.scipy_fit_kwargs, scipy_rv)
        X = self._check_X(X)

        # MLE fit based on scipy implementation
        if scipy_rv.numargs == 0 and 'floc' in scipy_fit_kwargs and 'fscale' in scipy_fit_kwargs:
            params=(scipy_fit_kwargs['floc'], scipy_fit_kwargs['fscale'])
        else:
            try:
                params = scipy_rv.fit(X.ravel(), **scipy_fit_kwargs)
            except RuntimeError as e:
                warnings.warn('Unable to fit to data using scipy_rv so attempting to use default '
                              'parameters for the distribution. Original error:\n%s' % str(e))
                params = self._get_default_params()
            except ValueError as e:
                #warnings.warn('Trying to use fixed parameters instead. Original error:\n%s' % str(e))
                # try to extract fixed parameters in a certain order
                params = []
                for k in ['fa', 'f0', 'fb', 'f1', 'floc', 'fscale']:
                    try:
                        params.append(scipy_fit_kwargs.pop(k))
                    except KeyError:
                        pass

        # Avoid degenerate case when scale = 0
        if len(params) >= 2 and params[-1] == 0:
            params = list(params)
            if isinstance(X.dtype, np.floating):
                params[-1] = np.finfo(X.dtype).eps
            else:
                params[-1] = 1  # Integer types
            params = tuple(params)

        # Create "frozen" version of random variable so that parameters do not need to be
        # specified
        self.rv_ = scipy_rv(*params)
        # Check for a fit error in the domain of the parameters
        try:
            self.rv_.rvs(1)
        except ValueError as e:
            warnings.warn('Parameters discovered by fit are not in the domain of the '
                          'parameters so attempting to use default parameters for the '
                          'distribution.')
            self.rv_ = scipy_rv(*self._get_default_params())
        return self

    def _get_default_params(self):
        if self._is_special(['beta']):
            return [1, 1]
        elif self._is_special(['uniform', 'norm', 'expon', 'lognorm']):
            return []  # Empty since no parameters needed
        else:
            raise NotImplementedError('The distribution given by the `scipy_rv = %s` does not '
                                      'have any associated default parameters.'
                                      % str(self._get_scipy_rv_or_default()))

    def sample(self, n_samples=1, random_state=None):
        self._check_is_fitted()
        rng = check_random_state(random_state)
        return np.array(self.rv_.rvs(size=n_samples, random_state=rng)).reshape((n_samples, 1))

    def score_samples(self, X, y=None):
        self._check_is_fitted()
        X = self._check_X(X)
        return self.rv_.logpdf(X.ravel()).reshape((-1, 1))

    def cdf(self, X, y=None):
        self._check_is_fitted()
        X = self._check_X(X)
        return self.rv_.cdf(X.ravel()).reshape((-1, 1))

    def inverse_cdf(self, X, y=None):
        self._check_is_fitted()
        X = self._check_X(X, inverse=True)
        return self.rv_.ppf(X.ravel()).reshape((-1, 1))

    def get_support(self):
        # Assumes density is univariate
        try:
            self._check_is_fitted()
        except NotFittedError:
            # Get upper and lower bounds of support from scipy random variable properties
            if self.scipy_rv is None:
                default_rv = ScipyUnivariateDensity._get_default_scipy_rv()
                return np.array([[default_rv.a, default_rv.b]])
            else:
                return np.array([[self.scipy_rv.a, self.scipy_rv.b]])
        else:
            # Scale and shift if fitted
            try:
                loc = self.rv_.args[-2]
            except IndexError:
                try:
                    loc = self.rv_.args[-1]
                except IndexError:
                    loc = 0
                scale = 1
            else:
                scale = self.rv_.args[-1]
            if scale == 0:  # Handle special degenerate case to avoid nans in domain
                scale += np.finfo(float).eps
            return loc + scale * np.array([[self.rv_.a, self.rv_.b]])

    def _check_X(self, X, inverse=False):
        # Check that X is univariate or warn otherwise
        X = super(ScipyUnivariateDensity, self)._check_X(X, inverse)

        # Move away from support/domain boundaries if necessary
        scipy_rv = self._get_scipy_rv_or_default()
        if inverse and (np.any(X <= 0) or np.any(X >= 1)):
            warnings.warn(BoundaryWarning(
                'Some probability values (input to inverse functions) are either 0 or 1. Bounding '
                'values away from 0 or 1 to avoid infinities in output.  For example, the inverse '
                'cdf of a Gaussian at 0 will yield `-np.inf`.'))
            X = make_interior_probability(X)
        if self._is_special(SCIPY_RV_UNIT_SUPPORT) and (np.any(X <= 0) or np.any(X >= 1)):
            warnings.warn(BoundaryWarning(
                'Input to random variable function has at least one value either 0 or 1 '
                'but all input should be in (0,1) exclusive. Bounding values away from 0 or 1 by '
                'eps=%g'))
            X = make_interior_probability(X)
        if self._is_special(SCIPY_RV_STRICLTY_POSITIVE) and np.any(X <= 0):
            warnings.warn(BoundaryWarning(
                'Input to random variable function has at least one value less than or equal to '
                'zero but all input should be strictly positive. Making all input greater than or '
                'equal to some small positive constant.'))
            X = make_positive(X)
        if np.any(np.isinf(X)):
            warnings.warn(BoundaryWarning(
                'Input to random variable function has at least one value that is `np.inf` or '
                '`-np.inf`. Making all input finite via a very large constant.'))
            X = make_finite(X)
        return X

    def _get_scipy_rv_or_default(self):
        if self.scipy_rv is None:
            return ScipyUnivariateDensity._get_default_scipy_rv()
        else:
            return self.scipy_rv

    @staticmethod
    def _get_default_scipy_rv():
        return scipy.stats.norm

    def _is_special(self, scipy_str_set):
        scipy_rv = self._get_scipy_rv_or_default()
        return np.any([
            '.'+dstr+'_gen' in str(scipy_rv)
            for dstr in scipy_str_set
        ])

    def _check_is_fitted(self):
        check_is_fitted(self, ['rv_'])


with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=UserWarning)
    STANDARD_NORMAL_DENSITY = ScipyUnivariateDensity(
        scipy_rv=scipy.stats.norm,
        scipy_fit_kwargs=dict(floc=0, fscale=1)
    ).fit(np.array([[0]]))


class PiecewiseConstantUnivariateDensity(UnivariateDensity):
    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        """Should fit parameters of piecewise constant estimator."""
        raise NotImplementedError()

    def sample(self, n_samples=1, random_state=None):
        # Inverse cdf sampling via uniform samples
        rng = check_random_state(random_state)
        u = rng.rand(n_samples)
        # Find insertion point
        idx = np.searchsorted(self.cdf_query_, u)
        # Sample uniformly from that bar
        x = (rng.rand(n_samples) - 0.5) * self.query_width_ + self.x_query_[idx]
        return x.reshape((-1, 1))

    def score_samples(self, X, y=None):
        self._check_is_fitted()
        X = self._check_X(X)
        X = X.ravel()

        # Interp nearest neighbor
        f_X = np.zeros(X.shape)
        interp_callable = interp1d(
            self.x_query_, self.pdf_query_,
            kind='nearest', copy=False,
            bounds_error=False, fill_value=(0, 0), assume_sorted=True
        )
        f_X = interp_callable(X)
        # Bump away from zero to avoid error in np.log
        f_X = np.maximum(f_X, np.finfo(f_X.dtype).tiny)
        return np.log(f_X).reshape((-1, 1))

    def cdf(self, X, y=None):
        self._check_is_fitted()
        self._check_X(X)
        X = X.ravel()

        # Interp nearest neighbor
        F_X = np.zeros(X.shape)
        interp_callable = interp1d(
            self.x_query_[:-1] + self.query_width_ / 2.0, self.cdf_query_[:-1],
            kind='linear', copy=False,
            bounds_error=False, fill_value=(0, 1), assume_sorted=True
        )
        F_X = interp_callable(X)
        return F_X.reshape((-1, 1))

    def inverse_cdf(self, X, y=None):
        self._check_is_fitted()
        self._check_X(X)
        X = X.ravel()

        # Interp nearest neighbor
        Finv_X = np.zeros(X.shape)
        interp_callable = interp1d(
            self.cdf_query_[:-1], self.x_query_[:-1] + self.query_width_ / 2.0,
            kind='linear', copy=False,
            bounds_error=False, fill_value=(0, 1), assume_sorted=True
        )
        Finv_X = interp_callable(X)
        return Finv_X.reshape((-1, 1))

    def get_support(self):
        # Make [[a,b]] so that it is explicitly a univariate density
        return np.array([self._check_bounds()])

    def _check_X(self, X, inverse=False):
        # Check that X is univariate or warn otherwise
        X = super(PiecewiseConstantUnivariateDensity, self)._check_X(X, inverse)
        return X

    def _check_bounds(self, X=None, extend=True):
        # If bounds is extension
        if np.isscalar(self.bounds):
            if X is None:
                # If no X than just return -inf, inf
                return _DEFAULT_SUPPORT
            else:
                # If X is not None than extract bounds and extend as necessary
                perc_extension = self.bounds
                _domain = np.array([np.min(X), np.max(X)])
                center = np.mean(_domain)
                _domain = (1 + perc_extension) * (_domain - center) + center
                return _domain
        # If bounds is just an array then directly return it
        else:
            _domain = column_or_1d(self.bounds).copy()
            if _domain.shape[0] != 2:
                raise ValueError('Domain should either be a two element array-like or a'
                                 ' scalar indicating percentage extension of domain')
            return _domain

    def _check_is_fitted(self):
        check_is_fitted(self, ['bounds_', 'x_query_', 'query_width_',
                               'pdf_query_', 'cdf_query_'])

    def _normalize_f_query(self, f_query, query_width):
        # noinspection PyAugmentAssignment
        f_query = f_query/np.sum(f_query)  # Normalize to 1
        f_query /= query_width  # Adjust by bin width to make valid pdf
        return f_query

    def _compute_F_query(self, f_query):
        F_query = np.cumsum(f_query)
        F_query = F_query / F_query[-1]  # Normalize to sum to 1
        F_query[0] = 0  # Ensure initial point is exact
        F_query[-2:-1] = 1  # Ensure last points are exact
        return F_query

    def _check_is_fitted(self):
        check_is_fitted(self, ['bounds_', 'x_query_', 'query_width_',
                               'pdf_query_', 'cdf_query_'])


class HistogramUnivariateDensity(PiecewiseConstantUnivariateDensity):
    """Bounds can be percentage extension or a specified interval [a,b].
    Parameter `bins` can take any value as the same parameter of `numpy.histogram`
    """
    def __init__(self, bins=None, bounds=0.1, alpha=1e-6):
        self.bins = bins
        self.bounds = bounds
        self.alpha = alpha

    def fit(self, X, y=None, **fit_params):
        X = self._check_X(X)
        # Get perc_extension but do not modify bounds
        bounds = self._check_bounds(X)
        bins = self.bins if self.bins is not None else 'auto'

        # Fit numpy histogram
        hist, bin_edges = np.histogram(X, bins=bins, range=bounds)
        hist = np.array(hist, dtype=float)  # Make float so we can add non-integer alpha
        hist += self.alpha  # Smooth histogram by alpha so no areas have 0 probability

        return self._fit(hist, bin_edges)
    
    def fit_from_probabilities(self, prob):
        bounds = self._check_bounds()
        bins = self.bins if self.bins is not None else 'auto'
        prob = column_or_1d(prob)

        # Fit numpy histogram
        n_features = prob.shape[0]
        X_temp = np.mean(self.bounds)*np.ones((1, n_features))
        hist, bin_edges = np.histogram(X_temp, bins=bins, range=bounds)
        hist = prob

        return self._fit(hist, bin_edges)

    def _fit(self, hist, bin_edges):
        """Fit given probabilities for histogram and bin edges."""
        bounds = np.array([bin_edges[0], bin_edges[-1]])
        bin_width = bin_edges[1] - bin_edges[0]
        x_query = bin_edges[:-1] + bin_width/2.0

        # Add endpoints fixed at 0
        x_query = np.concatenate(([bin_edges[0] - bin_width/2.0],
                                  x_query,
                                  [bin_edges[-1] + bin_width/2.0]))
        hist = np.concatenate(([0], hist, [0]))

        f_query = self._normalize_f_query(hist, bin_width)
        F_query = self._compute_F_query(f_query)

        self.bounds_ = bounds
        self.x_query_ = x_query
        self.query_width_ = bin_width
        self.pdf_query_ = f_query
        self.cdf_query_ = F_query
        return self


class ApproximateUnivariateDensity(PiecewiseConstantUnivariateDensity):
    """Bounds can be percentage extension or a specified interval [a,b]."""
    def __init__(self, univariate_density=None, n_query=1000, bounds=0.1):
        self.univariate_density = univariate_density
        self.n_query = n_query
        self.bounds = bounds

    def fit(self, X, y=None, **fit_params):
        # Validate parameters
        X = self._check_X(X)
        bounds = self._check_bounds(X)
        univariate_density = self._get_univariate_density_or_default()
        if not (float(self.n_query).is_integer() and self.n_query > 0):
            raise ValueError('n_query must be positive whole number')

        # Fit density for each dimension
        def unwrap_estimator(est):
            # Unwrap CV estimator
            if hasattr(est, 'best_estimator_'):
                return est.best_estimator_
            return est

        density = unwrap_estimator(clone(univariate_density).fit(X))

        # Find query points based on domain and number query points
        domain_size = bounds[1] - bounds[0]
        query_width = domain_size / self.n_query
        # Add 2 more points that are fixed to 0 outside of the domain
        x_query = np.linspace(bounds[0] - query_width / 2.0,
                              bounds[1] + query_width / 2.0,
                              self.n_query + 2)

        f_query = np.zeros(x_query.shape)
        # Skip endpoints which should be 0
        X_query_without_ends = x_query[1:-1].reshape((-1, 1))
        f_query[1:-1] = np.exp(density.score_samples(X_query_without_ends))
        f_query = self._normalize_f_query(f_query, query_width)

        F_query = self._compute_F_query(f_query)

        # Save important fitted values
        self.density_estimator_ = density
        self.bounds_ = bounds
        self.x_query_ = x_query
        self.query_width_ = query_width
        self.pdf_query_ = f_query
        self.cdf_query_ = F_query

        return self

    @abstractmethod
    def _get_univariate_density_or_default(self):
        raise NotImplementedError()


class KernelUnivariateDensity(ApproximateUnivariateDensity):
    def __init__(self, bandwidth=None, n_query=1000, bounds=0.1):
        super(KernelUnivariateDensity, self).__init__()
        self.bandwidth = bandwidth
        self.n_query = n_query
        self.bounds = bounds

    def _get_univariate_density_or_default(self):
        if self.bandwidth is None:
            # bandwidth = np.logspace(-3, 4, 50)
            bandwidth = 0.1
        else:
            bandwidth = self.bandwidth

        # Grid search if bandwidth is given
        if len(np.array(bandwidth).shape) > 0:
            return GridSearchCV(
                estimator=KernelDensity(),
                param_grid={'bandwidth': bandwidth},
            )
        else:
            return KernelDensity(bandwidth=bandwidth)

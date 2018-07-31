"""Module for univariate densities (see also :mod:`ddl.independent`)."""
from __future__ import division, print_function

import logging
import warnings
from abc import abstractmethod

import numpy as np
import scipy.stats
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, clone
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


class _UnivariateDensity(BaseEstimator, ScoreMixin):
    """Univariate density (abstract class)."""

    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        """[Placeholder].

        Parameters
        ----------
        X :
        y :
        fit_params :

        """
        pass

    @abstractmethod
    def sample(self, n_samples=1, random_state=None):
        """[Placeholder].

        Parameters
        ----------
        n_samples :
        random_state :

        """
        pass

    @abstractmethod
    def score_samples(self, X, y=None):
        """[Placeholder].

        Parameters
        ----------
        X :
        y :

        """
        pass

    @abstractmethod
    def cdf(self, X, y=None):
        """[Placeholder].

        Parameters
        ----------
        X :
        y :

        """
        pass

    @abstractmethod
    def inverse_cdf(self, X, y=None):
        """[Placeholder].

        Parameters
        ----------
        X :
        y :

        """
        pass

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


class ScipyUnivariateDensity(_UnivariateDensity):
    """Density estimator via random variables defined in :mod:`scipy.stats`.

    A univariate density estimator that can fit any distribution defined in
    :mod:`scipy.stats`.  This includes common distributions such as Gaussian,
    laplace, beta, gamma and log-normal distributions but also many other
    distributions as well.

    Note that this density estimator is strictly univariate and therefore
    expects the input data to be a single array with shape (n_samples, 1).

    Parameters
    ----------
    scipy_rv : object or None, default=None
        Default random variable is a Gaussian (i.e.
        :func:`scipy.stats.norm`) if `scipy_rv=None`. Other examples include
        :func:`scipy.stats.gamma` or :func:`scipy.stats.beta`.

    scipy_fit_kwargs : dict or None, optional
        Keyword arguments as a dictionary for the fit function of the scipy
        random variable (e.g. ``dict(floc=0, fscale=1)`` to fix the location
        and scale parameters to 0 and 1 respectively). Defaults are
        different depending on `scipy_rv` parameter. For example for the
        `scipy.stats.beta` we set `floc=0` and `fscale=1`, i.e. fix the
        location and scale of the beta distribution.

    Attributes
    ----------
    rv_ : object
        Frozen :mod:`scipy.stats` random variable object. Fitted parameters
        of distribution can be accessed via `args` property.

    See Also
    --------
    scipy.stats

    """

    def __init__(self, scipy_rv=None, scipy_fit_kwargs=None):
        self.scipy_rv = scipy_rv
        self.scipy_fit_kwargs = scipy_fit_kwargs

    def fit(self, X, y=None, **fit_params):
        """Fit estimator to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, 1)
            Training data, where `n_samples` is the number of samples. Note
            that the shape must have a second dimension of 1 since this is a
            univariate density estimator.

        y : None, default=None
            Not used in the fitting process but kept for compatibility.

        fit_params : dict, optional
            Optional extra fit parameters.

        Returns
        -------
        self : estimator
            Returns the instance itself.

        """
        def _check_scipy_kwargs(kwargs, _scipy_rv):
            if kwargs is None:
                if self._is_special(SCIPY_RV_UNIT_SUPPORT):
                    return dict(floc=0, fscale=1)
                elif self._is_special(SCIPY_RV_NON_NEGATIVE + SCIPY_RV_STRICLTY_POSITIVE):
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
            params = (scipy_fit_kwargs['floc'], scipy_fit_kwargs['fscale'])
        else:
            try:
                params = scipy_rv.fit(X.ravel(), **scipy_fit_kwargs)
            except RuntimeError as e:
                warnings.warn('Unable to fit to data using scipy_rv so attempting to use default '
                              'parameters for the distribution. Original error:\n%s' % str(e))
                params = self._get_default_params()
            except ValueError as e:
                # warnings.warn(
                #     'Trying to use fixed parameters instead. Original error:\n%s' % str(e))
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
        return np.array(self.rv_.rvs(size=n_samples, random_state=rng)).reshape((n_samples, 1))

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
        X = self._check_X(X)
        return self.rv_.logpdf(X.ravel()).reshape((-1, 1))

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
        return self.rv_.cdf(X.ravel()).reshape((-1, 1))

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
        return self.rv_.ppf(X.ravel()).reshape((-1, 1))

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
            '.' + dstr + '_gen' in str(scipy_rv)
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


class _PiecewiseConstantUnivariateDensity(_UnivariateDensity):
    """Piecewise constant univariate density (e.g. histogram and tree)."""

    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        """Should fit parameters of piecewise constant estimator."""
        raise NotImplementedError()

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
        # Inverse cdf sampling via uniform samples
        rng = check_random_state(random_state)
        u = rng.rand(n_samples)
        # Find insertion point
        idx = np.searchsorted(self.cdf_query_, u)
        # Sample uniformly from that bar
        x = (rng.rand(n_samples) - 0.5) * self.query_width_ + self.x_query_[idx]
        return x.reshape((-1, 1))

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
        # Make [[a,b]] so that it is explicitly a univariate density
        return np.array([self._check_bounds()])

    def _check_X(self, X, inverse=False):
        # Check that X is univariate or warn otherwise
        X = super(_PiecewiseConstantUnivariateDensity, self)._check_X(X, inverse)
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

    def _normalize_f_query(self, f_query, query_width):
        # noinspection PyAugmentAssignment
        f_query = f_query / np.sum(f_query)  # Normalize to 1
        f_query /= query_width  # Adjust by bin width to make valid pdf
        return f_query

    # noinspection PyPep8Naming
    def _compute_F_query(self, f_query):
        F_query = np.cumsum(f_query)
        F_query = F_query / F_query[-1]  # Normalize to sum to 1
        F_query[0] = 0  # Ensure initial point is exact
        F_query[-2:-1] = 1  # Ensure last points are exact
        return F_query

    def _check_is_fitted(self):
        check_is_fitted(self, ['bounds_', 'x_query_', 'query_width_',
                               'pdf_query_', 'cdf_query_'])


class HistogramUnivariateDensity(_PiecewiseConstantUnivariateDensity):
    """Histogram univariate density estimator.

    Parameters
    ----------
    bins : int or sequence of scalars or str, optional
        Same ase the parameter of :func:`numpy.histogram`. Copied from numpy
        documentation:

        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a
        sequence, it defines the bin edges, including the rightmost
        edge, allowing for non-uniform bin widths.
        .. versionadded:: 1.11.0
        If `bins` is a string from the list below, `histogram` will use
        the method chosen to calculate the optimal bin width and
        consequently the number of bins (see `Notes` for more detail on
        the estimators) from the data that falls within the requested
        range. While the bin width will be optimal for the actual data
        in the range, the number of bins will be computed to fill the
        entire range, including the empty portions. For visualisation,
        using the 'auto' option is suggested. Weighted data is not
        supported for automated bin size selection.
        'auto'
            Maximum of the 'sturges' and 'fd' estimators. Provides good
            all around performance.
        'fd' (Freedman Diaconis Estimator)
            Robust (resilient to outliers) estimator that takes into
            account data variability and data size.
        'doane'
            An improved version of Sturges' estimator that works better
            with non-normal datasets.
        'scott'
            Less robust estimator that that takes into account data
            variability and data size.
        'rice'
            Estimator does not take variability into account, only data
            size. Commonly overestimates number of bins required.
        'sturges'
            R's default method, only accounts for data size. Only
            optimal for gaussian data and underestimates number of bins
            for large non-gaussian datasets.
        'sqrt'
            Square root (of data size) estimator, used by Excel and
            other programs for its speed and simplicity.

    bounds : float or array-like of shape (2,)
        Specification for the finite bounds of the histogram. Bounds can be
        percentage extension or a specified interval [a,b].

    alpha : float
        Regularization parameter corresponding to the number of
        pseudo-counts to add to each bin of the histogram. This can be seen
        as putting a Dirichlet prior on the empirical bin counts with
        Dirichlet parameter alpha.

    Attributes
    ----------
    bounds_ : array of shape (2,)
        Fitted bounds for the histogram where `bounds_[0]` is the minimum and
        `bounds_[1]` is the maximum.

    x_query_ : array of shape (n_bins + 1,)
        Query points along domain corresponding to the middle of bins.

    query_width_ : array of shape (n_bins + 1,)
        Bin width or spacing between query points. Used with linear
        interpolation to compute pdf, cdf and inverse cdf.

    pdf_query_ : array of shape (n_bins + 1,)
        pdf values at query points. Note that histograms have a constant pdf
        value within each bin.

    cdf_query_ : array of shape (n_bins + 1,)
        cdf values at query points. Used with linear interpolation to
        compute pdf, cdf and inverse cdf.

    """

    def __init__(self, bins=None, bounds=0.1, alpha=1e-6):
        self.bins = bins
        self.bounds = bounds
        self.alpha = alpha

    def fit(self, X, y=None, **fit_params):
        """Fit estimator to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, 1)
            Training data, where `n_samples` is the number of samples. Note
            that the shape must have a second dimension of 1 since this is a
            univariate density estimator.

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
        # Get percent extension but do not modify bounds
        bounds = self._check_bounds(X)
        bins = self.bins if self.bins is not None else 'auto'

        # Fit numpy histogram
        hist, bin_edges = np.histogram(X, bins=bins, range=bounds)
        hist = np.array(hist, dtype=float)  # Make float so we can add non-integer alpha
        hist += self.alpha  # Smooth histogram by alpha so no areas have 0 probability

        return self._fit(hist, bin_edges)

    def fit_from_probabilities(self, prob):
        """[Placeholder].

        Parameters
        ----------
        prob :

        Returns
        -------
        obj : object

        """
        bounds = self._check_bounds()
        bins = self.bins if self.bins is not None else 'auto'
        prob = column_or_1d(prob)

        # Fit numpy histogram
        n_features = prob.shape[0]
        X_temp = np.mean(self.bounds) * np.ones((1, n_features))
        hist, bin_edges = np.histogram(X_temp, bins=bins, range=bounds)
        hist = prob

        return self._fit(hist, bin_edges)

    def _fit(self, hist, bin_edges):
        """Fit given probabilities for histogram and bin edges."""

        bounds = np.array([bin_edges[0], bin_edges[-1]])
        bin_width = bin_edges[1] - bin_edges[0]
        x_query = bin_edges[:-1] + bin_width / 2.0

        # Add endpoints fixed at 0
        x_query = np.concatenate(([bin_edges[0] - bin_width / 2.0],
                                  x_query,
                                  [bin_edges[-1] + bin_width / 2.0]))
        hist = np.concatenate(([0], hist, [0]))

        f_query = self._normalize_f_query(hist, bin_width)
        F_query = self._compute_F_query(f_query)

        self.bounds_ = bounds
        self.x_query_ = x_query
        self.query_width_ = bin_width
        self.pdf_query_ = f_query
        self.cdf_query_ = F_query
        return self


class _ApproximateUnivariateDensity(_PiecewiseConstantUnivariateDensity):
    """Bounds can be percentage extension or a specified interval [a,b]."""

    def __init__(self, univariate_density=None, n_query=1000, bounds=0.1):
        self.univariate_density = univariate_density
        self.n_query = n_query
        self.bounds = bounds

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
        # Validate parameters
        X = self._check_X(X)
        bounds = self._check_bounds(X)
        univariate_density = self._get_univariate_density_or_default()
        if not (float(self.n_query).is_integer() and self.n_query > 0):
            raise ValueError('n_query must be positive whole number')

        # Fit density for each dimension
        def unwrap_estimator(est):
            """

            Parameters
            ----------
            est :

            Returns
            -------

            """
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


class _KernelUnivariateDensity(_ApproximateUnivariateDensity):
    """Kernel univariate density.

    """

    def __init__(self, bandwidth=None, n_query=1000, bounds=0.1):
        super(_KernelUnivariateDensity, self).__init__()
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

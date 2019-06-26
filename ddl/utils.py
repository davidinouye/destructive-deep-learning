"""Module for utility functions and classes."""
from __future__ import division, print_function

import itertools
import logging
import warnings

import numpy as np
from sklearn.exceptions import DataConversionWarning

_INF_SPACE = np.array([-np.inf, np.inf])
_UNIT_SPACE = np.array([0, 1])
_DEFAULT_DOMAIN = _INF_SPACE
_DEFAULT_SUPPORT = _INF_SPACE

logger = logging.getLogger(__name__)


def get_support_or_default(density, warn=False):
    """Get the support of the density or return `DEFAULT_SUPPORT`.

    Default support is [-infty, infty].

    Parameters
    ----------
    density : estimator
        Density estimator.

    warn : bool, default=False
        Whether to warn if there the estimator does not implement
        :func:`get_support`.

    Returns
    -------
    support : array-like, shape (2,) or (2, n_features)
        The support of the density as returned by :func:`get_support` or
        just return the default support.

    """
    if has_method(density, 'get_support', warn=False):
        return density.get_support()
    else:
        if warn:
            msg = ('Support is assumed to be %s since '
                   'dens.get_support() is not implemented.'
                   % str(_DEFAULT_SUPPORT))
            warnings.warn(msg)
        return _DEFAULT_SUPPORT


def get_domain_or_default(destructor, warn=False):
    """Get the domain of the density or return `DEFAULT_DOMAIN`.

    Default domain is [-infty, infty].

    Parameters
    ----------
    destructor : estimator
        Destructor estimator.

    warn : bool, default=False
        Whether to warn if there the estimator does not implement
        :func:`get_domain`.

    Returns
    -------
    domain : array-like, shape (2,) or (2, n_features)
        The domain of the density as returned by :func:`get_domain` or
        just return the default domain.

    """
    if has_method(destructor, 'get_domain', warn=False):
        return destructor.get_domain()
    else:
        if warn:
            msg = ('Domain is assumed to be %s since '
                   'trans.get_domain() is not implemented.'
                   % str(_DEFAULT_DOMAIN))
            warnings.warn(msg)
        return _DEFAULT_DOMAIN


def check_domain(domain, n_features):
    """Check and return domain, broadcasting domain if necessary.

    Parameters
    ----------
    domain : array-like, shape (2,) or (2, n_features)
        The minimum and maximum for each dimension. If shape is (2,) then
        the minimum and maximum are assumed to be the same for every
        dimension.

    n_features : int
        The number of features. Used to check domain shape or broadcast
        domain if necessary.

    Returns
    -------
    domain : array, shape (2, n_features)
        Domain after error checking and broadcasting as necessary.

    >>> check_domain([0, 1], 3)
    array([[0, 1],
           [0, 1],
           [0, 1]])

    """
    domain = np.array(domain)
    if len(domain.shape) == 1:
        domain = np.array([domain for i in range(n_features)])
    if np.any(np.isnan(domain)):
        raise ValueError('The domain/support should not contain NaN values.')
    if len(domain) != n_features:
        warnings.warn(DataConversionWarning(
            'Domain had %d dimensions but requested `n_features` was %d. Using `domain = '
            'itertools.islice(itertools.cycle(domain), n_features)`.'
            % (len(domain), n_features)))
        domain = list(itertools.islice(itertools.cycle(domain), n_features))
    return domain


def check_X_in_interval(X, interval):
    """Check if the input X lies in the specified interval.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data matrix to check.

    interval : array-like, shape (2,) or (2, n_features)
        Interval to check. See :func:`check_domain` for interval types.

    Returns
    -------
    X : array, shape (n_samples, n_features)
        Data matrix as numpy array after checking and possibly
        shifting/scaling data as necessary to fit within specified interval.

    """
    msg_suffix = ('Thus, the original values will be clipped to the given domain: '
                  '%s.\n(Ideally, this would be an exception instead of a warning but the '
                  'current implementation of `sklearn.utils.check_estimator` (sklearn version '
                  '0.19.1) will fail if an exception is raised while calling fit, transform, '
                  'etc.  Therefore, we only require that an warning is issued.)'
                  % str(interval.tolist()))
    n_samples, n_features = np.shape(X)
    if n_samples == 0:
        return X  # Trivial case of no samples
    dom = check_domain(interval, n_features)
    copied = False
    for i, (low_domain, high_domain), low, high in zip(range(n_features), dom, np.min(X, axis=0),
                                                       np.max(X, axis=0)):
        if low < low_domain:
            warnings.warn(DataConversionWarning(
                'The minimum of dimension %d is not in the interval: %g (X_min) < %g ('
                'interval_min), diff = %g. %s'
                % (i, low, low_domain, low-low_domain, msg_suffix)))
        if high > high_domain:
            warnings.warn(DataConversionWarning(
                'The maximum of dimension %d is not in the interval: %g (X_max) > %g ('
                'interval_max), diff = %g. %s'
                % (i, high, high_domain, high-high_domain, msg_suffix)))
        # Rescale values if either too low or too high
        if low < low_domain or high > high_domain:
            if not copied:
                X = X.copy()
                copied = True
            # Clip to high and low domain values
            X[:, i] = np.minimum(high, np.maximum(low, X[:, i]))
    return X


def check_X_in_interval_decorator(func):
    """Decorate functions such as `transform` to check domain."""
    def wrapper(trans, X, *args, **kwargs):
        """[Placeholder].

        Parameters
        ----------
        trans :
        X :
        args :
        kwargs :

        Returns
        -------
        obj : object

        """
        X = check_X_in_interval(X, get_domain_or_default(trans))
        return func(trans, X, *args, **kwargs)

    return wrapper


def has_method(est, method_name, warn=True):
    """Check if an estimator has a method and possibly warn if not.

    Parameters
    ----------
    est : estimator
        Estimator to check.

    method_name : str
        Method to check.

    warn : bool
        Whether to warn if the method is not found.

    Returns
    -------
    has_method : bool
        Whether the estimator has the specified method.

    """
    if hasattr(est, method_name) and callable(getattr(est, method_name)):
        return True
    elif hasattr(est, method_name) and not callable(getattr(est, method_name)):
        raise TypeError(
            'While %s has the attribute %s, it is not callable (i.e. it is not a method).'
            % (est.__class__, method_name))
    elif not hasattr(est, method_name):
        if warn:
            warnings.warn(
                '%s does not have the specified attribute/method `%s` so skipping tests that '
                'require method `%s`.' % (est.__class__, method_name, method_name))
        return False
    else:
        raise NotImplementedError('Must have missed a logical case---bug in this function.')


def make_finite(X):
    """Make the data matrix finite by replacing -infty and infty.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data matrix.

    Returns
    -------
    X : array, shape (n_samples, n_features)
        Data matrix as numpy array after checking and possibly replacing
        -infty and infty with min and max of floating values respectively.

    """
    X = _check_floating(X)
    return np.minimum(np.maximum(X, np.finfo(X.dtype).min), np.finfo(X.dtype).max)


def make_positive(X):
    """Make the data matrix positive by clipping to +epsilon if not positive.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data matrix.

    Returns
    -------
    X : array, shape (n_samples, n_features)
        Data matrix as numpy array after checking and possibly replacing
        non-positive numbers to +epsilon.

    """
    X = _check_floating(X)
    return np.maximum(X, np.finfo(X.dtype).tiny)


def make_interior_probability(X, eps=None):
    """Convert data to probability values in the open interval between 0 and 1.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data matrix.
    eps : float, optional
        Epsilon for clipping, defaults to ``np.info(X.dtype).eps``

    Returns
    -------
    X : array, shape (n_samples, n_features)
        Data matrix after possible modification.

    """
    X = _check_floating(X)
    if eps is None:
        eps = np.finfo(X.dtype).eps
    return np.minimum(np.maximum(X, eps), 1-eps)


def make_interior(X, bounds, eps=None):
    """Scale/shift data to fit in the open interval given by `bounds`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data matrix.
    bounds : array-like, shape (2,)
        Minimum and maximum of bounds.
    eps : float, optional
        Epsilon for clipping, defaults to ``np.info(X.dtype).eps``

    Returns
    -------
    X : array, shape (n_samples, n_features)
        Data matrix after possible modification.

    """
    X = _check_floating(X)
    if eps is None:
        eps = np.finfo(X.dtype).eps
    left = bounds[0] + np.abs(bounds[0] * eps)
    right = bounds[1] - np.abs(bounds[1] * eps)
    return np.minimum(np.maximum(X, left), right)


def _check_floating(X):
    if not np.issubdtype(X.dtype, np.floating):
        X = np.array(X, dtype=np.float)
    return X

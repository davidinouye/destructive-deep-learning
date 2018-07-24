from __future__ import print_function
from __future__ import division
import warnings
import logging
import itertools

import numpy as np
from sklearn.exceptions import DataConversionWarning

_INF_SPACE = np.array([-np.inf, np.inf])
_UNIT_SPACE = np.array([0, 1])
_DEFAULT_DOMAIN = _INF_SPACE
_DEFAULT_SUPPORT = _INF_SPACE

logger = logging.getLogger(__name__)


def get_support_or_default(dens, warn=False):
    """Get the support of the density or return `DEFAULT_SUPPORT`."""
    if has_method(dens, 'get_support', warn=False):
        return dens.get_support()
    else:
        if warn:
            msg = ('Support is assumed to be %s since '
                   'dens.get_support() is not implemented.'
                   % str(_DEFAULT_SUPPORT))
            warnings.warn(msg)
        return _DEFAULT_SUPPORT


def get_domain_or_default(trans, warn=False):
    """Get the domain of the destructor or return `DEFAULT_DOMAIN`."""
    if has_method(trans, 'get_domain', warn=False):
        return trans.get_domain()
    else:
        if warn:
            msg = ('Domain is assumed to be %s since '
                   'trans.get_domain() is not implemented.'
                   % str(_DEFAULT_DOMAIN))
            warnings.warn(msg)
        return _DEFAULT_DOMAIN


def check_domain(domain, n_features):
    """Utility that returns domain after expanding to the specified number of dimensions if
    necessary.

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
    """Utility function to quickly check if the input X lies in the specified domain."""
    msg_suffix = ('Thus, the original values will be shifted and scaled to the given domain: '
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
            if high == low:  # Constant values
                # Use halfway between high and low of domain
                X[:, i] = (high_domain+low_domain)/2.0
            else:
                # Scale and shift as necessary into the proper domain
                u = (X[:, i] - low)/(high-low)
                X[:, i] = (high_domain - low_domain)*u + low_domain
    return X


def check_X_in_interval_decorator(func):
    """Decorator utility for destructors to check domain."""

    def wrapper(trans, X, *args, **kwargs):
        X = check_X_in_interval(X, get_domain_or_default(trans))
        return func(trans, X, *args, **kwargs)

    return wrapper


def has_method(est, method_name, warn=True):
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
    X = _check_floating(X)
    return np.minimum(np.maximum(X, np.finfo(X.dtype).min), np.finfo(X.dtype).max)


def make_positive(X):
    X = _check_floating(X)
    return np.maximum(X, np.finfo(X.dtype).tiny)


def make_interior_probability(X, eps=None):
    X = _check_floating(X)
    if eps is None:
        eps = np.finfo(X.dtype).eps
    return np.minimum(np.maximum(X, eps), 1-eps)


def _check_floating(X):
    if not np.issubdtype(X.dtype, np.floating):
        X = np.array(X, dtype=np.float)
    return X

"""Module for very simple autoregressive destructors."""
from __future__ import division, print_function

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array

from .base import BaseDensityDestructor
from .gaussian import GaussianDensity
# noinspection PyProtectedMember
from .utils import (_UNIT_SPACE, check_X_in_interval, get_domain_or_default,
                    make_interior_probability)


class AutoregressiveDestructor(BaseDensityDestructor):
    """Autoregressive destructor using densities that can compute conditionals.

    The density estimator should implement the method :func:`conditional_densities` that will return conditional densities,
    :func:`marginal_cdf` that will return the marginal cdf for a particular feature index, and :func:`marginal_inverse_cdf` that will
    return the marginal inverse cdf for a particular feature index. For an example of this type of density, see :class:`ddl.gaussian.GaussianDensity`
    or :class:`ddl.mixture.GaussianMixtureDensity`.

    Note that this interface has not been fully standardized yet and is likely to change in the future.

    Parameters
    ----------
    density_estimator : estimator
        Density estimator to be used for the autoregressive destructor. Note that this estimator must implement
        :func:`conditional_densities`, :func:`marginal_cdf`, and :func:`marginal_inverse_cdf`.

    order : {None, 'random', array-like with shape (n_features,)}, default=None
        If None, then simply choose the original index order. If 'random',
        then use the random number generator defined by the `random_state`
        parameter to generate a random permutation of feature indices. If an
        array-like is given, then use this as the order of features to
        regress.

    random_state : int, RandomState instance or None, optional (default=None)
        Used to determine random feature order if order is not given.

        If int, `random_state` is the seed used by the random number
        generator; If :class:`~numpy.random.RandomState` instance,
        `random_state` is the random number generator; If None, the random
        number generator is the :class:`~numpy.random.RandomState` instance
        used by :mod:`numpy.random`.

    """

    def __init__(self, density_estimator=None, order=None, random_state=None):
        self.density_estimator = density_estimator
        self.order = order
        self.random_state = random_state

    def _get_density_estimator(self):
        """Get density estimator.

        Returns
        -------
        density : estimator

        """
        if self.density_estimator is None:
            return GaussianDensity()
        else:
            return clone(self.density_estimator)

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
        return self._autoregress(X, y, inverse=False)

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
        return self._autoregress(X, y, inverse=True)

    def _autoregress(self, X, y=None, inverse=False):
        X = check_array(X, copy=True)
        try:
            self._check_is_fitted()
        except NotFittedError:
            pass
        else:
            n_features = self._get_n_features()
            if X.shape[1] != n_features:
                raise ValueError('Incorrect number of dimensions.')
        if inverse:
            X = check_X_in_interval(X, _UNIT_SPACE)
            X = make_interior_probability(X)
        else:
            X = check_X_in_interval(X, get_domain_or_default(self))

        order = self._get_order_or_default(X.shape[1])
        Z = np.zeros(X.shape)
        # Could be parallel for non-inverse
        for i in range(len(order)):
            target_idx = order[i]
            cond_idx_arr = order[:i]
            not_cond_idx_arr = order[i:]
            # Target index is always 0 *after* conditioning because of the construction of
            # cond_idx_arr and not_cond_idx_arr
            cond_target_idx = 0

            # Get conditional densities
            if inverse:
                A = Z
            else:
                A = X
            conditionals = self.density_.conditional_densities(
                A, cond_idx_arr, not_cond_idx_arr)

            if not hasattr(conditionals, '__len__'):
                # Handle case where conditional are all the same (i.e. independent dimensions)
                z = (conditionals.marginal_inverse_cdf(X[:, target_idx], cond_target_idx)
                     if inverse else conditionals.marginal_cdf(X[:, target_idx], cond_target_idx))
            else:
                # Handle dependent conditional_densities
                z = np.array([
                    (cond.marginal_inverse_cdf(x, cond_target_idx)
                     if inverse else cond.marginal_cdf(x, cond_target_idx))
                    for x, cond in zip(X[:, target_idx], conditionals)
                ])
            Z[:, i] = z

        if not inverse:
            # Clean up probabilities from numerical errors
            Z = np.minimum(Z, 1)
            Z = np.maximum(Z, 0)
        return Z

    def _get_order_or_default(self, n_features):
        if self.order is None:
            return np.array(list(range(n_features)))
        elif self.order == 'random':
            rng = check_random_state(self.random_state)
            return rng.permutation(n_features)
        elif len(self.order) == n_features:
            return np.array(self.order)
        else:
            raise ValueError('`order` should be either None, \'random\', or something that has '
                             'length = n_features')

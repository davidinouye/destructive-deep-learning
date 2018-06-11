from __future__ import print_function
from __future__ import division
import numbers
import logging
import warnings
import collections
from itertools import islice, cycle

import numpy as np
import scipy.stats
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import KFold, check_cv
from sklearn.exceptions import NotFittedError

from .base import DestructorMixin, ScoreMixin, get_implicit_density, get_n_dim
from .independent import IndependentDestructor, IndependentInverseCdf
from .utils import make_interior_probability
# noinspection PyProtectedMember
from .utils import _DEFAULT_SUPPORT

logger = logging.getLogger(__name__)


def _take(iterable, n):
    """Return first n items of the iterable as a list"""
    return list(islice(iterable, n))


def _consume(iterator, n):
    """Advance the iterator n-steps ahead. If n is none, consume entirely."""
    # Use functions that consume iterators at C speed.
    if n is None:
        collections.deque(iterator, maxlen=0)
    else:
        next(islice(iterator, n, n), None)


class CompositeDestructor(BaseEstimator, DestructorMixin):
    """Joins multiple destructors (or relative destructors like
    LinearProjector) into a composite destructor.
    """

    def __init__(self, destructors=None, random_state=None):
        """`random_state` is needed if any of the atomic destructors are random-based.
        By seeding the global np.random via random_state and then resetting to its previous
        state, we can avoid having to carefully pass around random_states for random atomic
        destructors.
        """
        self.destructors = destructors
        self.random_state = random_state

    def fit(self, X, y=None, **fit_params):
        self.fit_transform(X, y, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        rng = check_random_state(self.random_state)
        # Save old random state and seed via internal random number generator
        # saved_random_state = np.random.get_state()
        # np.random.seed(rng.randint(2 ** 32, dtype=np.uint32))

        Z = check_array(X, copy=True)

        # Fit and transform all destructors
        destructors = []
        for d in self._get_destructor_iterable():
            Z = self._single_fit_transform(d, Z, y)
            destructors.append(d)
            if np.any(np.isnan(Z)):
                raise RuntimeError('Need to check')

        self.fitted_destructors_ = np.array(destructors)
        self.density_ = get_implicit_density(self)

        # Reset random state
        # np.random.set_state(saved_random_state)
        return Z

    def _single_fit_transform(self, d, Z, y):
        if y is not None:
            warnings.warn('y is not None but this is not an adversarial composite/deep destructor. '
                          'Did you mean to use an adversarial version of this destructor?')
        return d.fit_transform(Z, y)

    def transform(self, X, y=None, partial_idx=None):
        self._check_is_fitted()
        Z = check_array(X, copy=True)

        fitted_destructors = self._get_partial_destructors(partial_idx)
        for d in fitted_destructors:
            Z = d.transform(Z, y)
        return Z

    def inverse_transform(self, X, y=None, partial_idx=None):
        self._check_is_fitted()
        Z = check_array(X, copy=True)

        fitted_destructors = self._get_partial_destructors(partial_idx)
        for d in reversed(fitted_destructors):
            Z = d.inverse_transform(Z, y)
        return Z

    def sample(self, n_samples=1, random_state=None):
        """Nearly the same as `DestructorMixin.sample` but n_dim is found from first 
        fitted destructor to avoid recursion.
        """
        self._check_is_fitted()
        rng = check_random_state(random_state)
        n_dim = get_n_dim(self.fitted_destructors_[-1])
        U = rng.rand(n_samples, n_dim)
        X = self.inverse_transform(U)
        return X

    def score_samples(self, X, y=None, partial_idx=None):
        return np.sum(self.score_samples_layers(X, y, partial_idx=partial_idx), axis=1)

    def score_samples_layers(self, X, y=None, partial_idx=None):
        self._check_is_fitted()
        X = check_array(X, copy=True)

        fitted_destructors = self._get_partial_destructors(partial_idx)
        log_likelihood_layers = np.zeros((X.shape[0], len(fitted_destructors) ))
        for i, d in enumerate(fitted_destructors):
            log_likelihood_layers[:, i]= d.score_samples(X)
            # Don't transform for the last destructor
            if i < len(fitted_destructors) - 1:
                X = d.transform(X, y)
        return log_likelihood_layers

    def score(self, X, y=None, partial_idx=None):
        """Overrides super class to allow for partial_idx"""
        return np.mean(self.score_samples(X, y, partial_idx=partial_idx))

    def score_layers(self, X, y=None, partial_idx=None):
        """Overrides super class to allow for partial_idx"""
        return np.mean(self.score_samples_layers(X, y, partial_idx=partial_idx), axis=0)

    def get_domain(self):
        # Get the domain of the first destructor (or relative destructor like LinearProjector)
        return next(iter(self._get_destructor_iterable())).get_domain()

    def _get_partial_destructors(self, partial_idx):
        if partial_idx is not None:
            return np.array(self.fitted_destructors_)[partial_idx]
        else:
            return self.fitted_destructors_

    def _get_destructor_iterable(self):
        if self.destructors is None:
            return [IndependentDestructor()]
        elif isinstance(self.destructors, (list, tuple, np.array)):
            return [clone(d) for d in self.destructors]
        else:
            raise ValueError('`destructors` must be a list, tuple or numpy array. Sets are not '
                             'allowed because order is important and general iterators/generators '
                             'are not allowed because we need the estimator parameters to stay '
                             'constant after inspecting.')

    def _check_is_fitted(self):
        check_is_fitted(self, ['fitted_destructors_'])


class DeepDestructor(CompositeDestructor):
    """Add destructors/layers until the validation log-likelihood stops going down."""

    # noinspection PyMissingConstructor
    def __init__(self, canonical_destructors=None, init_destructor=None,
                 n_canonical_destructors=1, canonical_destructor=None, random_state=None):
        self.canonical_destructors = canonical_destructors
        self.init_destructor = init_destructor
        self.n_canonical_destructors = n_canonical_destructors
        self.random_state = random_state

        # Deprecated
        self.canonical_destructor = canonical_destructor

    def _get_canonical_destructors(self):
        if self.canonical_destructor is not None and self.canonical_destructors is not None:
            raise ValueError('`canonical_destructor` (Deprecated) and `canonical_destructors` '
                             'cannot be usedat the same time')
        elif self.canonical_destructor is not None:
            warnings.warn(DeprecationWarning('"canonical_destructor" is deprecated. Please use '
                                             '"canonical_destructors" (notice "s" at end) instead'))
            canonical_destructors = self.canonical_destructor
        elif self.canonical_destructors is not None:
            canonical_destructors = self.canonical_destructors
        else:
            canonical_destructors = CompositeDestructor(
                destructors=[IndependentInverseCdf(), IndependentDestructor()]
            )
        # If single, then update to list
        if len(np.array(canonical_destructors).shape) < 1:
            canonical_destructors = [canonical_destructors]
        return canonical_destructors

    def _get_destructor_iterable(self):
        destructors = []
        if self.init_destructor is not None:
            destructors.append(self.init_destructor)
        destructors.extend(_take(cycle(self._get_canonical_destructors()),
                                self.n_canonical_destructors))
        return np.array([clone(d) for d in destructors])


class DeepCVMixin(object):
    def fit(self, X, y=None, X_test=None, **fit_params):
        # Save old random state and seed via internal random number generator
        rng = check_random_state(self.random_state)
        if self.n_extend < 1:
            raise ValueError('n_extend should be greater than or equal to 1') 
        # saved_random_state = np.random.get_state()
        # np.random.seed(rng.randint(2 ** 32, dtype=np.uint32))

        # Setup parameters
        X = check_array(X)
        cv = check_cv(self.cv)
        splits = list(cv.split(X))
        # Could split based on y but since y is just an ancillary
        #  variable, just split based on X (ignore below)
        # if y is not None:
        #     cv = check_cv(self.cv, y=y, classifier=True)
        #     try:
        #        splits = list(cv.split(X, y))
        #    except ValueError:
        #        # If this fails because too few samples
        #        cv = check_cv(self.cv)
        #        splits = list(cv.split(X))
        # else:
        #    cv = check_cv(self.cv)
        #    splits = list(cv.split(X))

        # CV path fit and transform
        cv_destructors_arr = [[] for _ in splits]
        scores_arr = [[] for _ in splits]
        cv_destructors_arr, scores_arr, splits = self._fit_cv_destructors(
            X, cv_destructors_arr, scores_arr, splits, X_test=X_test,
            max_layers=self.n_canonical_destructors)

        # Add layers as needed up to max # of layers of all splits
        if not self.silent:
            logger.debug(self.log_prefix + 'Fitting extra needed layers')
        best_n_layers_over_folds = np.max([
            len(cv_destructors) for cv_destructors in cv_destructors_arr])
        cv_destructors_arr, scores_arr, splits = self._fit_cv_destructors(
            X, cv_destructors_arr, scores_arr, splits, X_test=X_test,
            max_layers=best_n_layers_over_folds)

        # Determine best number of layers
        scores_mat = np.array(scores_arr)
        if np.any(scores_mat.shape != np.array([len(splits), best_n_layers_over_folds, 2])):
            raise RuntimeError('scores_mat does not seem to be the correct shape')
        scores_avg = np.mean(scores_mat, axis=0)  # Average over different splits
        best_n_layers = int(1 + np.argmax(scores_avg[:, 1].ravel()))  # Best over cumulative test_score
        best_score = np.max(scores_avg[:, 1].ravel())
        # logger.debug(self.log_prefix + '\n%s' % str(scores_mat))
        # logger.debug(self.log_prefix + '\n%s' % str(scores_avg))

        # Final fitting with best # of layers
        if self.refit:
            if not self.silent:
                logger.debug(self.log_prefix + 'Fitting final model with %d layers with score=%g'
                            % (best_n_layers, best_score))
            destructors = []
            Z = X.copy()
            for i, d in enumerate(islice(iter(self._get_destructor_iterable()), best_n_layers)):
                d.fit(Z)
                score = d.score(Z)
                Z = d.transform(Z)
                destructors.append(d)
                if not self.silent:
                    logger.debug(self.log_prefix + '(Final fit layer=%d) local layer score=%g' % (i + 1, score))
        else:
            # Use already fitted destructor from CV array
            if len(cv_destructors_arr) > 1:
                warnings.warn('refit=False but len(cv_destructors_arr) > 1 so just using fitted '
                              'destructors from the first split.')
            destructors = np.array(cv_destructors_arr[0])[:best_n_layers]

        self.fitted_destructors_ = np.array(destructors)
        self.density_ = get_implicit_density(self)
        self.cv_train_scores_ = scores_mat[:, :, 0].transpose()
        self.cv_test_scores_ = scores_mat[:, :, 1].transpose()
        self.best_n_layers_ = best_n_layers

        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, y)

    def _fit_cv_destructors(self, X, cv_destructors_arr, scores_arr, splits, X_test=None,
                            max_layers=None):
        compute_test = X_test is not None
        for i, (cv_destructors, scores, (train, validation)) in enumerate(
                zip(cv_destructors_arr, scores_arr, splits)):

            Z_train = X[train, :].copy()
            Z_validation = X[validation, :].copy()
            if compute_test:
                Z_test = X_test.copy()
            else:
                Z_test = None

            # If some destructors are already fit
            destructor_iterator = iter(self._get_destructor_iterable())
            if max_layers is not None and len(cv_destructors) == max_layers:
                # Don't need to fit any more destructors
                if not self.silent:
                    logger.debug(self.log_prefix + 'Already done fitting cv=%i deep destructor' % i)
                continue
            elif len(cv_destructors) > 0:
                if not self.silent:
                    logger.debug(self.log_prefix + 'Re-fitting extra destructors for cv=%i deep destructor' % i)
                # Pop off destructors that were already fit from the destructor iterator
                _consume(destructor_iterator, len(cv_destructors))
                for d in cv_destructors:
                    Z_train = d.transform(Z_train)
                    Z_validation = d.transform(Z_validation)
                    if compute_test:
                        Z_test = d.transform(Z_test)

            stop = False
            cum_test_score = 0
            while not stop:  # Add layers until all are ready to stop
                # Fit only on training data
                destructor = next(destructor_iterator)
                destructor.fit(Z_train)

                # Score and then transform data
                train_score = destructor.score(Z_train)
                validation_score = destructor.score(Z_validation)
                Z_train = destructor.transform(Z_train)
                Z_validation = destructor.transform(Z_validation)
                if compute_test:
                    test_score = destructor.score(Z_test)
                    Z_test = destructor.transform(Z_test)
                    cum_test_score += test_score
                    test_score_str = ' test=%g, cum_test=%g' % (test_score, cum_test_score)
                else:
                    test_score_str = ''
                if not self.silent:
                    logger.debug(self.log_prefix + '(CV sp=%d, L=%d) Scores: train=%g val=%g%s'
                                 % (i + 1, len(cv_destructors) + 1, train_score, validation_score,
                                    test_score_str))

                # Update cv_destructors
                cv_destructors.append(destructor)

                # Maintain cumulative scores
                previous_scores = scores[-1] if len(scores) > 0 else np.array([0, 0])
                cum_scores = previous_scores + np.array([train_score, validation_score])
                scores.append(cum_scores)

                # Keep going if need to fit a certain number of layers no matter what
                if max_layers is not None:
                    if len(cv_destructors) == max_layers:
                        stop = True
                else:
                    # If we have n_extend + 1 layers then check cumulative scores
                    if len(cv_destructors) > self.n_extend:
                        cur_score = scores[-1][1]
                        max_previous_scores = np.max([sc[1] for sc in scores[:-self.n_extend]])
                        if max_previous_scores == 0:
                            rel_diff = cur_score - max_previous_scores
                        else:
                            rel_diff = (cur_score - max_previous_scores)/np.abs(max_previous_scores)
                        if not self.silent:
                            logger.debug(self.log_prefix + '(CV sp=%d, L=%d) Relative diff=%g'
                                         % (i + 1, len(cv_destructors), rel_diff))
                        if rel_diff < self.stop_tol:
                            # If the most recent cumulative score is less than the max score of much
                            # everything except the last n_extend layers, then stop
                            stop = True

        return cv_destructors_arr, scores_arr, splits

    def _get_destructor_iterable(self):
        """Yield an infinite sequence of destructors."""
        def _destructor_generator():
            if self.init_destructor is not None:
                yield clone(self.init_destructor)
            for d in cycle(self._get_canonical_destructors()):
                yield clone(d)
        return _destructor_generator()


class DeepDestructorCV(DeepCVMixin, DeepDestructor):
    # noinspection PyMissingConstructor
    def __init__(self, canonical_destructors=None, init_destructor=None, cv=None, stop_tol=1e-3,
                 n_canonical_destructors=None, n_extend=1, refit=True, silent=False, log_prefix='',
                 random_state=None, canonical_destructor=None):
        self.canonical_destructors = canonical_destructors
        self.init_destructor = init_destructor
        self.cv = cv
        self.stop_tol = stop_tol
        self.n_canonical_destructors = n_canonical_destructors
        self.n_extend = n_extend
        self.silent = silent
        self.log_prefix = log_prefix
        self.refit = refit
        self.random_state = random_state

        # Deprecated
        self.canonical_destructor = canonical_destructor


class NoisyDeepMixin(object):
    """Adds a small amount of Gaussian noise between each stage.
    Children must have `noise_std` and `random_state` property.
    """

    def fit_transform(self, X, y=None, **fit_params):
        # Save old random state and seed via internal random number generator
        rng = check_random_state(self.random_state)
        # saved_random_state = np.random.get_state()
        # np.random.seed(rng.randint(2 ** 32, dtype=np.uint32))

        destructors = [clone(d) for d in self._get_destructors_or_default()]
        Z = check_array(X, copy=True)

        # Fit and transform all destructors
        Z_noisy = Z.copy()
        for d in destructors:
            d.fit_transform(Z_noisy, y)
            Z = d.transform(Z, y)

            Z_noisy = Z.copy()
            if self.noise_std != 0:
                # Add Gaussian noise
                Z_noisy = make_interior_probability(Z_noisy)
                Z_noisy = scipy.stats.norm.ppf(Z_noisy, loc=0, scale=1)
                Z_noisy += scipy.stats.norm.rvs(size=Z_noisy.shape,
                                                random_state=rng) * self.noise_std
                # The new standard deviation if convolving a standard normal with a normal of
                # mean 0 and var = self.noise_std**2. Note if this is already standard Gaussian,
                # then this will not affect the distribution
                new_std = np.sqrt(1 ** 2 + self.noise_std ** 2)
                Z_noisy = scipy.stats.norm.cdf(Z_noisy, loc=0, scale=new_std)
                Z_noisy = make_interior_probability(Z_noisy)

            if np.any(Z_noisy > 1) or np.any(Z_noisy < 0):
                raise RuntimeError(Z_noisy)

            if np.any(np.isnan(Z_noisy)):
                raise RuntimeError('Need to check')

        self.fitted_destructors_ = np.array(destructors)
        self.density_ = get_implicit_density(self)

        # Reset random state
        # np.random.set_state(saved_random_state)
        return Z


class NoisyDeepDestructor(NoisyDeepMixin, DeepDestructor):
    def __init__(self, canonical_destructor=None, init_destructor=None, n_canonical_destructors=1,
                 noise_std=0.1, random_state=None):
        super(NoisyDeepDestructor, self).__init__(
            canonical_destructor=canonical_destructor,
            init_destructor=init_destructor,
            n_canonical_destructors=n_canonical_destructors,
        )
        self.noise_std = noise_std
        self.random_state = random_state


class UnboundedDeepDestructor(DeepDestructor):
    """Sets up a Gaussian deep destructor (one which works in the real-valued density space).
    Essentially this just adds an IndependentInverseCdf() relative destructor for the canonical
    destructor.
    """

    # noinspection PyMissingConstructor
    def __init__(self, unbounded_destructor=None, pre_destructor=None, n_canonical_destructors=1,
                 random_state=None):
        self.unbounded_destructor = unbounded_destructor
        self.pre_destructor = pre_destructor
        self.n_canonical_destructors = n_canonical_destructors
        self.random_state = random_state

        # Setup not None parameters for super class
        self.init_destructor = self._get_init_destructor()
        self.canonical_destructors = self._get_canonical_destructors()

    def _get_unbounded_destructor(self):
        # Define default unbounded destructor
        if self.unbounded_destructor is None:
            return IndependentDestructor()
        else:
            return self.unbounded_destructor

    def _get_canonical_destructors(self):
        # Add inverse cdf to unbounded destructor
        return [CompositeDestructor(
            destructors=[
                IndependentInverseCdf(),
                self._get_unbounded_destructor(),
            ],
        )]

    def _get_init_destructor(self):
        unbounded_destructor = self._get_unbounded_destructor()
        # Setup init_destructor
        if self.pre_destructor is not None:
            init_destructor = CompositeDestructor(
                destructors=[
                    self.pre_destructor,
                    IndependentInverseCdf(),
                    unbounded_destructor,
                ]
            )
        else:
            init_destructor = unbounded_destructor

        return init_destructor


class UnboundedDeepDestructorCV(DeepCVMixin, UnboundedDeepDestructor):
    # noinspection PyMissingConstructor
    def __init__(self, unbounded_destructor=None, pre_destructor=None, cv=None, stop_tol=0,
                 n_canonical_destructors=None, n_extend=1, random_state=None):
        self.unbounded_destructor = unbounded_destructor
        self.pre_destructor = pre_destructor
        self.cv = cv
        self.stop_tol = stop_tol
        self.n_canonical_destructors = n_canonical_destructors
        self.n_extend = n_extend
        self.random_state = random_state

        # Setup not None parameters for super class
        self.init_destructor = self._get_init_destructor()
        self.canonical_destructors = self._get_canonical_destructors()


class NoisyUnboundedDeepDestructor(NoisyDeepMixin, UnboundedDeepDestructor):
    def __init__(self, unbounded_destructor=None, pre_destructor=None, n_canonical_destructors=1,
                 noise_std=0.1, random_state=None):
        super(NoisyUnboundedDeepDestructor, self).__init__(
            unbounded_destructor=unbounded_destructor,
            pre_destructor=pre_destructor,
            n_canonical_destructors=n_canonical_destructors,
        )
        self.noise_std = noise_std
        self.random_state = random_state

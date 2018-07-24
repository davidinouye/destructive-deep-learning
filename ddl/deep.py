from __future__ import division, print_function

import collections
import logging
import warnings
from itertools import cycle, islice

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import check_cv
from sklearn.utils.validation import check_array

from .base import CompositeDestructor, IdentityDestructor, get_implicit_density

# noinspection PyProtectedMember

logger = logging.getLogger(__name__)


class DeepDestructor(CompositeDestructor):
    """Add destructors/layers until the validation log-likelihood stops going down."""

    # noinspection PyMissingConstructor
    def __init__(self, canonical_destructor=None, init_destructor=None,
                 n_canonical_destructors=1, random_state=None):
        """Parameter `canonical_destructor` can be a list of canonical destructors.
        The list will be cycled through to get as many canonical destructors as needed."""
        self.canonical_destructor = canonical_destructor
        self.init_destructor = init_destructor
        self.n_canonical_destructors = n_canonical_destructors
        self.random_state = random_state

    def _get_canonical_destructors(self):
        """Get canonical destructors as list and if only a single one then wrap in a list."""
        if self.canonical_destructor is not None:
            canonical_destructors = self.canonical_destructor
        else:
            canonical_destructors = IdentityDestructor()

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
        if self.n_extend < 1:
            raise ValueError('n_extend should be greater than or equal to 1')

        # Save old random state and seed via internal random number generator
        # saved_random_state = np.random.get_state()
        # np.random.seed(self.random_state)

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
        best_n_layers = int(
            1 + np.argmax(scores_avg[:, 1].ravel()))  # Best over cumulative test_score
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
                    logger.debug(self.log_prefix + '(Final fit layer=%d) local layer score=%g'
                                 % (i + 1, score))
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

        # Reset random state
        # np.random.set_state(saved_random_state)
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
                    logger.debug(self.log_prefix
                                 + 'Re-fitting extra destructors for cv=%i deep destructor' % i)
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
                            rel_diff = (cur_score - max_previous_scores) / np.abs(
                                max_previous_scores)
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
    def __init__(self, canonical_destructor=None, init_destructor=None, cv=None, stop_tol=1e-3,
                 n_canonical_destructors=None, n_extend=1, refit=True, silent=False, log_prefix='',
                 random_state=None):
        """Parameter `canonical_destructor` can be a list of canonical destructors.
        The list will be cycled through to get as many canonical destructors as needed."""
        self.canonical_destructor = canonical_destructor
        self.init_destructor = init_destructor
        self.cv = cv
        self.stop_tol = stop_tol
        self.n_canonical_destructors = n_canonical_destructors
        self.n_extend = n_extend
        self.silent = silent
        self.log_prefix = log_prefix
        self.refit = refit
        self.random_state = random_state


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

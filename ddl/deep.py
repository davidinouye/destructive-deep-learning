"""Deep destructors module."""
from __future__ import division, print_function

import collections
import logging
import warnings
from itertools import cycle, islice

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import check_cv
from sklearn.utils.validation import check_array

# noinspection PyProtectedMember
from .base import (CompositeDestructor, IdentityDestructor, _check_global_random_state,
                   get_implicit_density)

logger = logging.getLogger(__name__)


class DeepDestructor(CompositeDestructor):
    """Destructor formed by composing copies of some atomic destructor.

    This destructor creates a dynamic composite destructor that includes an
    optional initial destructor (parameter `init_destructor`) followed by
    multiple copies of a canonical destructor (parameter
    `canonical_destructor`). The `init_destructor` is often used for
    preprocessing steps such as standardization.

    If the training data's domain/support is not the unit hypercube,
    an initial destructor is required---this initial destructor should have
    a domain that matches the training data (by the definition of a
    destructor, the range of the destructor is the unit hypercube and thus
    the initial destructor will project the data onto the canonical domain.

    This is a relatively thin wrapper around
    :class:`~ddl.base.CompositeDestructor` that creates copies of the
    canonical destructor to create a deep composite destructor with
    destuctors (or "layers") that are similar in structure because they have
    the same hyperparameters.

    See Also
    --------
    DeepDestructorCV
        A deep destructor whose number of destructors/layers is chosen
        automatically based on cross-validation test likelihood.

    ddl.base.CompositeDestructor

    Parameters
    ----------
    canonical_destructor : estimator or list
        The canonical destructor(s) that will be cloned to build up a deep
        destructor. Parameter `canonical_destructor` can be a list of
        canonical destructors. The list will be cycled through to get as
        many canonical destructors as needed.

    init_destructor : estimator, optional
        Initial destructor (e.g. preprocessing or just to project to
        canonical domain).

    n_canonical_destructors : int, defaults to 1
        Number of cloned canonical destructors to add to the deep
        destructor.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, `random_state` is the seed used by the random number
        generator; If :class:`~numpy.random.RandomState` instance,
        `random_state` is the random number generator; If None, the random
        number generator is the :class:`~numpy.random.RandomState` instance
        used by `np.random`.

    Attributes
    ----------
    fitted_destructors_ : list
        List of fitted (sub)destructors. See `fitted_destructors_` of
        :class:`~ddl.base.CompositeDestructor`.

    density_ : estimator
        *Implicit* density of deep destructor.

    """

    # noinspection PyMissingConstructor
    def __init__(self, canonical_destructor=None, init_destructor=None,
                 n_canonical_destructors=1, random_state=None):

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


class DeepDestructorCV(DeepDestructor):
    """Deep destructor whose number of destructors/layers is determined by CV.

    Nearly the same as `DeepDestructor` except that the number of
    canonical destructors (i.e. the number of layers) is automatically
    determined using cross validation. The likelihood of held-out data in
    each CV fold is used to determine the number of parameters.

    This destructor is computationally more efficient than using
    `sklearn.model_selection.GridSearchCV` because the deep destructor can
    be built one layer at a time and the test likelihood can be accumulated
    one layer at a time.

    See Also
    --------
    DeepDestructor

    Parameters
    ----------

    canonical_destructor : estimator or list
        The canonical destructor(s) that will be cloned to build up a deep
        destructor. Parameter `canonical_destructor` can be a list of
        canonical destructors. The list will be cycled through to get as
        many canonical destructors as needed.

    init_destructor : estimator, optional
        Initial destructor (e.g. preprocessing or just to project to
        canonical domain).

    cv :

    stop_tol :

    max_canonical_destructors : int or None, defaults to None
        The maximum number of destructors (including the initial destructor)
        to add to the deep destructor. If set to None, then the number of
        destructors is unbounded.

    n_extend : int, defaults to 1
        The number of destructors/layers to extend even after the stopping
        tolerance defined by `stop_tol` has been reached. This could be
        useful if the destructors are random or not gauranteed to always
        increase likelihood. If `n_extend` is 1, then the optimization will
        stop as soon as the test log likelihood decreases.

    refit : bool, defaults to False
        Whether to refit the entire deep destructor with the selected number
        of layers or just extract the fit from the first fold.

    silent : bool, defaults to False
        Whether to output debug messages via :class:`logging.logger`. Note that
        logging messages are not output to standard out automatically.  Please
        see the Python module :mod:`logging` for more information.

    log_prefix : str, defaults to ''
        Prefix of debug logging messages via :class:`logging.logger`. See
        `silent` parameter.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    fitted_destructors_ : array, shape = [n_layers]
        Array of fitted destructors. See `fitted_destructors_` of
        `base.CompositeDestructor`.

    density_ : estimator
        *Implicit* density of deep destructor.

    cv_train_scores_ : array, shape = [n_layers, n_splits]
        Cross validation train scores (mean log-likelihood).

    cv_test_scores_ : array, shape = [n_layers, n_splits]
        Cross validation test scores (mean log-likelihood).

    best_n_layers_ : int
        Best number of layers as selected by cross validation.

    """
    # noinspection PyMissingConstructor
    def __init__(self, canonical_destructor=None, init_destructor=None, cv=None, stop_tol=1e-3,
                 max_canonical_destructors=None, n_extend=1, refit=True, silent=False,
                 log_prefix='', random_state=None):
        self.canonical_destructor = canonical_destructor
        self.init_destructor = init_destructor
        self.cv = cv
        self.stop_tol = stop_tol
        self.max_canonical_destructors = max_canonical_destructors
        self.n_extend = n_extend
        self.silent = silent
        self.log_prefix = log_prefix
        self.refit = refit
        self.random_state = random_state

    @_check_global_random_state
    def fit(self, X, y=None, X_test=None, **fit_params):
        # Setup parameters
        if self.n_extend < 1:
            raise ValueError('n_extend should be greater than or equal to 1')
        X = check_array(X)
        cv = check_cv(self.cv)
        splits = list(cv.split(X))

        # CV path fit and transform
        cv_destructors_arr = [[] for _ in splits]
        scores_arr = [[] for _ in splits]
        cv_destructors_arr, scores_arr, splits = self._fit_cv_destructors(
            X, cv_destructors_arr, scores_arr, splits, X_test=X_test)

        # Add layers as needed up to max # of layers of all splits
        if not self.silent:
            logger.debug(self.log_prefix + 'Fitting extra needed layers')
        best_n_layers_over_folds = np.max([
            len(cv_destructors) for cv_destructors in cv_destructors_arr])
        cv_destructors_arr, scores_arr, splits = self._fit_cv_destructors(
            X, cv_destructors_arr, scores_arr, splits, X_test=X_test,
            selected_n_layers=best_n_layers_over_folds)

        # Determine best number of layers
        scores_mat = np.array(scores_arr)
        if np.any(scores_mat.shape != np.array([len(splits), best_n_layers_over_folds, 2])):
            raise RuntimeError('scores_mat does not seem to be the correct shape')
        scores_avg = np.mean(scores_mat, axis=0)  # Average over different splits
        best_n_layers = int(
            1 + np.argmax(scores_avg[:, 1].ravel()))  # Best over cumulative test_score
        best_score = np.max(scores_avg[:, 1].ravel())

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
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, y)

    def _fit_cv_destructors(self, X, cv_destructors_arr, scores_arr, splits, X_test=None,
                            selected_n_layers=None):
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
            if selected_n_layers is not None and len(cv_destructors) == selected_n_layers:
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

                # Stop if global max layers is reached
                if self.max_canonical_destructors is not None:
                    global_max_layers = self.max_canonical_destructors
                    if self.init_destructor is not None:
                        global_max_layers += 1
                    if len(cv_destructors) == global_max_layers:
                        stop = True
                        continue

                if selected_n_layers is not None:
                    if len(cv_destructors) == selected_n_layers:
                        stop = True
                    else:
                        # Keep going if need to fit a certain number of layers no matter what
                        # (i.e. don't check n_extend in this case)
                        pass
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

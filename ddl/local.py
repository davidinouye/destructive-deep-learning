from __future__ import division, print_function

import logging
import warnings

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state

from .base import DestructorMixin
from .independent import IndependentDensity, IndependentDestructor
from .univariate import HistogramUnivariateDensity
# noinspection PyProtectedMember
from .utils import _UNIT_SPACE

logger = logging.getLogger(__name__)


class FeatureGroupsDestructor(BaseEstimator, DestructorMixin):
    def __init__(self, groups_estimator=None, group_canonical_destructor=None, n_jobs=1):
        self.groups_estimator = groups_estimator
        self.group_canonical_destructor = group_canonical_destructor
        self.n_jobs = n_jobs

    def fit(self, X, y=None, **fit_params):
        self.fit_transform(X, y, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        # Validate parameters
        groups_estimator = (
            clone(self.groups_estimator) if self.groups_estimator is not None
            else RandomFeaturePairs(random_state=0)
        )
        group_canonical_destructor = (
            clone(self.group_canonical_destructor)
            if self.group_canonical_destructor is not None
            else IndependentDestructor(
                independent_density=IndependentDensity(
                    univariate_estimators=HistogramUnivariateDensity()
                )
            )
        )
        X = check_array(X, copy=True)
        Z = np.asfortranarray(X)  # Convert to column major order for computational reasons

        # Fit and get groups list
        groups_estimator.fit(Z, y)
        groups = groups_estimator.groups_

        # Check that groups has no duplicates
        all_idx = np.array(groups).ravel()
        uniq = np.unique(all_idx)
        if len(uniq) != len(all_idx):
            raise ValueError('There seem to be duplicates in the same round of groups')

        # Fit destructors for each group
        group_destructors = [clone(group_canonical_destructor) for _ in groups]
        Z_groups_and_destructors = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform)(Z[:, group], group_destructor)
            for group, group_destructor in zip(groups, group_destructors)
        )

        # Old code
        # for group, (Z_group, _) in zip(groups, Z_groups_and_destructors):
        #    Z[:, group] = Z_group
        # group_destructors = [d for _, d in Z_groups_and_destructors]

        # Filter out destructors that do not make any changes (i.e. that are identity destructors)
        filtered_results = (
            (group, Z_group, d)
            for group, (Z_group, d) in zip(groups, Z_groups_and_destructors)
            if np.any(Z_group != Z[:, group])
        )
        logger.debug('n_groups before filter = %d' % len(groups))
        groups, Z_groups, group_destructors = (list(a) for a in zip(*filtered_results))
        logger.debug('n_groups after filter = %d' % len(groups))
        # Update Z and group_destructors
        for group, Z_group in zip(groups, Z_groups):
            Z[:, group] = Z_group

        # Save important variables
        self.groups_ = groups
        self.group_destructors_ = group_destructors
        self.n_features_ = X.shape[1]

        Z = np.ascontiguousarray(Z)  # Convert back to standard C-order array
        return Z

    def transform(self, X, y=None):
        self._check_is_fitted()
        X = check_array(X, copy=True)
        Z = np.asfortranarray(X)  # Convert to column-major
        # Group destructors
        Z_groups = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform)(Z[:, group], destructor)
            for group, destructor in zip(self.groups_, self.group_destructors_)
        )
        for group, Z_group in zip(self.groups_, Z_groups):
            Z[:, group] = Z_group
        Z = np.ascontiguousarray(Z)  # Convert back to row-major
        return Z

    def inverse_transform(self, X, y=None):
        self._check_is_fitted()
        X = check_array(X, copy=True)
        Z = np.asfortranarray(X)  # Convert to column-major
        # Group destructors
        Z_groups = Parallel(n_jobs=self.n_jobs)(
            delayed(_inverse_transform)(Z[:, group], destructor)
            for group, destructor in zip(self.groups_, self.group_destructors_)
        )
        for group, Z_group in zip(self.groups_, Z_groups):
            Z[:, group] = Z_group
        Z = np.ascontiguousarray(Z)  # Convert back to row-major
        return Z

    def get_domain(self):
        # We assume canonical destructors
        return _UNIT_SPACE

    def score_samples(self, X, y=None):
        self._check_is_fitted()
        X = check_array(X, copy=True)
        X = np.asfortranarray(X)
        group_score_samples = Parallel(n_jobs=self.n_jobs)(
            delayed(_score_samples)(X[:, group], d)
            for group, d in zip(self.groups_, self.group_destructors_)
        )
        return np.sum(group_score_samples, axis=0)

    def _check_is_fitted(self):
        check_is_fitted(self, ['groups_', 'group_destructors_'])


def _fit_transform(Z_group, d):
    Z_group = d.fit_transform(Z_group)
    return Z_group, d


def _transform(Z_group, d):
    return d.transform(Z_group)


def _inverse_transform(Z_group, d):
    return d.inverse_transform(Z_group)


def _score_samples(X_group, d):
    return d.score_samples(X_group)


class RandomFeaturePairs(BaseEstimator):
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)

        n_features = X.shape[1]
        perm = rng.permutation(n_features)
        if np.mod(n_features, 2) == 1:
            # Remove one pair
            perm = perm[:-1]
            logger.info('Odd number of dimensions so one dimension is not paired')

        self.groups_ = perm.reshape(-1, 2)
        return self


class ImageFeaturePairs(BaseEstimator):
    """Generate pairs of features for FeatureGroupsDestructor based on image layout.

    `image_shape` is the shape such that X[0,:].reshape(image_shape) is converted to an image.
        Note that image_shape could have any length depending on the number of image channels.

    `relative_position` is length len(image_shape) and is a relative relative_position to pair
    with a selected feature. For example, if `relative_position` = (1, 0), then the pixels will
    be paired horizontally whereas if `relative_position` = (0, 1), then the pixels will be
    paired vertically.

    `init_offset` is the amount to init_offset in all directions on the image. For example,
    one might first do a init_offset of (0, 0) and then a init_offset of (1, 0) to couple the all
    horizontal pixels.

    `wrap` means whether to wrap the pixels to the other side so that all features are paired.
    For example, if `relative_position = (1,0)` and `init_offset = (1,0)`, the last pixel on the
    row will match with the first pixel on the row.
    """

    def __init__(self, image_shape=None, relative_position=None, init_offset=None, step=None,
                 wrap=True):
        self.image_shape = image_shape
        self.relative_position = relative_position
        self.init_offset = init_offset
        self.step = step
        self.wrap = wrap

    def fit(self, X, y=None):
        def _check_image_shape(shape, x):
            if shape is None:
                return x.shape
            else:
                try:
                    image_x = x.reshape(shape)
                except ValueError:
                    raise ValueError('Coulc not reshape X[0,:] into image_shape')
                return image_x.shape

        def _check_relative_position(pos, shape):
            if pos is None:
                pos = np.zeros(shape)
                pos[0] = 1
            else:
                return np.array(pos)

        def _check_init_offset(offset, shape):
            if offset is None:
                offset = np.zeros(shape)
            offset = np.array(offset)
            # if np.sum(offset) > 1:
            #    raise ValueError('np.sum(init_offset) should be less than 1.')
            return offset

        def _check_step(_step, shape):
            _step = _check_relative_position(_step, shape)
            if np.all(_step == 0):
                raise ValueError('step should be a non-zero array-like')
            return _step

        # Validate inputs and parameters
        X = check_array(X)
        if X.shape[0] < 1:
            raise ValueError('X must have one row so that the image_shape can be checked.')

        image_shape = _check_image_shape(self.image_shape, X[0, :])

        relative_position = _check_relative_position(self.relative_position, image_shape)
        init_offset = _check_init_offset(self.init_offset, image_shape)
        step = _check_step(self.step, image_shape)

        if (len(image_shape) != len(relative_position)
                or len(image_shape) != len(init_offset)
                or len(image_shape) != len(step)):
            raise ValueError('length of image_shape, relative_position and init_offset should all '
                             'be the same')

        # Setup unpaired features
        n_features = np.prod(image_shape)
        if n_features < 2:
            raise ValueError('n_features < 2 but this means there are no pairs')
        unpaired_features = set(range(n_features))

        def _lin_idx(I):
            return np.ravel_multi_index(I, image_shape)

        def _wrap(I):
            if self.wrap:
                return np.mod(I, image_shape)
            return I

        def _check_I(I):
            # Check that it is within the bounds of the image
            # Wrapping should have already been performed
            if np.any(I / image_shape >= 1):
                return False
            # Check if this feature_idx has already been paired
            linear_idx = _lin_idx(I)
            if linear_idx not in unpaired_features:
                return False
            # Otherwise return true
            return True

        # Only wrap pair if allowed
        pairs = []
        cur_I = init_offset
        pair_I = _wrap(cur_I + relative_position)
        while len(unpaired_features) > 0:
            unpaired_idx = -1  # Whether searching through unpaired_features
            unpaired_arr = None
            found_pair = True
            while not _check_I(cur_I) or not _check_I(pair_I):
                if unpaired_idx == -1:
                    # If invalid pair, then step
                    cur_I += step
                    pair_I = _wrap(cur_I + relative_position)
                    if np.all(cur_I / image_shape < 1):
                        continue
                    else:
                        unpaired_idx = 0
                        unpaired_arr = np.sort(list(unpaired_features))

                # Get I from linear idx
                if unpaired_idx >= len(unpaired_arr):
                    # We have reached the end of the unpaired_idx so break
                    # No valid indices found
                    warnings.warn('Did not pair all features.')
                    found_pair = False
                    break
                cur_I = np.array(np.unravel_index(unpaired_arr[unpaired_idx], image_shape))
                pair_I = _wrap(cur_I + relative_position)
                unpaired_idx += 1

            # Break if did not find a pair
            if not found_pair:
                break

            # Add pair and remove
            new_pair = (_lin_idx(cur_I), _lin_idx(pair_I))
            pairs.append(new_pair)
            unpaired_features.remove(new_pair[0])
            unpaired_features.remove(new_pair[1])

        self.groups_ = pairs
        self.unpaired_features_ = np.sort(list(unpaired_features))
        return self

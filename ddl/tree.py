from __future__ import division, print_function

import logging
import warnings
from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.tree import ExtraTreeRegressor
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted

from .base import (BaseDensityDestructor, BoundaryWarning, IdentityDestructor, ScoreMixin,
                   get_inverse_canonical_destructor)
# noinspection PyProtectedMember
from .utils import _UNIT_SPACE, check_X_in_interval, get_support_or_default

logger = logging.getLogger(__name__)


class TreeDestructor(BaseDensityDestructor):
    def __init__(self, tree_density=None):
        self.tree_density = tree_density

    def get_density_estimator(self):
        if self.tree_density is None:
            return TreeDensity()
        else:
            return clone(self.tree_density)

    def transform(self, X, y=None):
        self._check_is_fitted()
        X = self._check_X(X, copy=True)
        return _tree_transform(self.density_.tree_, X, y)

    def inverse_transform(self, X, y=None):
        self._check_is_fitted()
        X = self._check_X(X, copy=True, inverse=True)
        tree_inverse = _get_inverse_tree(self.density_.tree_)
        return _tree_transform(tree_inverse, X, y)

    def _check_X(self, X, copy=False, inverse=False):
        X = check_array(X, copy=copy)
        # noinspection PyProtectedMember
        self.density_._check_dim(X)
        if inverse:
            domain = _UNIT_SPACE
        else:
            domain = get_support_or_default(self.density_)
        X = check_X_in_interval(X, domain)
        return X


class TreeDensity(BaseEstimator, ScoreMixin):
    def __init__(self, tree_estimator=None, get_tree=None, node_destructor=None,
                 uniform_weight=1e-6):
        self.tree_estimator = tree_estimator
        self.get_tree = get_tree
        self.node_destructor = node_destructor
        self.uniform_weight = uniform_weight

    def fit(self, X, y=None, fitted_tree_estimator=None):
        """Should fit and assign attribute `self.root_node_`."""
        # Should have internal checks for X
        X = self._check_X(X)
        get_tree = _check_get_tree(self.get_tree)

        # Fit estimator if needed
        if fitted_tree_estimator is None:
            fitted_tree_estimator = self._get_tree_estimator()
            fitted_tree_estimator.fit(X, y)

        # Given the tree structure, fit the actual density
        tree = get_tree(fitted_tree_estimator)
        self._fit_tree_density(tree, X, y)

        self.fitted_tree_estimator_ = fitted_tree_estimator
        self.tree_ = tree
        self.n_features_ = X.shape[1]
        return self

    def _get_tree_estimator(self):
        """Supplies default tree, can be overriden in subclasses."""
        return clone(
            self.tree_estimator) if self.tree_estimator is not None else RandomTreeEstimator(
            random_state=0)

    def _fit_tree_density(self, tree, X, y):
        """Fits the probability values for each leaf and each leaf destructor."""
        node_destructor = self.node_destructor

        def _update_stack(child):
            """Add left or right child to stack"""
            sel_new = sel.copy()
            if child == 'left':
                sel_new[sel] = X[sel, node.feature] < node.threshold
            else:
                sel_new[sel] = X[sel, node.feature] >= node.threshold
            stack.append(sel_new)
            return sel_new

        # Initialize
        X = check_array(X, copy=True)
        n_samples, n_features = X.shape
        tree.set_node_destructor(node_destructor)

        # Setup root node
        sel = np.ones(n_samples, dtype=np.bool)
        stack = [sel]
        for i, node in enumerate(tree):  # Must be depth-first and left-then-right traversal
            sel = stack.pop()
            if node.is_leaf():
                # Just simple empirical probability (uniform mixture component added later)
                node.value = np.sum(sel) / n_samples

                # Fit node destructors at leaves
                if node.destructor is not None:
                    node.destructor.fit(_to_unit(X[sel, :], node.domain))
            else:
                # Add children to stack with appropriate selections
                _update_stack('right')
                _update_stack('left')

                # Don't know relative probability yet (must compute below)
                node.value = np.nan

        # Convert absolute probabilities to relative probabilities
        _absolute_to_relative_probability(iter(tree))
        self._add_uniform_component(tree)
        return tree

    def _add_uniform_component(self, tree):
        """Adds a uniform mixture component modifying the tree in-place."""
        uniform_weight = self.uniform_weight
        if uniform_weight < 0 or uniform_weight > 1:
            raise ValueError('uniform_weight should be between 0 and 1')
        if uniform_weight == 0:
            # No modification necessary
            return

        # Convert to absolute probabilities at the leaves
        _relative_to_absolute_probability(iter(tree), 1)

        # Iterate and add volume
        for i, node in enumerate(tree):  # Must be depth-first and left-then-right traversal
            if node.is_leaf():
                volume = np.prod(node.domain[:, 1] - node.domain[:, 0])
                node.value = (1 - uniform_weight) * node.value + uniform_weight * volume

        # Convert back to relative probability
        _absolute_to_relative_probability(iter(tree))

    def sample(self, n_samples=1, random_state=None, shuffle=True):
        # Randomly sample leaf nodes based on their node_value
        # NOTE: This is slightly inefficient because there are
        #  binomial samples of O(n_samples) at each level of the
        #  tree.  Thus the complexity is O(n_levels*n_samples).
        #  However, this is a simple implementation that does not
        #  require getting absolute probabilities of leaves.
        rng = check_random_state(random_state)
        stack = [n_samples]
        X_arr = []
        for node in self.tree_:
            cur_n = stack.pop()
            if node.is_leaf():
                if cur_n > 0:
                    if node.destructor is not None:
                        cur_X = node.destructor.sample(cur_n, random_state=rng)
                    else:
                        # Uniform random samples if no destructor
                        cur_X = rng.rand(cur_n, node.domain.shape[0])
                    cur_X = _from_unit(cur_X, node.domain)  # Transform to node domain
                    X_arr.append(cur_X)
            else:
                # Sample binomial for n_samples left and right
                left_n = rng.binomial(cur_n, node.value)
                right_n = cur_n - left_n
                stack.append(right_n)
                stack.append(left_n)

        # Return permuted X
        X = np.vstack(X_arr)
        if shuffle:
            rng.shuffle(X)  # In-place shuffling
        return X

    def score_samples(self, X, y=None):
        def _update_stack(child):
            """Add left or right child to stack"""
            sel_new = sel.copy()
            if child == 'left':
                sel_new[sel] = X[sel, node.feature] < node.threshold
                log_prob_new = log_prob + np.log(node.value)
            else:
                sel_new[sel] = X[sel, node.feature] >= node.threshold
                log_prob_new = log_prob + np.log1p(-node.value)
            stack.append((sel_new, log_prob_new))
            return sel_new

        self._check_is_fitted()
        X = self._check_X(X)
        self._check_dim(X)

        # Get the nodes associated with each instance
        n_samples, n_features = X.shape
        sel = np.ones(n_samples, dtype=np.bool)
        stack = [(sel, np.log(1))]
        log_pdf = np.NaN * np.ones(n_samples)
        for node in self.tree_:
            sel, log_prob = stack.pop()
            if node.is_leaf():
                if np.any(sel):
                    # Get log density based on volume
                    width = node.domain[:, 1] - node.domain[:, 0]
                    log_volume = np.sum(np.log(width))
                    log_weight = log_prob - log_volume
                    # Get node destructor density score
                    if node.destructor is not None:
                        U_sel = _to_unit(X[sel, :], node.domain)
                        log_node_score = node.destructor.score_samples(U_sel)
                    else:
                        log_node_score = 0
                    # Add weight and node_score
                    log_pdf[sel] = log_weight + log_node_score
            else:
                # Set selection and add absolute probabilities to stack
                _update_stack('right')
                _update_stack('left')

        if np.any(np.isnan(log_pdf)):
            warnings.warn('score_samples contains NaN values')
        return log_pdf

    def _check_X(self, X):
        X = check_array(X)
        X = check_X_in_interval(X, self.get_support())
        return X

    def get_support(self):
        return _UNIT_SPACE

    def _get_node_destructor_or_default(self):
        if self.node_destructor is None:
            return IdentityDestructor()
        else:
            return self.node_destructor

    def _check_is_fitted(self):
        check_is_fitted(self, ['tree_', 'n_features_'])

    def _check_dim(self, X):
        if X.shape[1] != self.n_features_:
            raise ValueError('X does not have the same dimension as the original training data.')


class RandomTreeEstimator(BaseEstimator):
    def __init__(self, min_samples_leaf=0.1, max_leaf_nodes=None, random_state=None):
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state

    def fit(self, X, y=None, **fit_params):
        # Just make y a random gaussian variable
        X = check_array(X)
        rng = check_random_state(self.random_state)
        y_rand = rng.randn(X.shape[0])

        tree_est = ExtraTreeRegressor(
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            max_features=1,  # Completely random tree
            splitter='random',
            random_state=rng,
        )
        tree_est.fit(X, y_rand, **fit_params)
        self.tree_ = tree_est.tree_
        return self


def _absolute_to_relative_probability(tree_depth_iter):
    """Changes from absolute probabilities at the leaves to relative
    probabilities on non-leaves modifying the tree in-place.
    """
    node = next(tree_depth_iter)
    if node.is_leaf():
        # Return absolute probability of node and reset to nan
        leaf_absolute_prob = node.value
        node.value = np.nan
        return leaf_absolute_prob
    else:
        left_absolute_prob = _absolute_to_relative_probability(tree_depth_iter)
        right_absolute_prob = _absolute_to_relative_probability(tree_depth_iter)

        # Set relative probability of left child
        node.value = left_absolute_prob / (left_absolute_prob + right_absolute_prob)

        # Return total absolute probability up to caller
        return left_absolute_prob + right_absolute_prob


def _relative_to_absolute_probability(tree_depth_iter, prob):
    """Changes from relative probabilities at internal nodes to absolute
    probabilities at the leaves modifying the tree in-place.
    """
    node = next(tree_depth_iter)
    if node.is_leaf():
        # Return absolute probability of node and reset to nan
        node.value = prob
    else:
        _relative_to_absolute_probability(tree_depth_iter, prob * node.value)
        _relative_to_absolute_probability(tree_depth_iter, prob * (1 - node.value))
        node.value = np.nan


class _ArrayedTreeWrapper:
    """A simple wrapper for an array-based tree like in scikit-learn.
    Note this is not an estimator but just a good wrapper object to expose iterators etc.
    Fitting will need to take this object or iterator as input.
    """

    def __init__(self, tree):
        if not all(hasattr(tree, a) for a in [
            'n_features',
            'feature', 'threshold',
            'children_left', 'children_right',
        ]):
            raise ValueError('tree does not seem to be an arrayed tree (e.g. sklearn tree)')
        self.wrapped_tree = tree
        self.node_destructors = [None for _ in range(len(tree.feature))]
        self.node_values = [None for _ in range(len(tree.feature))]

    def set_node_destructor(self, node_destructor):
        if node_destructor is None:
            self.node_destructors = [None for _ in range(len(self.wrapped_tree.feature))]
        else:
            self.node_destructors = [clone(node_destructor) for _ in
                                     range(len(self.wrapped_tree.feature))]

    def __str__(self):
        def _domain_str():
            return ','.join('[' + ','.join('%.3g' % a for a in dom) + ']' for dom in node.domain)

        s_arr = []
        stack = [(True, '')]
        for node in self:
            is_left, indent = stack.pop()

            pos_str = 'Left' if is_left else 'Right'
            if node.is_leaf():
                s_arr.append('%s%s leaf domain=%s, value=%g, left_child_index=%g\n'
                             % (indent, pos_str, _domain_str(), node.value, node.left_child_index))
            else:
                s_arr.append(
                    '%s%s feat=%4d, prob_left=%.3g, thresh=%.3g, thresh_out=%.3g, domain=%s\n'
                    % (
                        indent, pos_str, node.feature, node.value, node.threshold,
                        node.threshold_out,
                        _domain_str()))
                stack.append((False, '  ' + indent))
                stack.append((True, '  ' + indent))

        return ''.join(s_arr)

    def __iter__(self):
        init_domain = np.array([[0, 1] for _ in range(self.wrapped_tree.n_features)],
                               dtype=np.float)
        node_stack = [(0, init_domain)]
        while len(node_stack) > 0:
            # Construct and yield node
            node_i, domain = node_stack.pop()
            cur_node = _SklearnNode(self.wrapped_tree, self.node_values, self.node_destructors,
                                    node_i, domain)
            yield cur_node

            # Push children onto stack
            if not cur_node.is_leaf():
                left_domain = domain.copy()
                left_domain[cur_node.feature, 1] = cur_node.threshold
                right_domain = domain.copy()
                right_domain[cur_node.feature, 0] = cur_node.threshold

                node_stack.append((cur_node.right_child_index, right_domain))
                node_stack.append((cur_node.left_child_index, left_domain))


class _SklearnNode:
    # Need to mutate actual tree object so keep reference to tree object
    def __init__(self, tree, node_values, node_destructors, node_i, domain):
        # Create list of values instead of matrix of values
        self._tree = tree
        self._node_values = node_values
        self._node_destructors = node_destructors
        self._node_i = node_i
        self._domain = domain

    def __str__(self):
        return 'node_i = %d, value = %g' % (self.node_i, self.value)

    def is_leaf(self):
        if np.isnan(self.left_child_index):
            return True
        elif self.left_child_index < 0:
            if self.right_child_index >= 0:
                raise RuntimeError('Expected both left and right index to be negative.')
            return True
        else:
            if self.right_child_index < 0:
                raise RuntimeError('Expected both left and right index to be non-negative.')
            return False

    # Read only properties
    @property
    def feature(self):
        return self._tree.feature[self._node_i]

    @property
    def left_child_index(self):
        return self._tree.children_left[self._node_i]

    @property
    def right_child_index(self):
        return self._tree.children_right[self._node_i]

    # Read/write properties
    @property
    def node_i(self):
        return self._node_i

    @node_i.setter
    def node_i(self, x):
        self._node_i = x

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, x):
        self._domain = x

    @property
    def value(self):
        return self._node_values[self._node_i]

    @value.setter
    def value(self, x):
        if x <= 0:
            warnings.warn(BoundaryWarning(
                'Numerical imprecision or faulty algorithm because `value` '
                'should never be 0 or negative, changing from %g to 1e-15.' % x))
            x = 1e-15
        elif x >= 1:
            if x == 1 and len(self._node_values) == 1:
                # Handles the case where there are no splits (i.e. just root node as leaf node)
                pass
            else:
                warnings.warn(BoundaryWarning(
                    'Numerical imprecision or faulty algorithm because `value` '
                    'should never be 1 or greater than 1, changing from %g to 1-1e-15.' % x))
                x = 1 - 1e-15
        self._node_values[self._node_i] = x

    @property
    def value_out(self):
        # Extract necessary variables
        (a, b) = self.domain[self.feature]
        t = self.threshold
        return (t - a) / (b - a)

    @property
    def threshold(self):
        return self._tree.threshold[self._node_i]

    @threshold.setter
    def threshold(self, x):
        if x <= self.domain[self.feature, 0]:
            warnings.warn(BoundaryWarning(
                'Numerical imprecision or faulty algorithm because `threshold` should '
                'never be the same as the edge of the domain which is %g, changing from %g to '
                '%g+1e-15. '
                % (self.domain[self.feature, 0], x, self.domain[self.feature, 0])))
            x += 1e-15
        elif x >= self.domain[self.feature, 1]:
            warnings.warn(BoundaryWarning(
                'Numerical imprecision or faulty algorithm because `threshold` should '
                'never be the same as the edge of the domain which is %g, changing from %g to '
                '%g+1e-15. '
                % (self.domain[self.feature, 1], x, self.domain[self.feature, 1])))
            x -= 1e-15
        self._tree.threshold[self._node_i] = x

    @property
    def threshold_out(self):
        # Extract necessary variables
        (a, b) = self.domain[self.feature]
        p = self.value
        return (b - a) * p + a

    @property
    def destructor(self):
        return self._node_destructors[self._node_i]

    @destructor.setter
    def destructor(self, x):
        self._node_destructors[self._node_i] = x


def _to_unit(X, bounding_box):
    """Scale from input bounding box to unit hypercube."""
    X = check_X_in_interval(X, bounding_box)
    width = bounding_box[:, 1] - bounding_box[:, 0]
    pos = bounding_box[:, 0]
    U = (X - pos) / width
    U = check_X_in_interval(U, _UNIT_SPACE)
    return U


def _from_unit(U, bounding_box):
    """Scale from unit hypercube to input bounding box."""
    U = check_X_in_interval(U, _UNIT_SPACE)
    width = bounding_box[:, 1] - bounding_box[:, 0]
    pos = bounding_box[:, 0]
    X = U * width + pos
    X = check_X_in_interval(X, bounding_box)
    return X


def _tree_transform(tree, X, y=None):
    """Transforms X based on generic tree object."""

    def _compose_linear(scale_shift_inner, scale_shift_outer):
        scale_inner, shift_inner = scale_shift_inner
        scale_outer, shift_outer = scale_shift_outer
        shift_new = scale_outer * shift_inner + shift_outer
        scale_new = scale_outer * scale_inner
        scale_shift_new = np.array([scale_new, shift_new])
        return scale_shift_new

    def _update_stack(child):
        """Add left or right child by computing scale, shift and selection.
        Note this is late binding for outer variables so it should only be called
        when sel, node, a, b, t, t_out, etc have been defined (i.e. inside
        the node loop.
        Defined here so that the compiler doesn't keep creating new functions each time.
        """
        sel_new = sel.copy()
        if child == 'left':
            sel_new[sel] = X[sel, node.feature] < t
            scale_new_local = (t_out - a) / (t - a)
            shift_new_local = a - a * scale_new_local
        else:
            sel_new[sel] = X[sel, node.feature] >= t
            scale_new_local = (b - t_out) / (b - t)
            shift_new_local = t_out - t * scale_new_local

        # Update full scale, shift array
        scale_shift_arr_new = scale_shift_arr.copy()
        scale_shift_arr_new[node.feature] = _compose_linear(
            (scale_new_local, shift_new_local),
            scale_shift_arr[node.feature]
        )

        # Push onto stack
        stack.append((sel_new, scale_shift_arr_new))

    # Initialize
    X = check_array(X, copy=True, dtype=np.float)
    n_samples, n_features = X.shape

    # Setup root node
    sel = np.ones(n_samples, dtype=np.bool)
    scale_shift_arr = np.array([(1, 0) for _ in range(n_features)], dtype=np.float)
    stack = [(sel, scale_shift_arr)]
    for node in tree:  # Must be depth-first and left-then-right traversal
        sel, scale_shift_arr = stack.pop()
        if node.is_leaf():
            # Apply node destructor
            if node.destructor is not None and np.sum(sel) > 0:
                X[sel, :] = _to_unit(X[sel, :], node.domain)
                X[sel, :] = node.destructor.transform(X[sel, :])
                X[sel, :] = _from_unit(X[sel, :], node.domain)

            # Apply linear transformation
            X[sel, :] *= scale_shift_arr[:, 0]
            X[sel, :] += scale_shift_arr[:, 1]
        else:
            # Extract necessary variables
            (a, b) = node.domain[node.feature]
            t = node.threshold
            t_out = node.threshold_out

            # Add children with appropriate scale, shift and filtered selection
            _update_stack('right')
            _update_stack('left')

    # Cleanup values since numerical errors can cause them to fall
    #  outside of the destructor range of [0, 1]
    X = np.maximum(np.minimum(X, 1), 0)
    return X


def _get_inverse_tree(tree):
    """Computes the tree corresponding to the inverse of the transformation."""
    tree_out = deepcopy(tree)
    # Iterator starting at root (can be depth-first or breadth-first)
    for node_in, node_out in zip(tree, tree_out):
        # Implicitly changes a and b for children since bounds computed when traversing
        # Need to extract values before setting them since they are used internally
        if node_in.is_leaf():
            if node_in.destructor is not None:
                # Create view of the inverse destructor
                node_out.destructor = get_inverse_canonical_destructor(
                    node_out.destructor, copy=False)  # Destructor already deep copied
        else:

            (a, b) = node_in.domain[node_in.feature]
            (a_tilde, b_tilde) = node_out.domain[node_in.feature]
            p = node_in.value
            t = node_in.threshold

            # Output threshold if a = a_tilde and b = b_tilde
            #  (i.e. threshold relative to new bounds)
            node_out.threshold = (b_tilde - a_tilde) * p + a_tilde
            # Want a probability that inverts this function
            #  (i.e. such that alpha = 1/alpha)
            node_out.value = (t - a) / ((t - a) + (b - t))
    return tree_out


def _get_arrayed_tree(tree_estimator):
    """Nearly trivial wrapper to get wrapped tree from estimator."""
    return _ArrayedTreeWrapper(tree_estimator.tree_)


def _check_get_tree(_get_tree):
    return _get_tree if _get_tree is not None else _get_arrayed_tree

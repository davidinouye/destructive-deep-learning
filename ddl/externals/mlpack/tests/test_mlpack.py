"""Test the mlpack destructors."""
from __future__ import division, print_function

import numpy as np
# noinspection PyPackageRequirements
import pytest
from sklearn.utils import check_random_state

from ddl.datasets import make_toy_data
from ddl.externals.mlpack import MlpackDensityTreeEstimator
from ddl.tree import TreeDensity, TreeDestructor
from ddl.validation import check_destructor

try:
    # noinspection PyProtectedMember
    from ddl.externals.mlpack import _det as det
except ImportError:
    import warnings

    warnings.warn('In test script, could not import necessary mlpack wrappers.')


def test_mlpack_density_tree_destructor():
    destructor = TreeDestructor(
        tree_density=TreeDensity(
            tree_estimator=MlpackDensityTreeEstimator(max_leaf_nodes=10),
        )
    )
    assert check_destructor(destructor)


@pytest.mark.parametrize('test', ['canonical', 'complex', 'default'])
def test_mlpack_det(test):
    rng = check_random_state(0)
    if test == 'canonical':
        X = rng.rand(200, 2)
        min_vals = np.zeros(X.shape[1])
        max_vals = np.ones(X.shape[1])
        total_points = X.shape[0]
        tree = det.PyDTree(min_vals=min_vals, max_vals=max_vals, total_points=total_points)
        tree.fit(X)
        print(tree)
    elif test == 'complex':
        X = rng.randn(200, 2)
        tree = det.PyDTree(X=X)
        tree.fit(X)
        print(tree)
        tree.fit(X, max_depth=5)
        print(tree)
        tree.fit(X, max_leaf_nodes=8)
        print(tree)
    else:
        X = rng.randn(200, 2)
        tree = det.PyDTree(X)
        alpha = -1  # Just to initialize alpha
        for i in range(10):
            if i == 0:
                alpha = tree.grow(X)
            else:
                if tree.num_children() == 0:
                    break  # Stop since no more pruning allowed
                alpha = tree.prune_and_update(alpha, X.shape[0])
            print('alpha=%g' % alpha)
            print(tree.get_tree_str(show_leaves=True))


# noinspection SpellCheckingInspection
def test_mlpack_det_get_arrayed_tree():
    # Setup dataset
    D = make_toy_data('uniform_grid', n_samples=1000, random_state=0)
    X = D.X

    # Setup tree
    min_vals = np.zeros(X.shape[1])
    max_vals = np.ones(X.shape[1])
    total_points = X.shape[0]
    tree = det.PyDTree(min_vals=min_vals, max_vals=max_vals, total_points=total_points)

    # Fit tree
    tree.fit(X, max_leaf_nodes=5, min_leaf_size=5)
    arrayed_tree = tree.get_arrayed_tree()

    # print(tree.get_tree_str(show_leaves=True))
    # stack = [(0, None)]
    # while len(stack) > 0:
    #    print(stack)
    #    node_i, is_left = stack.pop()
    #    prefix = 'Left' if is_left else 'Right'
    #
    #    if arrayed_tree.feature[node_i] >= 0:
    #        print('%s dim=%3d, threshold=%.3f'
    #              % (prefix, arrayed_tree.feature[node_i], arrayed_tree.threshold[node_i]))
    #        stack.append((arrayed_tree.children_right[node_i], False))
    #        stack.append((arrayed_tree.children_left[node_i], True))
    #    else:
    #        print('%s' % prefix)

    # print(arrayed_tree)
    # print(repr(arrayed_tree.feature))
    # print(repr(arrayed_tree.threshold))
    # print(repr(arrayed_tree.children_left))
    # print(repr(arrayed_tree.children_right))

    # Test based on known values
    assert np.all(arrayed_tree.feature == np.array([0, -1, 1, 1, 1, -1, -1, -1, -1]))
    nan = np.nan
    expected_threshold = np.array(
        [0.5631866381580427, nan, 0.7914520226478026,
         0.40126660240012946, 0.19549050993708061, nan,
         nan, nan, nan]
    )
    np.testing.assert_allclose(arrayed_tree.threshold, expected_threshold, rtol=1e-16)
    assert np.all(arrayed_tree.children_left == [1, -1, 3, 4, 5, -1, -1, -1, -1])
    assert np.all(arrayed_tree.children_right == [2, -1, 8, 7, 6, -1, -1, -1, -1])

from __future__ import division, print_function

import logging
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array

try:
    from ._det import PyDTree
except ImportError:
    # Just ignoring import error because mlpack isn't required
    pass

logger = logging.getLogger(__name__)


class MlpackDensityTreeEstimator(BaseEstimator):
    def __init__(self, max_leaf_nodes=None, max_depth=None, min_samples_leaf=1):
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y=None):
        fit_params = dict()  # Leave as defaults unless overriden
        if self.max_leaf_nodes is not None:
            fit_params['max_leaf_nodes'] = self.max_leaf_nodes
        if self.max_depth is not None:
            fit_params['max_depth'] = self.max_depth
        if self.min_samples_leaf is not None:
            # Note the different parameter name
            fit_params['min_leaf_size'] = self.min_samples_leaf

        # Setup canonical decision tree
        X = check_array(X)
        n_samples, n_features = X.shape
        try:
            py_dtree = PyDTree(min_vals=np.zeros(n_features), max_vals=np.ones(n_features),
                               total_points=n_samples)
            # Make a copy so original data is not mutated when passed to fit below
            py_dtree.fit(X.copy(), **fit_params)
        except NameError:
            raise RuntimeError(
                'Mlpack estimator fitting failed because either mlpack or '
                'the corresponding wrappers were not installed correctly.')

        # Extract arrayed representation of the tree (similar representation to sklearn)
        self.tree_ = py_dtree.get_arrayed_tree()
        return self

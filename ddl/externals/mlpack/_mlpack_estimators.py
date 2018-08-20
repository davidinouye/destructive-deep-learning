"""Private module for loading mlpack estimators."""
from __future__ import division, print_function

import logging

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
    """Density tree estimator via mlpack (mlpack.org).

    This estimator leverages the methods for Density Estimation Trees (DET,
    see Ram & Gray 2011 paper below) that are implemented in mlpack (see the
    DET method in mlpack's documentation at `mlpack.org`_). Essentially,
    this class provides a simple wrapper around the C++ functions in mlpack
    and thus must be compiled with mlpack source code.

    .. _`mlpack.org`: http://mlpack.org/

    Parameters
    ----------
    max_leaf_nodes : int or None, default=None
        Maximum number of leaf nodes in final tree. The tree will be fully
        grown based on `min_samples_leaf` and then pruned until the number
        of leaf nodes is less than `max_leaf_nodes`. If None,
        then `max_leaf_nodes` is considered to be infinite. This parameter
        can be useful for simple regularization of the density tree.

    max_depth : int or None, default=None
        Maximum depth of final tree. The tree will be fully grown based on
        `min_samples_leaf` and then pruned until the depth of the tree is
        less than `max_depth`. If None, then `max_depth` is considered to be
        infinite. This parameter can be useful for simple regularization of
        the density tree.

    min_samples_leaf : int, default=1
        Minimum number of samples required at all leaf nodes. Main parameter
        for growing the tree initially before pruning. This parameter is
        mainly here for computational reasons on large datasets. This
        parameter could also be used as regularization.

    Attributes
    ----------
    tree_ : arrayed_tree
        The tree structure represented using arrays similar to the trees
        used in sklearn (e.g. :class:`sklearn.tree.DecisionTreeClassifier`).

    References
    ----------
    Ram, P. and Gray, A. G. Density Estimation Trees. In Proceedings of the
    17th ACM SIGKDD International Conference on Knowledge Discovery and Data
    Mining, 2011.

    """

    def __init__(self, max_leaf_nodes=None, max_depth=None, min_samples_leaf=1):
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y=None):
        """Fit estimator to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : None, default=None
            Not used in the fitting process but kept for compatibility.

        Returns
        -------
        self : estimator
            Returns the instance itself.

        """
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

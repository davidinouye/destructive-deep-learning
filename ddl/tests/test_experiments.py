# Simple and fast (but not comprehensive) tests 
# to sanity check that ICML 2018 experiments will run the same
##### Mostly copied from demo_toy_experiment.py notebook #####
from __future__ import division
from __future__ import print_function
import sys
import os
import logging
import time
import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from ddl.datasets import make_toy_data
from ddl.deep import DeepDestructorCV, CompositeDestructor
from ddl.independent import IndependentDestructor, IndependentDensity, IndependentInverseCdf
from ddl.univariate import ScipyUnivariateDensity, HistogramUnivariateDensity
from ddl.linear import (LinearProjector, RandomOrthogonalEstimator, 
                                                BestLinearReconstructionDestructor)
from ddl.autoregressive import AutoregressiveDestructor
from ddl.mixture import GaussianMixtureDensity, FirstFixedGaussianMixtureDensity
from ddl.tree import TreeDestructor, TreeDensity, RandomTreeEstimator
from ddl.externals.mlpack import MlpackDensityTreeEstimator

logger = logging.getLogger(__name__)

# Setup parameters for experiment
data_name = 'concentric_circles'
n_train = 1000
cv = 3  # Number of cv splits
random_state = 0

def _get_toy_destructors_and_names():
    # BASELINE SHALLOW DESTRUCTORS
    gaussian_full = CompositeDestructor(
        destructors=[
            LinearProjector(
                linear_estimator=PCA(),
                orthogonal=False,
            ),
            IndependentDestructor(),
        ],
    )
    mixture_20 = AutoregressiveDestructor(
        density_estimator=GaussianMixtureDensity(
            covariance_type='spherical',
            n_components=20,
        )
    )
    random_tree = CompositeDestructor(
        destructors=[
            IndependentDestructor(),
            TreeDestructor(
                tree_density=TreeDensity(
                    tree_estimator=RandomTreeEstimator(min_samples_leaf=20, max_leaf_nodes=50),
                    node_destructor=IndependentDestructor(
                        independent_density=IndependentDensity(
                            univariate_estimators=HistogramUnivariateDensity(
                                bins=10, alpha=10, bounds=[0,1]
                            )
                        )
                    )
                )
            )
        ]
    )
    density_tree = CompositeDestructor(
        destructors=[
            IndependentDestructor(),
            TreeDestructor(
                tree_density=TreeDensity(
                    tree_estimator=MlpackDensityTreeEstimator(min_samples_leaf=10),
                    uniform_weight=0.001,
                )
            )
        ]
    )
    baseline_destructors = [gaussian_full, mixture_20, random_tree, density_tree]
    baseline_names = ['Gaussian', 'Mixture', 'SingleRandTree', 'SingleDensityTree']

    # LINEAR DESTRUCTORS
    alpha_histogram = [10] #[1, 10, 100]
    random_linear_projector = LinearProjector(
        linear_estimator=RandomOrthogonalEstimator(), orthogonal=True
    )
    canonical_histogram_destructors = [
        IndependentDestructor(
            independent_density=IndependentDensity(
                univariate_estimators=HistogramUnivariateDensity(bins=20, bounds=[0, 1], alpha=a)
            )
        )
        for a in alpha_histogram
    ]
    linear_destructors = [
        DeepDestructorCV(
            init_destructor=IndependentDestructor(),
            canonical_destructor=CompositeDestructor(destructors=[
                IndependentInverseCdf(),  # Project to inf real space
                random_linear_projector,  # Random linear projector
                IndependentDestructor(),  # Project to canonical space
                destructor,  # Histogram destructor in canonical space
            ]),
            n_extend=20,  # Need to extend since random projections
        )
        for destructor in canonical_histogram_destructors
    ]
    linear_names = ['RandLin (%g)' % a for a in alpha_histogram]

    # MIXTURE DESTRUCTORS
    fixed_weight = [0.5] #[0.1, 0.5, 0.9]
    mixture_destructors = [
        CompositeDestructor(destructors=[
            IndependentInverseCdf(),
            AutoregressiveDestructor(
                density_estimator=FirstFixedGaussianMixtureDensity(
                    covariance_type='spherical',
                    n_components=20,
                    fixed_weight=w,
                )
            )
        ])
        for w in fixed_weight 
    ]
    # Make deep destructors
    mixture_destructors = [
        DeepDestructorCV(
            init_destructor=IndependentDestructor(),
            canonical_destructor=destructor,
            n_extend=5, 
        )
        for destructor in mixture_destructors
    ]
    mixture_names = ['GausMix (%.2g)' % w for w in fixed_weight]

    # TREE DESTRUCTORS
    # Random trees
    histogram_alpha = [10] #[1, 10, 100]
    tree_destructors = [
        TreeDestructor(
            tree_density=TreeDensity(
                tree_estimator=RandomTreeEstimator(
                    max_leaf_nodes=4
                ),
                node_destructor=IndependentDestructor(
                    independent_density=IndependentDensity(
                        univariate_estimators=HistogramUnivariateDensity(
                            alpha=a, bins=10, bounds=[0,1]
                        )
                    )
                ),
            )
        )
        for a in histogram_alpha
    ]
    tree_names = ['RandTree (%g)' % a for a in histogram_alpha]

    # Density trees using mlpack
    tree_uniform_weight = [0.5] #[0.1, 0.5, 0.9]
    tree_destructors.extend([
        TreeDestructor(
            tree_density=TreeDensity(
                tree_estimator=MlpackDensityTreeEstimator(min_samples_leaf=10),
                uniform_weight=w,
            )
        )
        for w in tree_uniform_weight
    ])
    tree_names.extend(['DensityTree (%.2g)' % w for w in tree_uniform_weight])

    # Add random rotation to tree destructors
    tree_destructors = [
        CompositeDestructor(destructors=[
            IndependentInverseCdf(),
            LinearProjector(linear_estimator=RandomOrthogonalEstimator()),
            IndependentDestructor(),
            destructor,
        ])
        for destructor in tree_destructors
    ]

    # Make deep destructors
    tree_destructors = [
        DeepDestructorCV(
            init_destructor=IndependentDestructor(),
            canonical_destructor=destructor,
            # Density trees don't need to extend as much as random trees
            n_extend=50 if 'Rand' in name else 5, 
        )
        for destructor, name in zip(tree_destructors, tree_names)
    ]
    # Collect all destructors and set CV parameter
    destructors = baseline_destructors + linear_destructors + mixture_destructors + tree_destructors
    destructor_names = baseline_names + linear_names + mixture_names + tree_names
    for d in destructors:
        if 'cv' in d.get_params():
            d.set_params(cv=cv)
        ############ Change from notebook to make faster ############
        if 'n_canonical_destructors' in d.get_params():
            d.set_params(n_canonical_destructors=2)

    return destructors, destructor_names



@pytest.mark.parametrize(
    'destructor_name', 
    ['Gaussian', 'Mixture', 'SingleRandTree', 'SingleDensityTree', 'RandLin (10)', 
     'GausMix (0.5)', 'RandTree (10)', 'DensityTree (0.5)']
)
def test_toy_destructor(destructor_name):
    # Make dataset and create train/test splits
    n_samples = 2 * n_train
    D = make_toy_data(data_name, n_samples=n_samples, random_state=random_state)
    X_train = D.X[:n_train]
    y_train = D.y[:n_train] if D.y is not None else None
    X_test = D.X[n_train:]
    y_test = D.y[n_train:] if D.y is not None else None

    def _fit_and_score(data_name, destructor, destructor_name, n_train, random_state=0):
        """Simple function to fit and score a destructor."""
        # Fix random state of global generator so repeatable if destructors are random
        rng = check_random_state(random_state)
        old_random_state = np.random.get_state()
        np.random.seed(rng.randint(2 ** 32, dtype=np.uint32))
        
        try:
            # Fit destructor
            start_time = time.time()
            destructor.fit(X_train)
            train_time = time.time() - start_time
        except RuntimeError as e:
            # Handle MLPACK error
            if 'mlpack' not in str(e).lower():
                raise e
            warnings.warn('Skipping density tree destructors because of MLPACK error "%s". '
                          'Using dummy IndependentDestructor() instead.' % str(e))
            destructor = CompositeDestructor([IndependentDestructor()]).fit(X_train)
            train_time = 0
            train_score = -np.inf
            test_score = -np.inf
            score_time = 0 
        else:
            # Get scores
            start_time = time.time()
            train_score = destructor.score(X_train)
            test_score = destructor.score(X_test)
            score_time = time.time() - start_time
            
        logger.debug('train=%.3f, test=%.3f, train_time=%.3f, score_time=%.3f, destructor=%s, data_name=%s' 
                     % (train_score, test_score, train_time, score_time, destructor_name, data_name))

        # Reset random state
        np.random.set_state(old_random_state)
        return dict(fitted_destructor=destructor,
                    destructor_name=destructor_name,
                    train_score=train_score,
                    test_score=test_score)

    # Get destructor
    destructors, destructor_names = _get_toy_destructors_and_names()
    ind = destructor_names.index(destructor_name)

    # Fit and score destructor
    result_dict = _fit_and_score(data_name, destructors[ind], destructor_names[ind], n_train, random_state=random_state)

    # Check that scores match expected values
    expected_train = [-4.436517571477812893e+00, -4.139799591807839185e+00, -4.135013171965160161e+00, 
                      -3.973179020102820758e+00, -4.367216508184048607e+00, -4.152734964599357426e+00,
                      -4.266537128263181877e+00, -4.058489689601575634e+00]
    expected_test = [-4.440092999888506142e+00, -4.241302023438742630e+00, -4.243528943385095786e+00,
                     -4.188421292899972670e+00, -4.400212807936888737e+00, -4.262652579336936753e+00,
                     -4.296448657246879854e+00, -4.233697219621666896e+00]
    assert(np.abs(expected_train[ind] - result_dict['train_score']) < 1e-15)
    assert(np.abs(expected_test[ind] - result_dict['test_score']) < 1e-15)

"""Simple tests for main destructors."""
from __future__ import division, print_function

import numpy as np
# noinspection PyPackageRequirements
import pytest

from ddl.datasets import make_toy_data


@pytest.mark.parametrize(
    'data_name', [
        'concentric_circles', 'grid', 'gaussian_grid', 'uniform_grid',
        'rotated_uniform', 'autoregressive', 'sin_wave', 'rbig_sin_wave',
        'quadratic'
    ])
def test_make_toy_data(data_name):
    """Trivial test of make_toy_data.

    Checks that the n_samples is correct and that random_state is initialized correctly
    so generated samples are exactly correct.
    """
    n_samples = 10
    random_state = 42
    data = make_toy_data(data_name, n_samples=n_samples, random_state=random_state)
    X, y, is_canonical_domain = data.X, data.y, data.is_canonical_domain

    # Check some properties of the created data
    assert len(X.shape) == 2, 'Should be a matrix.'
    assert X.shape[0] == n_samples, 'Wrong number of samples.'
    assert y is None or (len(y.shape) == 1 and y.shape[0] == X.shape[0]), 'y is incorrect'
    if is_canonical_domain:
        assert np.all(X.ravel() >= 0) and np.all(X.ravel() <= 1)

    # Check random state consistency of generated data
    data = make_toy_data(data_name, n_samples=n_samples, random_state=random_state)
    X_2 = data.X
    assert np.all(X.ravel() == X_2.ravel()), 'X and X_2 with same random state or different'

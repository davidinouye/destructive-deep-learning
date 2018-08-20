"""Tests for mixture destructors which are slow currently."""
from __future__ import division, print_function

from ddl.autoregressive import AutoregressiveDestructor
from ddl.mixture import GaussianMixtureDensity
from ddl.validation import check_destructor


def test_autoregressive_mixture_destructor():
    """Currently takes too long for regular testing."""
    destructor = AutoregressiveDestructor(
        density_estimator=GaussianMixtureDensity(
            covariance_type='spherical',
            max_iter=1,
            n_components=2,
            random_state=0,
        )
    )
    assert check_destructor(destructor, is_canonical=False)

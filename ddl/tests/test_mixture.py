"""Tests for mixture destructors which are slow currently."""
from __future__ import division, print_function

from sklearn.cluster import KMeans

from ddl.autoregressive import AutoregressiveDestructor
from ddl.independent import IndependentDensity
# noinspection PyProtectedMember
from ddl.mixture import GaussianMixtureDensity, _MixtureDensity
from ddl.validation import check_density, check_destructor


def test_autoregressive_mixture_density():
    """Test generic mixture density."""
    density = _MixtureDensity(
        cluster_estimator=KMeans(n_clusters=2, random_state=0),
        component_density_estimator=IndependentDensity()
    )
    assert check_density(density)


def test_autoregressive_sklearn_mixture_destructor():
    """Test autoregressive destructor and sklearn's Gaussian mixture."""
    destructor = AutoregressiveDestructor(
        density_estimator=GaussianMixtureDensity(
            covariance_type='spherical',
            max_iter=1,
            n_components=2,
            random_state=0,
        )
    )
    assert check_destructor(destructor, is_canonical=False)


if __name__ == '__main__':
    test_autoregressive_mixture_density()
    test_autoregressive_sklearn_mixture_destructor()

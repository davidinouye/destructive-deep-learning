"""Simple tests for main densities."""
from __future__ import division, print_function

# noinspection PyPackageRequirements
import pytest
from sklearn.cluster import KMeans

from ddl.gaussian import GaussianDensity
from ddl.independent import IndependentDensity
from ddl.mixture import _MixtureDensity
from ddl.univariate import HistogramUnivariateDensity
from ddl.validation import check_density


def test_histogram_univariate_density():
    density = HistogramUnivariateDensity(
        bins=10, alpha=10, bounds=[0, 1]
    )
    assert check_density(density)


def test_histogram_multivariate_density():
    density = IndependentDensity(
        univariate_estimators=HistogramUnivariateDensity(
            bins=10, alpha=10, bounds=[0, 1]
        )
    )
    assert check_density(density)


@pytest.mark.parametrize('covariance_type', ['full', 'tied', 'diag', 'spherical'])
def test_gaussian_density(covariance_type):
    density = GaussianDensity(
        covariance_type=covariance_type,
        reg_covar=1e-6,
    )
    assert check_density(density)


def test_autoregressive_mixture_density():
    """Test generic mixture density."""
    density = _MixtureDensity(
        cluster_estimator=KMeans(n_clusters=2, random_state=0),
        component_density_estimator=IndependentDensity()
    )
    assert check_density(density)

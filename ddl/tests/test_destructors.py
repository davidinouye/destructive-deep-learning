"""Simple tests for main destructors."""
from __future__ import division, print_function

# noinspection PyPackageRequirements
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state

from ddl.autoregressive import AutoregressiveDestructor
from ddl.base import CompositeDestructor, IdentityDestructor, create_inverse_canonical_destructor
from ddl.deep import DeepDestructor, DeepDestructorCV
from ddl.gaussian import GaussianDensity
from ddl.independent import IndependentDensity, IndependentDestructor
from ddl.linear import LinearProjector, RandomOrthogonalEstimator
from ddl.tree import RandomTreeEstimator, TreeDensity, TreeDestructor
from ddl.univariate import HistogramUnivariateDensity
from ddl.validation import check_destructor


def test_inverse_canonical_destructor():
    rng = check_random_state(0)
    fitted_canonical_destructor = IdentityDestructor().fit(rng.rand(10, 2))
    destructor = create_inverse_canonical_destructor(fitted_canonical_destructor)
    assert check_destructor(destructor)

    # Alpha must be high to pass the identity test
    fitted_canonical_destructor = create_inverse_canonical_destructor(
        TreeDestructor(TreeDensity(uniform_weight=0.99)).fit(rng.rand(10, 2))
    )
    destructor = create_inverse_canonical_destructor(fitted_canonical_destructor)
    assert check_destructor(destructor)

    # Alpha must be high to pass the identity test
    fitted_canonical_destructor = IndependentDestructor(
        independent_density=IndependentDensity(
            univariate_estimators=HistogramUnivariateDensity(bins=10, alpha=1000, bounds=[0, 1])
        )
    ).fit(rng.rand(10, 2))
    destructor = create_inverse_canonical_destructor(fitted_canonical_destructor)
    assert check_destructor(destructor)


def test_identity_destructor():
    assert check_destructor(IdentityDestructor())


def test_independent_destructor():
    assert check_destructor(IndependentDestructor(), is_canonical=False)


def test_tree_destructor():
    assert check_destructor(TreeDestructor())


def test_histogram_univariate_destructor():
    destructor = IndependentDestructor(
        independent_density=IndependentDensity(
            # Note only one univariate estimator
            univariate_estimators=[
                HistogramUnivariateDensity(
                    bins=4, alpha=10, bounds=[0, 1]
                )
            ]
        )
    )
    assert check_destructor(destructor)


def test_histogram_multivariate_destructor():
    destructor = IndependentDestructor(
        independent_density=IndependentDensity(
            univariate_estimators=HistogramUnivariateDensity(
                bins=4, alpha=10, bounds=[0, 1]
            )
        )
    )
    assert check_destructor(destructor)


def test_normal_independent_destructor():
    destructor = IndependentDestructor()
    assert check_destructor(destructor, is_canonical=False)


def test_tree_destructor_with_node_destructor():
    node_tree_destructor = IndependentDestructor(
        independent_density=IndependentDensity(
            univariate_estimators=HistogramUnivariateDensity(
                bins=10, alpha=100, bounds=[0, 1]
            )
        )
    )
    for node_destructor in [IdentityDestructor(), node_tree_destructor]:
        destructor = TreeDestructor(
            tree_density=TreeDensity(
                tree_estimator=RandomTreeEstimator(max_leaf_nodes=3, random_state=0),
                node_destructor=node_destructor,
                uniform_weight=0.9,
            )
        )
        assert check_destructor(destructor)


def test_random_linear_householder_destructor():
    destructor = CompositeDestructor(
        destructors=[
            LinearProjector(
                # Since n_components=1, a Householder matrix will be used
                linear_estimator=RandomOrthogonalEstimator(n_components=1),
                orthogonal=False,
            ),
            IndependentDestructor(),
        ],
    )
    assert check_destructor(destructor, is_canonical=False)


def test_pca_destructor():
    destructor = CompositeDestructor(
        destructors=[
            LinearProjector(
                linear_estimator=PCA(),
                orthogonal=False,
            ),
            IndependentDestructor(),
        ],
    )
    assert check_destructor(destructor, is_canonical=False)


def test_deep_destructor_with_tree_destructor():
    destructor = DeepDestructor(
        canonical_destructor=TreeDestructor(),
        n_canonical_destructors=2,
    )
    assert check_destructor(destructor)


def test_deep_destructor_cv_with_tree_destructor():
    destructor = DeepDestructorCV(
        canonical_destructor=TreeDestructor(),
        max_canonical_destructors=2,
        cv=2,
    )
    assert check_destructor(destructor)


def test_autoregressive_gaussian_destructor():
    """Currently takes too long for regular testing."""
    destructor = AutoregressiveDestructor(
        density_estimator=GaussianDensity(
            covariance_type='spherical',
            reg_covar=0.1
        )
    )
    assert check_destructor(destructor, is_canonical=False)

"""Module for validating destructor interface and testing properties."""
from __future__ import division, print_function

import logging
import sys
import warnings
from functools import wraps

import numpy as np
from sklearn.base import clone
from sklearn.utils import check_random_state

# noinspection PyProtectedMember
from .base import BoundaryWarning, UniformDensity, _NumDimWarning
from .utils import (check_domain, check_X_in_interval, get_domain_or_default,
                    get_support_or_default, has_method)

try:
    from sklearn.exceptions import SkipTestWarning, DataConversionWarning
    from sklearn.utils.estimator_checks import check_estimator
except ImportError:
    warnings.warn('Could not import sklearn\'s SkipTestWarning, '
                  'DataConversionWarning or check_estimator '
                  '(likely because nose is not installed) but continuing '
                  'so that documentation can be generated without nose.')
try:
    import ot  # Python optimal transport module (pot)
except ImportError:
    warnings.warn('Could not import python optimal transport (pip install pot) (import ot)')

logger = logging.getLogger(__name__)


# TODO What about self-evaluation?
# Sample from learned model and see if we can relearn current model?? -- shows how stable it is...
# Compare samples of "assumed" model with samples from relearned model
# Is this learnable from the number of samples given (vs null distribution)?
class _ShouldOnlyBeInTestWarning(UserWarning):
    """Warning that should only occur in testing."""

    pass


def _ignore_boundary_warnings(func):
    @wraps(func)
    def _wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', BoundaryWarning)
            return func(*args, **kwargs)

    return _wrapper


@_ignore_boundary_warnings
def check_density(density, random_state=0):
    """Check that an estimator implements density methods correctly.

    First, check that the estimator implements sklearn's estimator interface
    via :func:`sklearn.utils.estimator_checks.check_estimator`. Then,
    check for all of the following methods, and possibly check their basic
    functionality. Some methods will raise errors and others will issue
    warnings.

    Required methods (error if fails or does not exist):

    * :meth:`fit`

    Optional methods primary (warn if does not exist, error if fails):

    * :meth:`sample`
    * :meth:`score_samples`

    Optional methods secondary (warn if does not exist, warn if fails):

    * :meth:`score` (mixin)
    * :meth:`get_support` (required to pass tests if support is not real-valued unbounded)

    Parameters
    ----------
    density : estimator
        Density estimator to check.

    random_state : int, RandomState instance or None, optional (default=0)
        Note that random_state defaults to 0 so that checks are reproducible
        by default.

        If int, `random_state` is the seed used by the random number
        generator; If :class:`~numpy.random.RandomState` instance,
        `random_state` is the random number generator; If None, the random
        number generator is the :class:`~numpy.random.RandomState` instance
        used by :mod:`numpy.random`.

    Notes
    -----
    The :meth:`score` method must be equal to the average of
    :meth:`score_samples` for consistency. Note that some sklearn density
    estimators such as :class:`sklearn.mixture.GaussianMixture` return the
    total log probability instead of the average log probability. Thus,
    these estimators need to be subclassed and their :meth:`score` method
    overwritten.

    See Also
    --------
    check_destructor

    """
    # Standard check
    rng = check_random_state(random_state)
    density = clone(density)  # Remove fitted properties if exist
    _check_estimator_but_ignore_warnings(density)

    # Attempt to get support (show warning since optional)
    support = get_support_or_default(density, warn=True)
    logger.info('(Before fitting) Density support is %s' % str(np.array(support).tolist()))

    # Sample based on support
    n = 10
    d = _get_support_n_features(support)
    X_train = _sample_demo(support, n, d, random_state=rng)
    X_test = _sample_demo(support, n, d, random_state=rng)

    # Check that fit, sample, score_samples and score are implemented
    try:
        density.fit(X_train)
    except ValueError:
        raise ValueError('The demo data raised a value error when calling fit. This should not'
                         ' happen if the data is within the density support via the'
                         ' get_support method or the default real-valued support.')
    support = get_support_or_default(density, warn=True)
    logger.info('(After fitting) Density support is %s' % str(np.array(support).tolist()))

    # Run primary optional tests (error on failure)
    if has_method(density, 'sample'):
        # Check multiple samples
        X_sample = density.sample(n, random_state=rng)
        if isinstance(X_sample, tuple) and len(X_sample) == 2:
            raise TypeError('Sample returned tuple of length 2 instead of data matrix.'
                            ' This is likely because the density estimator returned (X,y)'
                            ' instead of just X.')
        if np.any(X_sample.shape != X_train.shape):
            raise RuntimeError('Samples from density.sample() do not match training data shape.')

        # Check single sample
        X_1 = density.sample(1, random_state=rng)
        if len(np.array(X_1).shape) != 2:
            raise TypeError(
                'A single sample should still return a 2D matrix with shape (1,n_features).')

    score_vec = None
    if has_method(density, 'score_samples'):
        score_vec = density.score_samples(X_test)
        if len(score_vec) != n:
            raise RuntimeError('Output of score_samples is not of length n_samples.')

    # Run secondary optional tests (warning on failure)
    _check_score_method(density, score_vec, X_test)
    logger.info(_success('check_density'))
    return True


@_ignore_boundary_warnings
def check_destructor(destructor, fitted_density=None, is_canonical=True, properties_to_skip=None,
                     random_state=0):
    """Check for the required interface and properties of a destructor.

    First, check the interface of the destructor (see below) and then check
    the 4 properties of a destructor (see Inouye & Ravikumar 2018 paper). A
    destructor estimator should have the following interface:

    Required methods (error if fails or does not exist):

    * :meth:`fit`
    * :meth:`transform`
    * :meth:`inverse_transform`

    Optional methods primary (warn if does not exist, error if fails):

    * :meth:`sample` (mixin, uniform->inverse)
    * :meth:`score_samples`

    Optional methods secondary (warn if does not exist, warn if fails):

    * :meth:`fit_from_density` (better for uniformability test than fit)
    * :meth:`get_domain` (mixin with :attr:`density_`, required to pass tests if \
            domain is not real-valued unbounded)
    * :meth:`get_support` (required to pass tests if support is not \
                                                     real-valued unbounded)
    * :meth:`score` (mixin)

    Optional attributes secondary (warn if does not exist, warn if fails):

    * :attr:`density_` attribute (provides default for uniformability test if \
            fitted_density not given)

    In addition to checking the interface, this function will empirically
    check to see if the destructor seems to have the following 4 properties
    (see Inouye & Ravikumar 2018 paper).  Note that these are just empirical
    sanity checks based on sampling or the like and thus do not ensure that
    this estimator always has these properties.

    1. Uniformability (required)
    2. Invertibility (required)
    3. Canonical domain (optional if canonical = False)
    4. Identity element (optional if canonical = False)

    Parameters
    ----------
    destructor : estimator
        Destructor estimator to check.

    fitted_density : estimator, optional
        A fitted density estimator to use in some checks. If None (default),
        then use simple but interesting distributions based on domain of
        destructor. Can be used to check a specific density and destructor
        pair.

    is_canonical : bool, default=True
        Whether this destructor should be checked for canonical destructor
        properties including canonical domain and identity element.

    properties_to_skip : list, optional
        Optional list of properties to skip designated by the following
        strings ['uniformability', 'invertibility', 'canonical-domain',
        'identity-element']

    random_state : int, RandomState instance or None, optional (default=0)
        Note that random_state defaults to 0 so that checks are reproducible
        by default.

        If int, `random_state` is the seed used by the random number
        generator; If :class:`~numpy.random.RandomState` instance,
        `random_state` is the random number generator; If None, the random
        number generator is the :class:`~numpy.random.RandomState` instance
        used by :mod:`numpy.random`.

    See Also
    --------
    check_density

    Notes
    -----
    *Uniformability check* - Numerically *and* approximately checks one
    density/destructor pair to determine if the destructor appropriately
    destroys the density (i.e. transforms the samples to be uniform). Note
    that this does not mean it will work for all cases but it does provide a
    basic sanity check on at least one case.
    This will also run sklearn's estimator checks, i.e.
    :func:`sklearn.utils.estimator_checks.check_estimator`.

    *Invertibility check* - Simple numerical check for invertiblity by applying the
    transformation and then applying the inverse transformation, and
    vice versa to see if the original data is recovered.

    *Canonical domain check* - Simply check if the domain is the unit
    hypercube. Checks if warnings are issued when data is passed that is not
    on the unit hypercube.

    *Identity element check* - Numerical approximation for checking if the
    destructor class includes an identity element. Note this check fits the
    destructor on uniform samples and then check whether the learned
    transformation is the identity. Thus, if the test fails, there can be
    *two* possible causes: 1) The fitting procedure overfitted the uniform
    samples such that the implied density is far from uniform. 2) The
    transformation does not appropriately produce an identity
    transformation. This is a bit stricter than the official property since
    we train on uniform samples. However, this is probably a better check
    because we want the destructor to fit an identity transformation if the
    distribution is truly uniform.

    `destructor.score_samples(x)` = abs(det(J(x))) should be close
    to `density.score_samples(x)` and equal if no approximation (such as
    a piecewise linear function) is made.

    `destructor.sample(n_samples)` will produce slightly different samples than
    `destructor.density_.sample(n_samples)` unless no approximation is used.

    Other fitted parameters other than `density_` should be strictly
    destructive destructorformation parameters.

    Some of the property tests rely on calculating the Wasserstein distance
    between two sets of samples and thus the python optimal transport module
    (pot) is used.

    References
    ----------
    D. I. Inouye, P. Ravikumar. Deep Density Destructors. In International
    Conference on Machine Learning (ICML), 2018.

    """

    def _check_property(p):
        if p in properties_to_skip:
            warnings.warn(SkipTestWarning('Skipping checks for property "%s" because '
                                          '`properties_to_skip` included this property.' % p))
            return False
        return True

    possible_properties = ['uniformability', 'invertibility',
                           'canonical-domain', 'identity-element']
    if properties_to_skip is None:
        properties_to_skip = []
    properties_to_skip = [prop.lower() for prop in properties_to_skip]
    for prop in properties_to_skip:
        if prop not in possible_properties:
            raise ValueError(
                'Properties to skip must be one of %s but property to skip was %s.'
                % (prop, str(possible_properties))
            )

    # Check destructive transformer interface
    _check_destructor_interface(destructor, fitted_density, random_state=random_state)

    # Check destructive transformer properties
    if _check_property('uniformability'):
        _check_uniformability(destructor, fitted_density, random_state=random_state)
    if _check_property('invertibility'):
        _check_invertibility(destructor, random_state=random_state)

    if not is_canonical:
        logger.info('Not testing canonical destructive properties because parameter '
                    '`is_canonical=False`.')
    else:
        try:
            # Check canonical destructive transformer properties
            if _check_property('canonical-domain'):
                _check_canonical_domain(destructor, random_state=random_state)
            if _check_property('identity-element'):
                _check_identity_element(destructor, random_state=random_state)
        except Exception as e:
            warnings.warn('An exception occured while testing for the canonical properties of the '
                          'destructor. Please fix the errors below. (If this is not a '
                          'canonical destructor, please set the option `is_canonical=False`.)')
            raise e
    logger.info(_success('check_destructor'))
    return True


@_ignore_boundary_warnings
def _check_destructor_interface(trans, fitted_density=None, random_state=None):
    def _check_density_attr(T, method_name='fit'):
        if hasattr(T, 'density_'):
            logger.info(_checking('transformer.density_'))
            check_density(T.density_)
            if np.any(get_domain_or_default(T) != get_support_or_default(T.density_)):
                raise ValueError('Transformer domain does not match density support'
                                 ' when using method `%s`.' % method_name)
            return True
        else:
            warnings.warn(SkipTestWarning(
                'Fitting destructive transformer with `%s` did not create a "density_" attribute. '
                'Skipping tests that require the "density_" attribute.' % method_name))
            return False

    rng = check_random_state(random_state)
    trans = clone(trans)  # Remove fitted properties if exist
    _check_estimator_but_ignore_warnings(trans)

    # Attempt to get domain (show warning since optional)
    domain = get_domain_or_default(trans, warn=True)
    logger.info('Transformer domain is %s' % str(np.array(domain).tolist()))

    # Call get_domain and sample some demo data
    n = 10
    d = _get_domain_n_features(domain)
    X_train = _sample_demo(domain, n, d, random_state=rng)
    X_test = _sample_demo(domain, n, d, random_state=rng)
    X_train_copy = X_train.copy()  # Needed for checking mutability
    X_test_copy = X_test.copy()  # Needed for checking mutability

    # # Check required methods
    # Fit
    try:
        trans.fit(X_train)
    except ValueError:
        raise ValueError('The demo data raised a value error when calling fit. This should not'
                         ' happen if the data is within the transformer domain via the'
                         ' get_domain method or the default real-valued domain.')
    else:
        np.testing.assert_array_equal(X_train, X_train_copy,
                                      'fit() function should not mutate input data matrix `X`')
    has_density_attr = _check_density_attr(trans)

    # Check that transform and inverse_transform work
    U_sample = trans.transform(X_test)
    np.testing.assert_array_equal(X_test, X_test_copy,
                                  'transform() function should not mutate input data matrix `X`')

    U_sample_copy = U_sample.copy()
    trans.inverse_transform(U_sample)
    np.testing.assert_array_equal(U_sample, U_sample_copy,
                                  'inverse_transform() function should not mutate input data '
                                  'matrix `X`')

    # # Check primary methods
    if has_method(trans, 'sample'):
        X_trans_sample = trans.sample(n, random_state=rng)
        if np.any(X_trans_sample.shape != X_train.shape):
            raise RuntimeError('Transformer samples do not have the same shape as the training '
                               'data.')

    score_vec = None
    if has_method(trans, 'score_samples'):
        score_vec = trans.score_samples(X_test)
        np.testing.assert_array_equal(X_test, X_test_copy,
                                      'score_samples() function should not mutate input data '
                                      'matrix `X`')
        if len(score_vec) != n:
            raise RuntimeError('Output of score_samples is not of length n')

    # # Check secondary methods
    # Check fit_from_density
    T2 = clone(trans)
    if has_method(T2, 'fit_from_density'):
        temp_dens = None
        if fitted_density is not None:
            logger.info(_checking('fitted_density'))
            check_density(fitted_density, random_state=rng)
            temp_dens = fitted_density
        elif has_density_attr:
            temp_dens = trans.density_

        if temp_dens is None:
            warnings.warn('Not testing `fit_for_density` because parameter `fitted_density`'
                          ' = None and `fit` did not provide `density_` property.')
        else:
            try:
                T2.fit_from_density(temp_dens)
            except ValueError:
                raise ValueError('The demo data raised a value error when calling '
                                 '`fit_from_density`. This should not happen if the data is '
                                 'within the transformer domain via the get_domain method or the '
                                 'default real-valued domain.')
            _check_density_attr(T2, method_name='fit_from_density')

    # Check that transformer "log-likelihood" is close to density log-likelihood
    if has_density_attr:
        dens_score_vec = trans.density_.score_samples(X_test)
        # Normalize by both number of dimensions (d) and by max likelihood from density
        avg_normalized_diff = np.mean(
            np.abs(np.exp(score_vec / d) - np.exp(dens_score_vec / d))
            / np.max(np.exp(dens_score_vec / d))
        )
        warn_thresh = 0.01
        if avg_normalized_diff > warn_thresh:
            warnings.warn(
                'Apparent mismatch between transformer log-likelihood (via `score_samples`)'
                ' and density log-likelihood;'
                ' Either there is an error in the log likelihood calculations'
                ' or the approximation in the transformer is bad. Average normalized'
                ' exp(score) = likelihood difference is %g (threshold is %g)'
                % (float(avg_normalized_diff), warn_thresh))
        logger.info(
            'Average normalized difference between density and transformer `score_samples`: %g'
            % avg_normalized_diff)

    _check_score_method(trans, score_vec, X_test)
    logger.info(_success('_check_destructor_interface'))
    return True


@_ignore_boundary_warnings
def _check_uniformability(destructor, fitted_density=None, random_state=0):
    # Init
    n = 100
    destructor = clone(destructor)
    rng = check_random_state(random_state)

    # Create a dummy density by fitting a transformer on demo data and getting the density_
    if fitted_density is not None:
        true_density = fitted_density
    else:
        logger.info('Fitting a simple density because `fitted_density` was not provided to '
                    '`check_uniformabilty`')
        # Sample demo data based on domain
        trans_temp = clone(destructor)
        domain = get_domain_or_default(trans_temp)
        d = _get_domain_n_features(domain)
        n_train = n
        X_temp = _sample_demo(domain, n_train, d, random_state=rng)

        # Fit transformer
        fitted_trans = trans_temp.fit(X_temp)

        # Check and extract "true" density from transformer
        if hasattr(fitted_trans, 'density_'):
            true_density = fitted_trans.density_
        else:
            raise RuntimeError('Parameter `fitted_density` is None and fitted destructor'
                               ' did not provide `density_` attribute'
                               ' so unable to continue with uniformability test.')

    # Extract number of dimensions
    n_features = np.array(true_density.sample(1, random_state=rng)).shape[1]

    # Fit transformer ideally via `fit_from_density` if it exists
    if has_method(destructor, 'fit_from_density'):
        destructor.fit_from_density(true_density)
    else:
        X_train = true_density.sample(n, random_state=rng)
        destructor.fit(X_train)
        msg = ('Because the `trans.fit_from_density` method does not exist, fitting the '
               'destructor on a separate training set. Note: This is not ideal for numerically '
               'checking uniformability because there are two sources of variability (training '
               'algorithm in `fit` and sampling rather than just sampling.')
        warnings.warn(msg)

    k = 30

    fitted_trans_domain = get_domain_or_default(destructor)
    logger.info('Fitted transformer domain is %s' % str(np.array(fitted_trans_domain).tolist()))

    def _transformer_emd(_trans, _true_density):
        # Sample from true densities
        X_true = _true_density.sample(n, random_state=rng)
        U_true = rng.rand(n, n_features)

        # Apply transforms to get (approximate) samples from the other distribution
        X_trans = _trans.inverse_transform(U_true)
        U_trans = _trans.transform(X_true)

        # Check output ranges
        _assert_X_in_interval(X_trans, fitted_trans_domain)
        _assert_X_in_interval(U_trans, np.array([0, 1]))

        # Check that inverse_transform domain is 0, 1
        _assert_unit_domain(rng, _trans.inverse_transform, n_features=n_features)

        # Compute emd between the transformed samples and independent samples from the true
        # distribution
        _X_emd = _compute_emd(X_trans, X_true)
        _U_emd = _compute_emd(U_trans, U_true)
        return _X_emd, _U_emd

    temp = np.array([
        _transformer_emd(destructor, true_density)
        for _ in range(k)
    ]).transpose()
    X_emd_vec = temp[0]
    U_emd_vec = temp[1]

    # Get samples of emd scores to compute a percentile
    X_emd_true_vec = _emd_sample(true_density, n, k, random_state=rng)
    U_emd_true_vec = _emd_sample(UniformDensity().fit(np.zeros((1, n_features))), n, k,
                                 random_state=rng)

    # Check whether these look like good samples
    def _p_val(val, true_vec):
        return np.sum(true_vec > val) / len(true_vec)

    def _avg_p_val(vec, true_vec):
        return np.mean([
            _p_val(val, true_vec)
            for val in vec
        ])

    X_avg_p_val = _avg_p_val(X_emd_vec, X_emd_true_vec)
    U_avg_p_val = _avg_p_val(U_emd_vec, U_emd_true_vec)
    logger.info('X_sampled average p value = %g (higher better in this case)'
                % float(X_avg_p_val))
    logger.info('U_sampled average p value = %g (higher better in this case)'
                % float(U_avg_p_val))

    # Check p-values
    avg_p_threshold = np.maximum(0.01, 1.0 / k)
    err_msg = ('The `trans.%s` of %s samples does not seem to yield samples from '
               'the %s density at the p value threshold of %g (i.e. the null hypothesis that the '
               'distributions are the same can be rejected with high confidence). This may be '
               'caused by a problem in `trans.%s`, `fitted_density.sample`, or too high '
               'of a p-value threshold.')
    if X_avg_p_val <= avg_p_threshold:
        def _plot_data_for_debug():
            n_samples = 1000
            X_true = true_density.sample(n_samples, random_state=rng)
            U_true = rng.rand(n_samples, n_features)
            X_trans = destructor.inverse_transform(U_true)
            U_trans = destructor.transform(X_true)

            # noinspection PyPackageRequirements
            import matplotlib.pyplot as plt
            axes = plt.subplots(2, 2)[1]
            for ax, title, X in zip(axes.ravel(),
                                    ['X_true', 'U_true', 'X_trans', 'U_trans'],
                                    [X_true, U_true, X_trans, U_trans]):
                ax.scatter(X[:, 0], X[:, 1], s=3)
                ax.axis('equal')
                ax.set_title(title)
            plt.show(block=False)
            raise _UniformabilityError(
                err_msg % ('inverse_transform', 'uniform', 'original (assumed)', avg_p_threshold,
                           'inverse_transform')
            )

        _plot_data_for_debug()
    if U_avg_p_val <= avg_p_threshold:
        raise _UniformabilityError(
            err_msg % ('transform', 'original (assumed)', 'uniform', avg_p_threshold,
                       'transform')
        )
    logger.info(_success('_check_uniformability'))
    return True


@_ignore_boundary_warnings
def _check_invertibility(destructor, random_state=0):
    # Sample and fit transformer
    destructor = clone(destructor)
    rng = check_random_state(random_state)
    domain = get_domain_or_default(destructor, warn=True)
    n = 100
    d = _get_support_n_features(domain)
    X_train = _sample_demo(domain, n, d, random_state=rng)
    destructor.fit(X_train)

    # Get demo samples and transform or inverse transform in both directions
    X_orig = _sample_demo(domain, n, d, random_state=rng)
    U_trans = destructor.transform(X_orig)
    X_back = destructor.inverse_transform(U_trans)

    U_orig = rng.rand(n, d)
    X_trans = destructor.inverse_transform(U_orig)
    U_back = destructor.transform(X_trans)

    # Check that the original and back transformed are nearly (numerically) the same
    def _check_nearly_equal(orig, back, middle):
        rel_diff = _relative_diff(orig, back)
        logger.info('Relative diff for invertibility check = %g' % rel_diff)
        if rel_diff > 1e-14:
            warnings.warn('Relative diff for invertibility check is larger than 1e-14. This may '
                          'be a stability issue.')
        if rel_diff > 1e-9:
            logger.debug(orig)
            logger.debug(middle)
            logger.debug(back)

            def _plot_debug_data():
                # noinspection PyPackageRequirements
                import matplotlib.pyplot as plt
                n_samples = 1000
                X_true = destructor.sample(n_samples=n_samples)
                U_true = rng.rand(n_samples, d)
                _X_trans = destructor.inverse_transform(U_true)
                _U_trans = destructor.transform(X_true)

                axes = plt.subplots(2, 2)[1]
                for ax, title, X in zip(axes.ravel(),
                                        ['X_true', 'U_true', 'X_trans', 'U_trans'],
                                        [X_true, U_true, _X_trans, _U_trans]):
                    ax.scatter(X[:, 0], X[:, 1], s=3)
                    ax.axis('equal')
                    ax.set_title(title)
                plt.show(block=False)

                axes = plt.subplots(2, 2)[1]
                for ax, title, X in zip(axes.ravel(),
                                        ['orig', 'middle', 'back'],
                                        [orig, middle, back]):
                    ax.scatter(X[:, 0], X[:, 1], s=3)
                    ax.axis('equal')
                    ax.set_title(title)
                plt.show(block=False)
                raise _InvertibilityError(
                    'Transforming and then inverse transforming does not seem to '
                    'give the original data. Relative difference = %g.' % rel_diff)

            _plot_debug_data()

    _check_nearly_equal(X_orig, X_back, U_trans)
    _check_nearly_equal(U_orig, U_back, X_trans)
    logger.info(_success('_check_invertibility'))
    return True


@_ignore_boundary_warnings
def _check_canonical_domain(destructor, random_state=0):
    # Setup functions to test
    rng = check_random_state(random_state)
    fitted = clone(destructor).fit(np.array([[0], [1], [0], [1], [0], [1]]))

    _assert_unit_domain(rng, clone(destructor).fit)
    _assert_unit_domain(rng, fitted.transform)

    if has_method(fitted, 'score', warn=False):
        _assert_unit_domain(rng, fitted.score)
    if has_method(fitted, 'score_samples', warn=False):
        _assert_unit_domain(rng, fitted.score_samples)

    logger.info(_success('_check_canonical_domain'))
    return True


@_ignore_boundary_warnings
def _check_identity_element(destructor, random_state=0):
    # Setup
    rng = check_random_state(random_state)
    domain = get_domain_or_default(destructor, warn=True)
    n = 1000
    d = _get_support_n_features(domain)

    # Sample uniform samples in order to fit destructor
    X_train = rng.rand(n, d)
    destructor.fit(X_train)

    # Transform data
    X = rng.rand(n, d)
    U = destructor.transform(X)

    diff = _relative_diff(X, U)
    if diff > 0.02:  # 2% movement of particles on average
        raise _IdentityElementError('Given that the transformer was trained on uniform samples, '
                                    'the transformation does not appear to be the identity. There '
                                    'are *two* possible causes: 1) The fitting procedure '
                                    'overfitted the uniform samples such that the implied density '
                                    'is far from uniform. 2) The transformation does not '
                                    'appropriately produce an identity transformation. Relative '
                                    'diff = %g' % diff)
    logger.info('Relative difference after fitting and then transforming uniform samples (shape '
                '%s) = %g' % (str(X_train.shape), diff))
    logger.info(_success('_check_identity_element'))
    return True


class _DestructorError(RuntimeError):
    """Destructor property error."""
    pass


class _UniformabilityError(_DestructorError):
    """Uniformability property error."""
    pass


class _InvertibilityError(_DestructorError):
    """Invertibility property error."""
    pass


class _CanonicalDomainError(_DestructorError):
    """Canonical domain property error."""
    pass


class _IdentityElementError(_DestructorError):
    """Identity element property error."""
    pass


def _sample_demo(support, n_samples, n_features, random_state=None):
    """Sample demo dataset based on support of density."""
    # Update to the number of dimensions as needed
    assert n_samples > 2, 'n_samples should be greater than 2 so that at' \
                          'least endpoints can be given supplied'
    rng = check_random_state(random_state)
    support = check_domain(support, n_features)
    X = np.vstack((
        rng.randn(n_samples) * 10
        if s[0] == -np.inf and s[1] == np.inf
        else np.hstack((s[0] + (s[1] - s[0]) * rng.beta(1, 0.5, size=n_samples - 2), s))
        if s[0] != -np.inf and s[1] != np.inf
        else np.hstack((rng.exponential(scale=10, size=n_samples - 1) + s[0], [s[0]]))
        if s[0] != -np.inf and s[1] == np.inf
        else np.hstack((s[1] - rng.exponential(scale=10, size=n_samples - 1), [s[1]]))
        if s[0] == -np.inf and s[1] != np.inf
        else np.nan  # Triggers error below since cannot explicitly raise error here
        for s in support
    )).transpose()
    if len(X.shape) == 1:
        raise ValueError('Given support is not implemented; only real-valued, left-bounded, '
                         'right-bounded, and bounded are implemented.')
    if np.any(np.isnan(X)):
        raise ValueError('Samples have NaN values.')
    if np.any(X.shape != np.array([n_samples, n_features])):
        raise RuntimeError('Demo data is different than specified shape.')
    return X


def _emd_sample(density, n_samples, n_emd_samples, random_state=None):
    """Computes samples of earth-mover distance (emd) between independent same-size
    samples of given density. Useful for statistically estimating whether a sample
    comes from the given density or not.
    """
    rng = check_random_state(random_state)
    X_arr = []
    i_sample = 0
    emd_sample = np.zeros(n_emd_samples)
    # Generate new samples only if needed for n_emd_samples
    while i_sample < n_emd_samples:
        X = density.sample(n_samples, random_state=rng)
        # Compare to all previous samples
        for X_other in X_arr:
            emd_sample[i_sample] = _compute_emd(X, X_other)
            i_sample += 1
            # Break if all needed samples have been computed
            if i_sample == n_emd_samples:
                return emd_sample
        X_arr.append(X)
    raise RuntimeError('Should have returned inside the nested loop. Must be bug in code.')


def _compute_emd(X1, X2):
    # Setup
    if np.any(X1.shape != X2.shape):
        raise ValueError('X1 and X2 should have the same shape.')
    n, d = X1.shape
    a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples

    # Compute cost matrix
    M = ot.dist(X1, X2, 'euclidean')
    M /= M.max()

    return ot.emd2(a, b, M)


def _check_score_method(est, score_vec, X_sample):
    if has_method(est, 'score'):
        X_sample_copy = X_sample.copy()
        score = est.score(X_sample)
        np.testing.assert_array_equal(X_sample, X_sample_copy,
                                      'score() function should not mutate input data matrix `X`')

        if score_vec is not None:
            # 1e-100 to avoid division by 0
            mean_diff = np.abs((score - np.mean(score_vec)) / np.maximum(np.abs(score), 1e-100))
            sum_diff = np.abs((score - np.sum(score_vec)) / np.maximum(np.abs(score), 1e-100))
            if mean_diff > 1e-14:
                if sum_diff < 1e-14:
                    warnings.warn(DeprecationWarning(
                        'It seems that the score method returns a sum of per-sample '
                        'log-likelihood. The current best practice is to return the average '
                        'per-sample log-likelihood. relative_sum_diff = %g, relative_mean_diff = %g'
                        % (sum_diff, mean_diff)
                    ))
                else:
                    warnings.warn(
                        'The est.score() method does not return the average log-likelihood. For '
                        'standardized estimators, the score method should be the average '
                        'per-sample log-likelihood. relative_mean_diff = %g' % mean_diff
                    )
        else:
            warnings.warn('Could not check score method output because score_samples not '
                          'implemented')


def _check_support(support):
    """Check support has dimension either (2,) or (n_features,2)."""
    support = np.array(support)
    if len(support.shape) == 1 and support.shape[0] == 2:
        return support
    elif len(support.shape) == 2 and support.shape[1] == 2:
        return support
    else:
        raise ValueError('The shape of support should either be (2,) or (n_features,2) but shape ')


def _get_support_n_features(support, default_n_features=2):
    """Gets the n_features based on support or returns the default"""
    support = _check_support(support)
    if len(support.shape) == 1:
        return default_n_features
    elif len(support.shape) == 2:
        return support.shape[0]


# Alias since domain and support are handled the same
_get_domain_n_features = _get_support_n_features


def _relative_diff(orig, back):
    return np.linalg.norm(np.abs(orig - back), ord='fro') / np.linalg.norm(orig, ord='fro')


def _clean_warning_registry():
    """
    (Copied from module `sklearn.utils.testing` (v. 0.19.1).)
    Safe way to reset warnings.
    """
    warnings.resetwarnings()
    reg = "__warningregistry__"
    for mod_name, mod in list(sys.modules.items()):
        if 'six.moves' in mod_name:
            continue
        if hasattr(mod, reg):
            getattr(mod, reg).clear()


def _assert_no_warnings(func, *args, allow_these_warnings=None, **kw):
    """
    (Edited from module `sklearn.utils.testing` (v. 0.19.1).)
    Asserts that no warnings are issued when calling function `func`.
    """
    if allow_these_warnings is None:
        allow_these_warnings = [BoundaryWarning, _ShouldOnlyBeInTestWarning]
    _clean_warning_registry()
    with warnings.catch_warnings(record=True) as w_arr:
        warnings.simplefilter('always')
        result = func(*args, **kw)
    # Only keep warning if not in allowed list
    w_arr = [w for w in w_arr
             if not np.any([isinstance(w.message, allowed) for allowed in allow_these_warnings])]
    if len(w_arr) > 0:
        raise AssertionError("Got warning(s) when calling %s: [%s]"
                             % (func.__name__,
                                ', '.join(str(w) for w in w_arr)))
    return result


def _assert_warns(warning_class, func, *args, **kw):
    """
    (Edited from module `sklearn.utils.testing` (v. 0.19.1).)
    Test that a certain warning occurs.
    Parameters
    ----------
    warning_class : the warning class
        The class to test for, e.g. UserWarning.
    func : callable
        Calable object to trigger warnings.
    *args : the positional arguments to `func`.
    **kw : the keyword arguments to `func`
    Returns
    -------
    result : the return value of `func`
    """
    # very important to avoid uncontrolled state propagation
    _clean_warning_registry()
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        result = func(*args, **kw)
        # Verify some things
        if not len(w) > 0:
            raise AssertionError("No warning raised when calling %s"
                                 % func.__name__)

        found = any(warning.category is warning_class for warning in w)
        if not found:
            raise AssertionError("%s did not give warning: %s( is %s)"
                                 % (func.__name__, warning_class, w))
    return result


def _assert_X_in_interval(X, interval):
    return _assert_no_warnings(check_X_in_interval, X, interval)


def _assert_unit_domain(rng, func, n_features=1):
    # Setup test cases
    X_good = [
        a * np.ones((10, n_features))
        for a in (0, 1, 0.5)
    ]
    X_good.append(rng.rand(1000, n_features))

    X_bad = [
        a * np.ones((10, n_features))
        for a in (2, 1.1, -0.1, -1, -2, -1e5, 1e5)
    ]

    # Attempt to fit data inside the canonical domain (should not raise warning)
    for X in X_good:
        _assert_no_warnings(func, X,
                            allow_these_warnings=[BoundaryWarning, _ShouldOnlyBeInTestWarning,
                                                  _NumDimWarning])

    # Attempt to fit data outside the canonical domain (should raise warning)
    for X in X_bad:
        _assert_warns(DataConversionWarning, func, X)

    return True


def _success(name):
    return 'Sucessfully passed `%s`' % name


def _checking(name):
    return 'Checking `%s`' % name


def _check_estimator_but_ignore_warnings(est):
    with warnings.catch_warnings(record=True) as w_arr:
        check_estimator(est)
    # Show warnings that are not data conversion warnings
    for w in w_arr:
        if isinstance(w.message, SkipTestWarning):
            warnings.warn(w.message)

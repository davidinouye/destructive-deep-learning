"""Module to handle linear projectors and destructors."""
from __future__ import division, print_function

import logging
import warnings

import numpy as np
import scipy.stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted

from .base import (CompositeDestructor, ScoreMixin, create_implicit_density,
                   create_inverse_transformer)
from .independent import IndependentDensity, IndependentDestructor, IndependentInverseCdf
from .univariate import ScipyUnivariateDensity
# noinspection PyProtectedMember
from .utils import _INF_SPACE

logger = logging.getLogger(__name__)


class LinearProjector(BaseEstimator, ScoreMixin, TransformerMixin):
    """A linear projector based on an underlying linear estimator.

    (Somewhat like a *relative* destructor with an implicit but unknown
    underlying density.)

    A few notes:
        1. By construction, `LinearProjector` is density-agnostic
        and therefore is not a valid destructor.
        2. However, if attached before any valid destructor, the
        joint transformer is a valid destructor, which implies a
        valid joint density.
        3. Thus, `LinearProjector` can be seen as a *relative*
        destructor that requires a base density to be useful.

    Parameters
    ----------
    linear_estimator : estimator, default=IdentityLinearEstimator
        A linear estimator that has either a `coef_` attribute or a
        `components_`. For example, :class:`sklearn.decomposition.PCA`.

    orthogonal : bool, default=False
        Whether to issue a warning if the matrix is not orthogonal.

    fit_bias : bool, default=True
        Whether to fit the bias term, i.e., the b of the linear transform
        y = Ax + b.

    save_linear_estimator : bool, default=False
        Whether to save the fitted linear estimator or not (mainly used
        for debugging).

    Attributes
    ----------
    A_ : object
        Representation of linear transformation, i.e. a matrix. The linear
        object that implements :func:`dot(X)`, :func:`logabsdet`,
        and :func:`inv`. Note that this can be a compact representation of a
        matrix such as merely storing the diagonal elements or a Householder
        transformation matrix.

    A_inv_ :
        Representation of the inverse linear transformation, i.e. a matrix.
        The linear object that implements :func:`dot(X)`, :func:`logabsdet`,
        and :func:`inv`. Note that this can be a compact representation of a
        matrix such as merely storing the diagonal elements or a Householder
        transformation matrix.

    """

    def __init__(self, linear_estimator=None, orthogonal=False, fit_bias=True,
                 save_linear_estimator=False):
        self.linear_estimator = linear_estimator
        self.orthogonal = orthogonal
        self.fit_bias = fit_bias
        self.save_linear_estimator = save_linear_estimator

    def fit(self, X, y=None, lin_est_fit_params=None, fitted_lin_est=None):
        """[Placeholder].

        Parameters
        ----------
        X :
        y :
        lin_est_fit_params :
        fitted_lin_est :

        Returns
        -------
        obj : object

        """
        # Find linear projection, project X (implicitly checks X)
        if fitted_lin_est is not None:
            lin_est = fitted_lin_est
            warnings.warn(DeprecationWarning('Class factory method `create_fitted` '
                                             'should be used instead.'))
        else:
            if lin_est_fit_params is None:
                lin_est_fit_params = {}
            if not isinstance(self.orthogonal, bool):
                raise ValueError('Parameter `orthogonal` should be a boolean value either '
                                 '``True`` or ``False``.')
            lin_est = (self.linear_estimator if self.linear_estimator is not None
                       else IdentityLinearEstimator())
            # noinspection PyArgumentList
            try:
                lin_est.fit(X, y, **lin_est_fit_params)
            except ValueError as e:
                # If univariate give default if fitting fails (e.g. FastICA)
                if np.array(X).shape[1] == 1:
                    lin_est.coef_ = np.array([1])
                    lin_est.intercept_ = np.array([0])
                else:
                    raise e

        # Extract from estimator and create matrix
        A, b = self._estimator_to_A_b(lin_est, self.orthogonal)
        if not self.fit_bias:
            b = np.zeros_like(b)

        if self.save_linear_estimator:
            self.fitted_linear_estimator_ = lin_est
        self.A_ = A
        self.A_inv_ = A.inv(copy=False)
        self.b_ = b
        self.n_features_ = A.shape[0]
        return self

    @classmethod
    def create_fitted(cls, A=None, b=None, coef=None, intercept=None,
                      fitted_linear_estimator=None, **kwargs):
        """Create fitted linear projector.

        Must provide (`A` and `b`) OR (`coef` and `intercept`)
        OR (`fitted_linear_estimator`) but only one combination
        should be provided.

        Parameters
        ----------
        A : array-like, shape (n_features, n_features), optional
            Matrix for projection.

        b : array-like, shape (n_features,), optional
            Shift vector for linear projection.

        coef : array-like, shape (n_features,), optional
            Vector of coefficents, will form scaled Householder reflector.

        intercept : float, optional
            Intercept for 1D linear projection.

        fitted_linear_estimator : fitted estimator, optional
            Must be a linear estimator that has `coef_` or `components_`
            fitted parameters.

        **kwargs
            Other parameters to pass to constructor.

        Returns
        -------
        fitted_transformer : Transformer
            Fitted transformer.

        """
        opt1 = A is not None and b is not None
        opt2 = coef is not None and intercept is not None
        opt3 = fitted_linear_estimator is not None
        if not (opt1 or opt2 or opt3):
            raise ValueError('Must supply one valid combination of fitted '
                             'parameters.')
        if np.sum([opt1, opt2, opt3]) > 1:
            raise ValueError('Cannot provide more than one combination of '
                             'fitted parameters.')

        projector = cls(**kwargs)

        if opt1:
            A, b = np.array(A), np.array(b)
            assert A.shape[0] == A.shape[1], 'Only square A are allowed currently.'
            assert b.ndim == 1 and b.shape[0] == A.shape[0], 'b should be (n_features,)'
            A = cls._coef_to_A(A, projector.orthogonal)
            b = b
        elif opt2:
            assert np.array(coef).ndim == 1, 'coef should be one dimensional array-like'
            intercept = np.array(intercept)
            if intercept.ndim > 1:
                raise ValueError('intercept should be scalar or 1 dimensional')
            elif intercept.ndim == 1:
                assert intercept.shape[0] == 1, 'intercept must be shape (1,) or scalar'
                intercept = intercept[0]
            A = cls._coef_to_A(coef, projector.orthogonal)
            n_features = len(coef)
            b = np.zeros(n_features)
            b[0] = intercept
        elif opt3:
            A, b = cls._estimator_to_A_b(fitted_linear_estimator, projector.orthogonal)
        else:
            raise RuntimeError('This should not be reached since checks '
                               'should raise error before')

        # Convert vector or matrix to matrix
        projector.A_ = A
        projector.A_inv_ = A.inv(copy=False)
        projector.b_ = b
        projector.n_features_ = A.shape[0]
        return projector

    def score_samples(self, X, y=None):
        """Compute log(det(Jacobian)) for each sample.

        Note that this is not the log-likelihood since this is not a
        destructor. This score is constant for all samples because the
        Jacobian is constant for linear operations and depends only on the
        transformation matrix.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples and n_features
            is the number of features.

        y : None, default=None
            Not used but kept for compatibility.

        Returns
        -------
        log_det_jacobian : array, shape (n_samples,)
            Log determinant of the Jacobian for each data point in X.

        """
        X = check_array(X)
        return self.A_.logabsdet() * np.ones((X.shape[0],))

    def transform(self, X, y=None):
        """Apply destructive transformation to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : None, default=None
            Not used in the transformation but kept for compatibility.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_features)
            Transformed data.

        """
        self._check_is_fitted()
        X = check_array(X)
        return self.A_.dot(X.T).T + self.b_

    def inverse_transform(self, X, y=None):
        """Apply inverse destructive transformation to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : None, default=None
            Not used in the transformation but kept for compatibility.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_features)
            Transformed data.

        """
        self._check_is_fitted()
        X = check_array(X)
        return self.A_inv_.dot((X - self.b_).T).T

    def get_domain(self):
        """Get the domain of this destructor.

        Returns
        -------
        domain : array-like, shape (2,) or shape (n_features, 2)
            If shape is (2, ), then ``domain[0]`` is the minimum and
            ``domain[1]`` is the maximum for all features. If shape is
            (`n_features`, 2), then each feature's domain (which could
            be different for each feature) is given similar to the first
            case.

        """
        return _INF_SPACE

    def _check_is_fitted(self):
        check_is_fitted(self, ['A_', 'A_inv_', 'b_'])

    @staticmethod
    def _coef_to_A(coef, orthogonal=False):
        # Create matrix object for projections
        # shape is (n_features,) or (n_components, n_features)
        coef = np.array(coef)

        if len(coef.shape) == 1 or coef.shape[0] == 1:
            w = np.array(coef.ravel(), dtype=np.float)  # Make vector
            w_norm = np.linalg.norm(w)
            if orthogonal:
                scale = 1
            else:
                scale = w_norm
            if w.shape[0] == 1:
                # Univariate case
                A = _IdentityWithScaling(n_features=1, scale=scale)
            elif np.all(w == np.eye(len(w), 1).ravel()):
                # Special case when w = e_1 (i.e. undefined reflection vector)
                A = _IdentityWithScaling(n_features=len(w), scale=scale)
            else:
                # Compute w - e_1
                #  (i.e. difference between target vector and standard basis [1,0,0,...])
                #  u = w - norm(w)*e_1
                #  u = u/norm(u)
                u = w.copy()
                u[0] -= w_norm
                u_norm = np.linalg.norm(u)
                u /= u_norm
                A = _HouseholderWithScaling(u, scale=scale, copy=False)
        elif coef.shape[0] != coef.shape[1]:
            raise NotImplementedError(
                'Projectors not implemented for 1 < n_components < n_features. '
                'Probably a series of Householder reflectors would be '
                'computationally the best.')
        else:
            if orthogonal:
                # Check to make sure provided matrix is orthogonal
                _, logdet = np.linalg.slogdet(coef)
                if logdet > 1e-12:
                    warnings.warn(
                        'Provided matrix does not seem to be orthogonal but orthogonal=True. '
                        'Non-orthogonal matrices are not converted automatically. abs(logdet)=%g'
                        % logdet)
            A = _SimpleMatrix(coef)
        return A

    @staticmethod
    def _estimator_to_A_b(lin_est, orthogonal=False):
        # Extract A
        try:
            # For sklearn.linear_model estimators such as LinearRegression, LogisticRegression, etc.
            coef = lin_est.coef_
        except AttributeError:
            try:
                # For PCA, ICA and possibly other decomposition methods
                # shape is (n_components, n_features)
                coef = lin_est.components_
            except AttributeError:
                raise ValueError('After fitting, the linear estimator does not have attribute '
                                 'coef_ or components_')
        A = LinearProjector._coef_to_A(coef, orthogonal)

        # Extract b
        try:
            intercept = lin_est.intercept_
        except AttributeError:
            try:
                b = lin_est.mean_  # For PCA
            except AttributeError:
                b = np.zeros(A.shape[0])
        else:
            b = np.zeros(A.shape[0])
            assert len(intercept) == 1, 'intercept should have shape (1,)'
            b[0] = intercept[0]
        b = np.array(b)

        return A, b


class BestLinearReconstructionDestructor(CompositeDestructor):
    """Best linear reconstruction after a linear destructor.

    Class that converts a linear -> destructor combination into a
    combination that returns the data to as close to the original space as
    possible. Essentially, linear -> destructor -> independent Gaussian
    inverse cdf -> inverse linear -> independent Gaussian cdf. For example,
    if the linear projector was PCA and the destructor was a independent
    Gaussian, then this would correspond to ZCA whitening.

    Parameters
    ----------
    linear_estimator : estimator, default=IdentityLinearEstimator
        A linear estimator that has either a `coef_` attribute or a
        `components_`. For example, :class:`sklearn.decomposition.PCA`.

    destructor : estimator
        Density destructor to use in between linear projections.

    linear_projector_kwargs : dict
        Keyword arguments to pass when constructing
        ``ddl.linear.LinearProjector``.

    Attributes
    ----------
    fitted_destructors_ : list
        List of fitted (sub)destructors. (Note that these objects are cloned
        via ``sklearn.base.clone`` from the ``destructors`` parameter so as
        to avoid mutating the ``destructors`` parameter.)

    density_ : estimator
        *Implicit* density of composite destructor.

    """

    def __init__(self, linear_estimator=None, destructor=None, linear_projector_kwargs=None):
        super(BestLinearReconstructionDestructor, self).__init__()
        self.linear_estimator = linear_estimator
        self.destructor = destructor
        self.linear_projector_kwargs = linear_projector_kwargs

    def fit_transform(self, X, y=None, **kwargs):
        """[Placeholder].

        Parameters
        ----------
        X :
        y :
        kwargs :

        Returns
        -------
        obj : object

        """
        lin_est, destructor = self._get_estimators_or_default()
        if self.linear_projector_kwargs is None:
            projector_kwargs = dict()
        else:
            projector_kwargs = self.linear_projector_kwargs
        if 'orthogonal' in projector_kwargs and not projector_kwargs['orthogonal']:
            warnings.warn(UserWarning('Overriding orthogonal to be True.'))
        projector_kwargs['orthogonal'] = True
        assert 'linear_estimator' not in projector_kwargs, ('`linear_estimator` should not be '
                                                            'set via `linear_projector_kwargs`')
        lin_proj = LinearProjector(linear_estimator=lin_est, **projector_kwargs)

        # Note only these two need to be fit, the others are fixed or related to these
        X = check_array(X, copy=True)
        X_proj = lin_proj.fit_transform(X, y)
        destructor.fit(X_proj, y)

        # Copy linear but fit as inverse
        lin_proj_inv = create_inverse_transformer(lin_proj)

        # Fit an independent inverse cdf
        ind_inverse_cdf = IndependentInverseCdf().fit(X, y)
        standard_normal = IndependentDestructor(
            independent_density=IndependentDensity(
                univariate_estimators=ScipyUnivariateDensity(
                    scipy_rv=scipy.stats.norm,
                    scipy_fit_kwargs=dict(floc=0, fscale=1),
                )
            )
        ).fit(X, y)  # Note that since this is fixed only X.shape matters

        # Setup fitted destructors (the only thing needed for CompositeDestructor)
        self.fitted_destructors_ = [
            lin_proj,
            destructor,
            ind_inverse_cdf,
            lin_proj_inv,
            standard_normal,
        ]
        self.density_ = create_implicit_density(self)

        # Transform original X now
        Z = self.transform(X, y)
        return Z

    def _get_estimators_or_default(self):
        lin_est = clone(self.linear_estimator) if self.linear_estimator is not None else PCA()
        destructor = clone(
            self.destructor) if self.destructor is not None else IndependentDestructor()
        return lin_est, destructor


class IdentityLinearEstimator(BaseEstimator):
    """Identity linear projection.

    Attributes
    ----------
    coef_ : array-like, shape (n_features, n_features)
        Just sets the identity matrix

    """

    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        """Fit estimator to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : None, default=None
            Not used in the fitting process but kept for compatibility.

        fit_params : dict, optional
            Optional extra fit parameters.

        Returns
        -------
        self : estimator
            Returns the instance itself.

        """
        X = check_array(X)
        # This will become the identity matrix in LinearProjector
        self.coef_ = np.eye(X.shape[1], 1).ravel()
        # Added so that it is compatible with more functions
        self.intercept_ = np.zeros(1)
        return self


class RandomOrthogonalEstimator(BaseEstimator):
    """Random linear orthogonal estimator.

    Parameters
    ----------
    n_components : int, default=n_features
        Number of components to return from random orthogonal estimator. If
        `n_components` is 1, then a random direction is returned. Otherwise,
        return an truncated orthogonal matrix by only returning the first
        `n_components` columns.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, `random_state` is the seed used by the random number
        generator; If :class:`~numpy.random.RandomState` instance,
        `random_state` is the random number generator; If None, the random
        number generator is the :class:`~numpy.random.RandomState` instance
        used by :mod:`numpy.random`.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Random matrix that is orthogonal if n_components = n_features.
        Otherwise, an orthogonal matrix that has been truncated to the first
        `n_components` columns.

    """

    def __init__(self, n_components=None, random_state=None):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None, **fit_params):
        """Fit estimator to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : None, default=None
            Not used in the fitting process but kept for compatibility.

        fit_params : dict, optional
            Optional extra fit parameters.

        Returns
        -------
        self : estimator
            Returns the instance itself.

        """
        X = check_array(X)
        rng = check_random_state(self.random_state)
        n_components = self.n_components if self.n_components is not None else X.shape[1]
        if n_components == 1:
            z = rng.randn(n_components)
            self.components_ = z / np.linalg.norm(z)
        else:
            components = np.linalg.qr(rng.randn(X.shape[1], X.shape[1]))[0]
            self.components_ = components[:n_components, :]
        return self


class _SimpleMatrix(object):
    """Thin wrapper around np.array.

    Implements `dot(X)`, `logabsdet()`, and `inv()`.
    """

    def __init__(self, A, copy=True):
        A = np.array(A)
        if copy:
            self.A = A.copy()
        else:
            self.A = A

    def dot(self, X):
        """[Placeholder].

        Parameters
        ----------
        X :

        Returns
        -------
        obj : object

        """
        return self.A.dot(X)

    def logabsdet(self):
        """[Placeholder].

        Returns
        -------
        obj : object

        """
        sign, logdet = np.linalg.slogdet(self.A)
        return logdet

    def inv(self, **kwargs):
        """[Placeholder].

        Parameters
        ----------
        kwargs :

        Returns
        -------
        obj : object

        """
        return _SimpleMatrix(np.linalg.inv(self.A), copy=False)

    def toarray(self):
        """[Placeholder].

        Returns
        -------
        obj : object

        """
        return self.A.copy()

    @property
    def shape(self):
        return self.A.shape


class _IdentityWithScaling(object):
    def __init__(self, n_features, scale=1):
        if not np.isscalar(scale):
            raise ValueError('Parameter `scale` should be a scalar value.')
        self.n_features = n_features
        self.scale = scale

    def dot(self, X):
        """[Placeholder].

        Parameters
        ----------
        X :

        Returns
        -------
        obj : object

        """
        Z = np.array(X, copy=True)
        if self.scale != 1:
            Z[0, :] *= self.scale
        return Z

    def logabsdet(self):
        """[Placeholder].

        Returns
        -------
        obj : object

        """
        if self.scale == 1 or self.scale == -1:
            return 0  # = np.log(|1|) = np.log(|-1|)
        else:
            # Only first dimension scaled so determinant only changes by `self.scale`
            return np.log(np.abs(self.scale))

    def inv(self, copy=True):
        """[Placeholder].

        Parameters
        ----------
        copy :

        Returns
        -------
        obj : object

        """
        # Just in case there is rounding error
        inv_scale = 1.0 / self.scale if self.scale != 1 else 1
        return _IdentityWithScaling(self.n_features, scale=inv_scale)

    def toarray(self):
        """Convert to array, but not implemented yet."""
        return np.eye(self.n_features)

    @property
    def shape(self):
        return (self.n_features, self.n_features)


class _HouseholderWithScaling(_IdentityWithScaling):
    """Linear object interface to a Householder reflector.

    Only stores the value of the `u` vector. Implements `dot(X)`,
    `logabsdet()`, and `inv()` in a computationally simple way.
    """

    def __init__(self, u, scale=1, copy=True):
        super(_HouseholderWithScaling, self).__init__(len(u), scale=scale)

        u = np.array(u, dtype=np.float)
        if np.abs(u.dot(u) - 1) > u.shape[0] * np.finfo(u.dtype).eps:
            raise ValueError('u should be a unit vector such that u.dot(u) == 1')
        if copy:
            self.u = u.copy()
        else:
            self.u = u

    def dot(self, X):
        """[Placeholder].

        Parameters
        ----------
        X :

        Returns
        -------
        obj : object

        """
        # NOTE: X.shape = (d, n), NOT (d, n)
        # u has shape (d,)
        Z = X - np.outer(2 * self.u, np.dot(self.u, X))
        # Only scale the first dimension (leave others alone)
        if self.scale != 1:
            Z[0, :] *= self.scale
        return Z

    def logabsdet(self):
        """[Placeholder].

        Returns
        -------
        obj : object

        """
        return super(_HouseholderWithScaling, self).logabsdet()

    def inv(self, copy=True):
        """[Placeholder].

        Parameters
        ----------
        copy :

        Returns
        -------
        obj : object

        """
        # Just in case there is rounding error
        inv_scale = 1.0 / self.scale if self.scale != 1 else 1
        return _HouseholderWithScaling(self.u, scale=inv_scale, copy=copy)

    def toarray(self):
        """[Placeholder].

        Returns
        -------
        obj : object

        """
        A = np.eye(len(self.u)) - np.outer(2 * self.u, self.u)
        A[0, :] *= self.scale
        return A


    @property
    def shape(self):
        n_features = len(self.u)
        return (n_features, n_features)

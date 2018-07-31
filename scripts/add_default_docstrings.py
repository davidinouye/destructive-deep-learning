"""Add default docstrings."""
# Read entire file
import os

replacements = [
    (
        """
    def fit(self, X, y=None):
        \"\"\"

        Parameters
        ----------
        X :
        y :

        Returns
        -------

        \"\"\"
        """,
        """
    def fit(self, X, y=None):
        \"\"\"Fit estimator to X.

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

        \"\"\"
        """
    ),
    (
        """
    def fit_transform(self, X, y=None, **fit_params):
        \"\"\"

        Parameters
        ----------
        X :
        y :
        fit_params :

        Returns
        -------

        \"\"\"
        """,
        """
    def fit_transform(self, X, y=None, **fit_params):
        \"\"\"Fit estimator to X and then transform X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : None, default=None
            Not used in the fitting process but kept for compatibility.

        fit_params : dict, optional
            Parameters to pass to the fit method.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_features)
            Transformed data.

        \"\"\"
        """
    ),
    (
        """
    def transform(self, X, y=None):
        \"\"\"

        Parameters
        ----------
        X :
        y :

        Returns
        -------

        \"\"\"
        """,
        """
    def transform(self, X, y=None):
        \"\"\"Apply destructive transformation to X.

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

        \"\"\"
        """
    ),
    (
        """
    def transform(self, X, y=None, partial_idx=None):
        \"\"\"

        Parameters
        ----------
        X :
        y :
        partial_idx :

        Returns
        -------

        \"\"\"
        """,
        """
    def transform(self, X, y=None, partial_idx=None):
        \"\"\"Apply destructive transformation to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : None, default=None
            Not used in the transformation but kept for compatibility.

        partial_idx : list or None, default=None
            List of indices of the fitted destructor to use in
            the transformation. The default of None uses all
            the fitted destructors. Mainly used for visualization
            or debugging.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_features)
            Transformed data (possibly only partial transformation).

        \"\"\"
        """
    ),
    (
        """
    def inverse_transform(self, X, y=None, partial_idx=None):
        \"\"\"

        Parameters
        ----------
        X :
        y :
        partial_idx :

        Returns
        -------

        \"\"\"
        """,
        """
    def inverse_transform(self, X, y=None, partial_idx=None):
        \"\"\"Apply inverse destructive transformation to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : None, default=None
            Not used in the transformation but kept for compatibility.

        partial_idx : list or None, default=None
            List of indices of the fitted destructor to use in
            the transformation. The default of None uses all
            the fitted destructors. Mainly used for visualization
            or debugging.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_features)
            Transformed data (possibly only partial transformation).

        \"\"\"
        """
    ),
    (
        """
    def inverse_transform(self, X, y=None):
        \"\"\"

        Parameters
        ----------
        X :
        y :

        Returns
        -------

        \"\"\"
        """,
        """
    def inverse_transform(self, X, y=None):
        \"\"\"Apply inverse destructive transformation to X.

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

        \"\"\"
        """
    ),
    (
        """
    def score_samples(self, X, y=None):
        \"\"\"

        Parameters
        ----------
        X :
        y :

        Returns
        -------

        \"\"\"
        """,
        """
    def score_samples(self, X, y=None):
        \"\"\"Compute log-likelihood (or log(det(Jacobian))) for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples and n_features
            is the number of features.

        y : None, default=None
            Not used but kept for compatibility.

        Returns
        -------
        log_likelihood : array, shape (n_samples,)
            Log likelihood of each data point in X.

        \"\"\"
        """
    ),
    (
        """
    def sample(self, n_samples=1, random_state=None):
        \"\"\"

        Parameters
        ----------
        n_samples :
        random_state :

        Returns
        -------

        \"\"\"
        """,
        """
    def sample(self, n_samples=1, random_state=None):
        \"\"\"Generate random samples from this density/destructor.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate. Defaults to 1.

        random_state : int, RandomState instance or None, optional (default=None)
            If int, `random_state` is the seed used by the random number
            generator; If :class:`~numpy.random.RandomState` instance,
            `random_state` is the random number generator; If None, the random
            number generator is the :class:`~numpy.random.RandomState` instance
            used by :mod:`numpy.random`.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample.

        \"\"\"
        """
    ),
    (
        """
    def get_domain(self):
        \"\"\"

        Returns
        -------

        \"\"\"
        """,
        """
    def get_domain(self):
        \"\"\"Get the domain of this destructor.

        Returns
        -------
        domain : array-like, shape (2,) or shape (n_features, 2)
            If shape is (2, ), then ``domain[0]`` is the minimum and
            ``domain[1]`` is the maximum for all features. If shape is
            (`n_features`, 2), then each feature's domain (which could
            be different for each feature) is given similar to the first
            case.

        \"\"\"
        """
    ),
    (
        """
    def get_support(self):
        \"\"\"

        Returns
        -------

        \"\"\"
        """,
        """
    def get_support(self):
        \"\"\"Get the support of this density (i.e. the positive density region).

        Returns
        -------
        support : array-like, shape (2,) or shape (n_features, 2)
            If shape is (2, ), then ``support[0]`` is the minimum and
            ``support[1]`` is the maximum for all features. If shape is
            (`n_features`, 2), then each feature's support (which could
            be different for each feature) is given similar to the first
            case.

        \"\"\"
        """
    ),
    (
        """
    def get_density_estimator(self):
        \"\"\"

        Returns
        -------

        \"\"\"
        """,
        """
    def get_density_estimator(self):
        \"\"\"Get the *unfitted* density associated with this destructor.

        NOTE: The returned estimator is NOT fitted but is a clone or new
        instantiation of the underlying density estimator. This is just
        a helper function that needs to be overridden by subclasses of
        :class:`~ddl.base.BaseDensityDestructor`.

        Returns
        -------
        density : estimator
            The *unfitted* density estimator associated wih this
            destructor.

        \"\"\"
        """
    ),
    (
        """
        \"\"\"

        Returns
        """,
        """
        \"\"\"[Placeholder].

        Returns
        """,
    ),
    (
        """
        \"\"\"

        Parameters
        """,
        """
        \"\"\"[Placeholder].

        Parameters
        """,
    ),
    (
        """
        Returns
        -------

        \"\"\"
        """,
        """
        Returns
        -------
        obj : object

        \"\"\"
        """,
    ),
    (
        """
    def score_samples(self, X, y=None, partial_idx=None):
        \"\"\"[Placeholder].

        Parameters
        ----------
        X :
        y :
        partial_idx :

        Returns
        -------
        obj : object

        \"\"\"
        """,
        """
    def score_samples(self, X, y=None, partial_idx=None):
        \"\"\"Compute log-likelihood (or log(det(Jacobian))) for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples and n_features
            is the number of features.

        y : None, default=None
            Not used but kept for compatibility.

        partial_idx : list or None, default=None
            List of indices of the fitted destructor to use in
            the computing the log likelihood. The default of None uses all
            the fitted destructors. Mainly used for visualization
            or debugging.

        Returns
        -------
        log_likelihood : array, shape (n_samples,)
            Log likelihood of each data point in X.

        \"\"\"
        """
    ),
    (
        """
    def fit(self, X, y=None, **fit_params):
        \"\"\"[Placeholder].

        Parameters
        ----------
        X :
        y :
        fit_params :

        Returns
        -------
        obj : object

        \"\"\"
        """,
        """
    def fit(self, X, y=None):
        \"\"\"Fit estimator to X.

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

        \"\"\"
        """
    ),
]

rootdir = 'ddl'
for subdir, dirs, files in os.walk(rootdir):
    for filename in files:
        filepath = os.path.join(subdir, filename)
        if filepath.endswith(".py"):
            print(filepath)

            # Read in file
            with open(filepath, 'r') as f:
                file_str = f.read()

            # Replace some strings
            for old, new in replacements:
                file_str = file_str.replace(old, new)

            # Write new file
            with open(filepath, 'w+') as f:
                f.write(file_str)

import warnings

import numpy as np
from sklearn.utils import shuffle, check_random_state

def make_toy_data(data_name, n_samples=1000, random_state=None, **maker_kwargs):
    try:
        maker = _makers_dict['_make_%s' % data_name]
    except KeyError:
        raise ValueError('Invalid data_name of "%s"' % data_name) 
    X, y, is_canonical_domain = maker(n_samples, random_state=random_state, **maker_kwargs)
    return _Data(X=X, y=y, data_name=data_name, is_canonical_domain=is_canonical_domain)


class _Data(object):
    """Simple class to hold data values and attributes"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _make_rotated_uniform(n_samples, scale=None, Q=None, random_state=0):
    n_features = 2
    rng = check_random_state(random_state)
    if scale is None:
        scale = np.array([1, 3])
    if Q is None:
        Q = np.linalg.qr(rng.randn(n_features, n_features))[0]
    U = rng.rand(n_samples, n_features) - 0.5
    X = np.dot(U * scale, Q)
    return X, None, False


def _make_autoregressive(
        n_samples, func=None, x_scale=2, y_std=1, flip_x_y=False,
        x_distribution='uniform', random_state=0
):
    rng = check_random_state(random_state)
    n_features = 2
    # Get x values
    if x_distribution == 'gaussian':
        x = x_scale * rng.randn(n_samples)
    elif x_distribution == 'abs-gaussian':
        x = np.abs(x_scale * rng.randn(n_samples))
    elif x_distribution == 'uniform':
        x = x_scale * rng.rand(n_samples)
    else:
        raise ValueError('x_distribution should be "gaussian", "uniform", or "abs-gaussian"')

    # Compute y from x and add some noise
    y = func(x) + y_std * rng.randn(n_samples)

    # Flip x and y
    if flip_x_y:
        X = np.array([y, x]).T
    else:
        X = np.array([x, y]).T

    return X, None, False


def _make_sin_wave(n_samples, x_scale=2 * np.pi, y_std=0.2, **kwargs):
    if 'func' in kwargs:
        raise ValueError('func is overridden by _make_sin_wave')
    return _make_autoregressive(n_samples, func=lambda x: np.sin(x),
                               x_scale=x_scale, y_std=y_std, **kwargs)


def _make_rbig_sin_wave(n_samples, random_state=0):
    # Example from [Laparra et al. 2011]
    # Code at https://www.uv.es/vista/vistavalencia/RBIG.htm
    return _make_sin_wave(n_samples, x_scale=2, y_std=0.25,
                         x_distribution='abs-gaussian', random_state=random_state)


def _make_quadratic(n_samples, random_state=0):
    # Example from [Papamakarios et al. 2017]
    return _make_autoregressive(n_samples, func=lambda x: (1 / 4) * x ** 2,
                               x_distribution='gaussian', x_scale=2, y_std=1, flip_x_y=True)


def _make_grid(n_samples, n_grid=5, sigma=None, Q=None, perc_filled=0.5, random_state=0,
              kind='gaussian'):
    rng = check_random_state(random_state)
    n_features = 2

    if Q is None:
        Q = np.eye(n_features, n_features)
    if kind == 'gaussian':
        if sigma is None:
            sigma = 0.2
        query = np.array(range(n_grid))
        is_bounded = False

        def sample(pos, n, d):
            return sigma * rng.randn(n, d) + pos
    elif kind == 'uniform':
        if sigma is not None:
            warnings.warn('sigma is ignored when kind="uniform"')
        query = np.linspace(0, 1, n_grid, endpoint=False)
        scale = 1 / n_grid
        is_bounded = True

        def sample(pos, n, d):
            return scale * rng.rand(n, d) + pos
    else:
        raise ValueError('kind should be "gaussian" or "uniform"')

    X_grid, Y_grid = np.meshgrid(query, query)
    positions = np.array([X_grid.ravel(), Y_grid.ravel()]).T
    positions = np.dot(positions, Q)

    # Filter to only certain components based on percent filled
    n_components = int(np.round(perc_filled * n_grid ** 2))
    perm_idx = rng.permutation(n_grid ** 2)
    positions = positions[perm_idx[:n_components], :]

    n_per_component = rng.multinomial(n_samples, 1 / n_components * np.ones(n_components))

    X = np.vstack([
        sample(pos, n, n_features)
        for pos, n in zip(positions, n_per_component)
    ])
    X, y = _get_y_and_shuffle(X, n_per_component)
    return X, y, is_bounded


def _make_gaussian_grid(n_samples, **kwargs):
    kwargs['kind'] = 'gaussian'
    return _make_grid(n_samples, **kwargs)


def _make_manifold_gaussian_grid(n_samples, sigma=0.05, **kwargs):
    kwargs['random_state'] = 2
    return _make_grid(n_samples, sigma=sigma, **kwargs)


def _make_uniform_grid(n_samples, **kwargs):
    kwargs['kind'] = 'uniform'
    kwargs['random_state'] = 1
    return _make_grid(n_samples, **kwargs)


def _make_concentric_circles(n_samples, n_circles=4, noise_std=0.1, random_state=0):
    rng = check_random_state(random_state)
    radius = np.array(range(n_circles)) + 1
    circum = np.array([
        2 * np.pi * rad for rad in radius
    ])
    perc_per_circle = circum / np.sum(circum)

    n_per_component = rng.multinomial(n_samples, perc_per_circle)

    theta = [
        2 * np.pi * rng.rand(n)
        for n in n_per_component
    ]
    X = np.vstack([
        ((r + noise_std * rng.randn(len(th))) * np.array([np.cos(th), np.sin(th)])).T
        for th, r in zip(theta, radius)
    ])
    X, y = _get_y_and_shuffle(X, n_per_component)
    return X, y, False


def _get_y_and_shuffle(X, n_per_component):
    y = np.concatenate([
        j * np.ones(n)
        for j, n in enumerate(n_per_component)
    ])
    # Shuffle
    X, y = shuffle(X, y, random_state=0)
    return X, y

_makers_dict = {key: val for key, val in locals().items() if '_make_' in key}

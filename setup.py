#!/usr/bin/env python
import os

def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg:
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('ddl')
    config.add_subpackage('ddl.tests')
    config.add_subpackage('ddl.externals')

    # I think this should automatically call setup.py in mlpack
    config.add_subpackage('ddl.externals.mlpack')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    metadata = dict(
        name='ddl',
        packages=['ddl'],
        version='0.0.1',
        description='Destructive deep learning estimators and functions. Estimators are compatible '
                    'with scikit-learn.',
        url='https://github.com/davidinouye/destructive-deep-learning',
        author='David I. Inouye',
        license='BSD 3-clause',
        zip_safe=False,
        # Nose needed by 0.19.1 version of scikit-learn for importing testing module
        # I think this was fixed for upcoming version 0.20.X to avoid dependency
        install_requires=['six', 'numpy', 'scipy', 'scikit-learn', 'matplotlib', 'plotly', 'seaborn',
                          'nose', # See above
                          'pot',
                          'anytree',
                          ],
        # Cython, numpy and pypandoc needed to install pot package (bug in installing pot from scratch)
        # Should do the following before trying to install
        # $ pip install setuptools
        # $ pip install Cython
        # $ pip install numpy
        # $ pip install pypandoc
        setup_requires=['setuptools', 'Cython', 'numpy', 'pypandoc'],
        extras_require={
            'test': ['pytest'],  # Testing framework
        },
    )

    metadata['configuration'] = configuration

    setup(**metadata)

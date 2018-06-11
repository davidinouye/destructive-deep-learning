#!/usr/bin/env python
import os

import numpy
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
from Cython.Build import cythonize


def configuration(parent_package='', top_path=None):
    config = Configuration('mlpack', parent_package, top_path)
    libraries = ['mlpack', 'boost_serialization']
    if os.name == 'posix':
        libraries.append('m')
    for pyx in ['_arma_numpy.pyx', '_det.pyx']:
        config.add_extension(
            pyx.split('.')[0],
            sources=[pyx],
            language='c++',
            #include_dirs=[np.get_include(), os.path.join(*package_list)], # Needed for arma_numpy.pyx
            include_dirs=[numpy.get_include()], # Needed for arma_numpy.pyx
            libraries=libraries,
            extra_compile_args=('-DBINDING_TYPE=BINDING_TYPE_PYX ' 
                                '-std=c++11 -Wall -Wextra -ftemplate-depth=1000 -O3 -fopenmp').split(' '),
            extra_link_args=['-fopenmp'],
            undef_macros=[] if len("") == 0 else ''.split(';')

        )
    # Cythonize files (i.e. create .cpp files and return cpp sources)
    config.ext_modules = cythonize(config.ext_modules)

    config.add_subpackage('tests')

    return config

if __name__ == "__main__":
    setup(**configuration(top_path='ddl/externals/').todict())

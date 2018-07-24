#!/usr/bin/env python
import os
import shutil
import subprocess
import sys
import tarfile
from sklearn.externals.six.moves import urllib

import numpy
from Cython.Build import cythonize
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration


def build_mlpack():
    # Setup mlpack paths
    version = '3.0.2'
    package_path = os.path.dirname(os.path.realpath(__file__))
    tar_file_name = 'mlpack-%s.tar.gz' % version
    tar_file_path = os.path.join(package_path, tar_file_name)
    tar_url = 'https://github.com/mlpack/mlpack/archive/%s' % tar_file_name 
    repo_path = os.path.join(package_path, 'mlpack-mlpack-%s' % version)
    cmake_file_path = os.path.join(repo_path, 'CMakeLists.txt')
    build_path = os.path.join(repo_path, 'build')
    orig_cwd = os.getcwd()

    # Download and extract
    if not os.path.isdir(repo_path):
        print('Downloading mlpack source')
        if not os.path.isfile(tar_file_path):
            urllib.request.urlretrieve(tar_url, tar_file_path)

        print('Extracting tar.gz file')
        tar = tarfile.open(tar_file_path, 'r:gz')
        tar.extractall(package_path)
        tar.close()
        os.remove(tar_file_path)

    if not os.path.isdir(build_path):
        # Add flag needed for static library linking
        print('Prepending flag for PIC')
        with open(cmake_file_path, 'r') as f:
            cmake_file_str = f.read()
        with open(cmake_file_path, 'w') as f:
            f.write('set(CMAKE_POSITION_INDEPENDENT_CODE ON)\n' + cmake_file_str)

        # Setup build directory and run cmake
        #print('Removing old build directory if necessary and creating new build directory')
        print('Attempting to create build directory if it does not exist')
        #shutil.rmtree(build_path, ignore_errors=True)
        os.mkdir(build_path)

        print('Running cmake')
        os.chdir(build_path)
        if sys.platform == 'darwin':
            print('Exporting environment variables needed for cmake in Mac OS')
            os.environ['CC'] = '/usr/local/opt/llvm/bin/clang'
            os.environ['CXX'] = '/usr/local/opt/llvm/bin/clang++'
            os.environ['LDFLAGS'] = '-L/usr/local/opt/llvm/lib'
            os.environ['CPPFLAGS'] = '-I/usr/local/opt/llvm/include'
            subprocess.call(['cmake', 
                             '-D', 'BUILD_SHARED_LIBS=OFF', 
                             '-D', 'FORCE_CXX11=ON',
                             '-D', 'CMAKE_CXX_FLAGS=-std=c++11',
                             '../'])
        else:
            subprocess.call(['cmake', '-D', 'BUILD_SHARED_LIBS=OFF', '../'])

    # Actually build mlpack
    print('Building mlpack')
    subprocess.call(['make', 'mlpack'])
    os.chdir(orig_cwd)
    return build_path


def configuration(parent_package='', top_path=None):
    build_path = build_mlpack()
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
            include_dirs=[numpy.get_include(), os.path.join(build_path, 'include')], # Needed for arma_numpy.pyx
            library_dirs=[os.path.join(build_path, 'lib')],
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

# Generate classes.rst file
import errno
import importlib
import os
import pkgutil
import sys
from inspect import getmembers, isclass, isfunction

INDENT = '    '


def create_api_rst(package, generated_dir):
    """

    Parameters
    ----------
    package :
    generated_dir :

    Returns
    -------

    """
    # Build list of packages
    module_names = [
        module_name
        for _, module_name, is_pkg in pkgutil.walk_packages(
            path=package.__path__, prefix=package.__name__ + '.', onerror=lambda x: None)
        if ('._' not in module_name
            and module_name[0] != '_'
            and 'test' not in module_name
            and not module_name.endswith('setup'))
    ]

    # Get functions and classes
    def _get_module_data(module_name):
        module = importlib.import_module(module_name)
        classes = [
            o for o in getmembers(module)
            if isclass(o[1])
            and o[0][0] != '_'
            and o[1].__module__.startswith(module_name)
        ]
        functions = [
            o for o in getmembers(module)
            if isfunction(o[1])
            and o[0][0] != '_'
            and o[1].__module__.startswith(module_name)
        ]
        return dict(module=module, functions=functions, classes=classes)

    module_data = [
        _get_module_data(module_name)
        for module_name in module_names
    ]

    # Write to file
    try:
        os.makedirs(generated_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    with open(os.path.join(generated_dir, 'api.rst'), 'w+') as f:
        f.write(_show_header())
        for module_name, data in zip(module_names, module_data):
            f.write(_show_module(module_name))
            if len(data['classes']) > 0:
                f.write(_show_classes(module_name, data['classes'], generated_dir))
            if len(data['functions']) > 0:
                f.write(_show_functions(module_name, data['functions'], generated_dir))


def _show_header():
    s = '.. _api_ref:\n\n=============\nAPI Reference\n=============\n\n'
    s += 'This is the class and function reference of destructive-deep-learning (ddl).\n\n'
    return s


def _show_module(module_name):
    mod_str = ':mod:`%s`' % module_name
    s = mod_str + '\n'
    s += '=' * len(mod_str) + '\n\n'
    s += '.. automodule:: %s\n' % module_name
    s += ('%s:no-members:\n%s:no-inherited-members:\n\n.. _%s_ref:\n\n'
          % (INDENT, INDENT, module_name.replace('.', '_')))
    return s


def _show_classes(module_name, classes, generated_dir):
    return _show_arr(module_name, classes, generated_dir, kind='class')


def _show_functions(module_name, classes, generated_dir):
    return _show_arr(module_name, classes, generated_dir, kind='function')


def _show_arr(module_name, arr, generated_dir, kind='class'):
    if kind == 'function':
        plural_kind = 'functions'
    elif kind == 'class':
        plural_kind = 'classes'
    else:
        raise ValueError('Invalid kind')
    this_module = module_name.split('.')[-1]
    arr_str = '%s %s' % (this_module.title(), plural_kind)
    s = arr_str + '\n'
    s += '-' * len(arr_str) + '\n\n'
    s += '.. currentmodule:: %s\n\n' % '.'.join(module_name.split('.')[:-1])
    s += ('.. autosummary::\n%s:toctree: %s\n%s:template: %s.rst\n\n' %
          (INDENT, os.path.join('..', generated_dir), INDENT, kind))
    for name, obj in arr:
        s += '%s%s.%s (%s)\n' % (INDENT, this_module, name, obj.__module__)
    s += '\n'
    return s


if __name__ == '__main__':
    sys.path.append('..')
    import ddl

    create_api_rst(ddl, 'generated')

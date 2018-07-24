import argparse
import errno
import gzip
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib
import warnings

import numpy as np
from sklearn.datasets import fetch_mldata

# Handle pickled data for python 2 and python 3
try:
    import cPickle
    def _get_cifar10_data_and_labels(file_path):
        with open(file_path, 'rb') as f:
            dict_obj = cPickle.load(f)
        return (dict_obj['data'], dict_obj['labels'])

except ImportError:
    import pickle
    def _get_cifar10_data_and_labels(file_path):
        with open(file_path, 'rb') as f:
            dict_obj = pickle.load(f, encoding='bytes')
        return (dict_obj[b'data'], dict_obj[b'labels'])

# From MAF code
MNIST_ALPHA = 1.0e-6
CIFAR10_ALPHA = 0.05

# Gets data path wherever the script is executed
_DOWNLOAD_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 
    '..', 'data', 'download-cache'
)


def get_maf_data(data_name):
    if data_name == 'mnist':
        return _get_maf_mnist()
    elif data_name == 'cifar10':
        return _get_maf_cifar10()
    else:
        raise ValueError('Only "mnist" and "cifar10" datasets are supported')


def _get_maf_mnist():
    X, y = _get_mnist_raw()
    # This splits file was created by comparing MNIST downloaded to the 
    #  MAF MNIST dataset splits
    splits_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 
        'maf_mnist_splits.txt.gz'
    )
    with gzip.open(splits_file, 'r') as f:
        ind_arr = [
            # Decode is needed to be compatible with Python 2 and 3
            np.array(line.decode('utf-8').split(', '), dtype=np.int)
            for i, line in enumerate(f) if i > 0 # Skip first line
        ]

    # Ensure dequantization is the same as MAF paper
    #print('Preprocessing MNIST data')
    rng = np.random.RandomState(42)
    return _data_arr_to_dict([
        _preprocess_mnist(X[ind, :], y[ind], rng)
        for ind in ind_arr
    ])


def _get_maf_cifar10():
    # Download cifar10 data if needed
    def _report_hook(count, block_size, total_size):
        # Copied from https://blog.shichao.io/2012/10/04/progress_speed_indicator_for_urlretrieve_in_python.html
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = min(int(count * block_size * 100 / total_size), 100)
        sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                        (percent, progress_size / (1024 * 1024), speed, duration))
        sys.stdout.flush()
    _make_dir(_DOWNLOAD_DIR)
    path = os.path.join(_DOWNLOAD_DIR, 'cifar-10-batches-py')
    file_name = path + '.tar.gz'
    if not os.path.isdir(path):
        if not os.path.isfile(file_name):
            url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
            print('Downloading cifar10 via %s' % url)
            urllib.request.urlretrieve(url, file_name, _report_hook)
        print('Extracting cifar10 from tar.gz file')
        tar = tarfile.open(file_name, 'r:gz')
        tar.extractall(_DOWNLOAD_DIR)
        tar.close()
    
    # Load cifar10 and create splits
    # Mostly copied from MAF source except for Python 3 compatability
    #print('Loading cifar10 from previously downloaded pickle files')
    x = []
    l = []
    for i in range(1, 6):
        data, labels = _get_cifar10_data_and_labels(os.path.join(path, 'data_batch_' + str(i)))
        x.append(data)
        l.append(labels)
    x = np.concatenate(x, axis=0)
    l = np.concatenate(l, axis=0)

    # use part of the train batches for validation
    split = int(0.9 * x.shape[0])
    maf_train = (x[:split], l[:split])
    maf_validation = (x[split:], l[split:])

    # load test batch
    data, labels = _get_cifar10_data_and_labels(os.path.join(path, 'test_batch'))
    maf_test = (data, np.array(labels))

    # Preprocess the dataset
    #print('Preprocessing cifar10 dataset')
    rng = np.random.RandomState(42)
    return _data_arr_to_dict([
        _preprocess_cifar10(maf[0], maf[1], flip, rng)
        for maf, flip in zip((maf_train, maf_validation, maf_test), (True, False, False))
    ])


def _get_mnist_raw():
    def _download_from_other_source():
        # Attempt to download mnist data from another source
        url = 'http://www.cs.cmu.edu/~dinouye/data/mnist-original.mat'
        warnings.warn('Could not download from mldata.org, attempting '
                      'to download from <%s>.' % url)
        file_name = os.path.join(_DOWNLOAD_DIR, 'mldata/mnist-original.mat')
        urllib.request.urlretrieve(url, file_name)

    _make_dir(_DOWNLOAD_DIR)
    n_attempts = 3
    print('Attempting to load/fetch MNIST data via sklearn.datasets.fetch_mldata')
    for i in range(n_attempts):
        try:
            data_obj = fetch_mldata('MNIST original', data_home=_DOWNLOAD_DIR)
        except (ConnectionResetError, urllib.error.HTTPError):
            _download_from_other_source()
            if i == n_attempts - 1:
                warnings.warn('Attempted to retrieve MNIST data from mldata.org or alternative %d times '
                      'and failed each time, please check internet connection' % n_attempts)
                raise
        else:
            #print('Successfully fetched MNIST data')
            break
    return data_obj.data, data_obj.target


def _preprocess_mnist(X, y, rng):
    # Convert between 0 and 1
    X = X.astype(np.float32) / 256.0
    # Dequantize
    X = X + rng.rand(*X.shape) / 256.0
    # Logit transform
    X = _logit_transform(X, MNIST_ALPHA)
    # One hot encode labels
    #y = _one_hot_encode(y, 10).astype(np.int64)
    return (X, y.astype(np.int64))


def _preprocess_cifar10(X, y, flip, rng):
    # Dequantize and set values between 0 and 1
    X = (X + rng.rand(*X.shape).astype(np.float32)) / 256.0
    # Logit transform
    X = _logit_transform(X, CIFAR10_ALPHA)
    # Add horizontal flips
    if flip:
        X = _flip_augmentation(X)
        y = np.hstack([y, y])
    # One hot encode
    #y = _one_hot_encode(y, 10)
    return (X, y.astype(np.int64))


#############################################################
# Utilities
#############################################################
def _flip_augmentation(X):
    """Slight modification for Pythone 3 of static method from maf/datasets/cifar10.py."""
    D = int(X.shape[1] / 3)
    I = int(np.sqrt(D))
    r = X[:,    :D].reshape([-1, I, I])[:, :, ::-1].reshape([-1, D])
    g = X[:, D:2*D].reshape([-1, I, I])[:, :, ::-1].reshape([-1, D])
    b = X[:,  2*D:].reshape([-1, I, I])[:, :, ::-1].reshape([-1, D])
    X_flip = np.hstack([r, g, b])
    return np.vstack([X, X_flip])


def _logit_transform(X, alpha):
    def _logit(X):
        return np.log(X / (1.0 - X))
    return _logit(alpha + (1 - 2 * alpha) * X)


def _one_hot_encode(labels, n_labels):
    """Copied with slight modifications for Python 3 from original MAF code."""
    assert np.min(labels) >= 0 and np.max(labels) < n_labels
    y = np.zeros([labels.size, n_labels])
    y[np.arange(labels.size, dtype=np.int), labels.astype(np.int)] = 1
    return y


def _make_dir(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def _data_arr_to_dict(data_arr):
    d = {}
    for data, name in zip(data_arr, ['train', 'validation', 'test']):
        d['X_%s' % name] = data[0]
        d['y_%s' % name] = data[1]
    return d


def _data_dict_to_arr(data_dict):
    d = data_dict # Short alias
    return [
        (d['X_train'], d['y_train']),
        (d['X_validation'], d['y_validation']),
        (d['X_test'], d['y_test'])
    ]


#############################################################
# Testing/Checking functions
#############################################################
def _check_maf_data():
    warnings.warn('This function should generally not be called because it '
                  'requires special setup but is kept here in order to reproduce functions if needed.')
    for data_name in ['mnist', 'cifar10']:
        print('Loading %s data directly and via maf code' % data_name)
        direct = _data_dict_to_arr(get_maf_data(data_name))
        original = _data_dict_to_arr(_get_maf_original(data_name))
        print('Comparing dtypes and values of returned arrays for %s' % data_name)
        for i, (x_direct, x_original) in enumerate(zip(np.array(direct).ravel(), np.array(original).ravel())):
            # Check that they have the same dtype
            assert x_direct.dtype == x_original.dtype, 'dtypes not equal for index %d' % i
            # Check that they are equal
            assert np.all(x_direct == x_original), 'Arrays not equal for index %d' % i
            print('Array of index %d are equal' % i)
    print('All arrays are equal! :-)')


def _get_maf_original(data_name):
    warnings.warn('This function should generally not be called because it '
                  'requires special setup but is kept here in order to reproduce functions if needed.')
    if sys.version_info < (3,):
        # Load MNIST from MAF code
        maf_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 
            '..', '..', 'maf'
        )
        sys.path.append(maf_path)
        import datasets  # maf/datasets/*

        # Reset datasets root directory relative to this file
        datasets.root = os.path.join(maf_path, 'data') + '/'

        # Copied from maf/experiments.py
        if data_name == 'mnist':
            data = datasets.MNIST(logit=True, dequantize=True)
        elif data_name == 'bsds300':
            data = datasets.BSDS300()
        elif data_name == 'cifar10':
            data = datasets.CIFAR10(logit=True, flip=True, dequantize=True)
        elif data_name == 'power':
            data = datasets.POWER()
        elif data_name == 'gas':
            data = datasets.GAS()
        elif data_name == 'hepmass':
            data = datasets.HEPMASS()
        elif data_name == 'miniboone':
            data = datasets.MINIBOONE()
        else:
            raise ValueError('Unknown dataset')

        # Make a dictionary instead of pickled object for better compatibility
        if hasattr(data.trn, 'labels'):
            data_dict = dict(
                X_train=data.trn.x,
                y_train=data.trn.labels,
                X_validation=data.val.x,
                y_validation=data.val.labels,
                X_test=data.tst.x,
                y_test=data.tst.labels,
                data_name=data_name,
            )
        else:
            data_dict = dict(
                X_train=data.trn.x,
                X_validation=data.val.x,
                X_test=data.tst.x,
                data_name=data_name,
            )
    else:
        raise RuntimeError('Must create data using Python 2 to load data since MAF is written for '
                           'Python 2')
    return data_dict


def _save_mnist_recreation_indices():
    """Code to find MNIST train, validation and test indices for recreation of 
    MNIST MAF dataset.
    Note this should not be called directly.  This is only here for reproducibility."""
    warnings.warn('This function should generally not be called because it '
                  'requires special setup but is kept here in order to reproduce functions if needed.')
    # Import maf data
    datasets_root = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 
        '..', '..', 'maf', 'data',
    )
    mnist_path = os.path.join(datasets_root, 'mnist', 'mnist.pkl.gz')
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    with gzip.open(mnist_path, 'rb') as f:
        maf_train, maf_val, maf_test = pickle.load(f)

    # Import raw mnist data
    data_obj = fetch_mldata('MNIST original')#, data_home=custom_data_home)

    # Prepare comparison matrices
    X_all = data_obj.data/256.0
    y_all = data_obj.target

    maf_data_tuple = (maf_train[0], maf_val[0], maf_test[0])
    n_maf = [X.shape[0] for X in maf_data_tuple]
    X_maf = np.vstack(maf_data_tuple)
    y_maf = np.concatenate((maf_train[1], maf_val[1], maf_test[1]))

    # Sort maf using all columns
    mnist_ind = np.lexsort(np.hstack((X_all, y_all.reshape(-1,1))).T)
    maf_ind = np.lexsort(np.hstack((X_maf, y_maf.reshape(-1,1))).T)
    rev_maf_ind = np.argsort(maf_ind)

    # Show that matrices match when sorted by indices
    print('Checking if the datasets are the same (should all be 0)')
    def n_diff(X, Y):
        return np.count_nonzero(X - Y)
    def print_n_diff(X, Y):
        print('Number different = %d' % n_diff(X, Y))
    print_n_diff(X_all[mnist_ind], X_maf[maf_ind])
    print_n_diff(y_all[mnist_ind], y_maf[maf_ind])

    # Retrieve indices and show that they are the same
    train_idx, val_idx, test_idx = (
        mnist_ind[rev_maf_ind[np.sum(n_maf[:i], dtype=np.int):np.sum(n_maf[:(i+1)], dtype=np.int)]]
        for i in range(3)
    )
    for idx, maf in zip((train_idx, val_idx, test_idx), (maf_train, maf_val, maf_test)):
        print_n_diff(X_all[idx], maf[0])
        print_n_diff(y_all[idx], maf[1])

    gzip_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 
        'maf_mnist_splits.txt.gz'
    )
    with gzip.open(gzip_file, 'w+') as f:
        f.write('# Indices of MNIST dataset retrieved using '
                'sklearn.datasets.fetch_mldata(\'MNIST original\') that correspond to the train, '
                'validation and test sets of the MAF paper (one line each).\n')
        for i, idx in enumerate([train_idx, val_idx, test_idx]):
            s = str(idx.tolist())
            s = s[1:-1] # Trim off ends
            f.write(s)
            if i < 2:
                f.write('\n')


if __name__ == '__main__':
    all_data_names = ['mnist', 'cifar10']
    parser = argparse.ArgumentParser(description='Creates cache of datasets.')
    parser.add_argument(
        'data_names', default=','.join(all_data_names),
        help='One or more data names separated by commas from the list %s' % str(all_data_names))
    args = parser.parse_args()
    for data_name in args.data_names.split(','):
        # Load data into cache
        get_maf_data(data_name)
        print('Finished caching %s' % data_name)
    # Uncomment to recreate mnist train, validation and test indices from MAF code
    #_save_mnist_recreation_indices()
    # Uncomment to check that data is the same as MAF code
    #_check_maf_data()

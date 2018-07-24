import os
import sys
import logging
import argparse
import time
import subprocess
import warnings
try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import scipy.stats # Needed for standard error of the mean scipy.stats.sem
from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA, FastICA
from sklearn.tree import DecisionTreeClassifier


# Add the directory of this script
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
# Add directory for ddl library
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

from ddl.base import CompositeDestructor
from ddl.univariate import HistogramUnivariateDensity, ScipyUnivariateDensity
from ddl.independent import IndependentDestructor, IndependentDensity, IndependentInverseCdf
from ddl.deep import DeepDestructorCV
from ddl.linear import BestLinearReconstructionDestructor
from ddl.local import FeatureGroupsDestructor, ImageFeaturePairs
from ddl.tree import TreeDestructor, TreeDensity
from ddl.externals.mlpack import MlpackDensityTreeEstimator

from maf_data import get_maf_data, MNIST_ALPHA, CIFAR10_ALPHA

logger = logging.getLogger(__name__)

def run_experiment(data_name, model_name, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {}
    # Setup
    experiment_filename = model_kwargs['experiment_filename']
    experiment_label = model_kwargs['experiment_label']
    _setup_loggers(experiment_filename)
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii')[:-1]
    except subprocess.CalledProcessError:
        git_hash = 'unknown'
    logger.debug('Current git hash = %s' % git_hash)

    # Load data
    logger.debug('Loading data for %s' % experiment_label)
    data_dict = get_maf_data(data_name)
    X_train, X_validation, X_test = (
        data_dict['X_train'], data_dict['X_validation'], data_dict['X_test'])
    if 'perc_train' in model_kwargs:
        X_train, X_validation = _perc_resample(model_kwargs['perc_train'])
    n_train, n_validation, n_test= (_X.shape[0] for _X in (X_train, X_validation, X_test))

    # Setup cv and refit parameters
    X_train_val = np.vstack((X_train, X_validation))
    model_kwargs['cv'] = [(np.arange(n_train), n_train + np.arange(n_validation))]
    model_kwargs['refit'] = False

    # Load model
    deep_destructor = _get_model(data_name, model_name, model_kwargs=model_kwargs)

    # Fit destructor
    logger.debug('Starting training for %s' % experiment_label)
    start_time = time.time()
    deep_destructor.fit(X_train_val, y=None, X_test=X_test)
    train_time = time.time() - start_time
    logger.debug('Finished training for %s' % experiment_label)
    logger.debug('%s: Time to train = %g s or %g minutes or %g hours'
                 % (experiment_label, train_time, train_time/60, train_time/60/60))

    # Get test score
    start_time = time.time()
    test_scores = deep_destructor.score_samples(X_test)
    score_time = time.time() - start_time
    test_score = np.mean(test_scores)
    test_score_stderr = scipy.stats.sem(test_scores)
    logger.debug('%s: Final test score=%g with std_err=%g computed in %g s'
                 % (experiment_label, test_score, test_score_stderr, score_time))
    date_time_completed = time.strftime("%Y_%m_%d-%H_%M_%S")
    logger.debug('Date/time completed (just before saving): %s' % date_time_completed)

    # Prepare results in dictionary
    result_dict = dict(
        # Data statistics
        data_name=data_name, n_features=X_train.shape[1],
        n_train=n_train, n_validation=n_validation, n_test=n_test,
        # Model
        destructor=deep_destructor, model_name=model_name, model_kwargs=model_kwargs,
        # Time
        train_time=train_time, score_time=score_time, date_time_completed=date_time_completed,
        # Test scores
        test_score=test_score, test_score_stderr=test_score_stderr, test_scores=test_scores,
        git_hash=git_hash,
    )

    # Save results to pickle file
    with open(experiment_filename + '.pkl', 'wb') as f:
        pickle.dump(result_dict, f)
    logger.debug('%s: Saved results to file %s' % (experiment_label, experiment_filename))
    return result_dict


def load_experiment_results(data_name, model_name=None, model_kwargs=None, notebook=False):
    experiment_filename, _ = _get_experiment_filename_and_label(data_name, model_name=model_name,
                                              model_kwargs=model_kwargs)
    if notebook:
        experiment_filename = os.path.join('..', experiment_filename)

    with open(experiment_filename + '.pkl', 'rb') as f:
        result_dict = pickle.load(file=f)
    logger.debug('Loaded results from file %s' % experiment_filename)
    return result_dict


def _get_model(data_name, model_name, model_kwargs):
    if 'is_test' not in model_kwargs:
        model_kwargs['is_test'] = False
    # Init destructor is shared with all models
    init_destructor = CompositeDestructor(
        destructors=[
            _get_inverse_logit_destructor(data_name),
            IndependentDestructor(
                independent_density=IndependentDensity(
                    univariate_estimators=HistogramUnivariateDensity(
                        bins=256, bounds=[0, 1], alpha=1)
                )
            )
        ],
        random_state=0,
    )

    # Setup canonical destructor for various models
    if model_name == 'deep-copula':
        deep_stop_tol=0.001
        canonical_destructor = _get_copula_destructor()
    else:
        deep_stop_tol = 0.0001
        n_jobs = model_kwargs['n_jobs']

        # Get pair estimators (i.e. pairs of pixels in a spiral pattern)
        pair_estimators = _get_pair_estimators(data_name, n_uniq_dir=8)

        # Setup the local/pair destructor
        pair_canonical_destructor = _get_pair_canonical_destructor(model_name)

        # Setup a list of canonical destructors that destroy in each pixel direction
        canonical_destructor = [
            FeatureGroupsDestructor(
                groups_estimator=pair_estimator,
                group_canonical_destructor=clone(pair_canonical_destructor),
                n_jobs=n_jobs
            )
            for pair_estimator in pair_estimators
        ]

    # Shared DeepDestructorCV
    return DeepDestructorCV(
        init_destructor=init_destructor,
        canonical_destructor=canonical_destructor,
        stop_tol=deep_stop_tol,
        n_extend=1,
        cv=model_kwargs['cv'],
        refit=model_kwargs['refit'],
        silent=False,
        log_prefix='',
        random_state=0,
        # Set maximum number of layers (None for infinite)
        n_canonical_destructors=None if not model_kwargs['is_test'] else 2,
    )


def _get_inverse_logit_destructor(data_name):
    if data_name == 'mnist':
        alpha = MNIST_ALPHA
    elif data_name == 'cifar10':
        alpha = CIFAR10_ALPHA
    else:
        raise ValueError('dataset should either be mnist or cifar10')
    inverse_logit = CompositeDestructor(
        destructors=[
            IndependentDestructor(
                independent_density=IndependentDensity(
                    univariate_estimators=ScipyUnivariateDensity(
                        scipy_rv=scipy.stats.logistic,
                        scipy_fit_kwargs=dict(floc=0, fscale=1)
                    )
                )
            ),
            IndependentDestructor(
                independent_density=IndependentDensity(
                    univariate_estimators=ScipyUnivariateDensity(
                        scipy_rv=scipy.stats.uniform,
                        scipy_fit_kwargs=dict(floc=MNIST_ALPHA, fscale=1 - 2 * MNIST_ALPHA)
                    )
                )
            )
        ]
    )
    return inverse_logit


def _get_copula_destructor(hist_kwargs=None):
    if hist_kwargs is None:
        hist_kwargs = dict(bins=40, bounds=[0, 1], alpha=100)
    return CompositeDestructor(
        destructors=[
            IndependentDestructor(
                independent_density=IndependentDensity(
                    univariate_estimators=HistogramUnivariateDensity(**hist_kwargs)
                )
            ),
            IndependentInverseCdf(),
            BestLinearReconstructionDestructor(
                linear_estimator=PCA(),
                destructor=IndependentDestructor(),
            )
        ],
        random_state=0,
    )


def _get_pair_canonical_destructor(model_name):
    if model_name == 'image-pairs-tree':
        return TreeDestructor(
            tree_density=TreeDensity(
                tree_estimator=MlpackDensityTreeEstimator(
                    max_depth=None,
                    min_samples_leaf=100,
                    max_leaf_nodes=50,
                ),
                get_tree=None,
                node_destructor=None,
                uniform_weight=0.5,
            )
        )
    elif model_name == 'image-pairs-copula':
        return _get_copula_destructor()
    else:
        raise ValueError('Invalid model name "%s"') 


def _get_pair_estimators(data_name, n_uniq_dir):
    """Returns `n_uniq_dir` pair estimators in a spiral pattern."""
    def _generate_pixel_circle(radius=1):
        cur = radius*np.array([1, 1])  # Start in top right
        d = []
        d.append(cur)
        for step in np.array([[0, -1], [-1, 0], [0,1], [1, 0]]):
            for i in range(2*radius):
                cur = cur + step
                d.append(cur)
        d.pop(-1) # remove last that is a repeat
        def _rotate(a, n):
            return a[n:] + a[:n]
        return _rotate(d, radius) # Rotate to make directly east the first direction
    def _generate_pixel_spiral(n_spirals=2):
        d = []
        for i in range(n_spirals):
            d.extend(_generate_pixel_circle(radius=i+1))
            return d
    directions = np.array(_generate_pixel_spiral(n_spirals=10))

    if data_name == 'mnist':
        directions = directions[:n_uniq_dir]
        return [
            ImageFeaturePairs(
                image_shape=(28, 28), relative_position=r,
                init_offset=(0,0), step=(1,0), wrap=True
            )
            for r in directions
        ]
    elif data_name == 'cifar10':
        # Make 3d coordinates
        directions = [(d2[0], d2[1], 0) for d2 in directions[:n_uniq_dir]]
        init_offset = [(0,0,0) for _ in directions]
        # Handle color channels
        directions.extend([(0, 0, 1), (0, 0, 1), (0, 0, 1)])
        init_offset.extend([(0, 0, 0), (0, 0, 1), (0, 0, 2)])
        return [
            ImageFeaturePairs(
                image_shape=(32, 32, 3), relative_position=r,
                init_offset=io, step=(1,0,0), wrap=True
            )
            for r, io in zip(directions, init_offset)
        ]
    else:
        raise RuntimeError('Only mnist and cifar10 are supported')


def _setup_loggers(experiment_filename):
    # Setup log file and console to have same format
    log_formatter = logging.Formatter(
        fmt='%(asctime)s:%(levelname)s:%(name)s:%(process)d: %(message)s')
    log_file = logging.FileHandler(experiment_filename + '.log')
    log_file.setFormatter(log_formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)

    # Add handlers to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(log_file)

    # Adjust settings for loggers
    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger('ddl').setLevel(logging.DEBUG)


def _get_experiment_filename_and_label(data_name, model_name=None, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {}
    data_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..', 'data', 'results')
    try:
        os.makedirs(data_dir)
    except OSError:
        pass
    arg_str = '_'.join(['%s-%s' % (k, str(v)) for k, v in model_kwargs.items()])
    arg_str = arg_str.replace('.', '_')
    if len(arg_str) > 0:
        arg_str = '_' + arg_str
    filename = ('data-%s_model-%s%s'
                % (str(data_name), str(model_name), arg_str))
    pickle_filename = os.path.join(data_dir, filename)
    
    arg_str = ', '.join(['%s=%s' % (k, str(v)) for k, v in model_kwargs.items()])
    if len(arg_str) > 0:
        arg_str = ', ' + arg_str
    experiment_label = '(data=%s, model=%s%s)' % (data_name, str(model_name), arg_str)

    return pickle_filename, experiment_label

# Add fast sanity-check tests for mnist dataset
try:
    import pytest
except ImportError:
    pass
else:
    @pytest.mark.parametrize(
        'model_name', 
        # 'image-pairs-tree' not needed since covered by other tests
        ['deep-copula', 'image-pairs-copula']
    )
    def test_mnist_experiment(model_name):
        data_name = 'mnist'
        model_kwargs = dict(is_test=True, n_jobs=1)
        model_kwargs['experiment_filename'], model_kwargs['experiment_label'] = _get_experiment_filename_and_label(
            data_name, model_name=model_name, model_kwargs=model_kwargs)
        result_dict = run_experiment(data_name, model_name, model_kwargs=model_kwargs)

        # Check if test likelihood/score is as expected
        model_names = ['deep-copula', 'image-pairs-copula', 'image-pairs-tree']
        expected_test_scores = [-1.060270463188296844e+03, -1.155477974922050180e+03, -1.134326498390250208e+03]
        ind = model_names.index(model_name)
        assert(np.abs(expected_test_scores[ind] - result_dict['test_score']) < 1e-12)


if __name__ == '__main__':
    # Parse args
    all_data_names = ['mnist', 'cifar10']
    all_model_names = ['deep-copula', 'image-pairs-copula', 'image-pairs-tree']
    parser = argparse.ArgumentParser(description='Sets up and/or runs MAF experiments.')
    parser.add_argument(
        '--model_names', default=','.join(all_model_names),
        help='One or more model names separated by commas from the list %s' % str(all_model_names))
    parser.add_argument(
        '--data_names', default=','.join(all_data_names),
        help='One or more data names separated by commas from the list %s' % str(all_data_names))
    parser.add_argument(
        '--parallel_subprocesses', default=False, type=bool,
        help='Whether to use parallel subprocesses for each (model, data) experiment '
             'pair or run directly (default is False).')
    parser.add_argument(
        '--n_jobs', default=1, type=int,
        help='Number of parallel jobs to use for image-pairs models (default is 1).')
    args = parser.parse_args()
    print('Parsed args = %s' % str(args))
    print('----------------------')

    # Run experiments
    model_kwargs = vars(args).copy() # Extract model_kwargs as dictionary
    model_names = model_kwargs.pop('model_names').split(',')
    data_names = model_kwargs.pop('data_names').split(',')
    is_parallel = model_kwargs.pop('parallel_subprocesses')
    processes = []
    for data_name in data_names:
        # Make sure data has already been cached
        get_maf_data(data_name)
        for model_name in model_names:
            model_kwargs['experiment_filename'], model_kwargs['experiment_label'] = _get_experiment_filename_and_label(
                data_name, model_name=model_name, model_kwargs=model_kwargs)
            if not is_parallel:
                # Just run the experiment directly
                try:
                    run_experiment(data_name, model_name, model_kwargs)
                except RuntimeError as e:
                    if 'mlpack' not in str(e).lower():
                        raise e
                    else:
                        warnings.warn('Skipping %s because of error "%s"' % (model_name, str(e)))
            else:
                # Generate script to run experiment in parallel in separate subprocesses
                script_str = (
                    'import os\n'
                    'os.chdir(\'%s\')\n'
                    'from icml_2018_experiment import run_experiment\n'
                    'run_experiment(\'%s\', \'%s\', model_kwargs=%s)\n'
                ) % (
                    os.path.dirname(os.path.realpath(__file__)), 
                    data_name, model_name, str(model_kwargs)
                )
                echo_args = ['echo', '-e', script_str]

                # Launch subprocess which can run in parallel
                DEVNULL = open(os.devnull, 'w')
                echo = subprocess.Popen(['echo', '-e', script_str], stdout=subprocess.PIPE)
                python = subprocess.Popen(['python'], stdin=echo.stdout, stdout=DEVNULL)
                processes.append(echo)
                processes.append(python)
                print('Started subprocess for experiment %s' % model_kwargs['experiment_label'])
                print('  Appending to end of log file %s.log' % model_kwargs['experiment_filename'])

            # Remove filenames and labels for next round
            model_kwargs.pop('experiment_filename')
            model_kwargs.pop('experiment_label')

    if is_parallel:
        # Wait for all processes to finish
        print('Waiting for all subprocesses to finish')
        for p in processes:
            p.wait()
        print('All subprocesses finished!')

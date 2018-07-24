SHELL:=/bin/bash
PYTEST_FLAGS=--maxfail=1 --ignore=ddl/externals/mlpack/mlpack-mlpack-3.0.2/ --doctest-modules --pyargs

all:
	python setup.py build_ext --inplace
cleanmlpack:
	rm -rf ddl/externals/mlpack/mlpack-mlpack-3.0.2
	rm ddl/externals/mlpack/mlpack-3.0.2.tar.gz
test:
	pytest --cov=ddl $(PYTEST_FLAGS) ddl
	codecov
test/mlpack:
	pytest -k test_mlpack $(PYTEST_FLAGS) ddl
test/other:
	pytest --ignore=ddl/externals/mlpack $(PYTEST_FLAGS) ddl
test/experiments/toy:
	pytest -k test_toy_experiment $(PYTEST_FLAGS) ddl scripts/test_toy_experiment.py
test/experiments/mnist:
	# Need capture=no flag to enable output so that test will not time out in circleci
	pytest --capture=no -k test_mnist_experiment $(PYTEST_FLAGS) ddl scripts/icml_2018_experiment.py
test/special:
	echo -e "from ddl.tests.test_all import *\ntest_adversarial_tree_destructor()" | python
data/mnist: scripts/maf_data.py
	python scripts/maf_data.py mnist
data/cifar10: scripts/maf_data.py
	python scripts/maf_data.py cifar10
clean:
	python setup.py clean --all
sing-experiment:
	singularity exec --pwd ~/ /home/dinouye/davidinouye-research-latest.simg "cd research/destructive-deep-learning/scripts && python large_experiment.py --tree_alpha=0.5 --min_samples_leaf=1 tree mnist"
show-all-files:
	find . -type f | grep -v ".git" | grep -v "un~" | grep -v ".pyc"

SHELL:=/bin/bash
PYTEST_FLAGS=--maxfail=1 --ignore=ddl/externals/mlpack/mlpack-mlpack-3.0.2/ --doctest-modules --pyargs ddl scripts/test_toy_experiment.py

all:
	python setup.py build_ext --inplace
cleanmlpack:
	rm -rf ddl/externals/mlpack/mlpack-mlpack-3.0.2
	rm ddl/externals/mlpack/mlpack-3.0.2.tar.gz
test:
	pytest --cov=ddl $(PYTEST_FLAGS)
	codecov
testmlpack:
	pytest -k test_mlpack $(PYTEST_FLAGS)
testother:
	pytest --ignore=ddl/externals/mlpack $(PYTEST_FLAGS)
testexperiments:
	pytest -k test_toy_experiment $(PYTEST_FLAGS)
testspecial:
	echo -e "from ddl.tests.test_all import *\ntest_adversarial_tree_destructor()" | python
data/maf_cache: scripts/large_experiment.py
	source activate python2 && cd scripts && python large_experiment.py nomodel create_cache
clean:
	python setup.py clean --all
sing-experiment:
	singularity exec --pwd ~/ /home/dinouye/davidinouye-research-latest.simg "cd research/destructive-deep-learning/scripts && python large_experiment.py --tree_alpha=0.5 --min_samples_leaf=1 tree mnist"
show-all-files:
	find . -type f | grep -v ".git" | grep -v "un~" | grep -v ".pyc"

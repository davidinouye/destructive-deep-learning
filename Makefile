SHELL:=/bin/bash
PYTEST_FLAGS=--cov=ddl --maxfail=1 --ignore=ddl/externals/mlpack/mlpack-mlpack-3.0.2/ --doctest-modules --pyargs
PYDOCSTYLE_FILES=ddl
PYDOCSTYLE_FLAGS=--add-ignore D107

all:
	python setup.py build_ext --inplace
cleanmlpack:
	rm -rf ddl/externals/mlpack/mlpack-mlpack-3.0.2
	rm ddl/externals/mlpack/mlpack-3.0.2.tar.gz
test:
	isort --recursive --diff --check-only
	flake8
	-pydocstyle $(PYDOCSTYLE_FLAGS) $(PYDOCSTYLE_FILES)
	pytest --ignore=ddl/tests/test_mixture.py $(PYTEST_FLAGS) ddl scripts/test_toy_experiment.py
pydocstyle:
	pydocstyle $(PYDOCSTYLE_FLAGS) $(PYDOCSTYLE_FILES)
test/mixture:
	pytest --cov-append $(PYTEST_FLAGS) ddl/tests/test_mixture.py
test/mlpack:
	pytest -k test_mlpack $(PYTEST_FLAGS) ddl
test/other:
	pytest --ignore=ddl/externals/mlpack $(PYTEST_FLAGS) ddl
test/experiments/toy:
	pytest $(PYTEST_FLAGS) scripts/test_toy_experiment.py
test/experiments/mnist:
	# Need capture=no flag to enable output so that test will not time out in circleci
	pytest --cov-append --capture=no $(PYTEST_FLAGS) scripts/icml_2018_experiment.py
codecov:
	codecov
test/special:
	echo -e "from ddl.tests.test_mixture import *\ntest_autoregressive_mixture_destructor()" | python
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
add-blank-line-docstrings:
	# Reads everything into buffer and then replaces 
	# Remove one dollar sign if running directly from command line
	sed -i ':a;N;$$!ba;s/ :\n        """/ :\n\n        """/g' ddl/*.py
profile:
	#pprofile --exclude /Users/dinouye/ --exclude "<frozen" --exclude "<string" --include /Users/dinouye/research/destructive-deep-learning/ddl -o profile.out test_mixture.py
	pprofile -o profile.out test_mixture.py
push/release:
	git push -u public public_release:release
push/master:
	git push -u public public_master:master

SHELL:=/bin/bash
all:
	python setup.py build_ext --inplace
test:
	pytest --cov=ddl --maxfail=1 --doctest-modules --pyargs ddl
	codecov
testmlpack:
	pytest --maxfail=1 --ignore=ddl/externals/mlpack/mlpack-mlpack-3.0.2/ -k test_mlpack --doctest-modules --pyargs ddl
testother:
	pytest --maxfail=1 --ignore=ddl/externals/mlpack -k 'not test_mlpack' --pyargs ddl
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

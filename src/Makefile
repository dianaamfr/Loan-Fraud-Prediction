SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

PYTHON = python

.PHONY: clean train test clean_models clean_cache

RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
$(eval $(RUN_ARGS):;@:)

# clean <submission_name>
# make clean sub2
clean:
	@$(PYTHON) clean.py $(filter-out $@, $(RUN_ARGS))

# train <classifier> <submission_name>
# make train logistic_regression sub2
train:
	@$(PYTHON) train.py $(filter-out $@, $(RUN_ARGS))

# test <classifier> <submission_name>
# make test logistic_regression sub2
test:
	@$(PYTHON) test.py $(filter-out $@, $(RUN_ARGS))

# explore <table>
# make explore [account, card, client, disp, district, loan, trans]
explore:
	@-$(PYTHON) ./data_understanding/$(word 1,$(RUN_ARGS))_dataset.py || (echo "Wrong table name. Available: account, card, client, disp, district, loan, trans")

# make clustering
clustering:
	@$(PYTHON) clustering.py

clean_models:
	rm -rf models/*

clean_cache:
	rm -rf __pycache__

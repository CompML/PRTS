default: | help

POETRY_RUN := poetry run
PYTHON := $(POETRY_RUN) python

# -------------------------
# install
# -------------------------
.PHONY: install
install: ## install this project
	pip install poetry
	poetry install --no-dev

.PHONY: develop
develop: ## setup project for development
	pip install poetry
	poetry install

# -------------------------
# test
# -------------------------
.PHONY: unittest
unittest: ## run unit test for api with coverage
	$(PYTHON) -m pytest --durations=0 --cov=prts tests

.PHONY: test
test:  ## run all test
	make unittest

# -------------------------
# coding style
# -------------------------
.PHONY: lint
lint: ## type check
	$(PYTHON) -m flake8 prts

#typecheck: ## typing check
#	$(PYTHON) -m mypy \
#		--allow-redefinition \
#		--ignore-missing-imports \
#		--disallow-untyped-defs \
#		--warn-redundant-casts \
#		--no-implicit-optional \
#		--html-report ./mypyreport \
#		prts

.PHONY: format
format: ## auto format
	$(PYTHON) -m autoflake \
		--in-place \
		--remove-all-unused-imports \
		--remove-unused-variables \
		--recursive prts
	$(PYTHON) -m isort prts
	$(PYTHON) -m black \
		--line-length=119 \
		prts

.PHONY: pre-push
pre-push:  ## run before `git push`
	make test
	make format
	make lint


help:  ## Show all of tasks
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


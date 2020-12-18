default: | help

POETRY_RUN := poetry run
PYTHON := $(POETRY_RUN) python

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

format: ## auto format
	$(PYTHON) -m autoflake \
		--in-place \
		--remove-all-unused-imports \
		--remove-unused-variables \
		--recursive prts
	$(PYTHON) -m isort prts
	$(PYTHON) -m black --check --diff --line-length 119 prts


help:  ## Show all of tasks
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

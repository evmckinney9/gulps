PYTHON_VERSION = python3.12
PIP = .venv/bin/pip
PYTEST = .venv/bin/pytest
PRE_COMMIT = .venv/bin/pre-commit

.DEFAULT_GOAL := help

help:  ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} /^[a-zA-Z_-]+:.*?##/ {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

init:  ## Create venv, install deps, set up pre-commit hooks (removes existing .venv/)
	rm -rf .venv
	$(PYTHON_VERSION) -m venv .venv
	@$(PIP) install --upgrade pip
# 	@$(PIP) install setuptools_rust
	$(PIP) install -e .[cplex,dev] --quiet
	$(PIP) install -r requirements-monodromy.txt --quiet
	@$(PRE_COMMIT) install && $(PRE_COMMIT) install --hook-type commit-msg
	@$(PRE_COMMIT) autoupdate
	chmod +x .git/hooks/pre-commit

upgrade:  ## Upgrade all packages to latest versions
	$(PIP) install --upgrade pip
	$(PIP) install -e .[cplex,dev] --upgrade
	$(PIP) install -r requirements-monodromy.txt --upgrade

clean:  ## Remove temporary files and build artifacts
	@find ./ -type f -name '*.pyc' -exec rm -f {} \; 2>/dev/null || true
	@find ./ -type d -name '__pycache__' -exec rm -rf {} \; 2>/dev/null || true
	@find ./ -type f -name 'Thumbs.db' -exec rm -f {} \; 2>/dev/null || true
	@find ./ -type f -name '*~' -exec rm -f {} \; 2>/dev/null || true
	@rm -rf .cache
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -rf htmlcov
	@rm -rf .tox/
	@rm -rf docs/_build
	@rm -rf .ruff_cache
	@rm -rf src/__pycache__
	@rm -rf src/*.egg-info

ab:
	.venv/bin/python ./scripts/simple_speed.py
	.venv/bin/python ./scripts/xx_compare.py
	.venv/bin/python ./scripts/weyl_speed.py

test:  ## Run pytest
	@$(PIP) install -e .[test] --quiet
	$(PYTEST) src/tests

format:  ## Run all pre-commit hooks on all files
	@$(PIP) install -e .[format] --quiet
	$(PRE_COMMIT) run --all-files

precommit:  ## Run tests, then format

# 	@$(PIP) install -e .[test] --quiet
	$(PYTEST) src/tests
# 	@$(PIP) install -e .[format] --quiet
	$(PRE_COMMIT) run --all-files

.PHONY: help init upgrade clean test precommit format

PYTHON_VERSION = python3.12
VENV           = .venv
RUST_BIN      ?= $(HOME)/.cargo/bin
BENCH_THREADS ?= 1
BENCH_ARGS    ?= --benchmark-autosave
export PATH   := $(RUST_BIN):$(PATH)

UV := $(shell command -v uv 2>/dev/null)
ifdef UV
  INSTALL = uv pip install --python $(VENV)/bin/python
else
  INSTALL = $(VENV)/bin/pip install
endif

.DEFAULT_GOAL := help
.PHONY: help init submodules update-hooks reset-venv rebuild test bench format lint docs docs-serve docs-draft clean

help:  ## List targets
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*?##/ {printf "  %-24s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

$(VENV)/bin/python:
	$(PYTHON_VERSION) -m venv $(VENV)

submodules:  ## Fetch the solver commit pinned by this checkout
	git submodule update --init --recursive

init: submodules $(VENV)/bin/python  ## Create the development environment and install git hooks
	@$(VENV)/bin/pip install --upgrade pip --quiet
	$(INSTALL) -e .[dev] --quiet
	@$(VENV)/bin/pre-commit install --hook-type pre-commit --hook-type commit-msg

update-hooks:  ## Bump hook revisions in .pre-commit-config.yaml
	@$(VENV)/bin/pre-commit autoupdate

reset-venv:  ## Delete the environment
	rm -rf $(VENV)

rebuild:  ## Rebuild the Rust extension
	$(INSTALL) -e . --quiet --no-deps

test: rebuild  ## Run Rust and Python tests
	cargo test --manifest-path crates/Cargo.toml --locked --workspace --all-targets
	$(VENV)/bin/pytest tests

bench: rebuild  ## Measure fixed compiler workloads and save timings (.benchmarks/)
	PYTHONHASHSEED=0 RAYON_NUM_THREADS=$(BENCH_THREADS) OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
	QISKIT_IN_PARALLEL=FALSE QISKIT_FORCE_THREADS=FALSE \
	$(VENV)/bin/pytest tests/test_benchmarks.py --benchmark-enable --benchmark-only \
		--benchmark-warmup=on --benchmark-disable-gc --benchmark-min-rounds=20 \
		--benchmark-min-time=0.005 --benchmark-max-time=0.5 \
		--benchmark-save-data --benchmark-columns=median,iqr,mean,stddev,rounds,iterations $(BENCH_ARGS)

format:  ## Format Rust and Python in place
	cargo fmt --manifest-path crates/Cargo.toml -p gulps-core -p gulps-pyext
	$(VENV)/bin/pre-commit run --all-files

lint:  ## Check formatting, clippy, and ruff without changing anything
	cargo fmt --manifest-path crates/Cargo.toml -p gulps-core -p gulps-pyext --check
	cargo clippy --manifest-path crates/Cargo.toml --workspace --all-targets --all-features -- -D warnings
	$(VENV)/bin/ruff check src tests docs
	$(VENV)/bin/ruff format --check src tests docs

docs: rebuild  ## Run the doc examples and build docs/_build/html
	@rm -rf docs/_build/html docs/_build/jupyter_execute
	$(VENV)/bin/sphinx-build -b html -W --keep-going docs docs/_build/html

docs-serve: rebuild  ## Serve docs at http://localhost:8000/ and rebuild on every edit
	$(VENV)/bin/sphinx-autobuild --port 8000 docs docs/_build/html

docs-draft: rebuild  ## Like docs-serve, but examples are shown unexecuted (about 1 s per rebuild)
	GULPS_DOCS_NOEXEC=1 $(VENV)/bin/sphinx-autobuild --port 8000 docs docs/_build/draft

clean:  ## Remove build artifacts and caches
	rm -rf build dist src/*.egg-info .pytest_cache .ruff_cache crates/target \
	       docs/_build docs/apidocs/stubs src/gulps/_accelerate*.so src/gulps/_accelerate*.pyd
	find src tests docs -name __pycache__ -type d -prune -exec rm -rf {} +

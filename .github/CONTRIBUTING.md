# Contributing

## Setup

Development uses Python 3.12 and the Rust version in
[rust-toolchain.toml](../rust-toolchain.toml), which rustup installs automatically.

```sh
git clone --recurse-submodules https://github.com/evmckinney9/gulps.git
cd gulps
make init
```

For an existing checkout, run `make submodules` to fetch the solver.

## Making changes

Run `make test` and `make lint` before submitting changes. Tests rebuild the
Rust extension. Use `make format` for formatting and `make help` for other commands.

Test behavior through the public API, allowing different valid decompositions.
Solver tests belong in `crates/can_sandwich`; GULPS tests cover its integration.

The oracle tests in `tests/test_analysis.py` compare against monodromy, which
needs lrslib. CI requires them; elsewhere they are skipped when monodromy is
not installed. To run them:

```sh
.venv/bin/python -m pip install "monodromy @ git+https://github.com/qiskit-community/monodromy"
sudo apt-get install lrslib
```

Commit hooks enforce [Conventional Commits](https://www.conventionalcommits.org/).

Qiskit source updates are documented in
[crates/qiskit-pyo3-ffi/README.md](../crates/qiskit-pyo3-ffi/README.md#upgrade)
and [crates/qiskit-numerics/README.md](../crates/qiskit-numerics/README.md#update).

## Timing benchmarks

`make bench` saves timings in `.benchmarks/`:

```sh
make bench BENCH_ARGS='--benchmark-save=before'
make bench BENCH_ARGS='--benchmark-save=after --benchmark-compare=0001'
```

Replace `0001` with the saved baseline number. Compare the same workloads
and thread count on an idle machine; repeat each version in three processes.
Use `BENCH_THREADS=4` to change the thread count.

## Documentation

Edit the [user guide](../docs/index.rst) and preview it with `make docs-serve`.
Use `jupyter-execute` blocks for Python examples. Run `make docs` to check them.

For writing and presentation, follow the
[Qiskit documentation style guide](https://github.com/Qiskit/documentation/blob/main/style-guide.md)
and the [Qiskit Sphinx theme documentation](https://qiskit.github.io/qiskit_sphinx_theme/index.html).

## Upgrade the solver

`crates/can_sandwich` is a separate Git repository; commit solver changes there.
Before updating GULPS to a new solver commit, run `make rebuild`, `make test`,
and `make lint` from the GULPS root. If they pass, commit the submodule reference
and any lockfile changes in GULPS.

## Releases

Push a stable `vX.Y.Z` tag matching the version in `pyproject.toml`.
The [release workflow](../.github/workflows/release.yml) builds and tests distributions.
Review the draft GitHub release and its distributions before publishing;
publication triggers the [PyPI upload](../.github/workflows/publish.yml).

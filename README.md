![Project Banner](.github/banner.png)

![Tests](https://github.ibm.com/evm9/zz_parallel_drive/actions/workflows/tests.yml/badge.svg?branch=main)
![Format Check](https://github.ibm.com/evm9/zz_parallel_drive/actions/workflows/format-check.yml/badge.svg?branch=main)

## 📌 Project Overview
- [Link to paper:](https://arxiv.org/) Not yet :)
- **Overview**: 2024 IBM Summer Internship code
- **Objective**: Advancements in transpilation tools for supporting next-generation quantum hardware experiments utilizing heterogenous ISAs. Requires fast 2Q gate decomposition into arbitrary ISAs and optimized decompositions using phase conjugated 1Q gates.
- **Strategy**: Use monodromy polytopes quantum Littlewood-Richardson linear inequality constraints in a linear programming solver. The resulting intermediate Cartan trajectory coordinates allows for faster numerical decomposition in tractable circuit sizes.

## 📋 Repository Structure
1. **Python modules**
- `src/monodromy_lp/` Code for monodromy linear programming formulation.
- `src/zz_parallel_drive/` Utilities for parallel-driven Hamiltonians.

2. **Jupyter notebooks**. My notebooks contain a combinaion of partially developed code that was subsequently moved into python modules as it grew; as well as more polished experiments using the completed modules. Notebooks prefixed with `dev_` are primarily for scratch work and meaningful work in these notebooks is later moved into their own.

- `crates/` Support for [Python bindings to Rust](https://blog.evm9.dev/posts/01_rust_wheels/) (currently used but could speed up the code in the future).
- `Makefile` Helper command for building the python virtual environment. Comes from my [template repository](https://github.com/evmckinney9/python-template).  
    - The most useful command `make init`
- `pyroject.toml` Standard configuration file for the repository to be defined as an installable python package. Primarily responsible for defining package dependencies.

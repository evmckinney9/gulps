# GULPS

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyPI - Version](https://img.shields.io/pypi/v/gulps)](https://pypi.org/project/gulps/)
[![CI](https://github.com/evmckinney9/gulps/actions/workflows/ci.yml/badge.svg)](https://github.com/evmckinney9/gulps/actions/workflows/ci.yml)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FQCE68830.2026.00091-blue)](https://doi.org/10.1109/QCE68830.2026.00091)
[![Qiskit Ecosystem](https://qisk.it/e-0a8128d0)](https://qisk.it/e)
[![Docs](https://github.com/evmckinney9/gulps/actions/workflows/docs.yml/badge.svg?branch=main)](https://evm9.dev/gulps/)

GULPS is a two-qubit gate synthesis package for arbitrary native instruction sets.
Given a set of native two-qubit gates and a cost for each, it selects the least-cost sentence, an ordered list of native gates with single-qubit gates before, between, and after them, that implements a target unitary.
It then constructs those single-qubit gates with a dedicated solver, [can_sandwich](https://github.com/evmckinney9/can_sandwich).

```bash
pip install gulps
```

The native gates can be any two-qubit gates with fixed parameters: Qiskit standard gates with bound parameters, or your own matrices wrapped in `UnitaryGate`.

```python
from qiskit.circuit.library import CSXGate, iSwapGate
from qiskit.quantum_info import random_unitary
from gulps.decomposition import GulpsDecomposer

decomposer = GulpsDecomposer([CSXGate(), iSwapGate().power(1 / 3)], costs=[90, 120])
circuit = decomposer(random_unitary(4, seed=0))
print(circuit.draw("text"))
```

A cost can be a gate duration or any other quantity that adds up along a sentence.

- [Compilation](https://evm9.dev/gulps/compile.html): charge for single-qubit layers, run GULPS in a Qiskit pass manager, and query synthesis costs from other transpiler passes.
- [Pulse duration calibration](https://evm9.dev/gulps/calibration.html): choose which native gate durations to calibrate for your workload.
- [How synthesis works](https://evm9.dev/gulps/synthesis.html): how reachable regions guide sentence selection and circuit construction.

The [paper](https://doi.org/10.1109/QCE68830.2026.00091) describes the method, which builds on [monodromy](https://github.com/qiskit-community/monodromy).

> [!IMPORTANT]
> This software is provided as-is with no guarantee of support or maintenance.
> AI tools were used to assist in writing code in this repository.
> Bug reports and pull requests are welcome. For more information, see [the contribution guide](https://github.com/evmckinney9/gulps/blob/main/.github/CONTRIBUTING.md).

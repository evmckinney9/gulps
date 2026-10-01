# Rust crates

## Crates

| Directory | Contents | How it is updated |
|---|---|---|
| `core` | GULPS math: gate classes, KAK frames, max-plus reachability and cost-ordered search, compilation | Edit here |
| `pyext` | GULPS Python bindings and all Qiskit interop through the C API: native gates, DAG unitaries, circuit emission | Edit here |
| `can_sandwich` | The solver, a Git submodule of a separate repository | [Upgrade the solver](../.github/CONTRIBUTING.md#upgrade-the-solver) |
| [`qiskit-numerics`](qiskit-numerics/README.md) | Qiskit two-qubit numerical routines, vendored | [Update](qiskit-numerics/README.md#update) |
| [`qiskit-pyo3-ffi`](qiskit-pyo3-ffi/README.md) | Generated bindings to the installed Python Qiskit | [Upgrade](qiskit-pyo3-ffi/README.md#upgrade) |

All crates are local Cargo path dependencies. There is no Git dependency on
Qiskit. `pyext` uses `core`, `qiskit-numerics`, and `qiskit-pyo3-ffi`. `core`
uses `can_sandwich` and `qiskit-numerics`, and does not use Python or the
generated bindings.

`can_sandwich` and `qiskit-pyo3-ffi` are outside the Cargo workspace.
`crates/Cargo.lock` pins the complete GULPS Rust build, so the source
distribution omits the lockfile of the generated bindings.

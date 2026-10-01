# Qiskit numerics

Qiskit's canonical two-qubit gate, Makhlin invariants, magic basis, closest
unitary, and utility types, copied from Qiskit revision
[`c1c01ada`](https://github.com/Qiskit/qiskit/tree/c1c01ada399af13e495c27b9b22b4ff942bbad7e).
Qiskit's C API does not expose these routines. When it does, GULPS will call
them through `qiskit-pyo3-ffi` and this crate will be removed. The KAK
decomposition itself is in `gulps-core`, which needs the local rotations in the
magic basis rather than single-qubit factors.

| File | Qiskit source |
|---|---|
| `src/two_qubit_decompose/common.rs` | `crates/synthesis/src/two_qubit_decompose/common.rs` |
| `src/linalg/mod.rs` | `crates/synthesis/src/linalg/mod.rs` |
| `src/linalg/cos_sin_decomp.rs` | `crates/synthesis/src/linalg/cos_sin_decomp.rs` |
| `src/util.rs` | `crates/util/src/lib.rs` |

Each file is the complete upstream file with [local.patch](local.patch)
applied. The patch makes these changes:

- It removes the items that need pyo3, numpy, or `qiskit_circuit`: Python
  functions and the gate constants taken from the circuit crate. Qiskit's
  internal crates are not dependencies. They are unstable, and their Python
  classes would be defined a second time in the GULPS extension.
- It makes `ud` and `MAGIC` public. `two_qubit_local_invariants` and
  `local_equivalence` take array views instead of NumPy arrays.
- It imports `PyErr` and `PyResult` from this crate instead of pyo3.

`src/lib.rs` and `Cargo.toml` are local. `lib.rs` declares the upstream module
paths, names the crate `qiskit_util` for the upstream imports, and replaces
Qiskit's Python exception types with message strings. Vendored files keep
upstream formatting and are not checked by the GULPS lint configuration.

## Reproduce

From this directory, with a Qiskit Git repository that contains the revision:

```sh
qiskit=/path/to/qiskit
rev=c1c01ada399af13e495c27b9b22b4ff942bbad7e
out=$(mktemp -d)
mkdir -p "$out/src/two_qubit_decompose" "$out/src/linalg"
for f in two_qubit_decompose/common.rs linalg/mod.rs linalg/cos_sin_decomp.rs; do
    git -C "$qiskit" show "$rev:crates/synthesis/src/$f" > "$out/src/$f"
done
git -C "$qiskit" show "$rev:crates/util/src/lib.rs" > "$out/src/util.rs"
git -C "$out" apply "$PWD/local.patch"
diff -r --exclude=lib.rs src "$out/src"
```

## Update

Copy the files at the new revision, apply `local.patch`, and resolve any
conflicts. Keep the patch limited to the changes listed above. Regenerate it
with `git diff --no-index` from the upstream files to `src`, excluding
`lib.rs`.

To verify, compare the source with upstream and run `make test` and
`make lint` from the GULPS root. Compare `make bench` results with the
[timing procedure](../../.github/CONTRIBUTING.md#timing-benchmarks),
including boundary decomposition and local-only targets. Check that the
source distribution includes the licenses listed in
[MANIFEST.in](../../MANIFEST.in).

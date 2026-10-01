"""Shared assertions for the gulps test suite."""

import numpy as np
from qiskit.quantum_info import Operator

# The emitted circuit reproduces the closest unitary of its target to working
# precision; the worst observed error over Haar and boundary targets is 8e-15.
RECONSTRUCT_ATOL = 1e-12


def two_qubit_ops(circuit):
    """The two-qubit instructions of a circuit, in order."""
    return [op for op in circuit.data if op.operation.num_qubits == 2]


def count_2q(circuit):
    """Number of two-qubit instructions in a circuit."""
    return len(two_qubit_ops(circuit))


def two_qubit_names(circuit):
    """Names of the two-qubit instructions, the observable gate sequence."""
    return tuple(op.operation.name for op in two_qubit_ops(circuit))


def assert_implements(target, circuit):
    """Compare matrices, including global phase, and return the entrywise error.

    The comparison is against the closest unitary of the target: a target that
    is unitary only to input precision is realized through its closest unitary.
    """
    u, _, vt = np.linalg.svd(Operator(target).data)
    error = np.abs(Operator(circuit).data - u @ vt).max()
    assert error <= RECONSTRUCT_ATOL, f"reconstruction error {error:.2e}"
    return error

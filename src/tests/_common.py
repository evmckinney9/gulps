"""Shared helpers across the gulps test suite."""

from qiskit.quantum_info import Operator, average_gate_fidelity

FIDELITY_TOL = 1 - 1e-8


def assert_fidelity(target, circuit, label=""):
    """Assert circuit matches target above FIDELITY_TOL; label appears on failure."""
    fid = average_gate_fidelity(Operator(target), Operator(circuit))
    assert fid > FIDELITY_TOL, (
        f"Fidelity too low{' (' + label + ')' if label else ''}: {fid}"
    )


def count_2q(circuit):
    """Count two-qubit operations in a QuantumCircuit."""
    return sum(1 for op in circuit.data if op.operation.num_qubits == 2)

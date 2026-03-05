"""fSim gate factory for use as a GULPS ISA base gate."""

from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate


def fsim(theta, phi):
    """Create fSim gate."""
    qc = QuantumCircuit(2, name="fsim")
    qc.append(XXPlusYYGate(2 * theta), [0, 1])
    qc.cp(phi, 0, 1)
    return qc.to_gate()

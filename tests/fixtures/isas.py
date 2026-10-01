"""The instruction sets every decomposer test runs against."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    CXGate,
    SwapGate,
    UnitaryGate,
    XXPlusYYGate,
    iSwapGate,
)
from qiskit.quantum_info import Operator

from gulps.decomposition import GulpsDecomposer


def fsim(theta, phi):
    """The fSim gate as a UnitaryGate (gulps emits only standard gates and unitaries)."""
    qc = QuantumCircuit(2)
    qc.append(XXPlusYYGate(2 * theta), [0, 1])
    qc.cp(phi, 0, 1)
    return UnitaryGate(Operator(qc).data, label="fsim")


ISA_BUILDERS = {
    "cx": lambda: GulpsDecomposer([CXGate()], [1.0]),
    "sq4iswap": lambda: GulpsDecomposer([iSwapGate().power(1 / 4)], [0.25]),
    "sq3cx": lambda: GulpsDecomposer([CXGate().power(1 / 3)], [1 / 3]),
    "sq8cx": lambda: GulpsDecomposer([CXGate().power(1 / 8)], [0.125], max_depth=24),
    "sq2cx+sq3iswap": lambda: GulpsDecomposer(
        [CXGate().power(1 / 2), iSwapGate().power(1 / 3)],
        [0.5, 1 / 3],
    ),
    "iswap+sq2iswap+sq3iswap": lambda: GulpsDecomposer(
        [iSwapGate(), iSwapGate().power(1 / 2), iSwapGate().power(1 / 3)],
        [1.0, 0.5, 1 / 3],
    ),
    "cx+sq2cx+sq4cx": lambda: GulpsDecomposer(
        [CXGate(), CXGate().power(1 / 2), CXGate().power(1 / 4)],
        [1.0, 0.5, 0.25],
    ),
    "fsim+sq4iswap": lambda: GulpsDecomposer(
        [fsim(np.pi / 2, np.pi / 6), iSwapGate().power(1 / 4)],
        [1.0, 0.25],
    ),
    "cx+sq2iswap+free_swap": lambda: GulpsDecomposer(
        [CXGate(), iSwapGate().power(1 / 2), SwapGate()],
        [1.0, 0.5, 0.0],
    ),
}

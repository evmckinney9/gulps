from __future__ import annotations

import numpy as np
from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from qiskit.circuit.library import iSwapGate
from qiskit.quantum_info import Operator


def _iswap_power_matrix(d: int) -> np.ndarray:
    """Return the 4x4 unitary matrix for iSwap^(1/d)."""
    return Operator(iSwapGate().power(1 / d)).data


class Sqrt3iSwapGate(ConstantGate, QubitGate):
    _num_qudits = 2
    _qasm_name = "sq3iswap"
    _utry = UnitaryMatrix(_iswap_power_matrix(3))


class Sqrt4iSwapGate(ConstantGate, QubitGate):
    _num_qudits = 2
    _qasm_name = "sq4iswap"
    _utry = UnitaryMatrix(_iswap_power_matrix(4))

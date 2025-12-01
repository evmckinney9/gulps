from __future__ import annotations

from bqskit.ir.gates import ConstantUnitaryGate
from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class Sqrt4iSwapGate(ConstantGate, QubitGate):
    _num_qudits = 2
    _qasm_name = "sq4iswap"
    _utry = UnitaryMatrix(
        [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.92387953 + 0.0j, 0.0 + 0.38268343j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.38268343j, 0.92387953 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
        ]
    )

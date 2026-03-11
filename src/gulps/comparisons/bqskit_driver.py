# Copyright 2025-2026 Lev S. Bishop, Evan McKinney
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

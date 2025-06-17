"""Define BGate matrix using sqrt{CX} gates."""

# as far as I know this is not efine dexplicitly in Qiskit
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, UnitaryGate
from qiskit.quantum_info import Operator

# XXX this gives something local to B but is not the exact conventional unitary
# qc = QuantumCircuit(2)
# qc.append(CXGate().power(1 / 2), [0, 1])
# qc.append(CXGate().power(1 / 2), [0, 1])
# qc.append(CXGate().power(1 / 2), [1, 0])
# BGate = UnitaryGate(Operator(qc).data)


# BGate = qt.Qobj(Operator(qc).data)
# c1c2c3(qt.Qobj(Operator(qc).data)) (0.5, 0.25, 0)


class BGate(UnitaryGate):
    """BGate (continuous).

    Ref: https://threeplusone.com/pubs/on_gates.pdf
    """

    def __init__(self, theta=np.pi / 2):
        """BGate(theta) constructor."""
        c1 = np.cos(theta / 4)
        c3 = np.cos(3 * theta / 4)
        s1 = np.sin(theta / 4)
        s3 = np.sin(3 * theta / 4)
        return super().__init__(
            [
                [c1, 0, 0, 1j * s1],
                [0, c3, 1j * s3, 0],
                [0, 1j * s3, c3, 0],
                [1j * s1, 0, 0, c1],
            ],
            label="b",
        )

"""Gate invariants utilities for two-qubit gates (monodromy, Makhlin, Weyl, etc.)."""

import logging
from typing import Tuple

import numpy as np
from qiskit._accelerate import two_qubit_decompose
from qiskit.circuit.library import (
    IGate,
    SXdgGate,
    SXGate,
    UnitaryGate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.quantum_info import Operator
from qiskit.synthesis.two_qubit import TwoQubitWeylDecomposition

from gulps.config import GulpsConfig

logger = logging.getLogger(__name__)

# TODO, this works great and is general enough and robust
# but might be over-calculating some parts by reusing TwoQubitWeylDecomposition
# because we already know the Cartan KAK invariants...
# maybe there is something more efficient if we purely need the exterior locals?
# especially useful if we need something cleaner when dealing with Symbolic params.


def recover_local_equivalence(
    U_target: np.ndarray,
    U_basis: np.ndarray,
    config: GulpsConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Find single-qubit corrections k1, k2, k3, k4 and a global phase so that:.

        U_target ~= (k1 x k2) · U_basis · (k3 x k4) · exp(i * global_phase).

    Cases:
      1) exact Weyl match         -> identity corrections
      2) target=(a,b,b) & basis=(a,b,-b)
                                  -> insert X,Z before and I,Y after
      3) otherwise                -> ValueError
    """
    if config is None:
        config = GulpsConfig()
    tol = config.equiv_recovery_tol

    # Weyl decompose both unitaries
    spec = None
    T = TwoQubitWeylDecomposition(U_target, _specialization=spec)
    B = TwoQubitWeylDecomposition(U_basis, _specialization=spec)

    a1, b1, c1 = T.a, T.b, T.c
    a2, b2, c2 = B.a, B.b, B.c

    # 1) exact Weyl match?
    if (
        np.isclose(a1, a2, atol=tol)
        and np.isclose(b1, b2, atol=tol)
        and np.isclose(c1, c2, atol=tol)
    ):
        k4 = T.K1l @ B.K1l.conj().T
        k3 = T.K1r @ B.K1r.conj().T
        k2 = B.K2l.conj().T @ T.K2l
        k1 = B.K2r.conj().T @ T.K2r
        return k1, k2, k3, k4, (T.global_phase - B.global_phase)

    # 2) rho reflection case: target=(a,b,c), basis=(pi/2-a,b,-c)?
    # if a1==a2 or a1 == pi/2 - a2
    # and b==b
    # and c1=-c2
    if (
        (np.isclose(a1, a2, atol=tol) or np.isclose(a1, np.pi / 2 - a2, atol=tol))
        and np.isclose(b1, b2, atol=tol)
        and np.isclose(c1, -c2, atol=tol)
    ):
        logger.debug("Detected rho reflect; inserting Pauli corrections.")
        P0 = YGate().to_matrix()  # pre on qubit 0
        Q0 = XGate().to_matrix()  # post on qubit 0
        Q1 = ZGate().to_matrix()  # post on qubit 1

        k4 = T.K1l @ P0 @ B.K1l.conj().T
        k3 = T.K1r @ B.K1r.conj().T
        k2 = B.K2l.conj().T @ Q1 @ T.K2l
        k1 = B.K2r.conj().T @ Q0 @ T.K2r
        return k1, k2, k3, k4, (T.global_phase - B.global_phase)

    # 3) cannot recover
    diffs = np.abs([a1 - a2, b1 - b2, c1 - c2])
    raise ValueError(f"Cannot recover local equivalence; Weyl differences {diffs}")
    # proceed like normal
    logger.warning(
        f"Cannot recover local equivalence; Weyl differences {diffs}. Proceeding with best effort."
    )
    k4 = T.K1l @ B.K1l.conj().T
    k3 = T.K1r @ B.K1r.conj().T
    k2 = B.K2l.conj().T @ T.K2l
    k1 = B.K2r.conj().T @ T.K2r
    return k1, k2, k3, k4, (T.global_phase - B.global_phase)


if __name__ == "__main__":
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import CXGate, CZGate, UnitaryGate
    from qiskit.quantum_info import Operator, average_gate_fidelity

    target_gate = CXGate()
    basis_gate = CZGate()
    print(average_gate_fidelity(target_gate, basis_gate))

    k1, k2, k3, k4, gphase = recover_local_equivalence(target_gate, basis_gate)
    qc = QuantumCircuit(2, global_phase=gphase)
    qc.append(UnitaryGate(k1), [0])
    qc.append(UnitaryGate(k2), [1])
    qc.append(basis_gate, [0, 1])
    qc.append(UnitaryGate(k3), [0])
    qc.append(UnitaryGate(k4), [1])
    print(average_gate_fidelity(target_gate, Operator(qc)))
    print(qc.draw())
    # print("\n.")
    # print("\n.")

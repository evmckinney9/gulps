"""Some miscellaneous utilities."""

import numpy as np
from monodromy.coordinates import positive_canonical_to_monodromy_coordinate
from qiskit._accelerate.two_qubit_decompose import weyl_coordinates
from qiskit.synthesis.two_qubit import TwoQubitWeylDecomposition
from weylchamber import canonical_gate


def unitary_to_mono_coordinates(U):
    # NOTE this has more precision than monodromy's unitary_to_monodromy_coordinate
    # perhaps not necessarily more precision, but consistently rounding down
    # whereas monodromy appears to be rounding either up or down
    a, b, c = positive_canonical_to_monodromy_coordinate(*weyl_coordinates(U))
    return (a, b, c, -1.0 * (a + b + c))


def mono_coordinates_to_makhlin(x, y, z, _=None):
    """Convert monodromy coordinates to makhlin invariants.

    NOTE previously I was using mono->positive canonical -> invariants
    (qiskit) using qiskit.synthesis.two_qubit.local_equivalence() but
    this method has np.round() which we want to avoid for high precision
    in the root-finding.
    """
    normalizing_factor = np.pi
    weyl = np.array(
        [
            (x + y) / 2 * normalizing_factor,
            (z + x) / 2 * normalizing_factor,
            (y + z) / 2 * normalizing_factor,
        ],
        dtype=np.double,
    )
    g0_equiv = np.prod(np.cos(2 * weyl) ** 2) - np.prod(np.sin(2 * weyl) ** 2)
    g1_equiv = np.prod(np.sin(4 * weyl)) / 4
    g2_equiv = (
        4 * np.prod(np.cos(2 * weyl) ** 2)
        - 4 * np.prod(np.sin(2 * weyl) ** 2)
        - np.prod(np.cos(4 * weyl))
    )
    return np.array([g0_equiv, g1_equiv, g2_equiv], dtype=np.double)
    # return np.round([g0_equiv, g1_equiv, g2_equiv], 12) + 0.0


def mono_coordinates_to_CAN(x, y, z, _=None):
    normalizing_factor = 2.0
    c1, c2, c3 = np.array(
        [
            (x + y) / 2 * normalizing_factor,
            (z + x) / 2 * normalizing_factor,
            (y + z) / 2 * normalizing_factor,
        ],
        dtype=np.double,
    )
    return canonical_gate(c1, c2, c3).full()


def recover_local_equivalence(U_target, U_basis):
    """Find local 'exterior' gates to complete 2Q decomposition.

    U1 and U2 are locally equivalent, but need to find local gates such that
    local.U_basis.local is exactly equivalent to U_target.
           ┌────┐ ┌────────────┐┌─────────┐       ┌────────────┐
    q_0: ──┤ K1 ├─┤            ├┤ K3      ├      ─┤            ├─
         ┌─┴────┴┐│  U_basis   │└┬───────┬┘   :=  |  U_target  |
    q_1: ┤  K2   ├┤            ├─┤ K4    ├─      ─┤            ├─
         └───────┘└────────────┘ └───────┘        └────────────┘

    NOTE this method is essentially TwoQubitBasisDecomposer.decomp1()
    """
    target_decomp = TwoQubitWeylDecomposition(U_target, fidelity=1.0)
    basis_decomp = TwoQubitWeylDecomposition(U_basis, fidelity=1.0)

    closeness_vector = (
        target_decomp.a - basis_decomp.a,
        target_decomp.b - basis_decomp.b,
        target_decomp.c - basis_decomp.c,
    )
    # XXX this atol is a bit too generous
    # makes up for imperfect convergence...
    if not np.all(
        [np.isclose(closeness, 0.0, atol=1e-3) for closeness in closeness_vector]
    ):
        raise ValueError(
            f"Tried to recover local equivalence on gates that were not locally equivalent. \
            Often this is a matter of precision. The difference between expected and given \
            weyl coords was {closeness_vector}. If this is close to 0, then adjust tolerances."
        )
    k4 = target_decomp.K1l @ np.conjugate(basis_decomp.K1l).T
    k3 = target_decomp.K1r @ np.conjugate(basis_decomp.K1r).T
    k2 = np.conjugate(basis_decomp.K2l).T @ target_decomp.K2l
    k1 = np.conjugate(basis_decomp.K2r).T @ target_decomp.K2r
    global_phase = target_decomp.global_phase - basis_decomp.global_phase
    return k1, k2, k3, k4, global_phase


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

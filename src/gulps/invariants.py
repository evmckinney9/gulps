"""Some miscellaneous utilities."""

import warnings
from typing import Optional, Tuple

import numpy as np
from monodromy.coordinates import positive_canonical_to_monodromy_coordinate
from qiskit._accelerate.two_qubit_decompose import weyl_coordinates
from qiskit.circuit import Gate
from qiskit.quantum_info import Operator
from qiskit.synthesis.two_qubit import TwoQubitWeylDecomposition
from weylchamber import canonical_gate

LEN_GATE_INVARIANTS = 3


def unitary_to_mono_coordinates(U) -> Tuple[float, float, float, float]:
    # NOTE this has more precision than monodromy's unitary_to_monodromy_coordinate
    # perhaps not necessarily more precision, but consistently rounding down
    # whereas monodromy appears to be rounding either up or down
    # XXX breaking change? Previously I passed U directly into weyl_coordinates
    U_op = Operator(U).data
    a, b, c = positive_canonical_to_monodromy_coordinate(*weyl_coordinates(U_op))
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


class MonodromyLPGate:
    """Minimal gate class for monodromy LP with only fully-defined gates."""

    def __init__(
        self, logspec: Tuple[float, float, float, float], name: Optional[str] = None
    ):
        self.logspec = logspec  # Full logspec from canonical coordinates
        self._definition = logspec[
            :LEN_GATE_INVARIANTS
        ]  # Monodromy coordinates (first 3)
        self.name = name or "2QGate"

    @classmethod
    def from_unitary(cls, gate: Gate, name: Optional[str] = None) -> "MonodromyLPGate":
        coords = unitary_to_mono_coordinates(gate)
        return cls(logspec=coords, name=name)

    @property
    def definition(self) -> Tuple[float, float, float]:
        """Return monodromy coordinates."""
        return self._definition

    def rho_reflect(self) -> "MonodromyLPGate":
        """Return the rho-reflected version of this gate."""
        rho_coords = (
            self.logspec[2] + 0.5,
            self.logspec[3] + 0.5,
            self.logspec[0] - 0.5,
            self.logspec[1] - 0.5,
        )
        return MonodromyLPGate(logspec=rho_coords, name=f"*{self.name}")

    def __str__(self) -> str:
        return self.name


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

    # XXX this atol is a bit too generous
    # makes up for imperfect convergence...
    local_coords1 = np.array([target_decomp.a, target_decomp.b, target_decomp.c])
    local_coords2 = np.array([basis_decomp.a, basis_decomp.b, basis_decomp.c])
    if not np.all(
        np.isclose(np.abs(local_coords1 - local_coords2), np.zeros(3), atol=1e-4)
    ):
        if not np.isclose(np.abs(target_decomp.c), np.abs(basis_decomp.c)):
            raise ValueError(
                f"Tried to recover local equivalence on gates that were not locally equivalent. \
                Often this is a matter of precision. The difference between expected and given \
                weyl coords was {np.abs(local_coords1 - local_coords2)}. If this is close to 0, then adjust tolerances."
            )
        else:
            warnings.warn(
                "Tried to recover local equivalence, but the c parameter had a sign difference."
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

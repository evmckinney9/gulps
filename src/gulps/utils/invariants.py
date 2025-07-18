"""Gate invariants utilities for two-qubit gates (monodromy, Makhlin, Weyl, etc.)."""

import warnings
from typing import Optional, Tuple

import numpy as np
from monodromy.coordinates import (
    positive_canonical_to_monodromy_coordinate,
    unitary_to_monodromy_coordinate,
)
from qiskit._accelerate.two_qubit_decompose import weyl_coordinates
from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator
from qiskit.synthesis.two_qubit import TwoQubitWeylDecomposition
from weylchamber import canonical_gate

LEN_GATE_INVARIANTS = 3


class GateInvariants:
    """Unified representation of two-qubit gate invariants."""

    def __init__(
        self,
        logspec: Tuple[float, float, float, float],
        name: Optional[str] = None,
        unitary: Optional[np.ndarray] = None,
    ):
        if len(logspec) == LEN_GATE_INVARIANTS:
            logspec = logspec + (-1.0 * sum(logspec),)
        self.logspec = logspec
        self._monodromy = logspec[:LEN_GATE_INVARIANTS]  # Monodromy
        self.name = name or "2QGate"
        self._unitary = unitary  # Optional reference to original unitary

        self._weyl = None
        self._makhlin = None
        self._canonical_matrix = None

    @classmethod
    def from_unitary(
        cls, gate: Gate | np.ndarray, name: Optional[str] = None
    ) -> "GateInvariants":
        U = Operator(gate).data
        if not isinstance(gate, Gate):
            gate = UnitaryGate(gate, label=name)
        coords = cls._unitary_to_mono_coordinates(U)
        return cls(logspec=coords, name=name, unitary=gate)

    @classmethod
    def from_weyl(cls, coords: Tuple[float, float, float]):
        """Create from weyl coordinates."""
        positive_canonical = np.pi / 2 * np.array(coords)
        return cls(positive_canonical_to_monodromy_coordinate(*positive_canonical))

    @staticmethod
    def _unitary_to_mono_coordinates(U) -> Tuple[float, float, float, float]:
        # using the convention breaks things # XXX ???
        # return tuple(unitary_to_monodromy_coordinate(U))
        a, b, c = positive_canonical_to_monodromy_coordinate(*weyl_coordinates(U))
        return (a, b, c, -1.0 * (a + b + c))

    @property
    def unitary(self) -> np.ndarray:
        """Return the unitary matrix of this gate."""
        if self._unitary is None:
            return self.canonical_matrix
        return self._unitary

    @property
    def monodromy(self) -> Tuple[float, float, float]:
        """Monodromy invariants."""
        return self._monodromy

    @property
    def weyl(self) -> np.ndarray:
        """Weyl invariants (lazy computed from monodromy).

        Matches the normalized convention from weylchamber.c1c2c3
        """
        if self._weyl is None:
            self._weyl = np.array(
                [
                    (self.monodromy[0] + self.monodromy[1]),
                    (self.monodromy[2] + self.monodromy[0]),
                    (self.monodromy[1] + self.monodromy[2]),
                ],
                dtype=np.double,
            )
        return np.abs(self._weyl)

    @property
    def makhlin(self) -> np.ndarray:
        """Makhlin invariants (lazy computed from Weyl).

        Matches the normalized convention from weylchamber.g1g2g3
        """
        from qiskit._accelerate.two_qubit_decompose import (
            two_qubit_local_invariants as tqli_rs,
        )

        return tqli_rs(Operator(self.unitary).data)

        # if self._makhlin is None:
        #     weyl = np.pi * self.weyl / 2  # Normalize back for Makhlin formula
        #     g0 = np.prod(np.cos(2 * weyl) ** 2) - np.prod(np.sin(2 * weyl) ** 2)
        #     g1 = np.prod(np.sin(4 * weyl)) / 4
        #     g2 = (
        #         4 * np.prod(np.cos(2 * weyl) ** 2)
        #         - 4 * np.prod(np.sin(2 * weyl) ** 2)
        #         - np.prod(np.cos(4 * weyl))
        #     )
        #     self._makhlin = np.array([g0, g1, g2], dtype=np.double)
        # return self._makhlin

    @property
    def canonical_matrix(self) -> np.ndarray:
        """Canonical gate matrix (lazy computed from Weyl)."""
        if self._canonical_matrix is None:
            self._canonical_matrix = canonical_gate(*self.weyl).full()
        return self._canonical_matrix

    @property
    def strength(self) -> float:
        return min(sum(self.monodromy), sum(self.rho_reflect().monodromy))

    def rho_reflect(self) -> "GateInvariants":
        """Rho-reflected version of this gate."""
        # TODO XXX double check this.
        rho_coords = (
            self.logspec[2] + 0.5,
            self.logspec[3] + 0.5,
            self.logspec[0] - 0.5,
            self.logspec[1] - 0.5,
        )
        return GateInvariants(logspec=rho_coords, name=f"*{self.name}")

    def __str__(self) -> str:
        return self.name


if __name__ == "__main__":
    from qiskit.circuit.library import iSwapGate
    from weylchamber import c1c2c3, g1g2g3

    u = iSwapGate().power(1 / 2).to_matrix()
    g = GateInvariants.from_unitary(u)
    assert np.allclose(g1g2g3(u), g.makhlin)
    assert np.allclose(c1c2c3(u), g.weyl)


def recover_local_equivalence(U_target, U_basis):
    """Find local gates such that local.U_basis.local = U_target."""
    target_decomp = TwoQubitWeylDecomposition(U_target, fidelity=1.0)
    basis_decomp = TwoQubitWeylDecomposition(U_basis, fidelity=1.0)

    local_coords1 = np.array([target_decomp.a, target_decomp.b, target_decomp.c])
    local_coords2 = np.array([basis_decomp.a, basis_decomp.b, basis_decomp.c])
    if not np.allclose(np.abs(local_coords1 - local_coords2), 0, atol=1e-2):
        if not np.isclose(np.abs(target_decomp.c), np.abs(basis_decomp.c)):
            raise ValueError(
                f"Gates are not locally equivalent. Difference: {np.abs(local_coords1 - local_coords2)}"
            )
        print(
            "Warning: Possible sign difference in c parameter during local equivalence recovery."
        )
        warnings.warn(
            "Possible sign difference in c parameter during local equivalence recovery."
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

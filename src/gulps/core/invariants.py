"""Gate invariants utilities for two-qubit gates (monodromy, Makhlin, Weyl, etc.)."""

import logging
from typing import Optional, Tuple

import numpy as np
from monodromy.coordinates import (
    positive_canonical_to_monodromy_coordinate,
    unitary_to_monodromy_coordinate,
)
from qiskit._accelerate.two_qubit_decompose import two_qubit_local_invariants as tqli_rs
from qiskit._accelerate.two_qubit_decompose import weyl_coordinates
from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

logger = logging.getLogger(__name__)

LEN_GATE_INVARIANTS = 3


class GateInvariants:
    """Unified representation of two-qubit gate invariants."""

    def __init__(
        self,
        logspec: Tuple[np.float64, np.float64, np.float64, np.float64],
        name: Optional[str] = None,
        unitary: Optional[np.ndarray] = None,
        rho_reflect: Optional["GateInvariants"] = None,
    ):
        # if logspec is a np.ndarray, convert to tuple
        if isinstance(logspec, np.ndarray):
            logspec = tuple(logspec.tolist())
        if len(logspec) == LEN_GATE_INVARIANTS:
            logspec = logspec + (-1.0 * sum(logspec),)
        self.logspec = logspec
        self._monodromy = np.array(
            logspec[:LEN_GATE_INVARIANTS], dtype=np.float64
        )  # Monodromy
        self.name = name or "2QGate"
        self._unitary = unitary  # Optional reference to original unitary
        self._rho_reflect = rho_reflect

        # Quantize ONCE
        self._key = tuple(np.rint(self.monodromy * 1e12).astype(np.int64))

        self._weyl = None
        self._makhlin = None
        self._canonical_matrix = None

    @classmethod
    def from_unitary(
        cls, gate: Gate | np.ndarray, enforce_alcove=False, name: Optional[str] = None
    ) -> "GateInvariants":
        U = Operator(gate).data
        if not isinstance(gate, UnitaryGate):
            gate = UnitaryGate(U, label=name)
        if enforce_alcove:
            coords = tuple(unitary_to_monodromy_coordinate(U))
        else:
            coords = cls._unitary_to_mono_coordinates(U)
        return cls(logspec=coords, name=name, unitary=gate)

    @classmethod
    def from_weyl(cls, coords: Tuple[np.float64, np.float64, np.float64]):
        """Create from weyl coordinates."""
        positive_canonical = np.pi / 2 * np.array(coords)
        return cls(positive_canonical_to_monodromy_coordinate(*positive_canonical))

    @staticmethod
    def _unitary_to_mono_coordinates(
        U,
    ) -> Tuple[np.float64, np.float64, np.float64, np.float64]:
        return tuple(unitary_to_monodromy_coordinate(U))
        # NOTE, backup conventon. the above used to give some unexplained precision issues
        # a, b, c = positive_canonical_to_monodromy_coordinate(*weyl_coordinates(U))
        # return (a, b, c, -1.0 * (a + b + c))

    @property
    def unitary(self) -> UnitaryGate:
        """Return the unitary matrix of this gate."""
        if self._unitary is None:
            return UnitaryGate(self.canonical_matrix, check_input=False)
        return self._unitary

    @property
    def monodromy(self) -> Tuple[np.float64, np.float64, np.float64]:
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
                dtype=np.float64,
            )
        return self._weyl

    @property
    def makhlin(self) -> np.ndarray:
        """Makhlin invariants (lazy computed from Weyl).

        Matches the normalized convention from weylchamber.g1g2g3
        """
        return tqli_rs(Operator(self.unitary).data)

    @property
    def canonical_matrix(self) -> np.ndarray:
        """Canonical gate matrix from Weyl coordinates."""
        if self._canonical_matrix is None:
            c1, c2, c3 = self.weyl
            a = np.pi / 2 * c1
            b = np.pi / 2 * c2
            g = np.pi / 2 * c3
            eig = np.exp(1j * g)
            eig_c = np.exp(-1j * g)
            cam = np.cos(a - b)
            sam = np.sin(a - b)
            cap = np.cos(a + b)
            sap = np.sin(a + b)
            self._canonical_matrix = np.array(
                [
                    [eig * cam, 0, 0, 1j * eig * sam],
                    [0, eig_c * cap, 1j * eig_c * sap, 0],
                    [0, 1j * eig_c * sap, eig_c * cap, 0],
                    [1j * eig * sam, 0, 0, eig * cam],
                ],
                dtype=np.complex128,
            )
        return self._canonical_matrix

    @property
    def strength(self) -> np.float64:
        return min(sum(self.monodromy), sum(self.rho_reflect.monodromy))

    @property
    def rho_reflect(self) -> "GateInvariants":
        """Rho-reflected version of this gate. Cached bidirectionally."""
        if self._rho_reflect is not None:
            return self._rho_reflect

        a, b, c, d = map(np.float64, self.logspec)
        rho_coords = (
            np.float64(c + 0.5),
            np.float64(d + 0.5),
            np.float64(a - 0.5),
            np.float64(b - 0.5),
        )

        # Create the reflected object with backward reference to this one
        self._rho_reflect = GateInvariants(
            logspec=rho_coords,
            name=f"*{self.name}",
            rho_reflect=self,
        )
        self._rho_reflect._rho_reflect = self
        return self._rho_reflect

    @property
    def is_identity(self) -> bool:
        return all(np.isclose(x, 0.0) for x in self.monodromy)

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        if not isinstance(other, GateInvariants):
            return NotImplemented
        return self._key == other._key

    def __hash__(self) -> int:
        # Round to 15-digit precision and convert to integers for stable hashing
        return hash(self._key)

    def plot(self):
        from gulps.viz.invariant_viz import scatter_plot

        return scatter_plot([self])


if __name__ == "__main__":
    from qiskit.circuit.library import iSwapGate
    from weylchamber import c1c2c3, g1g2g3

    u = iSwapGate().power(1 / 2).to_matrix()
    g = GateInvariants.from_unitary(u)
    assert np.allclose(g1g2g3(u), g.makhlin)
    assert np.allclose(c1c2c3(u), g.weyl)

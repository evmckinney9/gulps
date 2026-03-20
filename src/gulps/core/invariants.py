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

"""Gate invariants utilities for two-qubit gates (monodromy, Makhlin, Weyl, etc.)."""

import numpy as np
from qiskit._accelerate.two_qubit_decompose import two_qubit_local_invariants as tqli_rs
from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate

from gulps._accelerate import (
    canonical_matrix as _canonical_matrix,
    monodromy_from_weyl as _monodromy_from_weyl_rs,
    weyl_coordinates as _rust_weyl,
    weyl_from_monodromy as _weyl_from_monodromy_rs,
)


LEN_GATE_INVARIANTS = 3


class GateInvariants:
    """Unified representation of two-qubit gate invariants."""

    def __init__(
        self,
        logspec: tuple[np.float64, np.float64, np.float64, np.float64],
        name: str | None = None,
        rho_reflect: "GateInvariants | None" = None,
    ):
        """Create from a 3- or 4-element logspec tuple."""
        if isinstance(logspec, np.ndarray):
            logspec = tuple(logspec.tolist())
        if len(logspec) == LEN_GATE_INVARIANTS:
            logspec = logspec + (-1.0 * sum(logspec),)
        self.logspec = logspec
        self._monodromy = np.array(logspec[:LEN_GATE_INVARIANTS], dtype=np.float64)
        self.name = name or "2QGate"
        self._matrix = None  # np.ndarray complex128, set by from_unitary
        self._gate_ref = None  # original Qiskit Gate, for DAG insertion
        self._rho_reflect = rho_reflect

        # Quantize ONCE
        self._key = tuple(np.rint(self.monodromy * 1e12).astype(np.int64))

        self._weyl = None
        self._makhlin = None
        self._canonical_matrix = None
        self._strength = None

    @classmethod
    def from_unitary(
        cls, gate: Gate | np.ndarray, name: str | None = None
    ) -> "GateInvariants":
        """Construct from a Qiskit Gate or 4x4 unitary matrix."""
        if isinstance(gate, np.ndarray):
            U = gate
        elif isinstance(gate, Gate):
            try:
                U = gate.to_matrix()
            except Exception:
                from qiskit.quantum_info import Operator

                U = Operator(gate).data
        else:
            U = np.asarray(gate)
        U = np.asarray(U, dtype=np.complex128)

        c = _rust_weyl(U)
        coords = tuple(_monodromy_from_weyl_rs(float(c[0]), float(c[1]), float(c[2])))
        inv = cls(logspec=coords, name=name)
        inv._matrix = U
        inv._gate_ref = gate if isinstance(gate, Gate) else None
        if inv._gate_ref is not None and name and inv._gate_ref.name != name:
            inv._gate_ref = inv._gate_ref.copy(name=name)
        return inv

    @classmethod
    def from_weyl(cls, coords: tuple[np.float64, np.float64, np.float64]):
        """Create from weyl coordinates."""
        return cls(tuple(_monodromy_from_weyl_rs(*coords)))

    @property
    def matrix(self) -> np.ndarray:
        """4x4 unitary as np.ndarray complex128."""
        if self._matrix is None:
            self._matrix = self.canonical_matrix
        return self._matrix

    @property
    def gate(self) -> Gate:
        """Return the original Qiskit Gate for DAG circuit construction."""
        return self._gate_ref or UnitaryGate(self.matrix, check_input=False)

    @property
    def monodromy(self) -> tuple[np.float64, np.float64, np.float64]:
        """Monodromy invariants."""
        return self._monodromy

    @property
    def weyl(self) -> np.ndarray:
        """Weyl invariants (lazy computed from monodromy).

        Matches the normalized convention from weylchamber.c1c2c3
        """
        if self._weyl is None:
            mono = np.asarray(self.monodromy[:3], dtype=np.float64)
            self._weyl = np.asarray(_weyl_from_monodromy_rs(mono))
        return self._weyl

    @property
    def makhlin(self) -> np.ndarray:
        """Makhlin invariants via Qiskit Rust backend.

        Returns [Re(G1), Im(G1), Re(G2)] per Zhang et al. PRA 67, 042313.
        """
        if self._makhlin is None:
            self._makhlin = tqli_rs(self.matrix)
        return self._makhlin

    @property
    def canonical_matrix(self) -> np.ndarray:
        """Canonical gate matrix from Weyl coordinates."""
        if self._canonical_matrix is None:
            self._canonical_matrix = _canonical_matrix(*self.weyl)
        return self._canonical_matrix

    @property
    def strength(self) -> np.float64:
        """Minimum total monodromy weight of this gate or its rho-reflect."""
        if self._strength is None:
            self._strength = min(sum(self.monodromy), sum(self.rho_reflect.monodromy))
        return self._strength

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
        """Return True when this gate is locally equivalent to the identity."""
        m = self._monodromy
        return abs(m[0]) < 1e-8 and abs(m[1]) < 1e-8 and abs(m[2]) < 1e-8

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
        """Scatter this gate in a 3-D Weyl chamber plot."""
        from gulps.viz.invariant_viz import scatter_plot

        return scatter_plot([self])

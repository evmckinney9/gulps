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

"""Segment-wise synthesis: solve segments, stitch into circuit.

Architecture:
  Phase 1: Solve all segments (sequential or Rayon-parallel, decided in Rust).
  Phase 2: Accumulate P + intermediate recoveries (single Rust FFI call).
  Phase 3: Apply gates to DAG (Qiskit API, Python).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit

from gulps._accelerate import recover_local_equiv as recover_local_equivalence
from gulps._accelerate import solve_batch as _rust_solve_batch
from gulps._accelerate import stitch_segments as _rust_stitch
from gulps.config import GulpsConfig
from gulps.core.invariants import GateInvariants


@dataclass
class SegmentSolution:
    """Result of a single segment solve (local unitaries and diagnostics)."""

    u0: np.ndarray | None  # 2x2 or None if failure
    u1: np.ndarray | None  # 2x2 or None if failure
    max_residual: float  # worst-case residual component (Linf norm)
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class SegmentSynthesizer:
    """Solve all segments and stitch into a circuit.

    Each segment finds single-qubit unitaries (u0, u1) such that
    G @ kron(u1, u0) @ C is locally equivalent to the target invariants,
    where C is the prefix (accumulated product) and G is the basis gate.
    """

    def __init__(self, config: GulpsConfig | None = None):
        """Initialize with optional configuration (default if None)."""
        self.config = config or GulpsConfig()

    def synthesize_segments(
        self,
        gate_list: list[GateInvariants],
        invariant_list: list[GateInvariants],
        target: GateInvariants,
        *,
        return_dag: bool = False,
    ) -> QuantumCircuit | DAGCircuit:
        """Solve for local parameters of each interior segment."""
        if len(gate_list) != len(invariant_list):
            raise ValueError("Gate list and invariant list must have the same length.")
        if len(gate_list) < 1:
            raise ValueError("At least one gate is required for segment synthesis.")

        n_inner = len(gate_list) - 1
        dag = circuit_to_dag(QuantumCircuit(2, global_phase=0))
        qreg = dag.qregs["q"]

        # Phase 1: Solve (Rust handles sequential vs parallel)
        t_num0 = time.perf_counter()
        solutions = self._solve_segments(gate_list, invariant_list, n_inner)
        t_num1 = time.perf_counter()

        # Phase 2: Stitch into DAG (Rust P accumulation + Python DAG ops)
        P = self._stitch_into_dag(
            dag, qreg, gate_list, invariant_list, solutions, n_inner
        )

        # Final recovery to true target
        k1, k2, k3, k4, gphase = recover_local_equivalence(target.matrix, P)
        dag.global_phase += gphase
        dag.apply_operation_front(UnitaryGate(k1, check_input=False), [qreg[0]])
        dag.apply_operation_front(UnitaryGate(k2, check_input=False), [qreg[1]])
        dag.apply_operation_back(UnitaryGate(k3, check_input=False), [qreg[0]])
        dag.apply_operation_back(UnitaryGate(k4, check_input=False), [qreg[1]])
        t_stitch1 = time.perf_counter()

        self.last_phase_timing = {
            "numerics": t_num1 - t_num0,
            "stitch": t_stitch1 - t_num1,
        }

        return dag if return_dag else dag_to_circuit(dag)

    # -- Phase 1: segment solving --

    def _solve_segments(self, gate_list, invariant_list, n_inner):
        """Solve all segments via Rust (Rayon-parallel above threshold)."""
        prefixes = [
            gate_list[0] if idx == 0 else invariant_list[idx] for idx in range(n_inner)
        ]
        bases = [gate_list[idx + 1] for idx in range(n_inner)]
        targets = [invariant_list[idx + 1] for idx in range(n_inner)]

        makhlin_tol = self.config.makhlin_conv_tol
        weyl_tol = self.config.weyl_conv_tol

        results = _rust_solve_batch(
            [inv.matrix for inv in prefixes],
            [inv.matrix for inv in bases],
            [inv.matrix for inv in targets],
            makhlin_tol,
            weyl_tol,
            self.config.min_batch_size,
        )

        solutions = [
            SegmentSolution(
                u0=u0,
                u1=u1,
                max_residual=weyl_res,
                success=(weyl_res <= weyl_tol or makhlin_res <= makhlin_tol),
            )
            for u0, u1, weyl_res, makhlin_res in results
        ]

        for idx, sol in enumerate(solutions):
            if not sol.success:
                raise RuntimeError(
                    f"Segment {idx + 1} synthesis failed "
                    f"(residual norm={sol.max_residual:.2e})."
                )

        return solutions

    # -- Phase 2: stitch + DAG --

    @staticmethod
    def _stitch_into_dag(dag, qreg, gate_list, invariant_list, solutions, n_inner):
        """Rust stitch (P accumulation + recovery) then apply gates to DAG."""
        dag.apply_operation_back(gate_list[0].gate, qreg[:])

        if n_inner == 0:
            return gate_list[0].matrix.copy()

        # Single Rust FFI call for P accumulation + intermediate recoveries
        corrections, final_p = _rust_stitch(
            gate_list[0].matrix,
            [solutions[i].u0 for i in range(n_inner)],
            [solutions[i].u1 for i in range(n_inner)],
            [gate_list[i + 1].matrix for i in range(n_inner)],
            [
                np.asarray(invariant_list[i + 1].canonical_matrix, dtype=np.complex128)
                for i in range(n_inner - 1)
            ],
        )

        # Apply gates to DAG
        for i in range(n_inner):
            dag.apply_operation_back(
                UnitaryGate(solutions[i].u0, check_input=False), [qreg[0]]
            )
            dag.apply_operation_back(
                UnitaryGate(solutions[i].u1, check_input=False), [qreg[1]]
            )
            dag.apply_operation_back(gate_list[i + 1].gate, qreg[:])

            if i < n_inner - 1:
                k3, k4, gphase = corrections[i]
                dag.global_phase += gphase
                dag.apply_operation_back(UnitaryGate(k3, check_input=False), [qreg[0]])
                dag.apply_operation_back(UnitaryGate(k4, check_input=False), [qreg[1]])

        return np.asarray(final_p)

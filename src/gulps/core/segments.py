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
  Phase 1: Solve all segments with canonical prefixes (single Rust call,
    Rayon-parallel above min_batch_size).
  Phase 2: Stitch with intermediate KAK recoveries + final recovery (Rust).
  Phase 3: Build DAG from returned circuit data (Python/Qiskit).
"""

from __future__ import annotations

import time

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit

from gulps._accelerate import recover_local_equiv as recover_local_equivalence
from gulps._accelerate import solve_and_stitch as _rust_solve_and_stitch
from gulps.config import GulpsConfig
from gulps.core.invariants import GateInvariants


class SegmentSynthesizer:
    """Solve all segments and stitch into a circuit.

    Each segment finds single-qubit unitaries (u0, u1) such that
    G @ kron(u1, u0) @ P is locally equivalent to the target invariants,
    where P is the actual accumulated product and G is the basis gate.
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

        t0 = time.perf_counter()

        if n_inner == 0:
            # Single-gate case: just do the final recovery
            k1, k2, k3, k4, gphase = recover_local_equivalence(
                target.matrix, gate_list[0].matrix
            )
            u0s, u1s = [], []
        else:
            u0s, u1s, k1, k2, k3, k4, gphase = _rust_solve_and_stitch(
                gate_list[0].matrix,
                [gate_list[i + 1].matrix for i in range(n_inner)],
                [invariant_list[i + 1].matrix for i in range(n_inner)],
                target.matrix,
                self.config.makhlin_conv_tol,
                self.config.weyl_conv_tol,
                self.config.min_batch_size,
                [list(invariant_list[i + 1].weyl) for i in range(n_inner - 1)],
            )

        t1 = time.perf_counter()

        # Build DAG: front corrections, gate sequence, back corrections
        dag.global_phase += gphase
        dag.apply_operation_front(UnitaryGate(k1, check_input=False), [qreg[0]])
        dag.apply_operation_front(UnitaryGate(k2, check_input=False), [qreg[1]])
        dag.apply_operation_back(gate_list[0].gate, qreg[:])
        for i in range(n_inner):
            dag.apply_operation_back(UnitaryGate(u0s[i], check_input=False), [qreg[0]])
            dag.apply_operation_back(UnitaryGate(u1s[i], check_input=False), [qreg[1]])
            dag.apply_operation_back(gate_list[i + 1].gate, qreg[:])
        dag.apply_operation_back(UnitaryGate(k3, check_input=False), [qreg[0]])
        dag.apply_operation_back(UnitaryGate(k4, check_input=False), [qreg[1]])

        t2 = time.perf_counter()

        self.last_phase_timing = {
            "numerics": t1 - t0,
            "stitch": t2 - t1,
        }

        return dag if return_dag else dag_to_circuit(dag)

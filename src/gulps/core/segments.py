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

"""Segment-wise synthesis: solve the chain, recover against a target, build the DAG.

Three building blocks:

- :func:`solve_chain` is the chain. It dispatches to the Rust GN/LM solver
  and stitcher (Rayon-parallel above ``min_batch_size``) and returns
  ``(u0s, u1s, T)``: fused locals plus the chain's realized product matrix.
- :func:`recover_local_equiv` (FFI, used at call sites) is the recovery.
  Per-target ``(k1..k4, gphase)`` that lands ``T`` at ``target``.
- :func:`build_segment_dag` is the assembly. It assembles the standard
  sandwich circuit from chain + recovery.

:class:`SegmentSynthesizer.synthesize_segments` orchestrates all three for
single-target use; the pass-level batch calls them separately.
"""

from __future__ import annotations

import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.converters import dag_to_circuit
from qiskit.dagcircuit import DAGCircuit

from gulps._accelerate import recover_local_equiv, solve_and_stitch
from gulps.config import GulpsConfig
from gulps.core.invariants import GateInvariants


def solve_chain(
    config: GulpsConfig,
    sentence: list[GateInvariants],
    intermediates: list[GateInvariants],
) -> tuple:
    """Solve the chain; return fused locals and the realized product.

    Returns ``(u0s, u1s, T)``: fused locals plus the chain's realized unitary
    (intermediate-stitch phase absorbed). No target dependency; the chain is a property of the sentence and
    intermediates. Per-target recovery via :func:`recover_local_equiv` is
    the caller's job (one call for single-target, N calls for class-batch
    with N instances).

    Depth-0/1 short-circuits with no Rust call (``T`` is ``I_4`` or
    ``sentence[0].matrix``). Depth-2+ dispatches to Rust.
    """
    n_inner = len(sentence) - 1
    if n_inner < 1:
        T = sentence[0].matrix if sentence else np.eye(4, dtype=np.complex128)
        return [], [], T
    return solve_and_stitch(
        sentence[0].matrix,
        [sentence[i + 1].matrix for i in range(n_inner)],
        [intermediates[i + 1].matrix for i in range(n_inner)],
        config.makhlin_conv_tol,
        config.weyl_conv_tol,
        config.min_batch_size,
        [list(intermediates[i + 1].weyl) for i in range(n_inner - 1)],
        config.identity_warmstart,
    )


def build_segment_dag(
    gate_list: list[GateInvariants],
    u0s: list,
    u1s: list,
    k1,
    k2,
    k3,
    k4,
    gphase: float,
    *,
    return_dag: bool = False,
) -> QuantumCircuit | DAGCircuit:
    """Assemble the standard sentence circuit: k1⊗k2 front + 2q/1q sandwich + k3⊗k4 back.

    ``gate_list=[]`` (target locally equivalent to identity) reduces to just
    the front/back single-qubit layers, no 2q gate applied.
    """
    # Build DAGCircuit directly (skips the empty QuantumCircuit +
    # circuit_to_dag round-trip; ~35 μs/call).
    dag = DAGCircuit()
    qreg = QuantumRegister(2, "q")
    dag.add_qreg(qreg)
    dag.global_phase = gphase
    q0, q1 = qreg[0], qreg[1]
    qboth = [q0, q1]  # pre-bound; avoids per-gate qreg[:] slice
    dag.apply_operation_front(UnitaryGate(k1, check_input=False), [q0])
    dag.apply_operation_front(UnitaryGate(k2, check_input=False), [q1])
    for i, gate in enumerate(gate_list):
        if i > 0:
            dag.apply_operation_back(UnitaryGate(u0s[i - 1], check_input=False), [q0])
            dag.apply_operation_back(UnitaryGate(u1s[i - 1], check_input=False), [q1])
        dag.apply_operation_back(gate.gate, qboth)
    dag.apply_operation_back(UnitaryGate(k3, check_input=False), [q0])
    dag.apply_operation_back(UnitaryGate(k4, check_input=False), [q1])
    return dag if return_dag else dag_to_circuit(dag)


class SegmentSynthesizer:
    """Solve all segments and stitch into a circuit.

    Each segment is a tuple ``(C, G, T)``: solve for single-qubit unitaries
    ``(u0, u1)`` so that ``G @ kron(u1, u0) @ C`` is locally equivalent to
    ``T``, where ``C`` is the prefix (previous segment's reached matrix),
    ``G`` is the basis gate, and ``T`` is the segment's target invariant.
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
        """Solve the chain and stitch into a circuit against ``target``."""
        if len(gate_list) != len(invariant_list):
            raise ValueError("Gate list and invariant list must have the same length.")

        t0 = time.perf_counter()
        u0s, u1s, T = solve_chain(self.config, gate_list, invariant_list)
        k1, k2, k3, k4, gphase = recover_local_equiv(target.matrix, T)
        t1 = time.perf_counter()
        out = build_segment_dag(
            gate_list, u0s, u1s, k1, k2, k3, k4, gphase, return_dag=return_dag
        )
        t2 = time.perf_counter()
        self.last_phase_timing = {"numerics": t1 - t0, "stitch": t2 - t1}
        return out

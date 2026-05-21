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

"""Qiskit transpiler pass that decomposes every two-qubit block with GULPS.

Stages: collect 2q nodes, dedup by local-equivalence class, run LP and
numerics once per unique class, recover per instance, substitute back into
the DAG in collect order.
"""

import time

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks

from gulps._accelerate import recover_local_equiv
from gulps.core.invariants import GateInvariants
from gulps.core.segments import build_segment_dag, solve_chain
from gulps.gulps_decomposer import GulpsDecomposer
from gulps.linear_program.lp_dispatch import best_decomposition_batch


class GulpsDecompositionPass(TransformationPass):
    """Replace all two-qubit blocks with GULPS-synthesized circuits."""

    def __init__(self, decomposer: GulpsDecomposer, **kwargs) -> None:
        """Wrap decomposer as a Qiskit transformation pass."""
        super().__init__()
        self.requires = [Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)]
        self._decomposer = decomposer
        # Aggregate phase timing populated on each run().
        self.last_phase_timing: dict[str, float] = {}

    def _numerics_batch(
        self,
        lp_results: dict[GateInvariants, "ConstraintSolution"],  # type: ignore[name-defined]
    ) -> dict[GateInvariants, tuple]:
        """Solve the chain once per unique class.

        Returns ``{inv: (sentence, u0s, u1s, T)}``. :meth:`run` does the
        per-instance recovery against ``T``.
        """
        config = self._decomposer.config
        out: dict[GateInvariants, tuple] = {}
        for inv, lp_result in lp_results.items():
            u0s, u1s, T = solve_chain(
                config, lp_result.sentence, lp_result.intermediates
            )
            out[inv] = (lp_result.sentence, u0s, u1s, T)
        return out

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Decompose every two-qubit node in dag as a single batched pipeline."""
        t = [time.perf_counter()]

        # Stage 1: (node, inv) per eligible 2q DAG op, in topological order.
        tasks: list[tuple] = []
        for n in dag.op_nodes():
            if (
                n.op.num_qubits != 2
                or n.is_parameterized()
                or not hasattr(n.op, "to_matrix")
            ):
                continue
            tasks.append((n, GateInvariants.from_unitary(n.op.to_matrix())))
        if not tasks:
            self.last_phase_timing = {}
            return dag

        # Unique classes via GateInvariants.__hash__; first-occurrence order.
        unique_invs = list(dict.fromkeys(inv for _, inv in tasks))
        t.append(time.perf_counter())

        # LP once per unique class
        lp_results = best_decomposition_batch(self._decomposer, unique_invs)
        t.append(time.perf_counter())
        decomps = self._numerics_batch(lp_results)
        t.append(time.perf_counter())

        # Per-task recover and serial DAG substitute.
        for node, inv in tasks:
            sentence, u0s, u1s, T = decomps[inv]
            k1, k2, k3, k4, gphase = recover_local_equiv(inv.matrix, T)
            dag.substitute_node_with_dag(
                node,
                build_segment_dag(
                    sentence, u0s, u1s, k1, k2, k3, k4, gphase, return_dag=True
                ),
            )
        t.append(time.perf_counter())

        self.last_phase_timing = {
            "prep": t[1] - t[0],
            "lp": t[2] - t[1],
            "numerics": t[3] - t[2],
            "recover_substitute": t[4] - t[3],
            "total": t[4] - t[0],
            "n_nodes": len(tasks),
            "n_unique": len(unique_invs),
        }
        return dag

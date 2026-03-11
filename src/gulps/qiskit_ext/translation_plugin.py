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

"""Qiskit translation-stage plugin for GULPS.

Registers as ``translation_method="gulps"`` so that
``generate_preset_pass_manager(translation_method="gulps", ...)``
produces a fully self-contained translation stage.

Block collection (Collect2qBlocks, ConsolidateBlocks) is handled by
GulpsDecompositionPass.requires and runs automatically at execution time.
Optimize1qGatesDecomposition consolidates the 1q UnitaryGates that GULPS
emits, so the optimization loop doesn't waste its first iteration on them.
"""

from __future__ import annotations

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin

from gulps.config import GulpsConfig
from gulps.core.isa import DiscreteISA
from gulps.gulps_decomposer import GulpsDecomposer
from gulps.qiskit_ext.decomposer_pass import GulpsDecompositionPass


class GulpsTranslationPlugin(PassManagerStagePlugin):
    """Translation stage that decomposes 2q unitaries with GULPS and consolidates 1q gates.

    Assumes a uniform ISA: 2q gate properties are read from qargs (0, 1) and
    1q U-gate properties from qargs (0,).  Both must carry ``duration`` in the
    Target so that GULPS can minimise total gate time (an additive cost model).
    """

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Return a PassManager for the translation stage."""
        target = pass_manager_config.target
        if target is None:
            raise ValueError("GulpsTranslationPlugin requires a Target.")

        # Extract 2q gates and costs from the Target.
        qargs = (0, 1)
        gate_set, names, costs = [], [], []
        for name in target.operation_names_for_qargs(qargs):
            gate = target.operation_from_name(name)
            if gate.num_qubits != 2:
                continue
            props = target[name].get(qargs)
            if props is None or props.duration is None:
                raise ValueError(
                    f"Gate {name} on qubits {qargs} must have defined properties with a duration."
                )
            gate_set.append(gate)
            names.append(name)
            costs.append(props.duration)

        # Extract 1q gate duration as single_qubit_cost so gulps accounts for
        # the overhead of interleaving 1q layers between 2q gates.
        sq_cost = 0.0
        for name in target.operation_names_for_qargs((0,)):
            gate = target.operation_from_name(name)
            if gate.num_qubits == 1:
                props = target[name].get((0,))
                if props and props.duration is not None:
                    sq_cost = props.duration
                break

        isa = DiscreteISA(
            gate_set=gate_set,
            costs=costs,
            names=names,
            precompute_polytopes=None,
            single_qubit_cost=sq_cost,
        )

        # suppress warnings
        decomposer = GulpsDecomposer(
            isa=isa, config_options=GulpsConfig(flag_duration=0)
        )

        return PassManager(
            [
                GulpsDecompositionPass(decomposer),
                Optimize1qGatesDecomposition(target=target),
            ]
        )

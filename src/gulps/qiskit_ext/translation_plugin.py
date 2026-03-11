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

from gulps.gulps_decomposer import GulpsDecomposer
from gulps.qiskit_ext.decomposer_pass import GulpsDecompositionPass


class GulpsTranslationPlugin(PassManagerStagePlugin):
    """Translation stage that decomposes 2q unitaries with GULPS and consolidates 1q gates.

    Assumes the Target contains 2q gates with error rates that can be used as costs for GULPS.
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
            gate_set.append(gate)
            names.append(name)
            costs.append(props.error if props and props.error is not None else 1.0)

        decomposer = GulpsDecomposer(
            gate_set=gate_set,
            costs=costs,
            names=names,
            precompute_polytopes=None,
        )

        return PassManager(
            [
                GulpsDecompositionPass(decomposer),
                Optimize1qGatesDecomposition(target=target),
            ]
        )

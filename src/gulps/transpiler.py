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

"""The transpiler pass and the translation plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import PassManager, Target
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import (
    BasisTranslator,
    ConsolidateBlocks,
    Optimize1qGatesDecomposition,
)
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin

from gulps._accelerate import MAX_DEPTH, BlockSynthesizer
from gulps.decomposition import DecompositionError, GulpsDecomposer

if TYPE_CHECKING:
    from qiskit.transpiler import PassManagerConfig


class GulpsDecompositionPass(TransformationPass):
    """Replace each two-qubit block of a DAG with a GULPS circuit.

    The pass emits the single-qubit matrices required by each decomposition,
    leaving merging across blocks and translation to the single-qubit basis
    to Qiskit's passes. The translation plugin includes that cleanup in its
    pipeline.

    A Target must register each native instruction under its own ``gate.name``;
    aliases raise ``NotImplementedError``. Since a Target has one entry per
    name, it cannot register several fixed angles of the same standard gate
    under that name. For such instruction sets, pass a ``GulpsDecomposer``
    directly, with all native gates and their costs.

    Args:
        decomposer (GulpsDecomposer | qiskit.transpiler.Target): A fixed
            decomposer for every pair, or a Target from which to build one
            decomposer per distinct instruction set and cost model.
        cost (str): The Target property to minimize: ``"duration"`` or
            ``"error"``.
        max_depth (int): Maximum number of two-qubit gates in a sequence.
    """

    def __init__(
        self,
        decomposer: GulpsDecomposer | Target,
        *,
        cost: str = "duration",
        max_depth: int = MAX_DEPTH,
    ) -> None:
        """Use ``cost`` and ``max_depth`` only when configuring a Target."""
        super().__init__()
        self.requires = [ConsolidateBlocks(force_consolidate=True)]
        self._config = (decomposer, cost, max_depth)
        self._synthesizer = self._build()

    def _build(self) -> BlockSynthesizer:
        source, cost, max_depth = self._config
        if isinstance(source, Target):
            return BlockSynthesizer.from_target(source, cost, max_depth)
        return BlockSynthesizer.from_decomposer(source)

    def __getstate__(self) -> dict:
        # Qiskit's parallel pass manager pickles passes; the synthesizer is rebuilt.
        state = self.__dict__.copy()
        del state["_synthesizer"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._synthesizer = self._build()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Compile every two-qubit block in place.

        Raises:
            TranspilerError: If a block is invalid for the configured gates or
                cannot be compiled. Its ``__cause__`` is the original error.
            NotImplementedError: If the DAG has control flow, classical
                variables, or two-qubit gates without a matrix.
        """
        try:
            return self._synthesizer.run(dag)
        except (ValueError, DecompositionError) as error:
            raise TranspilerError(str(error)) from error


class GulpsTranslationPlugin(PassManagerStagePlugin):
    """Use GULPS for Qiskit's translation stage with ``translation_method="gulps"``.

    The plugin runs GULPS synthesis for the backend's Target, then Qiskit's
    single-qubit optimization and basis translation.
    """

    def pass_manager(
        self,
        pass_manager_config: PassManagerConfig,
        _optimization_level: int | None = None,
    ) -> PassManager:
        """The translation stage for ``pass_manager_config.target``."""
        target = pass_manager_config.target
        if target is None:
            raise TranspilerError("the translation plugin requires a Target")
        return PassManager(
            [
                GulpsDecompositionPass(target),
                Optimize1qGatesDecomposition(target=target),
                # Symbolic gates have no matrix, so neither pass above rewrites them.
                BasisTranslator(SessionEquivalenceLibrary, None, target=target),
            ]
        )

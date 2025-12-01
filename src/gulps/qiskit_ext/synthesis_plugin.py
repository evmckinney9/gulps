from __future__ import annotations

import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin

from gulps.synthesis.gulps_decomposer import GulpsDecomposer

from .synthesis_plugin import UnitarySynthesisPlugin


class GulpsSynthesisPlugin(UnitarySynthesisPlugin):
    """TODO Plugin documentation."""

    _decomposer = None

    @property
    def min_qubits(self):
        return 2

    @property
    def max_qubits(self):
        return 2

    @property
    def supports_target(self):
        return True

    ######################################
    # NOTE, I avoid using basis_gates and gate_errors to focus on implementation
    # that uses the target object
    @property
    def supports_basis_gates(self):
        return False

    @property
    def supports_gate_errors(self):
        return False

    # NOTE, using this one seems better than supports_gate_errors
    # but leave False for now
    @property
    def supports_gate_errors_by_qubit(self):
        return False

    @property
    def supports_coupling_map(self):
        return False

    @property
    def supports_natural_direction(self):
        return False

    @property
    def supports_pulse_optimize(self):
        return False

    @property
    def supports_gate_lengths(self):
        return False

    @property
    def supports_gate_lengths_by_qubit(self):
        return False

    @property
    def supported_bases(self):
        return None

    def run(self, unitary: np.ndarray, **options) -> DAGCircuit:
        # basis_gates = options.get("basis_gates", None)
        target = options.get("target")
        if target is None:
            raise ValueError("Target must be provided in options.")

        # parse target into gate_set list and cost list
        gate_set = []
        costs = []
        for idx, instruction in enumerate(target.operations_for_qargs((0, 1))):
            error = target.instruction_properties(idx).error
            gate_set.append(instruction)
            costs.append(error)

        if (
            GulpsSynthesisPlugin._decomposer is None
            or GulpsSynthesisPlugin._decomposer.isa.gate_set != gate_set
        ):
            GulpsSynthesisPlugin._decomposer = GulpsDecomposer(
                gate_set=gate_set,
                costs=costs,
                precompute_polytopes=None,
            )

        return GulpsSynthesisPlugin._decomposer(target=unitary, return_dag=True)

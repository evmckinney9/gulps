from __future__ import annotations

import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin

from .synthesis_plugin import UnitarySynthesisPlugin


class GulpsSynthesisPlugin(UnitarySynthesisPlugin):
    """TODO Plugin documentation."""

    # Generating basic approximations of single-qubit gates is computationally expensive.
    # We cache the instance of the Solovay-Kitaev class (which contains the approximations),
    # as well as the basis gates and the depth (used to generate it).
    # When the plugin is called again, we check if the specified basis gates and depth are
    # the same as before. If so, the stored basic approximations are reused, and if not, the
    # approximations are re-generated. In practice (when the plugin is run as a part of the
    # UnitarySynthesis transpiler pass), the basis gates and the depth do not change, and
    # basic approximations are not re-generated.
    _sk = None
    _basis_gates = None
    _depth = None

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

    def run(self, unitary, **options):
        raise NotImplementedError()
        # # configure settings
        # target = options.get("target")
        # if target is None:
        #     raise ValueError("Target must be provided in options.")

        # # XXX I'll just assume we are only working with 2 qubit targets for now
        # isa_tuple = []
        # for idx, instruction in enumerate(target.operations_for_qargs((0, 1))):
        #     error = target.instruction_properties(idx).error
        #     isa_tuple.append((instruction, error))

        # # call the synthesis function
        # dag_circuit = generate_dag_circuit_from_matrix(
        #     unitary, basis_gates, gate_errors
        # )
        # return dag_circuit

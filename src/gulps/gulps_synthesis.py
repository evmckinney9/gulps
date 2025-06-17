from __future__ import annotations

import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass

from .plugin import UnitarySynthesisPlugin


class GlobalUnitaryLinearProgram(TransformationPass):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        self._sk = GULPS(kwargs)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.op_nodes():
            # ignore operations on which the algorithm cannot run
            if (
                (node.op.num_qubits != 2
                or node.is_parameterized()
                or (not hasattr(node.op, "to_matrix"))
            ):
                continue

            # we do not check the input matrix as we know it comes from a Qiskit gate, as this
            # we know it will generate a valid SU(2) matrix
            check_input = not isinstance(node.op, Gate)

            # call solovay kitaev
            approximation = self._sk.run(
                node.op, self.recursion_degree, return_dag=True, check_input=check_input
            )

            # convert to a dag and replace the gate by the approximation
            dag.substitute_node_with_dag(node, approximation)

        return dag


class SolovayKitaevSynthesis(UnitarySynthesisPlugin):
    """A Solovay-Kitaev Qiskit unitary synthesis plugin.

    This plugin is invoked by :func:`~.compiler.transpile` when the ``unitary_synthesis_method``
    parameter is set to ``"sk"``.

    This plugin supports customization and additional parameters can be passed to the plugin
    by passing a dictionary as the ``unitary_synthesis_plugin_config`` parameter of
    the :func:`~qiskit.compiler.transpile` function.

    Supported parameters in the dictionary:

    basic_approximations (str | dict):
        The basic approximations for the finding the best discrete decomposition at the root of the
        recursion. If a string, it specifies the ``.npy`` file to load the approximations from.
        If a dictionary, it contains ``{label: SO(3)-matrix}`` pairs. If None, a default based on
        the specified ``basis_gates`` and ``depth`` is generated.

    basis_gates (list):
        A list of strings specifying the discrete basis gates to decompose to. If None,
        it defaults to ``["h", "t", "tdg"]``. If ``basic_approximations`` is not None,
        ``basis_set`` is required to correspond to the basis set that was used to
        generate it.

    depth (int):
        The gate-depth of the basic approximations. All possible, unique combinations of the
        basis gates up to length ``depth`` are considered. If None, defaults to 12.
        If ``basic_approximations`` is not None, ``depth`` is required to correspond to the
        depth that was used to generate it.

    recursion_degree (int):
        The number of times the decomposition is recursively improved. If None, defaults to 5.
    """

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
    def max_qubits(self):
        """Maximum number of supported qubits is ``1``."""
        return 1

    @property
    def min_qubits(self):
        """Minimum number of supported qubits is ``1``."""
        return 1

    @property
    def supports_natural_direction(self):
        """The plugin does not support natural direction, it does not assume
        bidirectional two qubit gates.
        """
        return True

    @property
    def supports_pulse_optimize(self):
        """The plugin does not support optimization of pulses."""
        return False

    @property
    def supports_gate_lengths(self):
        """The plugin does not support gate lengths."""
        return False

    @property
    def supports_gate_errors(self):
        """The plugin does not support gate errors."""
        return False

    @property
    def supported_bases(self):
        """The plugin does not support bases for synthesis."""
        return None

    @property
    def supports_basis_gates(self):
        """The plugin does not support basis gates. By default it synthesis to the
        ``["h", "t", "tdg"]`` gate basis.
        """
        return True

    @property
    def supports_coupling_map(self):
        """The plugin does not support coupling maps."""
        return False

    def run(self, unitary, **options):
        """Run the SolovayKitaevSynthesis synthesis plugin on the given unitary."""
        config = options.get("config") or {}
        basis_gates = options.get("basis_gates", None)
        depth = config.get("depth", 12)
        basic_approximations = config.get("basic_approximations", None)
        recursion_degree = config.get("recursion_degree", 5)

        # Check if we didn't yet construct the Solovay-Kitaev instance (which contains the basic
        # approximations) or if the basic approximations need need to be recomputed.
        if (SolovayKitaevSynthesis._sk is None) or (
            (basis_gates != SolovayKitaevSynthesis._basis_gates)
            or (depth != SolovayKitaevSynthesis._depth)
        ):
            SolovayKitaevSynthesis._basis_gates = basis_gates
            SolovayKitaevSynthesis._depth = depth
            SolovayKitaevSynthesis._sk = SolovayKitaevDecomposition(
                basic_approximations, basis_gates=basis_gates, depth=depth
            )
        approximate_circuit = SolovayKitaevSynthesis._sk.run(unitary, recursion_degree)
        return circuit_to_dag(approximate_circuit)

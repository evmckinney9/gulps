"""Qiskit transpiler pass that wraps GulpsDecomposer."""

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks

from gulps.gulps_decomposer import GulpsDecomposer


class GulpsDecompositionPass(TransformationPass):
    """Replace all two-qubit blocks with GULPS-synthesized circuits."""

    def __init__(self, decomposer: GulpsDecomposer, **kwargs) -> None:
        """Wrap decomposer as a Qiskit transformation pass."""
        super().__init__()
        self.requires = [Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)]
        self._decomposer = decomposer

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Decompose every two-qubit node in dag."""
        for node in dag.op_nodes():
            # ignore operations on which the algorithm cannot run
            if (
                node.op.num_qubits != 2
                or node.is_parameterized()
                or (not hasattr(node.op, "to_matrix"))
            ):
                continue

            # TODO ?
            # check_input = not isinstance(node.op, Gate)

            # call gulps and replace the decomposed op node
            decomposed_node = self._decomposer(node.op, return_dag=True)
            dag.substitute_node_with_dag(node, decomposed_node)

        return dag

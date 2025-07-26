from typing import List

from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks

from gulps.gulps_decomposer import GulpsDecomposer

# FIXME TODO, hash/cache for the decomposer? avoid constructing it at runtime


class GulpsDecompositionPass(TransformationPass):
    def __init__(self, gate_set: List[Gate], costs: List[float], **kwargs) -> None:
        self.decomposer = GulpsDecomposer(gate_set, costs, **kwargs)
        self.requires = [Collect2qBlocks(), ConsolidateBlocks()]
        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
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

            # call solovay kitaev
            decomposed_node = self.decomposer(node.op, return_dag=True)
            # convert to a dag and replace the gate by the approximation
            dag.substitute_node_with_dag(node, decomposed_node)

        return dag

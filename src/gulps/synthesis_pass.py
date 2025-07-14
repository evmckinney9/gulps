from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass

from gulps.gulps_synthesis import GulpsDecomposer


class GulpsDecompositionPass(TransformationPass):
    def __init__(self, gate_set, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.decomposer = GulpsDecomposer(gate_set, **kwargs)

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
            decomposed_node = self.decomposer(node.op, use_dag=True)
            # convert to a dag and replace the gate by the approximation
            dag.substitute_node_with_dag(node, decomposed_node)

        return dag

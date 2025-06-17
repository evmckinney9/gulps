"""Utils for numerically constructing polytopes for pulse-optmized decomps."""

from abc import ABC, abstractmethod
from itertools import product
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.axes import Axes
from monodromy.coordinates import (
    monodromy_to_positive_canonical_coordinate,
    unitary_to_monodromy_coordinate,
)
from qiskit import QuantumCircuit
from qiskit._accelerate.two_qubit_decompose import weyl_coordinates
from qiskit.circuit import Gate, Parameter
from qiskit.quantum_info import Operator
from tqdm.notebook import tqdm
from weylchamber import WeylChamber, c1c2c3


class CircuitAnsatz(ABC):
    """Class holds parameterized quantum circuit to be sampled or assigned."""

    def __init__(self, num_qubits: int, num_layers: int):
        """Construct the PQC."""
        assert num_qubits >= 1
        assert num_layers >= 1
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self._construct_circuit()
        self.num_params = len(self.qc.parameters)
        self.point_cloud = [[]]

    @abstractmethod
    def _construct_circuit(self) -> None:
        pass

    @property
    def weyl_coords(self):
        """Temporary work-around for mis-matched conventions.

        Assumes self.point_cloud is always 'monodromy', thus
        self.weyl_coords is 'positive-canonical'.
        """
        weyl_coords = []
        for coord in self.point_cloud:
            # XXX bad!, assumes never have a nested list of len(3)
            if len(coord) == 3:  # single
                weyl_coords.append(
                    np.array(monodromy_to_positive_canonical_coordinate(*coord))
                    / (np.pi / 2)
                )
            else:  # nested list
                temp = np.array(
                    [monodromy_to_positive_canonical_coordinate(*c) for c in coord]
                ) / (np.pi / 2)
                weyl_coords.append(temp)

        return weyl_coords

        # return np.array(
        #     [monodromy_to_positive_canonical_coordinate(*c) for c in self.point_cloud]
        # ) / (np.pi / 2)

    def _fit_polytope(self) -> None:
        raise NotImplemented()

    def _assign_parameters(self, param_values, inplace=False) -> QuantumCircuit:
        if len(param_values) != self.num_params:
            raise ValueError("parameter vectors does not match expected length.")
        param_values = {
            param: param_val
            for param, param_val in zip(self.qc.parameters, param_values)
        }
        return self.qc.assign_parameters(param_values, inplace=inplace)

    def _assign_random_parameters(self) -> QuantumCircuit:
        param_values = {
            param: np.random.uniform(-2 * np.pi, 2 * np.pi)
            for param in self.qc.parameters
        }
        return self.qc.assign_parameters(param_values)

    def generate_points(
        self, num_samples: int = 1_000, method: str = "monodromy"
    ) -> List[tuple]:
        """Randomly sample params then convert each to Weyl chamber coords."""
        # for _ in tqdm(range(num_samples)):

        if method != "monodromy":
            raise UserWarning("Only allowing monodromy convention for now.")

        for _ in range(num_samples):
            qc_sample = self._assign_random_parameters()

            if method == "monodromy":  # use monodromy coordinate convention
                point = unitary_to_monodromy_coordinate(Operator(qc_sample).data)[:3]
            elif method == "qiskit":  # use qiskit coordinate convention,
                # NOTE, this convention uses only c[0]<np.pi/2 (symmetry on y-axis)
                # and c[2] may be negative
                point = weyl_coordinates(Operator(qc_sample).data)
            elif method == "qucontrol":  # False: use qucontrol coordinate convention
                point = c1c2c3(Operator(qc_sample).data)
            else:
                raise ValueError("invalid coordinate convention method")

            self.point_cloud[0].append(point)
        return self.point_cloud


class BasicCircuitAnsatz(CircuitAnsatz):
    """Subclass of CircuitAnsatz for basic circuits.

    A basic ansatz has two qubits, a single 2Q basis gate interleaving
    with 1Qs."
    """

    def __init__(
        self,
        basis_gate: Gate | Operator,
        num_layers: int,
        u_sequence: List[tuple[int]] = None,
        outside_layer_u: tuple[int] = (0, 0),
    ):
        """Initialize by specifying a 2Q basis gate and number of layers."""
        self.basis_gate = basis_gate
        self.u_sequence = u_sequence or [(3, 3)] * num_layers
        self.outside_layer_u = outside_layer_u
        assert len(self.outside_layer_u) == 2
        super().__init__(2, num_layers)

    def _to_string(self, basis_str: str = "2Q", include_outside=False):
        # string-building :) gave up trying to make it look pythonic so just use simple for-loops
        GATE_LABELS = {0: "I", 1: "U1", 2: "U2", 3: "U3"}
        string_rep = ""

        if include_outside:
            string_rep += f"({GATE_LABELS[self.outside_layer_u[0]]}⊗{GATE_LABELS[self.outside_layer_u[1]]})-"

        string_rep += basis_str + "-"

        for idx in range(self.num_layers - 1):
            string_rep += f"({GATE_LABELS[self.u_sequence[idx][0]]}⊗{GATE_LABELS[self.u_sequence[idx][1]]})-"
            string_rep += basis_str + "-"

        if include_outside:
            string_rep += f"({GATE_LABELS[self.outside_layer_u[0]]}⊗{GATE_LABELS[self.outside_layer_u[1]]})-"

        return string_rep[:-1]

    def _add_u_gate(self, gate_type: int, qubit: int, param_idx: int) -> None:
        if gate_type == 3:
            # self.qc.u(
            # NOTE RV is much better at converging
            self.qc.rv(
                Parameter(f"x{param_idx}"),
                Parameter(f"x{param_idx + 1}"),
                Parameter(f"x{param_idx + 2}"),
                qubit,
            )
        elif gate_type == 2:
            self.qc.u(
                np.pi / 2,
                Parameter(f"x{param_idx}"),
                Parameter(f"x{param_idx + 1}"),
                qubit,
            )
        elif gate_type == 1:
            self.qc.u(0, 0, Parameter(f"x{param_idx}"), qubit)
        else:
            pass  # identity

    def _construct_circuit(self) -> None:
        self.qc = QuantumCircuit(self.num_qubits)
        param_idx = 0
        # every 'layer' contains a 2Q gate followed by 1Qs

        # handle optional outside at beginning
        self._add_u_gate(self.outside_layer_u[0], 0, param_idx)
        self._add_u_gate(self.outside_layer_u[1], 1, param_idx + 3)
        param_idx += 6

        self.qc.append(self.basis_gate, [0, 1])
        for idx in range(self.num_layers - 1):
            self._add_u_gate(self.u_sequence[idx][0], 0, param_idx)
            self._add_u_gate(self.u_sequence[idx][1], 1, param_idx + 3)
            self.qc.append(self.basis_gate, [0, 1])
            param_idx += 6

        # handle optional outside at end
        self._add_u_gate(self.outside_layer_u[0], 0, param_idx)
        self._add_u_gate(self.outside_layer_u[1], 1, param_idx + 3)


class PulseConstraintPolytopes:
    """Iterate all possible interleaving 1Q options for degrees of freedom.

    Interleaving gates can be U3, U2, or U1 realized by 2,1,0 X90
    pulses, respecitively.
    """

    def __init__(
        self,
        basis_gate: Gate | Operator,
        basis_gate_str: str,
        max_layers: int,
        include_identity: bool = False,
    ):
        """Initialize by specifying the 2Q basis gate and max_layers.

        The max_layers should be the number of 2Q to complete coverage,
        e.g. (CX=3, sqiSWAP=3, B=2).
        """
        self.basis_gate = basis_gate
        self.basis_str = basis_gate_str
        self.include_identity = include_identity
        self.u_set = self._construct_1q_sequences(max_layers)
        self.pulse_costs = self._construct_costs()
        self.circuit_ansatze = self._construct_ansatze()
        self.labels = self._construct_labels()
        self._sort_by_pulse_costs()

    def _construct_1q_sequences(self, max_layers: int) -> List[tuple]:
        gate_options = [0, 1, 2, 3] if self.include_identity else [1, 2, 3]
        num_qubits = 2  # XXX hardcoding
        unique_circuits = set()

        # NOTE num_layers = 0 corresponds to 1 2Q gate and thus 0 1Qs
        for num_layers in range(max_layers):
            circuit_layers = list(
                product(product(gate_options, repeat=num_qubits), repeat=num_layers)
            )

            # prune for symmetries
            for seq in circuit_layers:
                rev_seq = seq[::-1]
                swapped_seq = tuple((j, i) for i, j in seq)
                rev_swapped_seq = swapped_seq[::-1]
                min_seq = min(seq, rev_seq, swapped_seq, rev_swapped_seq)
                unique_circuits.add(min_seq)

        # don't sort, we will do that later by pulse cost
        # return sorted(unique_circuits, reverse=True)
        return unique_circuits

    def _construct_labels(self) -> List[str]:
        return [ansatz._to_string(self.basis_str) for ansatz in self.circuit_ansatze]

    def _construct_costs(self) -> List[tuple]:
        """Return tuple (2q count, 1q depth, 1q total).

        Agnotistic to relative costs between 2Q and 1Q gates, so return
        both can scale elsewhere.
        """
        pulse_costs = []
        costs_1q = {3: 2, 2: 1, 1: 0, 0: 0}
        # XXX assumes only given 2 wire circuits
        for seq in self.u_set:
            tot = sum([costs_1q[layer[0]] + costs_1q[layer[1]] for layer in seq])
            depth = sum([max(costs_1q[layer[0]], costs_1q[layer[1]]) for layer in seq])
            pulse_costs.append((len(seq) + 1, depth, tot))
        return pulse_costs

    def _construct_ansatze(self) -> List[BasicCircuitAnsatz]:
        circuit_ansatz_list = []
        for u_sequence in self.u_set:
            # NOTE offset len()+1, since len() is counting internal 1Qs
            ansatz = BasicCircuitAnsatz(
                self.basis_gate, len(u_sequence) + 1, u_sequence
            )
            circuit_ansatz_list.append(ansatz)
        return circuit_ansatz_list

    def _sort_by_pulse_costs(self) -> None:
        # sort (pulse_costs, labels, ansatz) by pulse_costs
        if len(self.labels) != len(self.pulse_costs) or len(self.labels) != len(
            self.circuit_ansatze
        ):
            raise ValueError("experiment list size mismatch!")
        self.pulse_costs, self.labels, self.circuit_ansatze = zip(
            *sorted(zip(self.pulse_costs, self.labels, self.circuit_ansatze))
        )

    @staticmethod
    def _plot_weyl_polytope_scatter(
        ansatz: CircuitAnsatz,
        ax: Axes = None,
        title: str = None,
        color: tuple = None,
        **kwargs,
    ):
        if ax is None:
            ax = plt.subplot(111, projection="3d", computed_zorder=False)
        w = WeylChamber()
        w.labels = {}
        w.render(ax)
        _flat = [coord for inner_cord in ansatz.weyl_coords for coord in inner_cord]
        w.ax.scatter3D(*zip(*_flat), color=color, zorder=-10)
        w.ax.set_title(title)

    @staticmethod
    def plot_weyl_scatter(
        cost_list,
        label_list,
        ansatz_list,
    ):
        """Create a colored grid of Weyl chambers for each ansatz."""
        n = len(ansatz_list)
        colors = cm.tab10(np.linspace(0, 1.0, 10))
        max_cols = 7
        num_rows = (n + max_cols - 1) // max_cols
        fig, axs = plt.subplots(
            num_rows,
            max_cols,
            subplot_kw={"projection": "3d"},
            figsize=(max_cols * 5, num_rows * 5),
        )
        axs = axs.flatten()
        for i, ansatz in enumerate(ansatz_list):
            ax = axs[i] if n > 1 else axs
            title = f"{label_list[i]}\n cost: {cost_list[i]}"
            color = colors[i % len(colors)]
            PulseConstraintPolytopes._plot_weyl_polytope_scatter(
                ansatz, ax, title, color
            )
        # remove blanks
        for j in range(i + 1, num_rows * max_cols):
            fig.delaxes(axs[j])

    def run(self, num_samples=1_000):
        """Generate points for each ansatz, then create plot."""
        for ansatz in tqdm(self.circuit_ansatze):
            ansatz.generate_points(num_samples, method="qucontrol")
            # ansatz.fit_polytope()

        PulseConstraintPolytopes.plot_weyl_scatter(
            self.pulse_costs, self.labels, self.circuit_ansatze
        )

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from monodromy.coordinates import monodromy_to_positive_canonical_coordinate
from qiskit.synthesis.two_qubit.local_invariance import local_equivalence
from scipy.optimize import OptimizeResult
from weylchamber import WeylChamber

from hetero_isas.monodromy_lp.invariants import mono_coordinates_to_makhlin
from hetero_isas.monodromy_lp.isa import ISAHandler, MonodromyLPGate
from hetero_isas.monodromy_lp.lp_constraints.lp_constraints import LPConstraints

GATE_INVARIANTS = 3  # define this constant to avoid magic values
len_gi = GATE_INVARIANTS


def plot_histograms(results: list["MonodromyLPDecomposerResult"]):
    """Generate histograms.

    Used for total attempts and successful attempts per segment,
    as well as a histogram of total function evaluations across all segments.

    Args:
    results (List[MonodromyLPDecomposerResult]): List of decomposition results
    """
    max_segments = max(result.n - 1 for result in results)
    # Create subplots: one for stacked bar chart, one for total evaluations
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

    # Prepare data for stacked bar chart
    success_counts = np.zeros(max_segments)
    fail_counts = np.zeros(max_segments)
    for result in results:
        for i, success in enumerate(result.success):
            if success:
                success_counts[i] += 1
            else:
                fail_counts[i] += 1
    # Plot stacked bar chart
    segment_labels = [f"Segment {i + 1}" for i in range(max_segments)]
    ax1.bar(segment_labels, success_counts, label="Success", color="g")
    ax1.bar(segment_labels, fail_counts, bottom=success_counts, label="Fail", color="r")
    ax1.set_ylabel("Count")
    ax1.set_title("Success vs Fail Count per Segment")
    ax1.legend()
    # Prepare data for total evaluations histogram
    total_evals = [sum(result.total_nfev) for result in results]
    success_evals = [sum(result.success_nfev) for result in results]
    # Plot histogram for total evaluations
    ax2.hist(total_evals, label="Total Evals")
    ax2.hist(success_evals, label="Evals on Success")
    # ax2.set_title("Distribution of Total Function Evaluations")
    ax2.set_xlabel("Number of Function Evaluations")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    plt.tight_layout()
    plt.show()


class MonodromyLPDecomposerResult:
    """Represents the result of a monodromy linear programming decomposition.

    This class encapsulates the solution of the LP problem and provides
    methods for visualizing the result.

    Methods:
        render_path: Visualize the decomposition path in the Weyl chamber.
    """

    def __init__(
        self,
        isa_handler: ISAHandler,
        target_u: MonodromyLPGate,
    ):
        """Initialize a new decomposition result."""
        self.isa_handler = isa_handler
        self.isa = self.isa_handler.isa
        self.color_map = self.isa.color_map
        self.target_u = target_u

        # set by render_path()
        self.traj_points = []

        # set by _setter_mono_lp()
        self.lp_result = None
        self.constraints = None
        self.lp_calls = None
        self.n = None
        self.isa_sequence = None
        self.mono_points = None
        self.invariants = None
        self.basis_gate_sequence = None
        self.total_strength = None
        self.color_seq = None

        # set by _setter_1q_numerics
        # each element is a different segment from cartan trajectory
        self.success = []
        self.restarts = []
        self.total_nfev = []
        self.success_nfev = []
        self.fun_val = []

        # set by _set_final_qc
        self.final_qc = None

    def __repr__(self):
        """For debugging."""
        return f"""MonodromyLPDecomposerResult:
            Number of segments: {self.n}
            Gates strength: {self.total_strength:.4f}
            LP calls: {self.lp_calls}
            1Q numerics success: {all(self.success)}
            1Q numerics restarts: {self.restarts}
            Total function evaluations: {self.total_nfev}
            Function evaluations on success: {self.success_nfev}
            ISA sequence: {[str(gate) for gate in self.isa_sequence]}
            """

    def _setter_trivial_edge_case(self):
        raise NotImplemented

    def _setter_mono_lp(self, basis_gate_sequence, mono_coords, lp_calls):
        self.isa_sequence = basis_gate_sequence
        self.n = len(basis_gate_sequence)
        self.basis_gate_sequence = [
            self.isa_handler.get_unitary(g) for g in self.isa_sequence
        ]
        self.lp_calls = lp_calls
        self.mono_points = mono_coords
        self.invariants = [mono_coordinates_to_makhlin(*m) for m in self.mono_points]
        self.total_strength = sum(gate.strength for gate in self.isa_sequence)
        self.color_seq = [self.color_map[gate] for gate in self.isa_sequence]
        return self

    def _setter_1q_numerics(self, success, restarts, total_nfev, success_nfev, fun_val):
        self.success.append(success)
        self.restarts.append(restarts)
        self.total_nfev.append(total_nfev)
        self.success_nfev.append(success_nfev)
        self.fun_val.append(fun_val)

        return self

    def _set_final_qc(self, decomposed_qc):
        self.final_qc = decomposed_qc
        return self

    def render_path(self):
        """Visualize the decomposition path in the Weyl chamber using named styles and per-gate primary+gap color pairs."""
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.lines import Line2D

        # Define named line styles
        dash_patterns = {
            "solid": None,
            "densely dotted": [0.75, 1],
            "densely dashed": [2.5, 2],
            "densely dashdot": [3, 1.5, 1, 1.5],
        }

        primary_colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
        gap_colors = ["tab:orange", "tab:cyan", "tab:pink", "tab:olive"]

        # Map each gate to a color + gapcolor + linestyle
        gates = list(self.color_map.keys())
        gate_styles = {
            gate: {
                "color": primary_colors[i % len(primary_colors)],
                "gap": gap_colors[i % len(gap_colors)],
                "style": list(dash_patterns.keys())[i % len(dash_patterns)],
            }
            for i, gate in enumerate(gates)
        }

        # Compute trajectory points if not already computed
        if not self.traj_points:
            self.traj_points = [
                abs(np.array(monodromy_to_positive_canonical_coordinate(*p)))
                / (np.pi / 2)
                for p in self.mono_points
            ]

        # Set up the 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
        w = WeylChamber()
        w.labels = {}
        w.render(ax)

        # Plot all trajectory points
        pts = np.array(self.traj_points)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], zorder=-1)

        # Plot each segment
        for i in range(len(self.traj_points) - 1):
            p0 = self.traj_points[i]
            p1 = self.traj_points[i + 1]
            gate = self.isa_sequence[i]
            styles = gate_styles.get(
                gate, {"color": "black", "gap": "gray", "style": "solid"}
            )
            dashes = dash_patterns.get(styles["style"])

            xs = [p0[0], p1[0]]
            ys = [p0[1], p1[1]]
            zs = [p0[2], p1[2]]

            (line,) = ax.plot(
                xs,
                ys,
                zs,
                color=styles["color"],
                linewidth=6,
                zorder=0,
                gapcolor=styles["gap"],
            )
            if dashes:
                line.set_dashes(dashes)

            # Endpoint marker
            ax.scatter(
                [p1[0]],
                [p1[1]],
                [p1[2]],
                color="black",
                edgecolor="black",
                marker="o",
                s=30,
                zorder=1,
                depthshade=False,
            )

        # Mark origin
        ax.scatter([0], [0], [0], color="black", s=30, zorder=1, depthshade=False)

        # Legend (just show primary colors and style)
        custom_lines = []
        legend_labels = []
        for gate, styles in gate_styles.items():
            line = Line2D(
                [0], [0], color=styles["color"], lw=6, linestyle=styles["style"]
            )
            custom_lines.append(line)
            legend_labels.append(gate)

        ax.legend(custom_lines, legend_labels, ncol=2, title="Basis gate")
        plt.savefig("trajn.svg", bbox_inches="tight")
        return ax

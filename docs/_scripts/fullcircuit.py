"""Full-circuit transpile time and output duration: GULPS, XXDecomposer, Qiskit default.

All-to-all target with RZZ(pi/2), RZZ(pi/4), RZZ(pi/6) and U gates. Each pipeline
unrolls, consolidates two-qubit blocks, synthesizes, and merges one-qubit gates.
Two rounds of PassManager.run per circuit with fresh pipelines, method order rotated.
"""

import argparse
import json
import math
import time
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RZZGate, UGate, efficient_su2, quantum_volume
from qiskit.circuit.random import random_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Operator, process_fidelity
from qiskit.synthesis import XXDecomposer
from qiskit.synthesis.qft import synth_qft_full
from qiskit.transpiler import (
    InstructionProperties,
    PassManager,
    Target,
    TransformationPass,
)
from qiskit.transpiler.passes import (
    ConsolidateBlocks,
    Optimize1qGatesDecomposition,
    UnitarySynthesis,
    Unroll3qOrMore,
)

from gulps.decomposition import GulpsDecomposer
from gulps.transpiler import GulpsDecompositionPass

HERE = Path(__file__).resolve().parent
QUBITS = [4, 8, 16, 32, 64]
GATES = [
    ("zz", np.pi / 2, 1.0),
    ("sq2_zz", np.pi / 4, 1 / 2),
    ("sq3_zz", np.pi / 6, 1 / 3),
]
DUR_BASE, DUR_1Q, ERR_BASE = 500e-9, 52.3e-9, 1e-3
ROUNDS = 2


def build_target(n: int) -> Target:
    """An all-to-all Target on ``n`` qubits with the three RZZ gates and U."""
    target = Target()
    pairs = list(permutations(range(n), 2))
    for name, angle, scale in GATES:
        props = InstructionProperties(
            duration=DUR_BASE * scale, error=1 - (1 - ERR_BASE) ** scale
        )
        target.add_instruction(RZZGate(angle), {p: props for p in pairs}, name=name)
    u_err = 1 - (1 - ERR_BASE) ** (DUR_1Q / DUR_BASE)
    u_props = {
        (q,): InstructionProperties(duration=DUR_1Q, error=u_err) for q in range(n)
    }
    target.add_instruction(
        UGate(Parameter("t"), Parameter("p"), Parameter("l")), u_props
    )
    return target


class XXPass(TransformationPass):
    """Qiskit's mixed-strength XXDecomposer on every consolidated two-qubit block."""

    def __init__(self) -> None:
        """Build one decomposer with the three RZZ strengths and their fidelities."""
        super().__init__()
        embodiments, fidelities = {}, {}
        for _, angle, scale in GATES:
            qc = QuantumCircuit(2)
            qc.h([0, 1])
            qc.rzz(angle, 0, 1)
            qc.h([0, 1])
            # XXDecomposer maximizes the product of fidelities. With fidelity
            # (1 - ERR_BASE)**scale it minimizes the summed scale, as GULPS does.
            embodiments[angle], fidelities[angle] = qc, (1 - ERR_BASE) ** scale
        self.xx = XXDecomposer(fidelities, euler_basis="U", embodiments=embodiments)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Replace each two-qubit unitary node with its decomposition."""
        for node in dag.op_nodes():
            if node.op.name == "unitary" and node.op.num_qubits == 2:
                dag.substitute_node_with_dag(
                    node, self.xx(node.op.to_matrix(), approximate=False, use_dag=True)
                )
        return dag


def pass_managers(target: Target) -> dict[str, PassManager]:
    """The GULPS, XXDecomposer, and default UnitarySynthesis pipelines."""
    unroll, merge1q = (
        Unroll3qOrMore(target=target),
        Optimize1qGatesDecomposition(target=target),
    )
    consolidate = ConsolidateBlocks(force_consolidate=True)
    # The decomposer is built from the same gates and durations as the Target.
    gulps = GulpsDecomposer(
        [RZZGate(angle) for _, angle, _ in GATES], [scale for *_, scale in GATES]
    )
    return {
        "gulps": PassManager([unroll, GulpsDecompositionPass(gulps), merge1q]),
        "xx": PassManager([unroll, consolidate, XXPass(), merge1q]),
        "qiskit": PassManager(
            [unroll, consolidate, UnitarySynthesis(target=target), merge1q]
        ),
    }


def circuits(n: int, seed: int = 42) -> dict[str, QuantumCircuit]:
    """The four benchmark circuits on ``n`` qubits."""
    esu2 = efficient_su2(n, reps=3, entanglement="circular")
    esu2 = esu2.assign_parameters(
        np.random.default_rng(seed).uniform(0, 2 * np.pi, esu2.num_parameters)
    )
    return {
        "QFT": synth_qft_full(n, approximation_degree=max(0, n - 4)),
        "EfficientSU2": esu2,
        "QV": quantum_volume(n, depth=n, seed=seed),
        "Random": random_circuit(n, depth=4 * n, max_operands=2, seed=seed),
    }


def duration_us(qc: QuantumCircuit, one_qubit: bool = True) -> float:
    """The sum of the gate durations, in microseconds, without scheduling.

    With ``one_qubit=False`` the sum covers the two-qubit gates only.
    """
    by_angle = {round(a, 10): DUR_BASE * s for _, a, s in GATES}
    by_name = {name: DUR_BASE * s for name, _, s in GATES}
    total = 0.0
    for inst in qc.data:
        op = inst.operation
        if op.num_qubits == 1:
            total += DUR_1Q if one_qubit else 0.0
        else:
            total += (
                by_name.get(op.name) or by_angle[round(abs(float(op.params[0])), 10)]
            )
    return total * 1e6


# GULPS and XXDecomposer both minimize the summed two-qubit duration. The sums add
# at most about 1e5 terms, so their rounding error is below 1e5 * 2**-53 relative.
# The smallest real difference, DUR_BASE / 6 (one RZZ(pi/4) against one RZZ(pi/6)),
# in a sum of at most about 1e5 * DUR_BASE, is above 1e-6 relative.
SAME_COST_RTOL = 1e-9


def audit_blocks(qc: QuantumCircuit, target: Target) -> dict:
    """Synthesize each consolidated block of ``qc`` alone with GULPS and XXDecomposer.

    Counts the blocks whose output is not equivalent to the block (Operator.equiv),
    the blocks where the two summed two-qubit durations differ, and among those the
    blocks where XXDecomposer's output is equivalent or GULPS's is not.
    """
    blocks = PassManager(
        [Unroll3qOrMore(target=target), ConsolidateBlocks(force_consolidate=True)]
    ).run(qc)
    managers = pass_managers(target)
    counts = dict(gulps_inexact=0, xx_inexact=0, differing=0, differing_unexplained=0)
    xx_infidelity = []
    for inst in blocks.data:
        if inst.operation.num_qubits != 2:
            continue
        block = QuantumCircuit(2)
        block.unitary(Operator(inst.operation), [0, 1])
        u = Operator(block)
        gulps, xx = managers["gulps"].run(block), managers["xx"].run(block)
        gulps_exact, xx_exact = Operator(gulps).equiv(u), Operator(xx).equiv(u)
        counts["gulps_inexact"] += not gulps_exact
        if not xx_exact:
            counts["xx_inexact"] += 1
            xx_infidelity.append(1 - process_fidelity(Operator(xx), u))
        if not math.isclose(
            duration_us(gulps, one_qubit=False),
            duration_us(xx, one_qubit=False),
            rel_tol=SAME_COST_RTOL,
        ):
            counts["differing"] += 1
            counts["differing_unexplained"] += xx_exact or not gulps_exact
    return {
        **{f"blocks_{k}": v for k, v in counts.items()},
        "xx_max_infidelity": max(xx_infidelity, default=0.0),
    }


def check(rows: list[dict]) -> None:
    """Assert that GULPS matches XXDecomposer's two-qubit cost wherever XX is exact.

    Every GULPS block output is equivalent to its block. A row with unequal summed
    two-qubit durations has at least one differing block, and in every differing
    block XXDecomposer's output is not equivalent to the block.
    """
    for r in rows:
        key = (r["circuit"], r["qubits"])
        assert r["blocks_gulps_inexact"] == 0, key
        assert r["same_cost"] or r["blocks_differing"] > 0, key
        assert r["blocks_differing_unexplained"] == 0, key


def benchmark() -> list[dict]:
    """Transpile every circuit at every size with the three pipelines."""
    rows = []
    for n in QUBITS:
        target = build_target(n)
        warm = QuantumCircuit(n)
        warm.unitary(np.eye(4), [0, 1])
        warm.unitary(Operator(RZZGate(0.3)), [0, 1])
        for i, (family, qc) in enumerate(circuits(n).items()):
            row = dict(circuit=family, qubits=n, gulps_s=[], xx_s=[], qiskit_s=[])
            for rnd in range(ROUNDS):
                # Fresh pipelines per round: GULPS keeps each block class's sentence
                # across calls, so a repeated circuit would be timed warm.
                managers = pass_managers(target)
                for pm in managers.values():
                    pm.run(warm)
                keys = list(managers)
                shift = (rnd + i) % len(keys)
                for key in keys[shift:] + keys[:shift]:
                    start = time.perf_counter()
                    out = managers[key].run(qc)
                    row[key + "_s"].append(time.perf_counter() - start)
                    if rnd == 0:
                        if n <= 8:
                            assert Operator(out).equiv(Operator(qc)), (key, family, n)
                        row[key + "_duration_us"] = duration_us(out)
                        row[key + "_2q_duration_us"] = duration_us(out, one_qubit=False)
            row["same_cost"] = math.isclose(
                row["xx_2q_duration_us"],
                row["gulps_2q_duration_us"],
                rel_tol=SAME_COST_RTOL,
            )
            row.update(audit_blocks(qc, target))
            rows.append(row)
            print(
                f"{family:12s} {n:2d}q  GULPS {min(row['gulps_s']):.4f}-{max(row['gulps_s']):.4f}s"
                f"  XX {min(row['xx_s']):.4f}-{max(row['xx_s']):.4f}s"
                f"  default {min(row['qiskit_s']):.4f}-{max(row['qiskit_s']):.4f}s"
                f"  duration GULPS {row['gulps_duration_us']:.4f} XX {row['xx_duration_us']:.4f}"
                f" default {row['qiskit_duration_us']:.4f} us"
                f"  two-qubit GULPS {row['gulps_2q_duration_us']:.4f}"
                f" XX {row['xx_2q_duration_us']:.4f} us"
                f"  XX inexact blocks {row['blocks_xx_inexact']}",
                flush=True,
            )
    return rows


FAMILIES = ["QFT", "EfficientSU2", "QV", "Random"]
MARKERS = ["o", "s", "D", "^"]


def plot(
    rows: list[dict], out: Path = HERE.parent / "_static" / "fullcircuit.svg"
) -> None:
    """Draw the XXDecomposer/GULPS ratio of transpile time."""
    with plt.rc_context({"svg.hashsalt": "fullcircuit"}):
        fig, ax = plt.subplots(figsize=(6.5, 3.4), constrained_layout=True)
        extent = []
        for family, marker in zip(FAMILIES, MARKERS):
            fam = sorted(
                (r for r in rows if r["circuit"] == family and r["xx_s"]),
                key=lambda r: r["qubits"],
            )
            qubits = [r["qubits"] for r in fam]
            rounds = np.array([np.divide(r["xx_s"], r["gulps_s"]) for r in fam])
            ratio = np.exp(np.log(rounds).mean(axis=1))
            extent += [rounds.min(), rounds.max()]
            ax.errorbar(
                qubits,
                ratio,
                yerr=(ratio - rounds.min(axis=1), rounds.max(axis=1) - ratio),
                marker=marker,
                capsize=1.5,
                elinewidth=0.8,
                label={"QV": "Quantum volume"}.get(family, family),
            )
        # The y range ends at the powers of two just outside the data.
        lo = int(np.floor(np.log2(min(extent))))
        hi = int(np.ceil(np.log2(max(extent))))
        if lo <= 0:
            ax.axhline(1, color="0.5", lw=0.8, ls="--", zorder=0)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_ylim(2.0**lo, 2.0**hi)
        ax.set_xticks(QUBITS, [str(q) for q in QUBITS])
        ticks = [2**k for k in range(lo, hi + 1)]
        ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("Qubits")
        ax.set_ylabel("Transpile time, XXDecomposer / GULPS")
        fig.legend(loc="outside upper center", ncol=4)
        fig.savefig(out, metadata={"Date": None})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot", action="store_true", help="replot from fullcircuit.json only"
    )
    data = HERE / "fullcircuit.json"
    if parser.parse_args().plot:
        rows = json.loads(data.read_text())
    else:
        rows = benchmark()
        data.write_text(json.dumps(rows, indent=2) + "\n")
        check(rows)
    plot(rows)

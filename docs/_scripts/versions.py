"""Transpile time of this GULPS against a published release, per circuit family.

Each version runs in its own Python environment with default settings, in
PROCESSES processes per version alternating between the versions. Each process
builds a fresh pipeline per round, runs it once on a small circuit, and then
times it on the circuits of fullcircuit.py. Make the baseline environment with

    python -m venv /tmp/gulps-baseline
    /tmp/gulps-baseline/bin/pip install gulps==0.3.6 qiskit==2.5.2

and run this script from the repository's environment:

    python docs/_scripts/versions.py --baseline /tmp/gulps-baseline/bin/python
"""

import argparse
import json
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import RZZGate, efficient_su2, quantum_volume
from qiskit.circuit.random import random_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Operator
from qiskit.synthesis.qft import synth_qft_full
from qiskit.transpiler import PassManager, TransformationPass
from qiskit.transpiler.passes import Optimize1qGatesDecomposition, Unroll3qOrMore

HERE = Path(__file__).resolve().parent
QUBITS = [4, 8, 16, 32, 64]
FAMILIES = ["QFT", "EfficientSU2", "QV", "Random"]
# The RZZ angles and their costs, in units of the RZZ(pi/2) duration.
GATES = [(np.pi / 2, 1.0), (np.pi / 4, 1 / 2), (np.pi / 6, 1 / 3)]
PROCESSES, ROUNDS = 3, 2
CHECK_QUBITS = 8


def circuits(n: int, seed: int = 42) -> dict[str, QuantumCircuit]:
    """The four circuits of fullcircuit.py on ``n`` qubits."""
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


def gulps_pass() -> TransformationPass:
    """GulpsDecompositionPass over the three RZZ gates, in either API."""
    gates, costs = [RZZGate(a) for a, _ in GATES], [s for _, s in GATES]
    try:
        from gulps.decomposition import GulpsDecomposer
        from gulps.transpiler import GulpsDecompositionPass
    except ImportError:  # 0.3.6
        from gulps import GulpsDecomposer, GulpsDecompositionPass

        decomposer = GulpsDecomposer(gates, costs)

        # 0.3.6 reads the column-major matrices of Qiskit 2.5 as row-major and
        # decomposes their transposes, so its pass gets a row-major copy.
        def row_major(
            op: Gate, return_dag: bool = False
        ) -> QuantumCircuit | DAGCircuit:
            return decomposer(
                np.ascontiguousarray(op.to_matrix()), return_dag=return_dag
            )

        return GulpsDecompositionPass(row_major)
    return GulpsDecompositionPass(GulpsDecomposer(gates, costs))


def cost_2q(qc: QuantumCircuit) -> float:
    """The summed cost of the two-qubit gates."""
    scale = {round(a, 10): s for a, s in GATES}
    return sum(
        scale[round(abs(float(i.operation.params[0])), 10)]
        for i in qc.data
        if i.operation.num_qubits == 2
    )


def worker() -> dict:
    """Time this environment's GULPS on every circuit; print the rows as JSON."""
    rows = []
    for n in QUBITS:
        warm = QuantumCircuit(n)
        warm.rzz(0.3, 0, 1)
        warm.cx(1, 0)
        for family, qc in circuits(n).items():
            row = {"circuit": family, "qubits": n, "seconds": []}
            for _ in range(ROUNDS):
                pm = PassManager(
                    [
                        Unroll3qOrMore(),
                        gulps_pass(),
                        Optimize1qGatesDecomposition(basis=["u"]),
                    ]
                )
                pm.run(warm)
                start = time.perf_counter()
                out = pm.run(qc)
                row["seconds"].append(time.perf_counter() - start)
            row["cost_2q"] = cost_2q(out)
            if n <= CHECK_QUBITS:
                row["equiv"] = bool(Operator(out).equiv(Operator(qc)))
            rows.append(row)
    return {"version": version("gulps"), "rows": rows}


def benchmark(baseline: str) -> dict:
    """Alternate worker processes of the baseline and this environment.

    Checks that this environment's outputs are equivalent to their circuits and
    that both versions reach the same summed two-qubit cost.
    """
    runs = {baseline: [], sys.executable: []}
    for p in range(PROCESSES):
        for python in list(runs)[:: 1 if p % 2 == 0 else -1]:
            out = subprocess.run(
                [python, __file__, "--worker"],
                check=True,
                capture_output=True,
                text=True,
            )
            runs[python].append(json.loads(out.stdout))
            print(f"{python} process {p} done", flush=True)
    versions = [results[0]["version"] for results in runs.values()]
    rows = {}
    for v, results in zip(versions, runs.values()):
        for result in results:
            for r in result["rows"]:
                row = rows.setdefault(
                    (r["circuit"], r["qubits"]),
                    {"circuit": r["circuit"], "qubits": r["qubits"]},
                )
                row.setdefault(v, {**r, "seconds": []})["seconds"] += r["seconds"]
    for row in rows.values():
        old, new = (row[v] for v in versions)
        assert new.get("equiv", True), row
        assert np.isclose(old["cost_2q"], new["cost_2q"]), row
    return {"versions": versions, "rows": list(rows.values())}


def plot(data: dict, out: Path = HERE.parent / "_static" / "versions.svg") -> None:
    """Draw transpile time against qubits, one panel per circuit family."""
    import matplotlib.pyplot as plt

    old, new = data["versions"]
    with plt.rc_context({"svg.hashsalt": "versions"}):
        fig, axes = plt.subplots(
            2, 2, figsize=(4.4, 3.8), sharex=True, sharey=True, constrained_layout=True
        )
        for ax, family in zip(axes.ravel(), FAMILIES):
            fam = sorted(
                (r for r in data["rows"] if r["circuit"] == family),
                key=lambda r: r["qubits"],
            )
            qubits = [r["qubits"] for r in fam]
            for v, marker in ((old, "o"), (new, "s")):
                ms = np.array([r[v]["seconds"] for r in fam]) * 1e3
                med = np.median(ms, axis=1)
                ax.errorbar(
                    qubits,
                    med,
                    yerr=(med - ms.min(axis=1), ms.max(axis=1) - med),
                    marker=marker,
                    capsize=1.5,
                    elinewidth=0.8,
                    label=v if family == FAMILIES[0] else None,
                )
            ratio = np.median(fam[-1][old]["seconds"]) / np.median(
                fam[-1][new]["seconds"]
            )
            ax.text(
                0.96,
                0.06,
                f"{ratio:.0f}× at {qubits[-1]}",
                transform=ax.transAxes,
                ha="right",
                fontsize=8,
            )
            ax.set_title(family, fontsize=9)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(QUBITS, [str(q) for q in QUBITS])
            ax.minorticks_off()
        ticks = [0.1, 1, 10, 100, 1000]
        axes[0, 0].set_yticks(ticks, [f"{t:g}" for t in ticks])
        for ax in axes[1]:
            ax.set_xlabel("qubits")
        for ax in axes[:, 0]:
            ax.set_ylabel("time (ms)")
        fig.legend(loc="outside upper center", ncol=2, fontsize=8)
        fig.savefig(out, metadata={"Date": None})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--baseline", help="Python of the release environment")
    group.add_argument("--plot", action="store_true", help="replot from versions.json")
    group.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    path = HERE / "versions.json"
    if args.worker:
        print(json.dumps(worker()))
        sys.exit()
    if args.plot:
        data = json.loads(path.read_text())
    else:
        data = benchmark(args.baseline)
        path.write_text(json.dumps(data, indent=2) + "\n")
    plot(data)

"""Two calibrated durations of iSWAP: every pair on the grid, for the calibration page.

Scores every pair of GRID fractions, with the page's local-layer cost, on Haar
targets and on the six-qubit QFT blocks, and writes the mean costs to
calibration_grid.json beside this script. The calibration page shows
``pair_landscapes`` and plots the JSON; the documentation build does not run
this script. The run takes several minutes.
"""

import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate, iSwapGate

from gulps.analysis.calibration import GRID
from gulps.analysis.coverage import coverage_report, empirical_cost, two_qubit_blocks
from gulps.decomposition import GulpsDecomposer

HERE = Path(__file__).resolve().parent
base = iSwapGate()
local_layer_cost = 0.1
workload = QuantumCircuit(6)
workload.append(QFTGate(6), range(6))
blocks = two_qubit_blocks(workload)


def instruction_set(
    strengths: list[float], layer_cost: float = local_layer_cost
) -> GulpsDecomposer:
    """A decomposer over iSWAP at each strength, costed by its strength."""
    return GulpsDecomposer(
        [base.power(k) for k in strengths],
        costs=list(strengths),
        local_layer_cost=layer_cost,
    )


def pair_landscapes() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Mean cost of every GRID pair, on Haar targets and on the QFT blocks."""
    grid = np.asarray(GRID)
    landscapes = {}
    for name in ("Haar targets", "QFT blocks"):
        costs = np.full((len(grid), len(grid)), np.nan)
        for i, first in enumerate(grid):
            for j in range(i, len(grid)):
                device = instruction_set(sorted({float(first), float(grid[j])}))
                costs[j, i] = (
                    coverage_report(device).expected_cost
                    if name == "Haar targets"
                    else empirical_cost(device, blocks).expected_cost
                )
        landscapes[name] = costs
    return grid, landscapes


if __name__ == "__main__":
    grid, landscapes = pair_landscapes()
    data = {
        "grid": grid.tolist(),
        "costs": {
            name: [[None if np.isnan(c) else c for c in row] for row in costs]
            for name, costs in landscapes.items()
        },
    }
    (HERE / "calibration_grid.json").write_text(json.dumps(data) + "\n")

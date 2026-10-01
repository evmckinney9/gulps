"""Draw the README example circuit into docs/_static/circuit.png."""

from pathlib import Path

from qiskit.circuit.library import CSXGate, iSwapGate

from gulps.decomposition import GulpsDecomposer
from gulps.invariants import LocalEquivalenceClass

decomposer = GulpsDecomposer([CSXGate(), iSwapGate().power(1 / 3)], costs=[90, 120])
target = LocalEquivalenceClass((3 / 8, 5 / 16, 1 / 8))
circuit = decomposer(target.matrix)
circuit.draw("mpl").savefig(
    Path(__file__).resolve().parent.parent / "_static" / "circuit.png",
    bbox_inches="tight",
)

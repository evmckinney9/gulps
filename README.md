##  GULPS (Global Unitary Linear Programming Synthesis)
**📌 Preprint:** [https://arxiv.org/abs/2505.00543](https://arxiv.org/abs/2505.00543)

#### Two-Qubit Gate Synthesis via Linear Programming for Heterogeneous Instruction Sets
We present GULPS (Global Unitary Linear Programming Synthesis), a segmented Cartan trajectory method that compiles arbitrary two-qubit unitaries into native gate sets by decomposing the synthesis problem into locally solvable sub-problems. Each segment corresponds to a depth-2 circuit synthesized from a linear program over canonical gate invariants, subject to quantum Littlewood-Richardson (QLR) constraints. The intermediate invariants are stitched together via nonlinear least-squares optimization to recover the local operations between segments. This hybrid LP-numerical method enables robust synthesis across parameterized instruction sets. As quantum hardware continues to evolve, GULPS provides a scalable, ISA-aware compilation strategy suitable for integration into platforms like Qiskit.

For basic usage see: `src/gulps/notebooks/main.ipynb`

TODO:
- if polytope lookup but LP feasibility check have fallback method.
- clean up ordering on working target to reduce redundant LP feasibility check
- GulpsDecomposer._run fix function scope consistency when handling rho reflects
- clean up polytope lookup, single call with rho() so don't have to enumerate twice
- timing analysis
- self._eval_edge_case(target_inv)
- include tests for monodromy (.5, y,z) because this is where issues occur with reflections
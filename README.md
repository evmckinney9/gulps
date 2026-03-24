##  [GULPS](https://c.tenor.com/NAwgHzRfK_wAAAAC/tenor.gif)
[![ecosystem](https://img.shields.io/badge/Qiskit-Ecosystem-blueviolet)](https://github.com/qiskit-community/ecosystem)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![CI](https://github.com/evmckinney9/gulps/actions/workflows/ci.yml/badge.svg)](https://github.com/evmckinney9/gulps/actions/workflows/ci.yml)
[![Release](https://github.com/evmckinney9/gulps/actions/workflows/release.yml/badge.svg)](https://github.com/evmckinney9/gulps/actions/workflows/release.yml)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2505.00543-blue)](https://doi.org/10.48550/arXiv.2505.00543)

GULPS (Global Unitary Linear Programming Synthesis) is the first open tool that **robustly compiles arbitrary two-qubit unitaries optimally into non-standard instruction sets**.  

Most existing compilers only target CNOT gates. Analytical rules exist for a few special cases like fractional CNOT (XX family), Berkeley (B), and $\sqrt{\text{iSWAP}}$, but nothing more general. Numerical methods can in principle handle arbitrary gates, but they are slow, unreliable, and do not scale as instruction sets grow.  **GULPS fills this gap** by combining linear programming with lightweight numerics to achieve:  
- Support for **fractional, continuous, or heterogeneous gate sets**.  
- **Scalability to larger ISAs**, unlike black-box numerical methods.  
- A fast, practical tool integrated with Qiskit if you study **gate compilation from two-body Hamiltonians** or parameterized unitary families.

#### 📌 Read the preprint: [Two-Qubit Gate Synthesis via Linear Programming for Heterogeneous Instruction Sets](https://arxiv.org/abs/2505.00543)

> [!IMPORTANT]
> GULPS is a general-purpose numerical method. If your ISA has a known analytical decomposition (e.g., Qiskit's `XXDecomposer` for CX/RZX families), prefer that - specialized solvers will always be faster and more precise for the gates they target. GULPS is for everything else.

______
### Getting Started

```bash
pip install gulps
```

**Optional extras:**
| Extra | Install | What it adds |
|-------|---------|--------------|
| `cplex` | `pip install "gulps[cplex]"` | CPLEX-based continuous LP solver. Works but slower than the discrete path. |
| `dev` | `pip install "gulps[dev]"` | Plotting (`matplotlib`, `SciencePlots`), Jupyter, linting, etc. |
| `test` | `pip install "gulps[test]"` | Adds `pytest`. |

**Monodromy polytopes (optional, separate install):**

Precomputes monodromy polytope coverage sets for direct lookup, bypassing LP enumeration. Only worth it for large ISAs with many small fractional gates. The `monodromy` package is not on PyPI, so we use a [fork](https://github.com/qiskit-community/monodromy/pull/18). Install the fork and its system dependency manually:
```bash
sudo apt-get install lrslib
pip install -r requirements-monodromy.txt
```

-  For usage examples, see the notebooks in `src/notebooks/`.
-  Report issues: [GitHub Issues](https://github.com/evmckinney9/gulps/issues)

To begin, define your instruction set architecture (ISA) to configure the decomposer. Alternatively, if the instruction set is specified in the properties of a Qiskit `Target`, you can use GULPS as a transpiler `translation` plugin.

Define an ISA as a list of Qiskit `Gate` objects, each with a cost and (optionally) a name. Names are only used in debug logs. Costs must be additive (affine cost model: each gate added to a sentence adds its cost) because that's what the LP enumeration and polytope coverage search assume. This is a good fit for durations or small-infidelity approximations where errors add linearly. I typically use normalized durations where fractional gates cost proportionally to their basis gate.

```python
from qiskit.circuit.library import iSwapGate
from gulps import GulpsDecomposer
from gulps.core.isa import DiscreteISA

isa = DiscreteISA(
    gate_set=[iSwapGate().power(1 / 2), iSwapGate().power(1 / 3)],
    costs=[1 / 2, 1 / 3],
    names=["sqrt2iswap", "sqrt3iswap"],
)
decomposer = GulpsDecomposer(isa=isa)
```

That's it. Once initialized, you can call the decomposer with either a Qiskit `Gate` or a 4x4 `np.ndarray` representing a two-qubit unitary:
```python
from qiskit.quantum_info import random_unitary
from qiskit import QuantumCircuit

u = random_unitary(4, seed=0)
v: QuantumCircuit = decomposer(u)
v.draw()
```

Alternatively, to compile a full `QuantumCircuit`, use the GULPS `TransformationPass`. Because GULPS leaves single-qubit gates unsimplified, I recommend appending `Optimize1qGatesDecomposition` to rewrite them into standard gate sets:

```python
from gulps import GulpsDecompositionPass
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit.circuit.random import random_circuit

input_qc = random_circuit(4, 4, max_operands=2)
pm = PassManager(
    [
        GulpsDecompositionPass(decomposer),
        Optimize1qGatesDecomposition(basis="u3"),
    ]
)
output_qc = pm.run(input_qc)
output_qc.draw("mpl")
```
___
### Overview of the Decomposition Process
The decomposition begins by identifying the cheapest feasible basis gate sentence (a sequence of native gates sufficient to construct the target unitary). We use [monodromy polytopes](https://github.com/qiskit-community/monodromy) to describe the reachable space of canonical invariants for each sentence in the ISA.

For example, this ISA:
```python
from gulps.core.isa import DiscreteISA

isa = DiscreteISA(
    gate_set=[iSwapGate().power(1 / 2), iSwapGate().power(1 / 3)],
    costs=[1 / 2, 1 / 3],
    names=["sqrt2iswap", "sqrt3iswap"],
    precompute_polytopes=True,
)
```
has the following coverage set:
```python
from gulps.core.coverage import coverage_report

coverage_report(isa.coverage_set)
```
![isa_coverage](images/isa_coverage.png)

Once a sentence is chosen, a linear program is used to determine a trajectory of intermediate invariants. These represent the cumulative two-qubit nonlocal action after each gate in the sentence, starting from the identity and ending at the target.
```python
from gulps.core.invariants import GateInvariants
from gulps.viz.invariant_viz import plot_decomposition

example_input = random_unitary(4, seed=31)
target_inv = GateInvariants.from_unitary(example_input)
constraint_sol = decomposer._best_decomposition(target_inv=target_inv)
plot_decomposition(
    constraint_sol.intermediates, constraint_sol.sentence, decomposer.isa
);
```
![example_cartan_trajectory](images/example_cartan_trajectory.png)

In this example, the optimal sentence is composed of 2 $\sqrt[3]{\texttt{iSWAP}}$ gates and 1 $\sqrt[2]{\texttt{iSWAP}}$. That is, the resulting circuit falls into a parameterized ansatz like this:
![full_ansatz](images/full_ansatz.png)

Unlike other decomposition techniques, the linear program contains additional information about the intermediate points used to reduce the problem into simpler subproblems, each corresponding to a depth-2 circuit segment. In this case, the circuit has three segments, although the first red segment (beginning at Identity is trivial). That leaves two segments requiring synthesis:

| ![ansatz_1](images/ansatz_1.png) | ![ansatz_2](images/ansatz_2.png) |
|:------------------------:|:------------------------:|
| Red(2)                   | Blue                     |

We solve for the local one-qubit gates in each segment using a Gauss-Newton solver on the Makhlin invariants, followed by a Weyl-coordinate polish. The solver (implemented in Rust) is tuned to work well across a broad range of ISAs, but there is no one-size-fits-all for every possible gate set, so edge-case performance may vary.
```python
solutions = decomposer._local_synthesis._solve_segments(
    constraint_sol.sentence,
    constraint_sol.intermediates,
    n_inner=len(constraint_sol.sentence) - 1,
)
solutions[0]
# SegmentSolution(u0=..., u1=..., weyl_residual=1.2e-16, max_residual=4.8e-09, success=True)
```
After solving the individual segments, we apply a final stitching step to handle orientation between segments and to promote local equivalence into global unitary equivalence:
```python
decomposer._local_synthesis.synthesize_segments(
    gate_list=constraint_sol.sentence,
    invariant_list=constraint_sol.intermediates,
    target=target_inv,
).draw("mpl")
```
![final](images/final.png)

___
See more:
 - https://quantum-journal.org/papers/q-2020-03-26-247/
 - https://quantum-journal.org/papers/q-2022-04-27-696/
 - https://threeplusone.com/pubs/on_gates.pdf
 - https://chromotopy.org/latex/papers/xx-synthesis.pdf
 - https://github.com/qiskit-advocate/qamp-spring-23/issues/33
 - https://github.com/Qiskit/qiskit/pull/9375
 - https://weylchamber.readthedocs.io/en/latest/readme.html

___
> [!NOTE]
> This software is provided as-is with no guarantee of support or maintenance. Bug reports and pull requests are welcome, but there is no commitment to respond or resolve issues on any timeline.

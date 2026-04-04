##  [GULPS](https://c.tenor.com/NAwgHzRfK_wAAAAC/tenor.gif)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyPI - Version](https://img.shields.io/pypi/v/gulps)](https://pypi.org/project/gulps/)
[![CI](https://github.com/evmckinney9/gulps/actions/workflows/ci.yml/badge.svg)](https://github.com/evmckinney9/gulps/actions/workflows/ci.yml)
[![Release](https://github.com/evmckinney9/gulps/actions/workflows/release.yml/badge.svg)](https://github.com/evmckinney9/gulps/actions/workflows/release.yml)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2505.00543-blue)](https://doi.org/10.48550/arXiv.2505.00543)
[![ecosystem](https://img.shields.io/badge/Qiskit-Ecosystem-blueviolet)](https://github.com/qiskit-community/ecosystem)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evmckinney9/gulps/blob/main/src/notebooks/00_quickstart.ipynb)

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
| `monodromy` | `pip install -r requirements-monodromy.txt` | Precomputes monodromy polytope coverage sets for direct lookup. Also requires `lrslib` (`sudo apt-get install lrslib`). |
| `cplex` | `pip install "gulps[cplex]"` | CPLEX-based continuous LP solver. Works but slower than the discrete path. |
| `dev` | `pip install "gulps[dev]"` | Plotting, Jupyter, linting, etc. |
| `test` | `pip install "gulps[test]"` | Adds `pytest`. |
___

#### Qiskit Transpiler Plugin
If your backend's ISA is already defined in a Qiskit `Target`, GULPS works as a drop-in `translation` stage plugin:
```python
from qiskit import transpile

output_qc = transpile(input_qc, target=my_target, translation_method="gulps")
```

#### Custom ISA
For full control, define your ISA manually. Gate costs must be additive (e.g., normalized durations where fractional gates cost proportionally to their basis gate).

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

Once initialized, call the decomposer with a Qiskit `Gate` or a 4x4 `np.ndarray`:
```python
from qiskit.quantum_info import random_unitary

u = random_unitary(4, seed=0)
v = decomposer(u)
v.draw()
```

To compile a full `QuantumCircuit`, use the `TransformationPass`. Because GULPS leaves single-qubit gates unsimplified, append `Optimize1qGatesDecomposition` to rewrite them:

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
![isa_coverage](https://raw.githubusercontent.com/evmckinney9/gulps/main/images/isa_coverage.png)

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
![example_cartan_trajectory](https://raw.githubusercontent.com/evmckinney9/gulps/main/images/example_cartan_trajectory.png)

In this example, the optimal sentence is composed of 2 $\sqrt[3]{\texttt{iSWAP}}$ gates and 1 $\sqrt[2]{\texttt{iSWAP}}$. That is, the resulting circuit falls into a parameterized ansatz like this:
![full_ansatz](https://raw.githubusercontent.com/evmckinney9/gulps/main/images/full_ansatz.png)

Unlike other decomposition techniques, the linear program contains additional information about the intermediate points used to reduce the problem into simpler subproblems, each corresponding to a depth-2 circuit segment. In this case, the circuit has three segments, although the first red segment (beginning at Identity is trivial). That leaves two segments requiring synthesis:

| ![ansatz_1](https://raw.githubusercontent.com/evmckinney9/gulps/main/images/ansatz_1.png) | ![ansatz_2](https://raw.githubusercontent.com/evmckinney9/gulps/main/images/ansatz_2.png) |
|:------------------------:|:------------------------:|
| Red(2)                   | Blue                     |

We solve for the local one-qubit gates in each segment using a Gauss-Newton solver on the Makhlin invariants, followed by a Weyl-coordinate polish. Segment solving and stitching (recovering global unitary equivalence from the local solutions) happen in a single Rust call. The solver is tuned to work well across a broad range of ISAs, but there is no one-size-fits-all for every possible gate set, so edge-case performance may vary.
```python
circuit = decomposer._local_synthesis.synthesize_segments(
    gate_list=constraint_sol.sentence,
    invariant_list=constraint_sol.intermediates,
    target=target_inv,
)
circuit.draw("mpl")
```
![final](https://raw.githubusercontent.com/evmckinney9/gulps/main/images/final.png)

___
### Notebooks
| | Topic |
|---|---|
| [00_quickstart](src/notebooks/00_quickstart.ipynb) | Getting started with GULPS |
| [01_decomposition_pipeline](src/notebooks/01_decomposition_pipeline.ipynb) | Step-by-step decomposition pipeline |
| [02_benchmarks](src/notebooks/02_benchmarks.ipynb) | LP and solver performance benchmarks |
| [03_continuous](src/notebooks/03_continuous.ipynb) | Continuous ISA with gate power as a free variable |
| [04_mixed_continuous](src/notebooks/04_mixed_continuous.ipynb) | Multiple continuous gate families in one ISA |
| [05_xxdecomposer](src/notebooks/05_xxdecomposer.ipynb) | Comparison with Qiskit's XXDecomposer |

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

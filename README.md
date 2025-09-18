##  GULPS (Global Unitary Linear Programming Synthesis)

#### Two-Qubit Gate Synthesis via Linear Programming for Heterogeneous Instruction Sets
📌Preprint: [https://arxiv.org/abs/2505.00543](https://arxiv.org/abs/2505.00543)

We present [GULPS](https://c.tenor.com/NAwgHzRfK_wAAAAC/tenor.gif) (Global Unitary Linear Programming Synthesis), a segmented Cartan trajectory method that **compiles two-qubit unitaries into abitrary native gate sets** by decomposing the synthesis problem into locally solvable sub-problems. Each segment corresponds to a depth-2 circuit synthesized from a linear program over canonical gate invariants, subject to quantum Littlewood-Richardson (QLR) constraints. The intermediate invariants are stitched together via nonlinear least-squares optimization to recover the local operations between segments. This hybrid LP-numerical method enables robust synthesis across parameterized instruction sets. As quantum hardware continues to evolve, GULPS provides a scalable, ISA-aware compilation strategy suitable for integration into platforms like Qiskit.

- 🐍 `pip install gulps @ git+https://github.com/evmckinney9/gulps`
-  For usage examples, see: `src/gulps/notebooks/main.ipynb`.
-  Report issues: [Github issue tracker](https://github.com/ajavadia/hetero_isas/issues/4)
______
### 🖥️ Getting Started
To begin, define your instruction set architecture (ISA) to configure the decomposer. Alternatively, if the instruction set is specified in the properties of a Qiskit `Target`, you can use GULPS as a `UnitarySynthesis` plugin.

In this example, we define an ISA as a list of Qiskit `Gate` objects, each with an associated cost and (optionally) a name. The name is only used in debugging logs. Costs are required to prioritize candidate circuit sentences and can be interpreted either as normalized durations or as fidelities. I typically use durations, where fractional gates incur a proportionally fractional cost relative to their basis gate, because currently the cost is taken to be additive.

```python
isa = [
    (CXGate(), 1.0, "cx"),
    (CXGate().power(1 / 2), 1 / 2, "sqrt2cx"),
    (iSwapGate().power(1 / 2), 1 / 2, "sqrt2iswap"),
    (iSwapGate().power(1 / 3), 1 / 3, "sqrt3iswap"),
]
gate_set, costs, names = zip(*isa)
decomposer = GulpsDecomposer(gate_set=gate_set, costs=costs, names=names)
```

That's it—once initialized, you can call the decomposer with either a Qiskit `Gate` or a 4×4 `np.ndarray` representing a two-qubit unitary:
```python
u = random_unitary(4, seed=idx)
v: QuantumCircuit = decomposer(u)
```

Alternatively, to compile a full `QuantumCircuit`, use the GULPS `TransformationPass`. Because GULPS leaves single-qubit gates in each segment as generic `Unitary` gates, I recommend appending `Optimize1qGatesDecomposition` to rewrite them into standard gate sets:

```python
pm = PassManager(
    [
        GulpsDecompositionPass(gate_set, costs),
        Optimize1qGatesDecomposition(),
    ]
)
output_qc = pm.run(input_qc)
```
___
### 🔧 Overview of the Decomposition Process
The decomposition begins by identifying the cheapest feasible basis gate sentence—a sequence of native gates sufficient to construct the target unitary. We use [monodromy polytopes](https://github.com/qiskit-community/monodromy) to describe the reachable space of canonical invariants for each sentence in the ISA.

For example, this ISA:
```python
isa = [
    (iSwapGate().power(1 / 2), 1 / 2, "sqrt2iswap"),
    (iSwapGate().power(1 / 3), 1 / 3, "sqrt3iswap"),
]
```
has the following coverage set:
```python
_plot_coverage_set(decomposer.isa.coverage_set)
```
![isa_coverage](images/isa_coverage.png)

Once a sentence is chosen, a linear program is used to determine a trajectory of intermediate invariants. These represent the cumulative two-qubit nonlocal action after each gate in the sentence—starting from the identity and ending at the target.
```python
example_input = random_unitary(4, seed=None)
example_sentence, example_intermediates = decomposer._best_decomposition(
    target_inv=GateInvariants.from_unitary(example_input, enforce_alcove=True)
)
render_path(example_intermediates);
```
![example_cartan_trajectory](images/example_cartan_trajectory.png)

In this example, the optimal sentence is composed of three $\sqrt[3]{\texttt{iSWAP}}$ gates. That is, the resulting circuit falls into a parameterized ansatz like this:
![full_ansatz](images/full_ansatz.png)

The intermediate points break the problem into simpler subproblems, each corresponding to a depth-2 circuit segment. In this case, the circuit has three segments, although the blue segment is fixed (e.g., identity). That leaves two segments requiring synthesis:

| ![ansatz_1](images/ansatz_1.png) | ![ansatz_2](images/ansatz_2.png) |
|:------------------------:|:------------------------:|
| Orange              | Green                |

We solve for the local one-qubit gates in each segment using a numerical root-finding routine:
```python
example_segment_solutions = decomposer._numerics._synthesize_segments(
    example_sentence, example_intermediates
)
print("Segment solutions:", example_segment_solutions)
Segment solutions:
[array([ 3.04980046, -3.97898785, -5.08187288,  5.42993702,  4.16191883,0.7034179 ]),
array([-2.49813347, -4.0929992 ,  0.14047136,  2.82952009,  4.63593378, 0.32678556])]
```
After solving the individual segments, we apply a final stitching step to handle orietation between segments and to promote local equivalence into global unitary equivalence:
```python
# Recover unitary equivalence by promoting local equivalence
ret = decomposer._numerics._stitch_segments(
    example_sentence, example_intermediates, example_segment_solutions
)
U, V = c1c2c3(example_input), c1c2c3(Operator(ret).data)
print("Input unitary weyl invariants:", U)
print("Output unitary weyl invariants:", V)
ret.draw()
Input unitary weyl invariants: (np.float64(0.44173763), np.float64(0.34153949), np.float64(0.15117788))
Output unitary weyl invariants: (np.float64(0.44173763), np.float64(0.34153949), np.float64(0.15117788))
```
![final](images/final.png)


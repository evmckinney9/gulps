.. meta::
   :description: Synthesize into your native gates with a local-layer cost, run GULPS in a Qiskit pass manager, query synthesis costs from other passes, and see which gate classes a sentence reaches.

Compilation
===========

.. _single-target-synthesis:
.. _cost-model:

Native gates and costs
----------------------

Each set of single-qubit gates in a sentence is a *local layer*, so a
sentence of :math:`n` native gates has :math:`n+1` local layers. The cost
of a sentence is

.. math::

   C(g_1,\ldots,g_n)=\sum_{i=1}^n c(g_i)+(n+1)c_{\mathrm{local}},

where :math:`c(g)` is the cost of native gate :math:`g` and
:math:`c_{\mathrm{local}}` is ``local_layer_cost``, which defaults to zero.
When costs are durations, set ``local_layer_cost`` to the duration of one
layer of single-qubit gates. It can also model a fixed cost per native
gate, such as pulse ramp times.

A local-layer charge can make a shorter sentence of more expensive native
gates the cheapest. For this target, four :math:`\sqrt{\mathrm{CX}}`
gates cost 360 ns, and one :math:`\sqrt{\mathrm{CX}}` with two
:math:`\sqrt[3]{\mathrm{iSWAP}}` gates costs 370 ns. A 20 ns charge for
each local layer adds 100 ns to the first sentence and 80 ns to the
second, so the three-gate sentence wins:

.. jupyter-execute::

   from qiskit.circuit.library import CSXGate, iSwapGate
   from qiskit.quantum_info import Operator, random_unitary
   from gulps.decomposition import GulpsDecomposer

   gates = [CSXGate(), iSwapGate().power(1 / 3)]
   target_unitary = random_unitary(4, seed=3)
   for local_layer_cost in [0, 20]:
       decomposer = GulpsDecomposer(
           gates, costs=[90, 140], local_layer_cost=local_layer_cost
       )
       cost, sentence = decomposer.select(target_unitary)
       print(f"local_layer_cost={local_layer_cost}: cost {cost:g} ns,",
             [gate.name for gate in sentence])

.. jupyter-execute::
   :hide-code:
   :hide-output:

   _costs = {
       ell: GulpsDecomposer(gates, costs=[90, 140], local_layer_cost=ell).select(
           target_unitary
       )
       for ell in [0, 20]
   }
   assert _costs[0][0] == 360 and [g.name for g in _costs[0][1]] == ["csx"] * 4
   assert _costs[20][0] == 450
   assert sorted(g.name for g in _costs[20][1]) == ["csx", "xx_plus_yy", "xx_plus_yy"]

.. jupyter-execute::
   :alt: A two-qubit circuit with one √CX gate, two ∛iSWAP gates, and four layers of single-qubit gates.

   from qiskit.quantum_info import process_fidelity

   compiled = decomposer(target_unitary)
   print(f"Process fidelity: {process_fidelity(Operator(compiled), target_unitary):.12f}")
   compiled.draw("mpl")

.. _qiskit-circuits:

Qiskit circuits
---------------

:class:`~gulps.transpiler.GulpsDecompositionPass` groups the gates of a
circuit into two-qubit blocks and synthesizes each block with the
decomposer.

.. jupyter-execute::
   :alt: A compiled three-qubit quantum-volume circuit: two blocks, one with two √CX gates and one ∛iSWAP, the other with one √CX and two ∛iSWAP, between single-qubit u gates.

   from qiskit.circuit.library import quantum_volume
   from qiskit.transpiler import PassManager
   from qiskit.transpiler.passes import (
       HighLevelSynthesis, Unroll3qOrMore, Optimize1qGatesDecomposition,
   )
   from gulps.transpiler import GulpsDecompositionPass

   decomposer = GulpsDecomposer(gates, costs=[90, 140], local_layer_cost=20)
   circuit = quantum_volume(3, depth=3, seed=5)
   pipeline = PassManager([
       HighLevelSynthesis(),
       Unroll3qOrMore(),
       GulpsDecompositionPass(decomposer),
       Optimize1qGatesDecomposition(basis=["u"]),
   ])
   compiled = pipeline.run(circuit)
   compiled.draw("mpl", fold=-1, scale=0.5)

.. jupyter-execute::
   :hide-code:
   :hide-output:

   assert Operator(compiled).equiv(Operator(circuit))

With a backend ``Target``, select GULPS as the translation method. The
plugin reads the native gates and their durations for each qubit pair from
the Target.

.. details:: Build a Target

   .. jupyter-execute::

      from itertools import permutations
      from qiskit.circuit import Parameter
      from qiskit.circuit.library import UGate
      from qiskit.transpiler import InstructionProperties, Target

      pairs = list(permutations(range(3), 2))
      target = Target(num_qubits=3)
      target.add_instruction(
          CSXGate(), {p: InstructionProperties(duration=90e-9, error=3e-3) for p in pairs}
      )
      target.add_instruction(
          iSwapGate().power(1 / 3),
          {p: InstructionProperties(duration=140e-9, error=8e-3) for p in pairs},
      )
      target.add_instruction(
          UGate(Parameter("a"), Parameter("b"), Parameter("c")),
          {(q,): InstructionProperties(duration=20e-9, error=1e-4) for q in range(3)},
      )

.. jupyter-execute::

   from qiskit import transpile

   compiled = transpile(
       circuit, target=target, translation_method="gulps", optimization_level=1
   )
   print(dict(compiled.count_ops()))

.. warning::

   Optimization levels 2 and 3 can undo the least-cost sentences. Their
   later passes resynthesize two-qubit blocks with Qiskit's own decomposers,
   which minimize the number of two-qubit gates instead of your costs.

A pass built from a Target minimizes the Target property named by ``cost``,
``"duration"`` (the default) or ``"error"``. With these error rates,
minimizing error uses only :math:`\sqrt{\mathrm{CX}}`:

.. jupyter-execute::

   pipeline = PassManager([
       HighLevelSynthesis(),
       Unroll3qOrMore(),
       GulpsDecompositionPass(target, cost="error"),
       Optimize1qGatesDecomposition(basis=["u"]),
   ])
   print(dict(pipeline.run(circuit).count_ops()))

.. _routing-costs:

Synthesis costs in other passes
-------------------------------

``select`` returns the cost of a block without constructing its circuit.
Other transpiler passes can use it to compare choices by their compiled
cost instead of by gate counts. `MIRAGE <https://arxiv.org/abs/2308.03874>`_
introduced this idea for routing: a router can implement a block :math:`U`
or its *mirror* :math:`\mathrm{SWAP}\,U`, which applies
:math:`U` and then exchanges the two qubits. Using the mirror updates the
qubit mapping instead of inserting a SWAP gate.

.. jupyter-execute::

   from qiskit.circuit.library import quantum_volume, SwapGate
   from gulps.analysis.coverage import two_qubit_blocks

   routing_circuit = quantum_volume(4, depth=2, seed=1)
   blocks = two_qubit_blocks(routing_circuit)
   swap = Operator(SwapGate()).data
   direct = decomposer.select(blocks)
   mirrored = decomposer.select([swap @ Operator(block).data for block in blocks])
   print("Block   Original (ns)   Mirrored (ns)")
   for i, ((cost, _), (mirror_cost, _)) in enumerate(zip(direct, mirrored)):
       print(f"{i:5} {cost:15g} {mirror_cost:15g}")

.. jupyter-execute::
   :hide-code:
   :hide-output:

   assert [c for c, _ in direct] == [350, 400, 400, 350]
   assert [c for c, _ in mirrored] == [400, 350, 350, 400]

Because a mirror moves the logical qubits, the router weighs
each saving against its effect on later gates.

.. _weyl-coordinates:

Local equivalence
-----------------

Two gates are *locally equivalent* when single-qubit gates before and
after one of them give the other, up to global phase. For example, Hadamard
gates on the target qubit before and after CZ give CX. GULPS searches over
these equivalence classes, not over matrices, because the local layers of a
sentence can absorb any difference within a class.

.. jupyter-execute::

   from qiskit.circuit.library import CXGate, CZGate
   from gulps.invariants import LocalEquivalenceClass

   cx, cz = LocalEquivalenceClass.from_unitaries([CXGate(), CZGate()])
   print("Same class:", cx == cz)
   print("Weyl coordinates:", cx.weyl.round(3) + 0.0)

Each class has one *canonical gate*, set by its three Weyl coordinates:

.. math::

   \operatorname{CAN}(c_1,c_2,c_3)
   = \exp\!\left[\frac{i\pi}{2}(c_1 XX+c_2 YY+c_3 ZZ)\right].

Here :math:`XX=X\otimes X`, and likewise for :math:`YY` and :math:`ZZ`.

.. _sentence-regions:

Reachable regions
-----------------

The *reachable region* of a sentence is the set of classes that it
implements as its local layers vary.

.. jupyter-execute::

   from gulps.analysis.region import ReachableRegion
   from gulps.analysis.viz.polytope_viz import plot_region

   sqrt_iswap = LocalEquivalenceClass.from_unitary(iSwapGate().power(1 / 2))
   swap_class = LocalEquivalenceClass.from_unitary(SwapGate())
   one = ReachableRegion.of([sqrt_iswap])
   two = ReachableRegion.of([sqrt_iswap, sqrt_iswap])
   print("One √iSWAP reaches CX:", one.reaches(cx))
   print("Two √iSWAP reach CX:", two.reaches(cx))
   print("Two √iSWAP reach SWAP:", two.reaches(swap_class))

.. jupyter-execute::
   :alt: Reachable classes for two √iSWAP gates in the Weyl chamber. CX and CZ share a point on the edge of the region, marked with a circle. SWAP is outside, marked with a cross.

   ax = plot_region(two, color="tab:blue")
   ax.scatter(*cx.weyl, color="tab:orange", s=60, label="CX / CZ")
   ax.scatter(*swap_class.weyl, color="tab:red", s=60, marker="x", label="SWAP")
   ax.set_title("Reachable with two √iSWAP gates")
   ax.legend();

The class of CX and CZ lies in the shaded region, so two
:math:`\sqrt{\mathrm{iSWAP}}` gates implement both. SWAP lies outside, so no
choice of local layers makes two :math:`\sqrt{\mathrm{iSWAP}}` gates implement
SWAP. The outer tetrahedron is the Weyl chamber, the set of all Weyl
coordinates, and the inner wireframe encloses the perfect entanglers. The
shaded region includes both
:ref:`global-phase representatives <global-phase-representatives>` of each
class.

:doc:`synthesis` describes how GULPS searches these regions in order of
cost.

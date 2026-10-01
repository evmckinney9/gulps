.. meta::
   :description: GULPS is a Qiskit transpiler plugin that decomposes two-qubit unitaries into least-cost circuits of arbitrary native gates, such as fractional iSWAP or √CX.

GULPS
=====

GULPS is a two-qubit gate synthesis package for arbitrary native
instruction sets. Given a set of native two-qubit gates and a cost for each,
it selects the least-cost *sentence*, an ordered list of native gates with
single-qubit gates before, between, and after them, that implements a target
unitary. It then constructs those single-qubit gates with a dedicated
solver, `can_sandwich <https://github.com/evmckinney9/can_sandwich>`_.

Synthesize into your instruction set
------------------------------------

Install GULPS with the plotting dependencies that the examples in this guide
use:

.. code-block:: bash

   pip install "gulps[viz]"

The native gates can be any two-qubit gates
with fixed parameters: Qiskit standard gates with bound parameters, or your
own matrices wrapped in ``UnitaryGate``.

.. jupyter-execute::
   :alt: A two-qubit circuit with two √CX gates and one ∛iSWAP gate, separated by single-qubit gates.

   from qiskit.circuit.library import CSXGate, iSwapGate
   from qiskit.quantum_info import random_unitary
   from gulps.decomposition import GulpsDecomposer

   target = random_unitary(4, seed=0)
   decomposer = GulpsDecomposer(
       [CSXGate(), iSwapGate().power(1 / 3)], costs=[90, 120]
   )
   circuit = decomposer(target)
   circuit.draw("mpl")

.. jupyter-execute::

   from qiskit.quantum_info import Operator, process_fidelity

   print(f"Process fidelity: {process_fidelity(Operator(circuit), target):.12f}")

A cost can be a gate duration or any other quantity that adds up along a
sentence. In this example, single-qubit gates cost nothing. :doc:`compile`
adds a charge for each layer of single-qubit gates, runs GULPS in a Qiskit
pass manager, and queries synthesis costs from other transpiler passes.
:doc:`calibration` uses synthesis costs to choose which native gate durations to
calibrate.

.. toctree::
   :caption: Use GULPS
   :hidden:
   :maxdepth: 2

   self
   compile
   calibration

.. toctree::
   :caption: Understand the method
   :hidden:
   :maxdepth: 2

   synthesis
   monodromy

.. toctree::
   :caption: Reference
   :hidden:

   performance
   apidocs/index

.. toctree::
   :caption: Links
   :hidden:

   Paper <https://doi.org/10.1109/QCE68830.2026.00091>
   GitHub <https://github.com/evmckinney9/gulps>
   PyPI <https://pypi.org/project/gulps/>

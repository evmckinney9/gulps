.. meta::
   :description: Reference for the GULPS compilation, gate invariant, instruction-set analysis, visualization, and error APIs.

Public API
==========

Compilation
-----------

The translation plugin and the transpiler pass compile circuits for a
Qiskit ``Target``. A :class:`~gulps.decomposition.GulpsDecomposer` compiles
one two-qubit target, and a pass built from it uses its gates on every pair.

.. currentmodule:: gulps

.. autosummary::
   :toctree: stubs/
   :nosignatures:
   :template: public-class.rst

   decomposition.GulpsDecomposer
   transpiler.GulpsTranslationPlugin
   transpiler.GulpsDecompositionPass

Gate invariants
---------------

.. autosummary::
   :toctree: stubs/
   :nosignatures:
   :template: public-class.rst

   invariants.LocalEquivalenceClass

Instruction-set analysis
------------------------

These objects measure the reachable regions, Haar coverage, and cost of an
instruction set, and choose calibrated strengths, without synthesizing
circuits.

.. autosummary::
   :toctree: stubs/
   :nosignatures:

   analysis.region.haar_mass
   analysis.coverage.coverage_report
   analysis.coverage.empirical_cost
   analysis.calibration.calibrate
   analysis.calibration.strength_sweep

.. autosummary::
   :toctree: stubs/
   :nosignatures:
   :template: public-class.rst

   analysis.coverage.CoverageReport
   analysis.coverage.SentenceCoverage
   analysis.coverage.SampledCost
   analysis.calibration.Calibration
   analysis.calibration.StrengthSweep
   analysis.region.ReachableRegion

Visualization
-------------

These functions require the ``gulps[viz]`` extra.

.. autosummary::
   :toctree: stubs/
   :nosignatures:

   analysis.viz.weyl_chamber.draw_chamber
   analysis.viz.invariant_viz.scatter_plot
   analysis.viz.polytope_viz.plot_region
   analysis.viz.polytope_viz.plot_coverage_set
   analysis.viz.polytope_viz.plot_waypoints

Errors
------

.. autosummary::
   :toctree: stubs/
   :nosignatures:
   :template: public-class.rst

   decomposition.DecompositionError
   decomposition.SearchDepthError

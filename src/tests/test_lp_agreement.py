"""Verify that LP sentence selection agrees with polytope lookup.

TODO: refactor
NOTE, currently this is trivial because try_discrete_lp uses polytope_lookup.
A better test would compare between the precompute True and False methods.
"""

# import pytest
# from qiskit.quantum_info import random_unitary

# from gulps.core.invariants import GateInvariants
# from gulps.gulps_decomposer import GulpsDecomposer
# from tests.fixtures.isas import get_slim_isas


# @pytest.fixture(params=get_slim_isas(precompute_polytopes=False))
# def decomposer(request):
#     return GulpsDecomposer(isa=request.param)


# def test_lp_agrees_with_polytope(decomposer):
#     """For random targets the LP should find the same sentence as polytope lookup."""
#     for seed in range(5):
#         target = random_unitary(4, seed=seed)
#         target_inv = GateInvariants.from_unitary(target, enforce_alcove=True)

#         sentence = decomposer.isa.polytope_lookup(target_inv)
#         assert sentence is not None, "Polytope lookup failed for reachable gate."

#         lp_result = decomposer._try_discrete_lp(target_inv)
#         assert lp_result.success, "LP failed even though polytope lookup succeeded."
#         assert lp_result.sentence == tuple(sentence), (
#             f"LP solution sentence does not match polytope lookup sentence. "
#             f"Expected {tuple(sentence)}, got {lp_result.sentence}"
#         )

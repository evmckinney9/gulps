# # test_decomposer_numerics.py

# import numpy as np
# import pytest
# from qiskit.quantum_info import Operator

# from gulps.gulps_decomposer import GulpsDecomposer
# from gulps.core.invariants import GateInvariants
# from gulps import logger
# from tests.fixtures.circuits import sampled_reachable_target
# from tests.fixtures.isas import get_all_test_isas

# N_tests = 4


# @pytest.fixture(params=get_all_test_isas())
# def decomposer_fixture(request):
#     gates, costs = request.param
#     return GulpsDecomposer(gates, costs, precompute_polytopes=True)


# def test_segment_synthesis(decomposer_fixture, sampled_reachable_target):
# XXX, I tried decoupling this unit test from the LP, however
# I can't figure out why the intermediate trajectory returned by the sampler
# has a hard time working with my numerics synthesizer.
# the LP will not return the same intermediates as the sampler, but works fine
# the sampler is doing something messy with reflections(?)
# _, lp_intermediates, _ = decomposer_fixture._try_lp(
#     sampled_sentence, target_inv
# )

# for _ in range(N_tests):
#     target_unitary, sampled_sentence, sampled_intermediates = (
#         sampled_reachable_target(decomposer_fixture)
#     )
#     true_target = GateInvariants.from_unitary(target_unitary)
#     # target_inv = GateInvariants.from_unitary(target_unitary, enforce_alcove=True)

#     # first get the local rotations
#     segment_sols = decomposer_fixture._numerics._synthesize_segments(
#         sampled_sentence, sampled_intermediates
#     )

#     # NOTE this is not a check for correctness
#     assert segment_sols is not None, "Segment synthesis failed."

#     # then stitch together, recovering nested local equivalences
#     output_qc = decomposer_fixture._numerics._stitch_segments(
#         sampled_sentence,
#         sampled_intermediates,
#         segment_sols,
#         true_target,
#     )
#     # NOTE this is not a check for correctness
#     assert output_qc is not None, "Stitching segments failed."

# # if these have same monodromy, already in alcove_c2
# target_in_ac2 = np.isclose(
#     target_inv.monodromy, true_target.monodromy, rtol=1e-14
# ).all()

# # Find the best decomposition using LP
# sentence_out, intermediates = decomposer_fixture._best_decomposition(target_inv)

# if not target_in_ac2 and intermediates[-1] is not true_target:
#     logger.debug("Trying reflection of intermediates")
#     intermediates = [x.rho_reflect for x in intermediates]

# assert np.isclose(
#     GateInvariants.from_unitary(Operator(output_qc).data).monodromy,
#     true_target.monodromy,
#     rtol=1e-8,
# ).all(), "Output circuit does not match target monodromy."

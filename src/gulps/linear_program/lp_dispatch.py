# Copyright 2025-2026 Lev S. Bishop, Evan McKinney
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Batch LP solving for GulpsDecomposer.

Single public entry: :func:`best_decomposition_batch`. State read or mutated
on the decomposer: ``_lp_cache``, ``_continuous_lp``, ``isa``, ``config``.

Design: invert the naive "for each target, walk sentences" structure into
"for each sentence in cost order, knock out every pending target it can
solve." One sentence walk total, N targets riding along. Correctness
relies on ``isa.candidate_sentences()`` yielding in cost-monotone order --
the first sentence that solves a target IS its cheapest decomposition, so
solved targets drop out of ``pending`` immediately.

Per sentence, three filter tiers cheapest-to-most-expensive:
  1. Strength prefilter (O(1) necessary condition; always available)
  2. Polytope membership (exact when precomputed; skips LP entirely)
  3. LP solve (the real decomposition attempt)

Continuous path is structurally different: one parametric LP/MILP with
per-target RHS swap, no sentence enumeration. Handled separately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gulps.core.invariants import GateInvariants
from gulps.linear_program.lp_abc import ConstraintSolution
from gulps.linear_program.lp_solver import MinimalOrderedISAConstraints

if TYPE_CHECKING:
    from gulps.gulps_decomposer import GulpsDecomposer


# Identity-equivalent targets short-circuit to this depth-0 synthetic solution.
# The true cost is ``isa.single_qubit_cost`` (one bare 1q layer), but cost is
# only ever consumed by reporting/stats in notebooks, not behavioral.
# Hardcoding 0 lets us keep this at module scope instead of rebuilding per
# call against per-decomposer state.
_IDENTITY_SOL = ConstraintSolution(
    success=True,
    sentence=(),
    intermediates=(),
    parameters=(),
    cost=0.0,
)


class LPInfeasibleError(RuntimeError):
    """No valid ISA sentence found for ``target``: LP infeasible across all candidates."""

    def __init__(self, target: GateInvariants):
        """Carry the offending target on the exception for downstream handling."""
        super().__init__(
            f"No valid ISA sentence found for target with monodromy {target.monodromy}. "
            "All candidates failed the LP feasibility check. "
            "For small fractional basis gates, consider increasing isa.max_sequence_length. "
            "This may indicate insufficient gate set strength or numerical issues."
        )
        self.target = target


def _build_continuous_model(decomposer: "GulpsDecomposer"):
    """Construct the appropriate continuous constraints object for ``decomposer.isa``.

    Lazy import: CPLEX is only required when this path is actually taken.
    """
    isa = decomposer.isa
    config = decomposer.config
    if isa.is_single_family:
        from gulps.linear_program.cplex_lp import ContinuousISAConstraints

        return ContinuousISAConstraints(
            base=isa.gate_set[0],
            max_sequence_length=isa.max_sequence_length,
            k_lb=isa.k_lb,
            single_qubit_cost=isa.single_qubit_cost,
            config=config,
        )
    raise NotImplementedError(
        "HeterogeneousContinuousISAConstraints not yet implemented."
    )


def _continuous_batch(
    decomposer: "GulpsDecomposer",
    targets: list[GateInvariants],
    log_output: bool,
) -> dict[GateInvariants, ConstraintSolution]:
    """Solve every target through the shared continuous LP/MILP model.

    Identity-equivalent targets short-circuit. The model is built once on
    first call; subsequent calls reuse it via per-target RHS swap.
    """
    if decomposer._continuous_lp is None:
        decomposer._continuous_lp = _build_continuous_model(decomposer)
    constraints = decomposer._continuous_lp

    results: dict[GateInvariants, ConstraintSolution] = {}
    for t in targets:
        if t.is_identity or t.rho_reflect.is_identity:
            results[t] = _IDENTITY_SOL
            continue
        r = constraints.solve(t, log_output=log_output)
        if not r.success:
            raise LPInfeasibleError(t)
        results[t] = r
    return results


def _discrete_batch(
    decomposer: "GulpsDecomposer",
    targets: list[GateInvariants],
    log_output: bool,
) -> dict[GateInvariants, ConstraintSolution]:
    """Walk sentences in cost order, knocking out pending targets per sentence.

    Iterates ``isa.candidate_sentences()`` cost-monotone. Identity-equivalent
    targets short-circuit.

    Raises ``LPInfeasibleError`` if any target remains unsolved after the
    sentence stream is exhausted.
    """
    isa = decomposer.isa
    config = decomposer.config
    tol = config.lp_feasibility_tol
    cost_dict = isa.cost_dict
    s_q_c = isa.single_qubit_cost
    lp_cache = decomposer._lp_cache
    moc_cache = decomposer._moc_cache

    results: dict[GateInvariants, ConstraintSolution] = {}
    pending: list[GateInvariants] = []
    for t in targets:
        if t.is_identity or t.rho_reflect.is_identity:
            results[t] = _IDENTITY_SOL
        else:
            pending.append(t)
    if not pending:
        return results

    # Cost-monotone sentence stream: first sentence that solves a target is
    # that target's cheapest decomposition, so solved targets drop out
    # immediately and never need re-evaluation against later sentences.
    for sentence, polytope, sentence_strength in isa.candidate_sentences():
        if not pending:
            break

        # Necessary-condition prefilter: cheap strength check, then exact
        # polytope membership if precomputed.
        def prefilter(target: GateInvariants) -> bool:
            if sentence_strength + tol < target.strength:
                return False
            if polytope is not None and not (
                polytope.has_element(target.monodromy)
                or polytope.has_element(target.rho_reflect.monodromy)
            ):
                return False
            return True

        to_solve: list[GateInvariants] = []
        rest: list[GateInvariants] = []
        for target in pending:
            (to_solve if prefilter(target) else rest).append(target)
        if not to_solve:
            pending = rest
            continue

        constraints = moc_cache.get(sentence)
        if constraints is None:
            constraints = MinimalOrderedISAConstraints(
                sentence,
                config=config,
                solver_cache=lp_cache,
                parameters=tuple(cost_dict[g] for g in sentence),
                single_qubit_cost=s_q_c,
            )
            moc_cache[sentence] = constraints
        for target in to_solve:
            r = constraints.solve(target, log_output=log_output)
            if r.success:
                results[target] = r
            else:
                rest.append(target)
        pending = rest

    if pending:
        raise LPInfeasibleError(pending[0])
    return results


def best_decomposition_batch(
    decomposer: "GulpsDecomposer",
    targets: list[GateInvariants],
    log_output: bool = False,
) -> dict[GateInvariants, ConstraintSolution]:
    """Batch-solve ``targets`` against ``decomposer.isa``, keyed by target.

    Identity-equivalent targets short-circuit to a depth-0 synthetic solution.
    The remainder go through the continuous LP (one shared model) or the
    discrete sentence walk depending on ``decomposer._is_continuous``.
    Raises ``LPInfeasibleError`` on any target the ISA cannot solve.
    """
    if decomposer._is_continuous:
        return _continuous_batch(decomposer, targets, log_output)
    return _discrete_batch(decomposer, targets, log_output)

try:
    import docplex  # or the precise submodule you need
except ModuleNotFoundError as exc:
    raise ImportError(
        "ContinuousISAConstraints requires the 'cplex' extra. "
        "Install with `pip install gulps[cplex]`."
    ) from exc

from typing import TYPE_CHECKING, List, Optional

import numpy as np
from docplex.mp.model import Model

from gulps import GateInvariants
from gulps.config import GulpsConfig
from gulps.linear_program.lp_abc import ConstraintSolution, ISAConstraints
from gulps.linear_program.qlr import len_qlr, qlr_inequalities

if TYPE_CHECKING:
    from gulps.core.isa import ContinuousISA


class ContinuousISAConstraints(ISAConstraints):
    """LP constraints for a single yet continuous ISA sequence.

    Continuous ordered-ISA constraints with a single base gate:
    g_i = k_i * base.monodromy,  k_i ∈ [0,1].

    This is a variation of the original CPLEX implementation
    https://github.com/ajavadia/hetero_isas/blob/main/src/hetero_isas/monodromy_lp/lp_constraints/docplex_constraints.py

    Crucially, NOTE we will assume the ISA contains a single gate described by a
    single continuous family: g = B * k, with k in [0,1]. The (duration/cost) is k.

    TODO, has the opposite monotonic constraint as the discrete ISA, I have found
    that it is slightly better to have shorter gates first.
    """

    _ci_block, _gi_block, _ciplus1_block, _bi = qlr_inequalities

    def __init__(
        self,
        base: GateInvariants,
        max_sequence_length: int = 6,
        offset: float = 1e-8,
        k_lb: float = 0.01,
        single_qubit_cost: float = 1e-8,
        config: GulpsConfig | None = None,
    ):
        self.N = max_sequence_length
        if self.N < 2:
            raise ValueError("sequence_length must be at least 2")
        self.base = base
        self.B = base.monodromy
        self.offset = offset
        self.k_lb = 0.001 if k_lb < 0.0 else k_lb
        self.single_qubit_cost = single_qubit_cost
        self.config = config or GulpsConfig()
        self.model = self._create_model()
        self._target_def: Optional[np.ndarray] = None
        self._target_cts: List = []  # holds the 3 equality constraints for c_N

    def set_target(self, target: GateInvariants) -> None:
        """Set the target gate invariants for the constraint RHS."""
        self._target_def = target.monodromy

        # remove previous target constraints (if any), then add new ones
        if self._target_cts:
            self.model.remove_constraints(self._target_cts)
            self._target_cts.clear()

        # ci_nested represents [c2, ..., cN]; last element is cN
        self._target_cts = [
            self.model.add_constraint(
                self.ci_nested[-1][j] == float(self._target_def[j])
            )
            for j in range(3)
        ]

    def solve(
        self, target: GateInvariants, log_output: bool = False
    ) -> ConstraintSolution:
        """Solve LP, trying both rho orientations and returning the cheaper solution.

        Unlike discrete ISAs where cost is fixed by the sentence, continuous ISAs
        can have very different costs for target vs rho(target). We must try both
        and return the cheaper one.

        TODO FIXME: this is doubling the cost of solving. We could use a binary variable to
        move the selection of rho to be incorporated into the LP itself.
        """
        self.set_target(target)
        result = self.solve_single(log_output=log_output)

        self.set_target(target.rho_reflect)
        rho_result = self.solve_single(log_output=log_output)

        if not result.success:
            return rho_result
        if not rho_result.success:
            return result

        return result if result.cost <= rho_result.cost else rho_result

    def solve_single(self, log_output: bool = False) -> ConstraintSolution:
        sol = self.model.solve(log_output=log_output)
        if not sol:
            return ConstraintSolution(success=False)

        # Extract k, g, c values from solution
        ks = [float(sol.get_value(self.k_vars[i])) for i in range(self.N)]
        cis = [
            np.array(
                [sol.get_value(self.ci_nested[i][j]) for j in range(3)], dtype=float
            )
            for i in range(self.N - 1)
        ]

        # Count active gates using y binary variables
        depth = sum(int(sol.get_value(self.y[i])) for i in range(self.N))

        gi_list = [
            GateInvariants.from_unitary(self.base.unitary.power(k), name=self.base.name)
            for k in ks[:depth]
        ]
        intermediate_invariants = (gi_list[0],) + tuple(
            GateInvariants(tuple(c)) for c in cis[: depth - 1]
        )

        return ConstraintSolution(
            success=True,
            sentence=tuple(gi_list),
            intermediates=intermediate_invariants,
            parameters=tuple(ks[:depth]),
            cost=sum(ks[:depth]) + self.single_qubit_cost * (depth + 1),
        )

    def _create_model(self):
        m = Model("ContinuousISAConstraints", ignore_names=True)
        # Define variables, constraints, and objective here
        # STEP 1. CREATE ALL VARIABLES
        # Variables
        self.y = m.binary_var_list(self.N)
        self.k_vars = m.semicontinuous_var_list(self.N, lb=self.k_lb)

        self.gi_nested = [
            m.continuous_var_list(3, lb=-1.0, ub=1.0) for _ in range(self.N)
        ]
        self.ci_nested = [
            m.continuous_var_list(3, lb=-1.0, ub=1.0) for _ in range(self.N - 1)
        ]
        # used for depth/nz counting
        for i in range(self.N):
            m.add_constraint(self.k_vars[i] >= self.k_lb * self.y[i])
            m.add_constraint(self.k_vars[i] <= self.y[i])

        # STEP 2. GATE PARAMETERIZATION + MONOTONIC
        for i in range(self.N):
            m.add_constraint(self.gi_nested[i][0] == self.B[0] * self.k_vars[i])
            m.add_constraint(self.gi_nested[i][1] == self.B[1] * self.k_vars[i])
            m.add_constraint(self.gi_nested[i][2] == self.B[2] * self.k_vars[i])
        for i in range(self.N - 1):
            m.add_constraint(self.k_vars[i] >= self.k_vars[i + 1])
            m.add_constraint(self.y[i] >= self.y[i + 1])

        # STEP 3. QLR CONSTRAINTS
        self._qlr_constraints(m)

        # STEP 4. OBJECTIVE FUNCTION
        # Primary objective: minimize total cost = ∑ k_i + single_qubit_cost * (∑ y_i + 1)
        # This integrates depth penalty into the primary cost objective
        # Note: depth + 1 because single-qubit layers = two-qubit depth + 1
        obj_cost = m.sum(self.k_vars) + self.single_qubit_cost * (m.sum(self.y) + 1)

        # Secondary objective: minimize depth (may become obsolete with 1q cost in primary)
        obj_depth = m.sum(self.y)
        # obj_leftpack = m.sum((i + 1) * self.y[i] for i in range(self.N))

        m.set_multi_objective(
            sense="min",
            exprs=[obj_cost, obj_depth],
            priorities=[2, 1],
            abstols=[self.offset, 0],
        )

        return m

    def _qlr_constraints(self, m: Model):
        # For each gate g_i (i=2..N), build L(c_{i-1}, g_i, c_i)
        # Array mappings: g_i = gi_nested[i-1], c_i = gi_nested[0] if i==1 else ci_nested[i-2]
        for i in range(2, self.N + 1):
            c_im1 = self.gi_nested[0] if i == 2 else self.ci_nested[i - 3]  # c_{i-1}
            g_i = self.gi_nested[i - 1]  # g_i
            c_i = self.ci_nested[i - 2]  # c_i
            for r in range(len_qlr):
                m.add_constraint(
                    m.scal_prod(c_im1, self._ci_block[r])
                    + m.scal_prod(g_i, self._gi_block[r])
                    + m.scal_prod(c_i, self._ciplus1_block[r])
                    <= float(self._bi[r])
                )


# class HeterogeneousContinuousISAConstraints(ContinuousISAConstraints):
#     """MILP constraints for heterogeneous continuous ISA with multiple gate families.

#     Uses binary selection matrix z[i,f] to select gate family f at position i,
#     combined with continuous parameter k[i] for the selected family.
#     """

#     def __init__(
#         self,
#         isa: "ContinuousISA",
#         max_sequence_length: int = 6,
#         offset: float = 1e-8,
#         config: GulpsConfig | None = None,
#     ):
#         if isa.is_single_family:
#             raise ValueError("Use ContinuousISAConstraints for single-family ISAs.")

#         self.isa = isa
#         self.num_families = len(isa.gate_set)
#         self.bases = [g.monodromy for g in isa.gate_set]
#         self.cost_rates = [isa.cost_dict[g] for g in isa.gate_set]

#         super().__init__(
#             base=isa.gate_set[0],
#             max_sequence_length=max_sequence_length,
#             offset=offset,
#             k_lb=isa.k_lb,
#             single_qubit_cost=isa.single_qubit_cost,
#             config=config,
#         )

#     def _create_model(self):
#         m = Model("HeterogeneousContinuousISAConstraints", ignore_names=True)
#         F = self.num_families

#         # STEP 1: VARIABLES
#         self.y = m.binary_var_list(self.N)  # position active?
#         self.k_vars = m.semicontinuous_var_list(self.N, lb=self.k_lb)  # parameter

#         # Binary selection: z[i,f] = 1 iff position i uses family f
#         self.z = m.binary_var_matrix(self.N, F)

#         self.gi_nested = [
#             m.continuous_var_list(3, lb=-1.0, ub=1.0) for _ in range(self.N)
#         ]
#         self.ci_nested = [
#             m.continuous_var_list(3, lb=-1.0, ub=1.0) for _ in range(self.N - 1)
#         ]

#         # STEP 2: LINK y, k, and z
#         for i in range(self.N):
#             # k is active iff y is active
#             m.add_constraint(self.k_vars[i] >= self.k_lb * self.y[i])
#             m.add_constraint(self.k_vars[i] <= self.y[i])

#             # Exactly one family selected iff position is active
#             m.add_constraint(m.sum(self.z[i, f] for f in range(F)) == self.y[i])

#         # STEP 3: GATE PARAMETERIZATION via indicators
#         # When z[i,f] == 1, enforce g_i = k_i * B[f]
#         for i in range(self.N):
#             for f in range(F):
#                 for j in range(3):
#                     m.add_indicator(
#                         self.z[i, f],
#                         self.gi_nested[i][j] == self.bases[f][j] * self.k_vars[i],
#                         active_value=1,
#                     )

#             # When inactive (y[i]=0), g_i = 0
#             for j in range(3):
#                 m.add_indicator(
#                     self.y[i],
#                     self.gi_nested[i][j] == 0,
#                     active_value=0,
#                 )

#         # STEP 4: MONOTONIC ORDERING
#         for i in range(self.N - 1):
#             m.add_constraint(self.k_vars[i] >= self.k_vars[i + 1])
#             m.add_constraint(self.y[i] >= self.y[i + 1])

#             # Optional: order by family strength (like old code)
#             # sum_{f'<f} z[i,f'] >= sum_{f'<f} z[i+1,f']
#             for f in range(F):
#                 m.add_constraint(
#                     m.sum(self.z[i, fp] for fp in range(f))
#                     <= m.sum(self.z[i + 1, fp] for fp in range(f))
#                 )

#         # STEP 5: QLR CONSTRAINTS
#         self._qlr_constraints(m)

#         # STEP 6: OBJECTIVE
#         # Cost = sum_i sum_f z[i,f] * cost_rate[f] * k[i]
#         # This is nonlinear, so we linearize using auxiliary variables w[i,f]
#         self.w = m.continuous_var_matrix(self.N, F, lb=0.0)

#         for i in range(self.N):
#             for f in range(F):
#                 # w[i,f] = z[i,f] * k[i] (linearized via big-M)
#                 # w[i,f] <= k[i]
#                 # w[i,f] <= z[i,f]  (since k <= 1)
#                 # w[i,f] >= k[i] - (1 - z[i,f])
#                 m.add_constraint(self.w[i, f] <= self.k_vars[i])
#                 m.add_constraint(self.w[i, f] <= self.z[i, f])
#                 m.add_constraint(self.w[i, f] >= self.k_vars[i] - (1 - self.z[i, f]))

#         obj_cost = m.sum(
#             self.cost_rates[f] * self.w[i, f] for i in range(self.N) for f in range(F)
#         ) + self.single_qubit_cost * (m.sum(self.y) + 1)
#         obj_depth = m.sum(self.y)

#         m.set_multi_objective(
#             sense="min",
#             exprs=[obj_cost, obj_depth],
#             priorities=[2, 1],
#             abstols=[self.offset, 0],
#         )

#         return m

#     def solve_single(self, log_output: bool = False) -> ConstraintSolution:
#         sol = self.model.solve(log_output=log_output)
#         if not sol:
#             return ConstraintSolution(success=False)

#         ks = [float(sol.get_value(self.k_vars[i])) for i in range(self.N)]
#         depth = sum(int(sol.get_value(self.y[i])) for i in range(self.N))

#         # Extract selected families
#         families = []
#         for i in range(depth):
#             for f in range(self.num_families):
#                 if sol.get_value(self.z[i, f]) > 0.5:
#                     families.append(f)
#                     break

#         # Build gate list
#         gi_list = []
#         for i in range(depth):
#             base_gate = self.isa.gate_set[families[i]]
#             gi_list.append(
#                 GateInvariants.from_unitary(
#                     base_gate.unitary.power(ks[i]), name=base_gate.name
#                 )
#             )

#         cis = [
#             np.array(
#                 [sol.get_value(self.ci_nested[i][j]) for j in range(3)], dtype=float
#             )
#             for i in range(self.N - 1)
#         ]
#         intermediate_invariants = (gi_list[0],) + tuple(
#             GateInvariants(tuple(c)) for c in cis[: depth - 1]
#         )

#         cost = sum(ks[i] * self.cost_rates[families[i]] for i in range(depth))
#         cost += self.single_qubit_cost * (depth + 1)

#         return ConstraintSolution(
#             success=True,
#             sentence=tuple(gi_list),
#             intermediates=intermediate_invariants,
#             parameters=tuple(ks[:depth]),
#             cost=cost,
#         )

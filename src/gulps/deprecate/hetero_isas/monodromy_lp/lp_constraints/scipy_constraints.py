"""Constraint definitions and factory for monodromy linear programming."""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from scipy.optimize import OptimizeResult, linprog

from hetero_isas.monodromy_lp.isa import MonodromyLPGate, MonodromyLPISA
from hetero_isas.monodromy_lp.lp_constraints.lp_constraints import LPConstraints
from hetero_isas.monodromy_lp.lp_constraints.qlr import len_qlr, qlr_inequalities

GATE_INVARIANTS = 3  # define this constant to avoid magic values
len_gi = GATE_INVARIANTS


class ScipyLPConstraints(LPConstraints):
    def __init__(self, isa_sequence: List[MonodromyLPGate]):
        super().__init__(isa_sequence)

    def _lp_solve(self, log_output) -> OptimizeResult:
        """Attempt to solve the LP problem for the given constraints."""
        # edge case, if there were no free variables in x_vec
        if len(self.A_ub[0]) == 0:
            if np.all(0 <= self.Ab_ub):
                return OptimizeResult(x=[], success=True)
            return OptimizeResult(x=[], success=False)

        return linprog(
            *(self.c, self.A_ub, self.b_ub, self.A_eq, self.b_eq),
            bounds=self.bounds,
            method="highs",
            integrality=self.integrality,
            options={"disp": log_output},
        )

    def _extract_solution(self):
        # (basis gate sequence, intermediate mono coords)
        # if integrality, means we need to remove
        # from the parameter results the decision variables
        if self.integrality is not None:
            c_vec = self._last_result.x[:-1]
        else:
            c_vec = self._last_result.x[:]

        if self.c2_index == 0:
            full_c_vec = np.concatenate(
                (
                    np.zeros(len_gi),
                    self.isa_sequence[0].definition,
                    c_vec,
                    self._target_def,
                )
            )
        else:
            raise NotImplementedError("todo: need to assign back to gate definitions")

        mono_points = list(zip(*[iter(full_c_vec)] * len_gi))
        return self.isa_sequence, mono_points

    def _create_model(self):
        self.c = self._create_objective_matrix()
        self.A_ub, self.b_ub = self._create_inequality_matrices()
        self.A_eq, self.b_eq = self._create_equality_matrices()

    @abstractmethod
    def _set_target(self, target_gate: MonodromyLPGate):
        raise NotImplementedError

    @abstractmethod
    def _create_objective_matrix(self):  # noqa: D102
        raise NotImplementedError

    @abstractmethod
    def _create_inequality_matrices(self):  # noqa: D102
        raise NotImplementedError

    def _create_equality_matrices(self):  # noqa: D102
        return None, None


class OrderedDefinedISAConstraints(ScipyLPConstraints):
    """Special case of LPConstraints for well-defined ISA.

    This class simplifies the constraint setup by removing redundant variables
    for gates with exact definitions.

    A_ub.x <= b_ub; A_eq.x=b_eq
    x = [c2,...c{n-1}],
    NOTE missing elements bc c1 is constant (g1) and cn is constant (T)
    which I include in the constraints proper.
    """

    def __init__(self, isa_sequence: List[MonodromyLPGate]):
        """Initialize ConstantSubLPConstraints."""
        super().__init__(isa_sequence)
        self.num_ineq = len_qlr * (self.n - 1)  # rows in A
        self.num_params = len_gi * (self.n - 2)  # rows in x
        self.g1_index = 0
        self.c2_index = 0
        self.last_iter_ct = np.zeros(len_qlr)
        self._create_model()

    def _set_target(self, target_gate: MonodromyLPGate):
        # second edge case, plug in constant values for cn:=T
        # add back the last iterations to reset from past target
        self._target_def = target_gate.definition
        ct = np.dot(self._ciplus1_block, self._target_def)
        self.b_ub[-len_qlr:] += self.last_iter_ct - ct
        self.last_iter_ct = ct

    def _create_objective_matrix(self):
        """Generate a zero objective matrix as gate parameters are constant."""
        return np.zeros(self.num_params)

    def _create_inequality_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate simplified inequality constraint matrices A_ub and b_ub."""
        A_ub = np.zeros((self.num_ineq, self.num_params), dtype=np.double)
        b_ub = np.zeros((self.num_ineq), dtype=np.double)

        # L_i (ci, gi, ci+1)
        # first edge case is L_1 (g1, g2, c2)
        # second edge case is Ln (ci, gi, cn), since cn is constant...
        for i in range(self.n - 1):
            temp_row_range = slice(len_qlr * i, len_qlr * (i + 1))

            # c_i
            if i > 0:  # first edge case
                offset = self.c2_index + len_gi * (i - 1)
                A_ub[temp_row_range, range(offset, len_gi + offset)] += self._ci_block

            # g_i
            gi_c = np.dot(self._gi_block, self.isa_sequence[i + 1].definition)
            if i == 0:  # edge case, need to subtract both g_i (g1) and c_i (g2)
                ci_c = np.dot(self._ci_block, self.isa_sequence[i].definition)
                gi_c += ci_c
            b_ub[temp_row_range] += self._bi - gi_c

            # c_i+1
            if i < self.n - 2:
                offset = self.c2_index + len_gi * i
                A_ub[
                    temp_row_range, range(offset, len_gi + offset)
                ] += self._ciplus1_block

        return A_ub, b_ub


class RhoInclusiveOrderedDefinedISAConstraints(OrderedDefinedISAConstraints):
    """Same as parent clas but for T OR rho(T) with a binary decision var."""

    def __init__(self, isa_sequence: List[MonodromyLPGate]):
        """Initialize ConstantSubLPConstraints."""
        super().__init__(isa_sequence)
        self.num_params += 1
        # the final decision variable is binary {0,1}
        self.bounds = [(None, None)] * (self.num_params - 1) + [(0, 1)]
        self.integrality = np.zeros(self.num_params)
        self.integrality[-1] = 1
        self.last_iter_ct12 = np.zeros(len_qlr)
        self.last_iter_ct2 = np.zeros(len_qlr)
        self._create_model()

    def _set_target(self, target_gate: MonodromyLPGate):
        self._target_def = target_gate.definition
        # second edge case, plug in constant values for cn:=T
        # simpify using interpolation x(t1-t2) + t2, where t1:T and t2:rho(T)
        t1 = np.array(self._target_def)
        t2 = np.array(target_gate.rho_reflect().definition)
        ct12 = np.dot(self._ciplus1_block, t1 - t2)
        # ct12 should be a column vector, gets inserted into A_ub
        # finally, move constants to right hand side of the equation
        self.A_ub[-len_qlr:, -1] += ct12 - self.last_iter_ct12
        ct2 = np.dot(self._ciplus1_block, t2)
        self.b_ub[-len_qlr:] += self.last_iter_ct2 - ct2
        self.last_iter_ct12 = ct12
        self.last_iter_ct2 = ct2


# class UnorderedDefinedISAConstraints(ScipyLPConstraints):
#     """Introduce binary decision variables.

#     These are used for selecting the basis_sequence without us
#     needing to iterate the priority queue at all.

#     A_ub.x <= b_ub; A_eq.x=b_eq
#     x = [kij, ..., knm|f1,...,fn|c2,...c{n-1}]

#     NOTE i don't think I need f_i, because I could have a cost vector
#     and then minimize cost.k to get a vector of costs, using
#     the fact that dottin k correctly is just like an index lookup
#     x = [kij, ..., knm|c2,...c{n-1}]

#     where k_ij is {0,1} which selects gi in gate sequence to use gate Gj in ISA.
#     with additional constraints such that for all i, sum_j kij = 1 (1 and only 1 gate is selected)
#     (^ each k row only has a single non-zero)
#     additionally, lexicographic ordering. assume Gj is ordered then
#     (could probably formulate this in a couple different ways)
#     (^ k{i+1}j must have >= j then k_ij)

#     NOTE these methods have moved around during refactoring,
#     no guarantee of correctness right now.
#     """

#     def __init__(self, isa: MonodromyLPISA, target_gate: MonodromyLPGate):
#         """Initialize ConstantSubLPConstraints."""
#         if not isa.defined:
#             raise ValueError("must be a discrete set.")
#         super().__init__(isa, target_gate)
#         self.num_ineq = len_qlr * (self.n - 1)  # num inequalities, rows in A
#         self.num_params = len_gi * (2 * self.n - 1)  # free parameters, cols in x
#         self.g1_index = n * m
#         self.c2_index = len_gi * self.n
#         self.bounds = [(None, None)] * (self.num_params - 1) + [(0, 1)]
#         self.integrality = np.zeros(self.num_params)

#     def _construct_objective_matrix(self):
#         """Min sum_i (gi_a + g_ib + g_ic)."""
#         c = np.zeros(self.num_params)
#         c[: self.c2_index] = 1  # minimize costs of the basis gates
#         return c

#     def _construct_inequality_matrices(self):
#         """Places (alpha,beta,gamma) cols into appropriate col blocks."""
#         A_ub = np.zeros((self.num_ineq, self.num_params))
#         b_ub = np.zeros((self.num_ineq, 1))

#         # construct L_i(ci, gi, ci+1)
#         # the only edge case is L_1 (g1, g2, c2)
#         # otherwise L_i (ci, gi, ci+1)
#         for i in range(self.n - 1):
#             temp_row_range = slice(len_qlr * i, len_qlr * (i + 1))

#             # c_i block
#             offset = self.c2_index + len_gi * (i - 1) if i > 0 else 0
#             A_ub[temp_row_range, range(offset, len_gi + offset)] += self._ci_block
#             # g_i block
#             offset = len_gi * (i + 1)
#             A_ub[temp_row_range, range(offset, len_gi + offset)] += self._gi_block
#             # c_{i+1} block
#             offset = self.c2_index + len_gi * i
#             A_ub[temp_row_range, range(offset, len_gi + offset)] += self._ciplus1_block

#             b_ub[temp_row_range] = self._bi

#         return A_ub, b_ub


# class OrderedContinuousISAConstraints(ScipyLPConstraints):
#     """Ordered continuous.

#     I'm not sure how well this can work. because we need,
#     to have linear relationship between gate cost f and its invariants,
#     which I think is only true for XX gates....

#     A_ub.x <= b_ub; A_eq.x=b_eq
#     x = [f1,..,fn|c2,...cn]
#     minimize sum f_i

#     NOTE these methods have moved around during refactoring,
#     no guarantee of correctness right now.
#     """

#     def __init__(
#         self, isa_sequence: List[MonodromyLPGate], target_gate: MonodromyLPGate
#     ):
#         super().__init__(isa_sequence, target_gate)
#         self.num_ineq = len_qlr * (self.n - 1)  # rows in A
#         self.num_params = len_gi * (2 * self.n - 1)  # free parameters, cols in x
#         self.g1_index = 0
#         self.c2_index = len_gi * self.n

#     def _construct_objective_matrix(self):
#         """Min sum_i (gi_a + g_ib + g_ic)."""
#         raise NotImplementedError("we need cost function to depend linearly on params")
#         c = np.zeros(self.num_params)
#         c[self.g1_index : self.c2_index] = 1  # minimize costs of the basis gates
#         return c

#     def _construct_inequality_matrices(self):
#         """Places (alpha,beta,gamma) cols into appropriate col blocks."""
#         A_ub = np.zeros((self.num_ineq, self.num_params))
#         b_ub = np.zeros((self.num_ineq, 1))

#         # construct L_i(ci, gi, ci+1)
#         # the only edge case is L_1 (g1, g2, c2)
#         # otherwise L_i (ci, gi, ci+1)
#         for i in range(self.n - 1):
#             temp_row_range = slice(len_qlr * i, len_qlr * (i + 1))

#             # c_i block
#             offset = self.c2_index + len_gi * (i - 1) if i > 0 else 0
#             A_ub[temp_row_range, range(offset, len_gi + offset)] += self._ci_block
#             # g_i block
#             offset = len_gi * (i + 1)
#             A_ub[temp_row_range, range(offset, len_gi + offset)] += self._gi_block
#             # c_{i+1} block
#             offset = self.c2_index + len_gi * i
#             A_ub[temp_row_range, range(offset, len_gi + offset)] += self._ciplus1_block

#             b_ub[temp_row_range] = self._bi

#         return A_ub, b_ub

#     def _construct_equality_matrices(self):
#         """Ax = b (m constraints for each gate, +1 for cn).

#         gi=Gi for i=1 to n; cn = T
#         """
#         num_rows = len_gi * (self.n + 1)  # +1 is for cn
#         A_eq = np.zeros((num_rows, self.num_params))
#         b_eq = np.zeros((num_rows, 1))

#         # g1, ... gm := G1, ... Gm
#         for i, gate in enumerate(self.isa_sequence):
#             for j, equality in enumerate(gate.equality_terms):
#                 a, b, c, d = equality
#                 gi_params_idx = slice(len_gi * i, len_gi * (i + 1))
#                 A_eq[len_gi * i + j, gi_params_idx] = a, b, c
#                 b_eq[len_gi * i + j] = d

#         # cn := T
#         for j, equality in enumerate(self.target_gate.equality_terms):
#             a, b, c, d = equality
#             A_eq[-len_gi + j, -len_gi:] = a, b, c
#             b_eq[-len_gi + j] = d

#         return A_eq, b_eq


# class ApproximateTargetConstraints(ConstantSubLPConstraints):
# # TODO
# # - approximation requires quadratic solver in order to define fidelity
# # requires modifying the loop in _best_decomposition() because
# # every gate sequence will be "feasible" since has no equality constraints
# # so instead of returning first answer we find we need to track scores
# # and return best score once we arrive at an exact solution...
# # I think the objective should just minimize the L1 norm (linear)
# # and then we can convert to fidelity in the decompoer's outer loop
# # is this fine? probably L1 vs L2 norm is ~about the same in the space anyway?

#     """Special case of ConstantSubLPConstraints for approximate objective.

#     NOTE this could directly subclass LPConstraints instead in order to be more general,
#     but I am starting with assuming I am working with special-case for a more simple version.

#     Attributes:
#         Inherits all attributes from LPConstraints.

#     Methods:
#         Overrides methods from LPConstraints to provide simplified constraints.
#     """

#     def __init__(
#         self, isa_sequence: List[MonodromyLPGate], target_gate: MonodromyLPGate
#     ):
#         """Initialize ConstantSubLPConstraints."""
#         super().__init__(isa_sequence, target_gate)

#     def _construct_objective_matrix(self):
#         """Generate an objective for maximizing approx.

#         fidelity to target.
#         """
#         # cn :~= T
#         for j, equality in enumerate(self.target_gate.equality_terms):
#             a, b, c, d = equality
#             A_eq[-len_gi + j, -len_gi:] = a, b, c
#             b_eq[-len_gi + j] = d
#         return np.zeros(self.num_params)

#     def _construct_equality_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
#         """Generate (zero) equality constraint matrices A_eq and b_eq."""
#         num_rows = len_gi
#         A_eq = np.zeros((num_rows, self.num_params))
#         b_eq = np.zeros((num_rows, 1))
#         return A_eq, b_eq

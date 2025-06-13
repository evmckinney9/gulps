# input is a list of gates
# first constraint should be that we need the list of basis gate to be true for some gate in the gate set
# N is the number of gates I need to select
# M is the number of gates that I am able to choose from
from typing import List

import numpy as np
from docplex.mp.dvar import Var
from docplex.mp.model import Model

from hetero_isas.monodromy_lp.isa import MonodromyLPGate, MonodromyLPISA
from hetero_isas.monodromy_lp.lp_constraints.lp_constraints import LPConstraints
from hetero_isas.monodromy_lp.lp_constraints.qlr import len_qlr, qlr_inequalities


class DocplexConstraints(LPConstraints):
    def __init__(self, gate_set: List[MonodromyLPGate], sequence_length=8):
        self.N = sequence_length
        self.M = len(gate_set)
        self._gate_set = gate_set

        self.model = self._create_model()
        self._set_model_params()
        self.warmstart = self.model.new_solution()
        for x in range(self.N):
            # self.warmstart.add_var_value(self.k_dict[(x, 0)], 1)
            self.warmstart.add_var_value(self.k_dict[(x, self.M - 1)], 1)
        self.model.add_mip_start(self.warmstart, complete_vars=True)

        # persistent reference to target cts
        self.target_cts = []
        self.r = self.model.binary_var()

    def _lp_solve(self, log_output):
        _ret = self.model.solve(log_output=log_output)
        if _ret:
            _ret.success = True
        return _ret

    def _set_model_params(self):
        # must have, help a ton
        self.model.parameters.threads = 1
        self.model.parameters.preprocessing.dual = 1
        self.model.parameters.mip.limits.cutpasses = -1
        self.model.parameters.mip.strategy.heuristicfreq = -1

        # could help a little bit
        self.model.parameters.mip.strategy.rinsheur = -1
        self.model.parameters.barrier.algorithm = 3
        self.model.parameters.emphasis.mip = 1
        self.model.parameters.mip.strategy.nodeselect = 2
        self.model.parameters.mip.strategy.branch = 1
        self.model.parameters.mip.cuts.nodecuts = 2
        self.model.parameters.preprocessing.symmetry = 0
        self.model.parameters.preprocessing.folding = 0

        # try these TODO
        # dont' seem to help (improvements in the noise limit)
        # self.model.parameters.mip.cuts.mircut = 2
        # self.model.parameters.mip.cuts.gomory = 2
        # self.model.parameters.preprocessing.aggregator = 1
        # self.model.parameters.mip.cuts.covers = 2
        # self.model.parameters.mip.cuts.cliques = 3
        # self.model.parameters.mip.cuts.disjunctive = 3
        # self.model.parameters.mip.cuts.liftproj = 3
        # self.model.parameters.mip.cuts.localimplied = 3
        # self.model.parameters.mip.strategy.probe = 3  # moderate probing, unclear
        # self.model.parameters.mip.polishafter.mipgap = 0
        # self.model.parameters.mip.limits.auxrootthreads = 0
        # self.model.parameters.preprocessing.reduce = 3
        # self.model.parameters.lpmethod = 0
        # self.model.parameters.mip.strategy.subalgorithm = 2
        # self.model.parameters.mip.tolerances.mipgap = 0.01  # XXX
        # self.model.parameters.mip.strategy.lbheur = 1

    def _extract_solution(self):
        # return list of basis gate definitions
        # followed by list of intermediate trajectory coordinates
        gate_sequence = []
        intermediate_coords = []
        for (i, j), v in self._last_result.get_value_dict(self.k_dict).items():
            if v == 1 and j != 0:  # j != self.M - 1:
                gate_sequence.append(self._gate_set[j])
                if i < self.N - 1:
                    intermediate_coords.append(
                        self._last_result.get_values(self.ci_nested[i])
                    )

        # intermediate_coords -> mono_points
        intermediate_coords.insert(0, self._gate_set[0].definition)
        intermediate_coords.insert(0, np.zeros(3))
        # intermediate_coords.append(self._target_def)

        return gate_sequence, intermediate_coords

    def _set_target(self, target_gate: MonodromyLPGate):
        # STEP 4. TARGET GATE CONSTRAINT
        # set cn = T OR rho(T)
        self._target_def = target_gate.definition
        rho_def = target_gate.rho_reflect().definition

        if self.target_cts:
            self.model.remove_constraints(self.target_cts)
            self.target_cts.clear()

        for c in range(3):
            self.target_cts.extend(
                [
                    self.model.add_indicator(
                        self.r,
                        self.ci_nested[-1][c] == self._target_def[c],
                        active_value=1,
                        name=f"t{c}",
                    ),
                    self.model.add_indicator(
                        self.r,
                        self.ci_nested[-1][c] == rho_def[c],
                        active_value=0,
                        name=f"r{c}",
                    ),
                ]
            )

    def _create_model(self):
        model = Model(ignore_names=True)
        # XXX HACK, no offset for last element that im assuming is identity
        isa_costs = np.array([g.cost + 0.001 for g in self._gate_set])
        isa_costs[-1] -= 0.001

        # STEP 1. CREATE ALL VARIABLES
        self.gi_nested = [
            model.continuous_var_list(3, lb=[0.0, 0.0, -1.0], ub=[1.0, 1.0, 0.0])
            for _ in range(self.N)
        ]
        self.ci_nested = [
            model.continuous_var_list(3, lb=-1.0, ub=1.0) for _ in range(self.N - 1)
        ]
        self.k_dict = model.binary_var_matrix(self.N, self.M)

        # STEP 2. INDICATORS
        self._indicator_variables(model)

        # STEP 3. QLR CONSTRAINTS
        self._qlr_constraints(model)

        # STEP 5. OBJECTIVE FUNCTION
        # finally, minimize objective over the basis gate costs
        model.minimize(
            model.sum(
                (
                    isa_costs.dot(
                        np.fromiter(
                            (self.k_dict[(n, m)] for m in range(self.M)),
                            dtype=Var,
                        )
                    )
                    for n in range(self.N)
                )
            )
        )

        return model

    def _indicator_variables(self, model):
        # STEP 2. DECISION VARIABLE CONSTRAINTS
        # if k_i_j then gi_nested[i] := gate_set[ j]
        model.add_indicators(
            [v for _ in range(3) for v in self.k_dict.values()],
            [
                (self.gi_nested[key[0]][c] == self._gate_set[key[1]].definition[c])
                for c in range(3)
                for key in self.k_dict.keys()
            ],
        )

        # then impose that sum of decision variables == 1 for each row
        for n in range(self.N):
            model.add_constraint(
                model.sum_vars_all_different(self.k_dict[(n, m)] for m in range(self.M))
                == 1
            )

        # constraint for unique gate ordering
        # e.g. most -> less expensive
        # strictly non-decreasing ISA selection
        # impose strongest constraint
        # for all i (0, N-1); for all k (0, M)
        # sum_1^k (x_ik - xi+1k) >= 0
        for n in range(self.N - 1):
            for m in range(self.M):
                # model.add_constraint(
                #     model.sum(self.k_dict[n, i] for i in range(m))
                #     <= model.sum(self.k_dict[n + 1, i] for i in range(m))
                # )
                model.add_constraint(
                    model.sum(
                        self.k_dict[n, i] - self.k_dict[n + 1, i] for i in range(m)
                    )
                    >= 0
                )

        # try putting constraints on the coordinates not the indicator variables.
        # for n in range(self.N - 1):
        #     model.add_constraints(
        #         [
        #             self.gi_nested[n][0] >= self.gi_nested[n + 1][0],
        #             self.gi_nested[n][1] >= self.gi_nested[n + 1][1],
        #             self.gi_nested[n][2] <= self.gi_nested[n + 1][2],
        #         ]
        #     )

        # NOTE SLOWER
        # alternative cumulative sum constraint
        # for n in range(self.N):
        #     model.add_constraint(
        #         model.sum_vars(self.gi_nested[i][0] for i in range(n, self.N))
        #         <= (self.N - n) * self.gi_nested[n][0]
        #     )
        #     model.add_constraint(
        #         model.sum_vars(self.gi_nested[i][1] for i in range(n, self.N))
        #         <= (self.N - n) * self.gi_nested[n][1]
        #     )
        #     model.add_constraint(
        #         model.sum_vars(self.gi_nested[i][2] for i in range(n, self.N))
        #         >= (self.N - n) * self.gi_nested[n][2]
        #     )

        # NOTE DOESN'T HELP - a little slower
        # cumulative sum constraint
        # the sum of the first n elements is >= n* the nth element
        # for n in range(self.N):
        #     model.add_constraint(
        #         model.sum_vars(self.gi_nested[i][0] for i in range(n))
        #         >= n * self.gi_nested[n][0]
        #     )
        #     model.add_constraint(
        #         model.sum_vars(self.gi_nested[i][1] for i in range(n))
        #         >= n * self.gi_nested[n][1]
        #     )
        #     model.add_constraint(
        #         model.sum_vars(self.gi_nested[i][2] for i in range(n))
        #         <= n * self.gi_nested[n][2]
        #     )

        # SLOW
        # bounding difference between consecutive elements
        # assumes total decrease is at most the first element
        # for n in range(self.N - 1):
        #     model.add_constraint(
        #         self.gi_nested[n][0] - self.gi_nested[n + 1][0] <= self.gi_nested[0][0]
        #     )
        #     model.add_constraint(
        #         self.gi_nested[n][1] - self.gi_nested[n + 1][1] <= self.gi_nested[0][1]
        #     )
        #     model.add_constraint(
        #         self.gi_nested[n][2] - self.gi_nested[n + 1][2] >= self.gi_nested[0][2]
        #     )

        # NOTE doesn't appear to help - a tiny bit slower
        # global sum,
        # the total sum is at least N times the smallest (last) element
        # model.add_constraint(
        #     model.sum_vars(self.gi_nested[i][0] for i in range(self.N))
        #     >= self.N * self.gi_nested[self.N - 1][0]
        # )
        # model.add_constraint(
        #     model.sum_vars(self.gi_nested[i][1] for i in range(self.N))
        #     >= self.N * self.gi_nested[self.N - 1][1]
        # )
        # model.add_constraint(
        #     model.sum_vars(self.gi_nested[i][2] for i in range(self.N))
        #     <= self.N * self.gi_nested[self.N - 1][2]
        # )

        # monotonicity
        # XXX wrong - this is only true if sequence is strictly decreasing
        # the rate of decrease is non-increasing
        # true for fractional basis gates converging closer to Identity
        # for n in range(self.N - 2):
        #     model.add_constraint(
        #         self.gi_nested[n][0] - self.gi_nested[n + 1][0]
        #         >= self.gi_nested[n + 1][0] - self.gi_nested[n + 2][0]
        #     )

        ####################################################################
        ####################################################################
        # the ones below are probably logically incorrect
        # NOTE the rest seems to be slower and/or redundant
        # impose sum of decision variables == 1 for each row
        # and get priority ordering using SOS
        # for n in range(self.N):
        #     dvar = list(self.k_dict[(n, m)] for m in range(self.M))
        #     model.add_sos(dvar, sos_arg=1, weights=range(self.M))
        # # helper cumulative variables
        # y = {
        #     (n, m): model.continuous_var(lb=0, ub=1)
        #     for n in range(self.N)
        #     for m in range(self.M)
        # }
        # # # cumulative constraints
        # for n in range(self.N):
        #     model.add_constraint(y[n, 0] == self.k_dict[n, 0])
        #     for m in range(1, self.M):
        #         model.add_constraint(y[n, m] == y[n, m - 1] + self.k_dict[n, m])
        #     model.add_constraint(y[n, self.M - 1] == 1)
        # # # global ordering constraint
        # for n1 in range(self.N):
        #     for n2 in range(n1 + 1, self.N):
        #         for m in range(self.M):
        #             model.add_constraint(y[n1, m] >= y[n2, m])

        # I thought this would work but it gives me infeasible solutions
        # for n in range(self.N - 1):
        #     model.add_constraints(
        #         [
        #             self.N - (n + 1) * self.gi_nested[n][0]
        #             >= model.sum_vars(
        #                 self.gi_nested[i][0] for i in range(n + 1, self.N)
        #             ),
        #             self.N - (n + 1) * self.gi_nested[n][1]
        #             >= model.sum_vars(
        #                 self.gi_nested[i][1] for i in range(n + 1, self.N)
        #             ),
        #             self.N - (n + 1) * self.gi_nested[n][2]
        #             <= model.sum_vars(
        #                 self.gi_nested[i][2] for i in range(n + 1, self.N)
        #             ),
        #         ]
        #     )

        # NOTE i am not sure I did this one correctly
        # implied bound constraints
        # for n in range(self.N):
        #     for m in range(self.M):
        #         model.add_constraint(
        #             m * self.k_dict[n, m]
        #             <= model.sum(i * self.k_dict[n, i] for i in range(m, self.M))
        #         )

    def _qlr_constraints(self, model):
        # first edge case L(g2, g1, c2)
        # note offset index, ci_nested = [c2,..., cn]
        # but gi_nested = [g1, ..., gn]
        model.add_constraints(
            (
                (
                    model.sum(
                        (
                            model.scal_prod(self.gi_nested[1], self._ci_block[i]),
                            model.scal_prod(self.gi_nested[0], self._gi_block[i]),
                            model.scal_prod(self.ci_nested[0], self._ciplus1_block[i]),
                        )
                    )
                    <= self._bi[i]
                )
                for i in range(len_qlr)
            )
        )

        # begin on second step
        # L(c2, g3, c3)
        for j in range(1, self.N - 1):
            model.add_constraints(
                (
                    (
                        model.sum(
                            (
                                model.scal_prod(
                                    self.ci_nested[j - 1], self._ci_block[i]
                                ),
                                model.scal_prod(
                                    self.gi_nested[j + 1], self._gi_block[i]
                                ),
                                model.scal_prod(
                                    self.ci_nested[j], self._ciplus1_block[i]
                                ),
                            )
                        )
                        <= self._bi[i]
                    )
                    for i in range(len_qlr)
                )
            )

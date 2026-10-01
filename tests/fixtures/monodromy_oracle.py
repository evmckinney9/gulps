"""The monodromy package as an independent oracle for gulps' polytopes and Haar integral.

monodromy enumerates vertices with lrs and integrates rationally; gulps replaces
both with closed forms. CI installs monodromy and lrslib.
"""


def region_vertices(gates):
    """Reachable vertices from monodromy's QLR rules and lrs enumeration."""
    from fractions import Fraction

    from monodromy.coordinates import (
        monodromy_to_positive_canonical_polytope,
        unitary_to_monodromy_coordinate,
    )
    from monodromy.coverage import deduce_qlr_consequences
    from monodromy.static.examples import (
        everything_polytope,
        exactly,
        identity_polytope,
    )
    from qiskit.quantum_info import Operator

    region = identity_polytope
    for gate in gates:
        coordinate = unitary_to_monodromy_coordinate(Operator(gate).data)[:-1]
        point = exactly(*(Fraction(x).limit_denominator(10_000) for x in coordinate))
        region = deduce_qlr_consequences(
            target="c",
            a_polytope=region,
            b_polytope=point,
            c_polytope=everything_polytope,
        ).reduce()
    canonical = monodromy_to_positive_canonical_polytope(region).reduce()
    return sorted(
        {tuple(round(float(x), 6) for x in v) for vs in canonical.vertices for v in vs}
    )


def expected_cost(isa):
    """Expected Haar cost through monodromy's own coverage pipeline."""
    from fractions import Fraction

    from monodromy.coordinates import unitary_to_monodromy_coordinate
    from monodromy.coverage import (
        CircuitPolytope,
        build_coverage_set,
        deduce_qlr_consequences,
    )
    from monodromy.haar import expected_cost
    from monodromy.static.examples import (
        everything_polytope,
        exactly,
        identity_polytope,
    )

    gate_cost = {g.name: cost for g, cost in zip(isa.gates, isa.costs, strict=True)}
    from qiskit.quantum_info import Operator

    operations = []
    for gate in isa.gates:
        coordinate = unitary_to_monodromy_coordinate(Operator(gate).data)[:-1]
        target = exactly(*(Fraction(x).limit_denominator(10_000) for x in coordinate))
        polytope = deduce_qlr_consequences(
            target="c",
            a_polytope=identity_polytope,
            b_polytope=target,
            c_polytope=everything_polytope,
        )
        operations.append(
            CircuitPolytope(
                operations=[gate.name],
                cost=gate_cost[gate.name] + isa.local_layer_cost,
                convex_subpolytopes=polytope.convex_subpolytopes,
            )
        )
    coverage = build_coverage_set(operations)
    for piece in coverage:
        gates = [g for g in isa.gates for name in piece.operations if name == g.name]
        piece.cost = (
            sum(gate_cost[g.name] for g in gates)
            + (len(gates) + 1) * isa.local_layer_cost
            if gates
            else 0.0
        )
    return expected_cost(coverage)

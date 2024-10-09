refactor:


1. first some obvious speedups
a. isa don't recompute ordered sequence, i found too fast to matter
b. don't recompute constraints (memoize?)
2. then add rho(target) with binary decision to half calls
3. then continue developing the mixed cases once we see how slow 1 binary var is.

############################################################
numerical_decomposer

- never optimize exterior gates, use KAK
- minimize use of Operator and QuantumCircuit
- special case between scipy.root and .minimize depending on if solution is guaranteed
- custom result object to store fidelity, parameter vector, num restarts, etc
- option to return with exterior gates or not (for example, if sequential solve we don't care)
- but if we are doing parallel solve then we want exterior gates at each intermediate to turn into CAN

then refactor: singleq_numerics:
- updated patterns for sequential and parallel iteration of parameters
- don't recompute prefactor unitaries using Operator(qc)
- use the method from numerical_decomposer with .root() finding


I am refactoring this such that we only have class but with multiple configurable options.
First, I should always find the exterior 1Q gates using KAK instead of numerics.

Second, I should use more efficient vectorization of ansatz. Using Operator(), QuantumCircuit()
are needlessly slow and inefficient with recomputing unitaries.

Third, have option for when an exact solution is known to exist (e..g with LP) this should use
scipy.root() rather than scipy.minimize()

Fourth, introduce a custom result class. Current version has some mess about modifying the resulting ansatz in place.
It would be easier if we returned an object that gave back the fidelity, vector solution and other metadata like restart attempts.
This is important because this class shouldn't be responsible for turning results back into a cirucit - instead this could be a method of the result class, e.g. to_quantum_circuit() -> QuantumCircuit.

Fifth, we want to use jax to compute differentials (forward/backwards Jacobian), and see if that helps the solver, which
by default will numerically estimate the functional's derivative.

Sixth, update solver's method to use, for example, Levenberg-Marquardt with root finding which
are themselves wrappers over (MINPACK).

Seven, MonodromyLPAnsatz is needlessly complicated. Get rid of iterable and just use a list.

LATER, use this with wrapper in singleq_numeris. Sequential should use the previous iteration's final
result as the precomputed part (recall this is in place of calling Operator) we can multiply in the new gates.
Parallel can avoid having to complicated stitching together at the end if we set each of its targets to CAN of the local gate.

__________
monodromy linear programming:
0. more unit tests case to verify accuracy in decompositions
 - we can compare XX to qiskit, we could compare iSWAP to Ali's PR, for arbitray we could test against monodromy polytope contains


1. approximate objective function, how can we get ~L1 norm using linear solver
    - later can try something like polytope distance using a quadratic solver options?
    - XXX big problem that I don't know best way to implement yet, is once we have solved for a decomposition sequence I need to take the solutions to the gates we found and convert them back to unitaries. except that our ISA is only defined in terms of invariants. so we don't have a unique mapping back to a specific unitary. presumably this will depend on the gate's realization - but this is a part of qiskit.Target that doesn't really exist yet - specificially for continuous-family basis gates.
    - to clarify, how do I tell the solved ISA how to differentiate between a CX-family and CZ-family? not that complicated but I can't think of a good way to write this in my transpiler code. For now - I can just assume I have some class to handle this???


2. mixed-integer linear programming. we should setup solver options to use cplex and reduce LP to a single call
where we describe x_vec using indicator variables
3. integrate into larger decomposition flow. first step would be with makhlin functionals to fit 1Q parameters
- later can put tgoether into a larger transpilation pass that gets informatoin from the qiskit.backend.Target
___
pulse-constrained 1Q polytopes:

open questions, as a function of single-qubit gate costs:
- integrate point clouds with targeted sampling. this is all done just hasn't been used to reproduce the large tables with volume calc
- how does it change our analysis of fractional gates? not all interleaving layers cost the same.
- currently I can delete layers I know should have cost 0 by larger basis gate. but what about u3,u2,u1...
- when does creating a BGate outweigh creating a sqiswap with parallel-drive. depends on pulse costs I assume because
bgate always deletes a full layer but is itself more costly.
- depends on hamiltonian. in snail I know iswap=cnot=bgate. but at ibm cnot=sqiswap!=bgate

use theorem 55 which an anyltical form of xy_alpha^2 volume
I wonder if I can analytically write the alcove sequences for xy_alpha^N
then make a better plot for expected count vs alpha
then use that to get expected pulse cost vs alpha which should be extended to include single-qubit gate costs
however not every layer is the same...(u3,u2,u1). before we can describe those volumes an easier way
is just limit to u3 and I by stating that alpha/2 can trivially be alpha in a hetero ISA.

the answer should be somewhere between sqiswap and continuous. sqiswap doesn't subdivide itself at all so can be improved but
continuous is a lower bound because 2Q gate is optimal to use least amount as possible, but doesn't contribute any 1Q costs.

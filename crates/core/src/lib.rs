//! Two-qubit synthesis over canonical gates. `class` holds the coordinates of
//! gate classes; `kak` computes a unitary's class and frames; `reachability`
//! and `search` select sentences; `compile` binds physical gates, realizes
//! sentences, and lists the resulting circuits. Python and Qiskit interop live
//! in gulps-pyext.
//!
//! # Tolerances
//!
//! Computational reuse keys a class by its exact coordinates, with signed zero
//! normalized (`Mono::key`). The Python class's equality uses `CLASS_TOL`
//! cells instead, and two classes in one cell can lie on opposite sides of a
//! reachability boundary and require different gate sequences. Do not key
//! cost aggregation, synthesis reuse, or native-gate pruning by those cells.
//!
//! Do not add a tolerance elsewhere. Derive it from one of these constants.
//! Never round a coordinate to a grid. Rounding each coordinate separately
//! breaks the linear relations between gates, and their products then miss
//! the strata the solver needs within `can_sandwich::SPECTRAL_TOLERANCE`.
//!
//! `make bench` records each decomposition workload's worst entrywise error
//! as `max_error` in the saved report's `extra_info`, and `make test` asserts
//! it below `RECONSTRUCT_ATOL` in `tests/_common.py`. Re-measure there before
//! changing any constant. The solver's precision floor is documented in the
//! [can_sandwich](https://github.com/evmckinney9/can_sandwich) repository.

use nalgebra::{Complex, Matrix2, Matrix4};

pub mod class;
mod compile;
mod kak;
mod reachability;
mod search;

pub use compile::{Binding, CompileError, CompiledTarget, Decomposer, Gate, Op, compile_all};
pub use kak::{Unitary, canonical_matrix};
pub use reachability::{FORMS, Region, covers, project, sentence_region};

/// Complex scalar.
pub type C64 = Complex<f64>;
/// One-qubit matrix.
pub type Mat2 = Matrix2<C64>;
/// Two-qubit matrix.
pub type Mat4 = Matrix4<C64>;

/// Coordinate-cell width for public class equality and hashing, also used for
/// chart folding and chamber validation.
pub const CLASS_TOL: f64 = 1e-12;
/// Reachability slack in phase turns: the solver's root-error ceiling
/// `can_sandwich::SPECTRAL_TOLERANCE` divided by 2π.
/// A small single-phase error `d` moves `exp(2πi m)` by approximately `2πd`.
/// Subset sums and repeated products accumulate errors, so this scale is a
/// numerical policy.
pub const MEMBERSHIP_TOL: f64 = can_sandwich::SPECTRAL_TOLERANCE / std::f64::consts::TAU;
/// Entrywise error accepted in `U†U − I` for an input matrix.
pub const INPUT_ATOL: f64 = 1e-8;
/// Working-precision error accepted when a factorization is recomposed: a KAK
/// frame and a local rotation split into single-qubit gates (both Frobenius),
/// and the entrywise input nonunitarity below which no closest-unitary repair
/// is needed.
pub(crate) const RECONSTRUCT_TOL: f64 = 1e-13;
/// Frobenius distance from the identity below which a compiled circuit omits
/// a single-qubit gate. An emitted local gate is the product of at most three
/// recovered factors, each accurate to about 1e-15.
pub(crate) const LOCAL_IDENTITY_TOL: f64 = 1e-14;
/// Serial work below which waking Rayon's workers costs more than it saves;
/// measured on 6 cores / 12 threads under WSL.
const PAR_WORK_NS: usize = 300_000;

/// Map a batch in input order, returning the first error in input order with
/// the index of its item. Runs in parallel once the batch holds `PAR_WORK_NS`
/// of work at `item_ns` per item, unless Qiskit has disabled threads.
pub fn map_batch<T: Sync, U: Send, E: Send>(
    items: &[T],
    item_ns: usize,
    f: impl Fn(&T) -> Result<U, E> + Send + Sync,
) -> Result<Vec<U>, (usize, E)> {
    let indexed = |(i, item)| f(item).map_err(|err| (i, err));
    if items.len() * item_ns >= PAR_WORK_NS && qiskit_numerics::getenv_use_multiple_threads() {
        use rayon::prelude::*;
        let results: Vec<_> = items.par_iter().enumerate().map(indexed).collect();
        results.into_iter().collect()
    } else {
        items.iter().enumerate().map(indexed).collect()
    }
}

#[cfg(test)]
pub(crate) const C0: C64 = C64::new(0.0, 0.0);

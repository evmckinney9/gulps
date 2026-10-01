//! Two-qubit synthesis over canonical gates. `class` holds the coordinates of
//! gate classes; `kak` computes a unitary's class and frames; `reachability`
//! and `search` select sentences; `compile` binds physical gates, realizes
//! sentences, and lists the resulting circuits. Python and Qiskit interop live
//! in gulps-pyext.
//!
//! # Tolerances
//!
//! The constants below are the crate's tolerances. `INPUT_ATOL`, the
//! unitarity tolerance at the Python input boundary, is in gulps-pyext.
//!
//! Computational reuse requires exact coordinate keys, with signed zero
//! normalized. Two classes in the same public equality cell can lie on
//! opposite sides of a reachability boundary and require different gate
//! sequences. Do not use the public equality key for cost aggregation,
//! synthesis reuse, or native-gate pruning.
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
pub use kak::{canonical_matrix, monodromies};
pub use reachability::{FORMS, Region, covers, project, sentence_region};

/// Complex scalar.
pub type C64 = Complex<f64>;
/// One-qubit matrix.
pub type Mat2 = Matrix2<C64>;
/// Two-qubit matrix.
pub type Mat4 = Matrix4<C64>;

/// Coordinate-cell width for public class equality and hashing, also used for
/// chart folding and chamber validation. Computational reuse requires exact
/// coordinates: nearby classes can lie on opposite reachability boundaries.
/// It never rounds a stored coordinate.
pub const CLASS_TOL: f64 = 1e-12;
/// Reachability slack in phase turns: the solver's root-error ceiling
/// `can_sandwich::SPECTRAL_TOLERANCE` divided by 2π.
/// A small single-phase error `d` moves `exp(2πi m)` by approximately `2πd`.
/// Subset sums and repeated products accumulate errors, so this scale is a
/// numerical policy, not a proof of infeasibility beyond every facet's slack.
pub const MEMBERSHIP_TOL: f64 = can_sandwich::SPECTRAL_TOLERANCE / std::f64::consts::TAU;
/// Working-precision error accepted when a factorization is recomposed: a KAK
/// frame and a local rotation split into single-qubit gates (both Frobenius),
/// and the entrywise input nonunitarity below which no closest-unitary repair
/// is needed.
pub(crate) const RECONSTRUCT_TOL: f64 = 1e-13;
/// Frobenius distance from the identity below which a compiled circuit omits
/// a single-qubit gate. Recovered local gates are accurate to about 1e-15.
pub(crate) const LOCAL_IDENTITY_TOL: f64 = 1e-14;
/// Fewest items in one parallel task. Waking a worker costs about as much as
/// a few dozen KAKs, so smaller tasks lose time; measured on 6 cores / 12
/// threads under WSL, splitting 128 KAKs into tasks this size halves the
/// overhead of Rayon's default split.
const PAR_TASK: usize = 32;

/// Map a batch in input order, stopping at an error. A batch that fills at
/// least two tasks of `PAR_TASK` items uses Rayon, unless Qiskit has disabled
/// threads inside its parallel workers.
pub fn map_batch<T: Sync, U: Send, E: Send>(
    items: &[T],
    f: impl Fn(&T) -> Result<U, E> + Send + Sync,
) -> Result<Vec<U>, E> {
    if items.len() >= 2 * PAR_TASK && qiskit_numerics::getenv_use_multiple_threads() {
        use rayon::prelude::*;
        items.par_iter().with_min_len(PAR_TASK).map(f).collect()
    } else {
        items.iter().map(f).collect()
    }
}

#[cfg(test)]
pub(crate) const C0: C64 = C64::new(0.0, 0.0);

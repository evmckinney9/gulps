//! Local-equivalence classes and their coordinate charts. The class of a
//! unitary is computed by the KAK in `kak`.

use std::f64::consts::PI;

use crate::CLASS_TOL;

/// Monodromy to Weyl coordinates.
pub fn weyl_from_monodromy(m: &[f64; 3]) -> [f64; 3] {
    [m[0] + m[1], m[0] + m[2], m[1] + m[2]]
}

/// Weyl to monodromy coordinates.
pub fn monodromy_from_weyl(c1: f64, c2: f64, c3: f64) -> [f64; 3] {
    [
        0.5 * (c1 + c2 - c3),
        0.5 * (c1 - c2 + c3),
        0.5 * (-c1 + c2 + c3),
    ]
}

/// Choose between the two chart representatives, retaining the frame reflection.
pub fn fold_weyl([c1, c2, c3]: [f64; 3]) -> ([f64; 3], bool) {
    // On c3=0, (c1,c2,0) and (1-c1,c2,0) are the same class, so the fold
    // decides by c1 within the class tolerance of that face. Coordinates are
    // never rounded: see `Mono`.
    let boundary = c3.abs() <= CLASS_TOL;
    let reflected = if boundary { c1 > 0.5 } else { c3 < 0.0 };
    let (c1, c3) = if reflected { (1.0 - c1, -c3) } else { (c1, c3) };
    ([c1, c2, c3], reflected)
}

/// Bits of `value` for exact comparison, with +0.0 and -0.0 equal.
pub(crate) fn exact_bits(value: f64) -> u64 {
    if value == 0.0 { 0 } else { value.to_bits() }
}

/// Monodromy coordinates: the eigenphase triple of `U·Ũ`, used by selection,
/// membership, and trajectories. Values are kept unrounded: rounding each
/// coordinate separately breaks the linear relations between gates, and their
/// products then miss the strata the realization solver needs within
/// `can_sandwich::SPECTRAL_TOLERANCE`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Mono(pub [f64; 3]);

impl Mono {
    /// Makhlin invariants `[Re(G1), Im(G1), Re(G2)]` of this class.
    pub fn makhlin(self) -> [f64; 3] {
        let weyl = weyl_from_monodromy(&self.0).map(|c| PI * c / 2.0);
        qiskit_numerics::two_qubit_decompose::common::local_equivalence(weyl.as_slice().into())
            .expect("local_equivalence has no error path")
    }

    /// Class key: the `CLASS_TOL` cell of each coordinate, for public equality
    /// and hashing only.
    #[allow(clippy::cast_possible_truncation)] // |c| <= 2, so the scaled key is far inside i64
    pub fn key(self) -> [i64; 3] {
        self.0.map(|c| (c / CLASS_TOL).round_ties_even() as i64)
    }

    /// Exact coordinate key for computational reuse. Approximate class equality
    /// can straddle a reachability boundary, so it cannot identify computations.
    pub(crate) fn exact_key(self) -> [u64; 3] {
        self.0.map(exact_bits)
    }

    /// The other unfolded-chart representative of this local-equivalence orbit.
    pub(crate) fn rho(self) -> Self {
        let [a, b, c] = self.0;
        Self([c + 0.5, -(a + b + c) + 0.5, a - 0.5])
    }
}

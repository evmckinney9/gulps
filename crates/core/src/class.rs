//! Local-equivalence classes and their coordinate charts. The class of a
//! unitary is computed by the KAK in `kak`.

use std::f64::consts::PI;

use crate::CLASS_TOL;

/// Monodromy to Weyl coordinates.
pub fn weyl_from_monodromy(m: &[f64; 3]) -> [f64; 3] {
    [m[0] + m[1], m[0] + m[2], m[1] + m[2]]
}

/// Weyl to monodromy coordinates.
pub fn monodromy_from_weyl(&[c1, c2, c3]: &[f64; 3]) -> [f64; 3] {
    [
        0.5 * (c1 + c2 - c3),
        0.5 * (c1 - c2 + c3),
        0.5 * (-c1 + c2 + c3),
    ]
}

/// The monodromy coordinates of eigenphases (in turns) of a determinant-one
/// matrix, and the eigen indices of `y`, `x`, `w`, `z`, the magic-basis order.
/// Each phase is lifted by a whole turn so the lifts sum to zero and span at
/// most one turn; sorted, they are `x >= y >= z >= w = -x - y - z`.
pub(crate) fn monodromy(turns: &[f64; 4]) -> ([f64; 3], [usize; 4]) {
    let mut lifted: [(f64, usize); 4] = std::array::from_fn(|k| (turns[k], k));
    let sort = |l: &mut [(f64, usize); 4]| l.sort_by(|a, b| b.0.total_cmp(&a.0));
    sort(&mut lifted);
    // The phases multiply to one, so their sum is a whole number of turns.
    let mut excess = lifted.iter().map(|&(t, _)| t).sum::<f64>().round();
    while excess > 0.0 {
        lifted[0].0 -= 1.0;
        excess -= 1.0;
        sort(&mut lifted);
    }
    while excess < 0.0 {
        lifted[3].0 += 1.0;
        excess += 1.0;
        sort(&mut lifted);
    }
    let [(x, xi), (y, yi), (z, zi), (_, wi)] = lifted;
    ([x, y, z], [yi, xi, wi, zi])
}

/// Bits of `value` for exact comparison, with +0.0 and -0.0 equal.
pub(crate) fn exact_bits(value: f64) -> u64 {
    if value == 0.0 { 0 } else { value.to_bits() }
}

/// Monodromy coordinates: the eigenphase triple of `U·Ũ`. Values are kept
/// unrounded: rounding each coordinate separately breaks the linear relations
/// between gates, and their products then miss the strata the realization
/// solver needs within `can_sandwich::SPECTRAL_TOLERANCE`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Mono(pub [f64; 3]);

impl Mono {
    /// Makhlin invariants `[Re(G1), Im(G1), Re(G2)]` of this class.
    pub fn makhlin(self) -> [f64; 3] {
        let weyl = weyl_from_monodromy(&self.0).map(|c| PI * c / 2.0);
        qiskit_numerics::two_qubit_decompose::common::local_equivalence(weyl.as_slice().into())
            .expect("local_equivalence has no error path")
    }

    /// Exact coordinate key for computational reuse. Approximate class equality
    /// can straddle a reachability boundary, so it cannot identify computations.
    pub(crate) fn key(self) -> [u64; 3] {
        self.0.map(exact_bits)
    }

    /// The class of sorted monodromy coordinates, folded into the chamber by
    /// choosing between the two chart representatives, and whether the fold
    /// reflected them.
    pub(crate) fn folded(m: &[f64; 3]) -> (Self, bool) {
        let [c1, c2, c3] = weyl_from_monodromy(m);
        // On c3=0, (c1,c2,0) and (1-c1,c2,0) are the same class, so the fold
        // decides by c1 within the class tolerance of that face.
        let reflected = if c3.abs() <= CLASS_TOL {
            c1 > 0.5
        } else {
            c3 < 0.0
        };
        let weyl = if reflected {
            [1.0 - c1, c2, -c3]
        } else {
            [c1, c2, c3]
        };
        (Self(monodromy_from_weyl(&weyl)), reflected)
    }

    /// The class of Weyl coordinates of either chart representative, sorted
    /// and folded as the KAK sorts and folds a matrix's class.
    pub fn from_weyl(weyl: [f64; 3]) -> Self {
        let [a, b, c] = monodromy_from_weyl(&weyl);
        let (m, _) = monodromy(&[a, b, c, -a - b - c]);
        Self::folded(&m).0
    }

    /// The other unfolded-chart representative of this local-equivalence orbit.
    pub(crate) fn rho(self) -> Self {
        let [a, b, c] = self.0;
        Self([c + 0.5, -(a + b + c) + 0.5, a - 0.5])
    }
}

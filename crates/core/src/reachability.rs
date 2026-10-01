//! Which classes can a native-gate sequence reach, and how can it reach them?
//!
//! A class has ordered eigenphases x >= y >= z >= w, with w = -x-y-z.
//! Horn inequalities bound sums of one, two, or three of these phases.
//! `Region::product` extends a reachable region by one gate. `trajectory`
//! walks backward through those regions to choose the intermediate classes.
//!
//! A region stores inequality right-hand sides, not an attainable spectrum.
//! Before the codimension shift, every allowed QLR row (I,J -> K,d) proposes
//! `next[K] = max(next[K], prefix[I] + gate[J] - d)`.
//! The fixed-size kernels below evaluate that rule table without a table loop.
//! See the tutorial's [expanded update](https://evm9.dev/gulps/monodromy.html#reference-recurrence)
//! and the explicit degree-carrying table in the rank-two regression test.

use std::ops::Range;

use crate::MEMBERSHIP_TOL;
use crate::class::Mono;

/// A class's ordered phases x >= y >= z >= w, as bits of a phase subset.
const X: u8 = 8;
const Y: u8 = 4;
const Z: u8 = 2;
const W: u8 = 1;

/// The phase subsets whose sums a region bounds, in coefficient order: the
/// singles by rank, the pairs [zw, yz, xy, xw] that cycle under `product` and
/// the pairs [yw, xz] that cross, and the triples, the complements of x, y,
/// z, and w.
const SUBSETS: [u8; 14] = [
    W,
    Z,
    Y,
    X,
    Z | W,
    Y | Z,
    X | Y,
    X | W,
    Y | W,
    X | Z,
    Y | Z | W,
    X | Z | W,
    X | Y | W,
    X | Y | Z,
];

// Coefficient ranges of the groups of `SUBSETS`.
const SINGLES: Range<usize> = 0..4;
const CYCLE: Range<usize> = 4..8;
const CROSS: Range<usize> = 8..10;
const TRIPLES: Range<usize> = 10..14;

/// The Schubert codimension of a subset: the sum of its phases' ranks, with
/// w of rank zero, less the smallest such sum for its size.
const fn codimension(subset: u8) -> u32 {
    let size = subset.count_ones();
    let mut ranks = 0;
    let mut bit = 0;
    while bit < 4 {
        if subset & (1 << bit) != 0 {
            ranks += bit;
        }
        bit += 1;
    }
    ranks - size * (size - 1) / 2
}

/// One if `phase` is in `subset`, else zero.
const fn member(subset: u8, phase: u8) -> f64 {
    if subset & phase != 0 { 1.0 } else { 0.0 }
}

/// Each subset's sum as `normal . [x, y, z]`, with w = -x-y-z, and the shift
/// of its coefficient, a quarter of its codimension.
pub const FORMS: [([f64; 3], f64); 14] = {
    let mut forms = [([0.0; 3], 0.0); 14];
    let mut k = 0;
    while k < 14 {
        let s = SUBSETS[k];
        let w = member(s, W);
        forms[k] = (
            [member(s, X) - w, member(s, Y) - w, member(s, Z) - w],
            codimension(s) as f64 / 4.0,
        );
        k += 1;
    }
    forms
};

/// Lower bounds on the subset sums in [`FORMS`], shifted by codimension/4.
/// The shift makes composition a max-plus product: add bounds along a rule,
/// then take the strongest resulting bound. These are QLR coefficient bounds,
/// not separately optimized extrema: the alcove and other bounds can make an
/// entry redundant until [`Region::tighten`] raises it.
#[derive(Clone, Copy, Debug)]
pub struct Region(pub [f64; 14]);

/// `a . q` in x, y, z order.
fn dot(a: &[f64; 3], q: &[f64; 3]) -> f64 {
    a[0] * q[0] + a[1] * q[1] + a[2] * q[2]
}

/// Cyclic max-plus convolution: `out[k] = max_i(a[i] + b[k-i])`.
fn convolve4(a: &[f64], b: &[f64]) -> [f64; 4] {
    std::array::from_fn(|k| {
        (a[0] + b[k])
            .max(a[1] + b[(k + 3) % 4])
            .max(a[2] + b[(k + 2) % 4])
            .max(a[3] + b[(k + 1) % 4])
    })
}

fn convolve2(a: &[f64], b: &[f64]) -> [f64; 2] {
    [
        (a[0] + b[0]).max(a[1] + b[1]),
        (a[0] + b[1]).max(a[1] + b[0]),
    ]
}

impl Region {
    /// Compose QLR coefficient states. For fixed-gate sentences this computes
    /// the exact reach bounds; arbitrary polytope composition is not implied.
    ///
    /// The rank-two basis is (1,p,p²,p³,s,ps), with
    /// p⁴=1, p²s=s, and s²=p+p³ after the codimension shift.
    /// These relations list QLR output support: two outputs contribute a
    /// candidate bound to each slot, and competing candidates combine by max.
    pub fn product(&self, gate: &Self) -> Self {
        let (a, b) = (&self.0, &gate.0);
        // `fold` merges input slots that lead to the same output slot, keeping
        // the strongest candidate.
        let fold = |v: &[f64]| [v[0].max(v[2]), v[1].max(v[3])];
        let cycle = convolve4(&a[CYCLE], &b[CYCLE]);
        let cross = convolve2(&a[CROSS], &b[CROSS]);
        let left = convolve2(&fold(&a[CYCLE]), &b[CROSS]);
        let right = convolve2(&a[CROSS], &fold(&b[CYCLE]));
        let mut out = [0.0; 14];
        out[SINGLES].copy_from_slice(&convolve4(&a[SINGLES], &b[SINGLES]));
        // xy: max(zw+xy, yz+yz, xy+zw, xw+xw, yw+xz, xz+yw),
        // with each sum taking its left term from self, right from gate.
        out[CYCLE].copy_from_slice(&[
            cycle[0].max(cross[1]),
            cycle[1].max(cross[0]),
            cycle[2].max(cross[1]),
            cycle[3].max(cross[0]),
        ]);
        out[CROSS].copy_from_slice(&[left[0].max(right[0]), left[1].max(right[1])]);
        out[TRIPLES].copy_from_slice(&convolve4(&a[TRIPLES], &b[TRIPLES]));
        Self(out)
    }

    /// The largest bound. If A's bounds are all <= B's, then
    /// `A.max_bound() <= B.max_bound()`, so it screens componentwise dominance.
    pub fn max_bound(&self) -> f64 {
        self.0.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    /// The intersection: the stronger bound of each coefficient.
    pub fn intersect(&self, other: &Self) -> Self {
        Self(std::array::from_fn(|k| self.0[k].max(other.0[k])))
    }

    /// Certify containment by componentwise comparison of lower bounds:
    /// sufficient for any states, and exact up to rounding for tightened ones;
    /// see [`Region::tighten`]. `product` preserves this order, which makes it
    /// safe for subtree pruning. Unlike [`covers`], it applies no tolerance.
    /// The two states come from different sequences of floating-point
    /// products and certificate sums, so a bound can round either way in its
    /// last bit, and the test can report or miss a containment at that scale,
    /// far below `MEMBERSHIP_TOL`, the precision at which selection decides
    /// membership.
    pub fn contains_by_bounds(&self, other: &Self) -> bool {
        self.0.iter().zip(&other.0).all(|(a, b)| a <= b)
    }
}

/// The alcove as constraints `n . [x, y, z] >= r`: x >= y, y >= z, z >= w,
/// and x - w <= 1, with w = -x-y-z.
const ALCOVE: [([f64; 3], f64); 4] = [
    ([1.0, -1.0, 0.0], 0.0),
    ([0.0, 1.0, -1.0], 0.0),
    ([1.0, 1.0, 2.0], 0.0),
    ([-2.0, -1.0, -1.0], -1.0),
];

/// The normals of a region's constraints `n . q >= r`: its bounds, then the
/// alcove.
const NORMALS: [[f64; 3]; 18] = {
    let mut normals = [[0.0; 3]; 18];
    let mut j = 0;
    while j < normals.len() {
        normals[j] = if j < FORMS.len() {
            FORMS[j].0
        } else {
            ALCOVE[j - FORMS.len()].0
        };
        j += 1;
    }
    normals
};

/// The index of a zero right-hand side that pads certificates of fewer than
/// three terms. Padding with zero weight on a real constraint would
/// multiply an infinite bound by zero.
const PAD: usize = NORMALS.len();

/// At most three terms `(index, weight)` whose weighted normals add up to the
/// target normal, padded with [`PAD`], so that their weighted
/// right-hand sides add up to a lower bound on the sum. An index before
/// [`PAD`] is a constraint, with a positive weight; one from [`PINNED`] on is
/// a pinned coordinate, whose weight may have either sign.
type Certificate = [(usize, f64); 3];

/// The index of the first pinned coordinate, x, in the right-hand sides of
/// [`feasible_point`]; y and z follow it.
const PINNED: usize = PAD + 1;

/// Room for the certificates of one bound. Constant evaluation fails if a
/// bound has more.
const ROOM: usize = 32;

/// The certificates of one bound, the first `len` entries of `list`.
struct Certificates {
    list: [Certificate; ROOM],
    len: usize,
}

/// The certificates of each subset sum. The sum bounding itself is left out:
/// `tighten` starts from that coefficient.
static CERTIFICATES: [Certificates; 14] = {
    let mut all = [Certificates::EMPTY; 14];
    let mut k = 0;
    while k < all.len() {
        all[k] = Certificates::of(FORMS[k].0, 0).without_lone(k);
        k += 1;
    }
    all
};

/// The certificates of the minimum of `sign` times coordinate `axis`, with
/// the earlier coordinates pinned. A `sign` of -1 bounds the maximum.
const fn axis_bound(axis: usize, sign: f64) -> Certificates {
    let mut target = [0.0; 3];
    target[axis] = sign;
    Certificates::of(target, axis)
}

// The certificates of the range ends that `feasible_point` takes: the minimum
// of x, of -y, and of z.
const X_MIN: [Certificate; axis_bound(0, 1.0).len] = axis_bound(0, 1.0).exact();
const Y_MAX: [Certificate; axis_bound(1, -1.0).len] = axis_bound(1, -1.0).exact();
const Z_MIN: [Certificate; axis_bound(2, 1.0).len] = axis_bound(2, 1.0).exact();

/// Column `c` of a certificate: a constraint's normal, zero at [`PAD`], or
/// from [`PINNED`] on, the unit vector of a pinned coordinate.
const fn column(c: usize) -> [f64; 3] {
    let mut column = [0.0; 3];
    if c < PAD {
        column = NORMALS[c];
    } else if c > PAD {
        column[c - PINNED] = 1.0;
    }
    column
}

impl Certificates {
    const EMPTY: Self = Self {
        list: [[(PAD, 0.0); 3]; ROOM],
        len: 0,
    };

    /// The certificates of `target . q`, one per support, with the first
    /// `pinned` coordinates of q fixed: from every triple of independent
    /// columns that includes each pinned coordinate and gives no constraint a
    /// negative weight. With k coordinates pinned, a basic dual optimum weights
    /// at most 3 - k constraints, so these triples reach it.
    const fn of(target: [f64; 3], pinned: usize) -> Self {
        let mut found = Self::EMPTY;
        let end = PINNED + pinned;
        let mut i = 0;
        while i < end {
            let mut j = i + 1;
            while j < end {
                let mut l = j + 1;
                while l < end {
                    found.add([i, j, l], target, pinned);
                    l += 1;
                }
                j += 1;
            }
            i += 1;
        }
        found
    }

    /// Add the certificate of `target` on the columns `triple`, if the columns
    /// are independent (a triple with the zero column [`PAD`] is not), include
    /// every pinned coordinate, and give constraints nonnegative weights. A
    /// support is listed once; supports compare by position, since `of`
    /// visits columns in increasing order and dropping zero weights keeps it.
    const fn add(&mut self, triple: [usize; 3], target: [f64; 3], pinned: usize) {
        let mut columns = [[0.0; 3]; 3];
        let mut pinned_columns = 0;
        let mut t = 0;
        while t < 3 {
            if triple[t] >= PINNED {
                pinned_columns += 1;
            }
            columns[t] = column(triple[t]);
            t += 1;
        }
        if pinned_columns != pinned {
            return;
        }
        let Some(weights) = cramer(columns, target) else {
            return;
        };
        let mut certificate = [(PAD, 0.0); 3];
        let mut len = 0;
        t = 0;
        while t < 3 {
            if triple[t] < PAD && weights[t] < 0.0 {
                return;
            }
            if weights[t] != 0.0 {
                certificate[len] = (triple[t], weights[t]);
                len += 1;
            }
            t += 1;
        }
        let mut k = 0;
        while k < self.len {
            let [(a, _), (b, _), (c, _)] = self.list[k];
            if a == certificate[0].0 && b == certificate[1].0 && c == certificate[2].0 {
                return;
            }
            k += 1;
        }
        self.list[self.len] = certificate;
        self.len += 1;
    }

    /// The same certificates without the one of constraint `k` alone.
    const fn without_lone(self, k: usize) -> Self {
        let mut kept = Self::EMPTY;
        let mut c = 0;
        while c < self.len {
            let [(first, _), (second, _), _] = self.list[c];
            if first != k || second != PAD {
                kept.list[kept.len] = self.list[c];
                kept.len += 1;
            }
            c += 1;
        }
        kept
    }

    /// The certificates as an array of length `N`, which must be `len`.
    const fn exact<const N: usize>(self) -> [Certificate; N] {
        assert!(N == self.len);
        let mut exact = [[(PAD, 0.0); 3]; N];
        let mut c = 0;
        while c < N {
            exact[c] = self.list[c];
            c += 1;
        }
        exact
    }
}

/// The lower bound that `certificate` gives from the right-hand sides `rhs`.
fn lower_bound(&[(i, a), (j, b), (l, c)]: &Certificate, rhs: &[f64]) -> f64 {
    a * rhs[i] + b * rhs[j] + c * rhs[l]
}

/// The strongest lower bound that `certificates` give from the right-hand
/// sides `rhs`, for the lists of [`feasible_point`]. Several running maxima,
/// rather than one, let the evaluations overlap. Inlined, a list of known
/// length unrolls, with its indices and weights folded in.
#[inline(always)]
fn strongest(certificates: &[Certificate], rhs: &[f64]) -> f64 {
    let mut lanes = [f64::NEG_INFINITY; 4];
    for chunk in certificates.chunks(lanes.len()) {
        for (lane, certificate) in lanes.iter_mut().zip(chunk) {
            *lane = lane.max(lower_bound(certificate, rhs));
        }
    }
    lanes.into_iter().fold(f64::NEG_INFINITY, f64::max)
}

/// The determinant of the matrix with columns `u`, `v`, `w`.
const fn det([u, v, w]: [[f64; 3]; 3]) -> f64 {
    u[0] * (v[1] * w[2] - v[2] * w[1])
        + u[1] * (v[2] * w[0] - v[0] * w[2])
        + u[2] * (v[0] * w[1] - v[1] * w[0])
}

/// The weights `w` with `w[0] columns[0] + w[1] columns[1] + w[2] columns[2]
/// = b`, if the columns are independent. The entries are integers, so the
/// determinant is an integer and `|det| < 1/2` means zero.
const fn cramer(columns: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let d = det(columns);
    if d.abs() < 0.5 {
        return None;
    }
    let mut weights = [0.0; 3];
    let mut i = 0;
    while i < 3 {
        let mut replaced = columns;
        replaced[i] = b;
        weights[i] = det(replaced) / d;
        i += 1;
    }
    Some(weights)
}

impl Region {
    /// The same region, with every coefficient raised to the minimum of its
    /// shifted phase sum over the region intersected with the alcove.
    ///
    /// Why this is safe for sentence composition: let `R` be the region of a
    /// fixed-gate prefix, `h` its QLR coefficient state and `s` any state with
    /// `s >= h` that every point of `R` satisfies, such as `h.tighten()`.
    /// Appending a gate is monotone, so `s.product(g)` describes a subset of
    /// `h.product(g)`. Since `s <= project(p)` for every `p` in `R`, it still
    /// contains the union of the two-factor Horn sets over `R`. By the
    /// multiple-factor theorem, `h.product(g)` is exactly that union, hence
    /// both states describe the same region. By induction every stored state
    /// `s_n` satisfies `s_n >= h_n` and describes `R_n`: `s_n.product(g) >=
    /// h_{n+1}` by monotonicity, describes `R_{n+1}` by the argument above,
    /// and tightening adds only inequalities valid on `R_{n+1}`. So tightening
    /// at every step never changes a sentence's region.
    ///
    /// Why it helps: with the subset-sum forms and the alcove fixed, tight
    /// states make [`Region::contains_by_bounds`] exact up to rounding instead
    /// of only sufficient.
    ///
    /// How: each certificate's weighted right-hand sides bound its sum from
    /// below, and by LP duality the largest of them is the sum's minimum. The
    /// result is tight only for a nonempty region; an empty one stays empty.
    /// A product of fixed gates is nonempty up to rounding.
    #[must_use]
    pub fn tighten(&self) -> Self {
        let rhs = self.constraint_rhs();
        Self(std::array::from_fn(|k| self.tight(&rhs, k)))
    }

    /// The right-hand sides `r` of the constraints `n . q >= r`, in the order
    /// of [`NORMALS`], then zero at [`PAD`].
    fn constraint_rhs(&self) -> [f64; PINNED] {
        std::array::from_fn(|j| {
            if j < FORMS.len() {
                self.0[j] + FORMS[j].1
            } else if j < PAD {
                ALCOVE[j - FORMS.len()].1
            } else {
                0.0
            }
        })
    }

    /// Coefficient `k` raised to the minimum of its shifted sum; see
    /// [`Region::tighten`].
    fn tight(&self, rhs: &[f64; PINNED], k: usize) -> f64 {
        let certificates = &CERTIFICATES[k];
        let bound = certificates.list[..certificates.len]
            .iter()
            .map(|certificate| lower_bound(certificate, rhs))
            .fold(f64::NEG_INFINITY, f64::max);
        self.0[k].max(bound - FORMS[k].1)
    }
}

impl Mono {
    /// Whether the empty sentence reaches this class.
    pub(crate) fn is_identity(self) -> bool {
        covers(&BASE_STATE, &project(&self.0))
    }

    /// Spectra that can precede this gate and reach `next`, with fixed lifts.
    pub(crate) fn preimage(self, next: &[f64; 3]) -> Region {
        let [x, y, z] = self.0;
        // Reverse a spectral product using the inverse's ordered phases
        // (-w, -z, -y, -x), where w = -x-y-z.
        project(next).product(&project(&[x + y + z, -z, -y]))
    }
}

/// Reachable bounds of a sentence.
pub fn sentence_region(monos: &[[f64; 3]]) -> Region {
    monos
        .iter()
        .fold(BASE_STATE, |state, mono| state.product(&project(mono)))
}

/// Multiplicative identity for the QLR recurrence (the empty sentence).
/// Only the subsets of codimension zero, w, zw, and yzw, have finite lower
/// bounds, all zero. Together with the alcove these force the identity
/// spectrum. This coefficient state differs from `project([0, 0, 0])`, which
/// supplies every subset bound.
pub const BASE_STATE: Region = {
    let mut bounds = [f64::NEG_INFINITY; 14];
    let mut k = 0;
    while k < 14 {
        if codimension(SUBSETS[k]) == 0 {
            bounds[k] = 0.0;
        }
        k += 1;
    }
    Region(bounds)
};

/// Shift the ordered eigenphase sums into the Horn coefficient basis.
pub fn project(c: &[f64; 3]) -> Region {
    Region(FORMS.map(|(normal, shift)| dot(&normal, c) - shift))
}

/// Whether `point` satisfies every bound of `region` within `MEMBERSHIP_TOL`.
pub fn covers(region: &Region, point: &Region) -> bool {
    (region.0.iter().zip(&point.0)).all(|(h, p)| *h <= p + MEMBERSHIP_TOL)
}

/// The classes reached after the second through the next-to-last gate of a
/// sentence that the search found to reach `endpoint`, recovered backward:
/// each is the [`feasible_point`] of its prefix region intersected with the
/// next gate's preimage of the class after it. Empty for two gates; `gates`
/// must not be empty.
pub fn trajectory(gates: &[Mono], endpoint: Mono) -> Option<Vec<Mono>> {
    let n = gates.len();
    let mut prefix = BASE_STATE;
    let mut prefixes = vec![prefix];
    for gate in &gates[..n - 1] {
        prefix = prefix.product(&project(&gate.0));
        prefixes.push(prefix);
    }
    let mut interior = vec![Mono([0.0; 3]); n.saturating_sub(2)];
    let mut next = endpoint.0;
    for i in (2..n).rev() {
        let point = feasible_point(&prefixes[i].intersect(&gates[i].preimage(&next)))?;
        interior[i - 2] = Mono(point);
        next = point;
    }
    Some(interior)
}

/// A point of `region`: x at the lower end of its range over the region, then
/// y at the upper end of its range with x pinned there, then z at the lower
/// end of its range with x and y pinned. The certificates give each range end
/// exactly, up to rounding. Pinning x low and y high tends to make bounds of
/// the gate preimage tight. When they are, the next depth-two problem has its
/// target on a boundary of the region its two inputs reach. The ends were
/// chosen from the eight combinations by median solver time on quantum-volume
/// and Haar workloads, not derived. `None` when the point fails the region's
/// bounds or the alcove in floating point, as it does when the region is
/// empty.
fn feasible_point(region: &Region) -> Option<[f64; 3]> {
    let mut rhs = [0.0; PINNED + 3];
    rhs[..PINNED].copy_from_slice(&region.constraint_rhs());
    rhs[PINNED] = strongest(&X_MIN, &rhs);
    rhs[PINNED + 1] = -strongest(&Y_MAX, &rhs);
    rhs[PINNED + 2] = strongest(&Z_MIN, &rhs);
    let point = [rhs[PINNED], rhs[PINNED + 1], rhs[PINNED + 2]];
    (point.iter().all(|v| v.is_finite())
        && covers(region, &project(&point))
        && ALCOVE
            .iter()
            .all(|(n, r)| dot(n, &point) >= r - MEMBERSHIP_TOL))
    .then_some(point)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)] // All finite values are exact multiples of 1/4.
    fn rank_two_product_matches_the_unshifted_qlr_table() {
        // Peterson--Crooks--Smith, Figure 14: Gr(2, 2), translated to phase
        // subsets. Test every ordered input pair, including absent outputs.
        // Unlike the implementation's cyclic factorization, this reference
        // retains each quantum degree explicitly.
        let [zw, yz, xy, xw, yw, xz] = [0, 1, 2, 3, 4, 5];
        let codim = [0.0, 2.0, 4.0, 2.0, 1.0, 3.0];
        let rules = [
            (zw, zw, zw, 0),
            (zw, yw, yw, 0),
            (zw, yz, yz, 0),
            (zw, xw, xw, 0),
            (zw, xz, xz, 0),
            (zw, xy, xy, 0),
            (yw, yw, yz, 0),
            (yw, yw, xw, 0),
            (yw, yz, xz, 0),
            (yw, xw, xz, 0),
            (yw, xz, xy, 0),
            (yw, xz, zw, 1),
            (yw, xy, yw, 1),
            (yz, yz, xy, 0),
            (yz, xw, zw, 1),
            (yz, xz, yw, 1),
            (yz, xy, xw, 1),
            (xw, xw, xy, 0),
            (xw, xz, yw, 1),
            (xw, xy, yz, 1),
            (xz, xz, xw, 1),
            (xz, xz, yz, 1),
            (xz, xy, xz, 1),
            (xy, xy, zw, 2),
        ];
        let basis = |index: usize| {
            let mut state = Region([f64::NEG_INFINITY; 14]);
            // The chosen label has unshifted bound zero; all others are absent.
            state.0[4 + index] = -codim[index] / 4.0;
            state
        };
        for left in 0..6 {
            for right in 0..6 {
                let mut expected = [f64::NEG_INFINITY; 6];
                for (i, j, out, degree) in rules {
                    if (i == left && j == right) || (j == left && i == right) {
                        expected[out] = -f64::from(degree) - codim[out] / 4.0;
                    }
                }
                let actual = basis(left).product(&basis(right));
                assert_eq!(actual.0[4..10], expected, "input labels {left}, {right}");
                assert!(
                    (actual.0[..4].iter())
                        .chain(&actual.0[10..])
                        .all(|x| *x == f64::NEG_INFINITY)
                );
            }
        }
    }

    #[test]
    fn two_swaps_reach_the_reflected_identity_lift() {
        // A(SWAP) = exp(i*pi/4) SWAP, so its square has Cartan double -I.
        // This needs positive-degree QLR rules, even though the physical circuit is local.
        let swap = project(&[0.25, 0.25, 0.25]);
        let twice = swap.product(&swap);
        let identity = Mono([0.0; 3]);
        let reflected = project(&identity.rho().0);
        assert!(twice.0.iter().eq(&reflected.0));
        assert!(!covers(&twice, &project(&identity.0)));
        assert!(covers(&twice, &reflected));
    }

    #[test]
    fn every_subset_direction_can_exclude_an_alcove_point() {
        // Two fixed input pairs suffice to show that no output direction can
        // be dropped universally. Each target satisfies the alcove and the
        // other 13 bounds, but fails the designated one. These rational
        // witnesses were checked independently against the unshifted QLR rules.
        // This is not a lower bound on the number of independent state parameters.
        let scaled = |v: [i32; 3], denominator: f64| v.map(|x| f64::from(x) / denominator);
        let regions = [
            sentence_region(&[scaled([51, 3, -13], 120.0), scaled([41, 37, -11], 172.0)]),
            sentence_region(&[scaled([49, 13, -15], 132.0), scaled([15, 11, 7], 68.0)]),
        ];
        // Coefficient order: w,z,y,x; zw,yz,xy,xw,yw,xz; yzw,xzw,xyw,xyz.
        let witnesses = [
            (0, 0, [1625, 1359, 551], 5160.0),
            (1, 0, [2591, 1359, -1975], 5160.0),
            (2, 0, [2313, -771, -771], 5160.0),
            (3, 0, [367, 367, -201], 5160.0),
            (4, 0, [2829, 1359, -1857], 5160.0),
            (5, 0, [3279, -653, -1313], 5160.0),
            (6, 0, [551, -9, -9], 5160.0),
            (7, 0, [1313, 1313, 671], 5160.0),
            (8, 1, [191, -4, -4], 374.0),
            (9, 1, [113, 113, -113], 561.0),
            (10, 0, [3351, -653, -889], 5160.0),
            (11, 0, [1975, 1975, -1857], 5160.0),
            (12, 0, [955, 955, 955], 5160.0),
            (13, 0, [1101, -367, -367], 5160.0),
        ];
        for (excluded, pair, numerator, denominator) in witnesses {
            let [x, y, z] = scaled(numerator, denominator);
            let w = -x - y - z;
            assert!(
                [x - y, y - z, z - w, 1.0 - x + w]
                    .iter()
                    .all(|v| *v >= -1e-12)
            );
            let point = project(&[x, y, z]);
            let failed: Vec<_> = (regions[pair].0.iter())
                .zip(&point.0)
                .enumerate()
                .filter_map(|(i, (bound, value))| (value < &(bound - 1e-12)).then_some(i))
                .collect();
            assert_eq!(failed, vec![excluded]);
        }
    }
}

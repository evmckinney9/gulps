//! The KAK decomposition in the magic basis: a class, and the local rotations
//! around its canonical gate.

use crate::class::{Mono, monodromy, weyl_from_monodromy};
use crate::compile::CompileError;
use crate::{C64, INPUT_ATOL, Mat2, Mat4, RECONSTRUCT_TOL};
use nalgebra::{Matrix4, SymmetricEigen, Vector4};
use ndarray::{ArrayView2, ShapeBuilder};
use qiskit_numerics::linalg::{closest_unitary_faer, ndarray_to_faer};
use std::f64::consts::{FRAC_PI_2, PI, TAU};

/// Single-qubit gates in Qiskit order: matrix = q1 ⊗ q0.
#[derive(Clone, Copy, Debug)]
pub struct LocalPair {
    pub q0: Mat2,
    pub q1: Mat2,
}

/// The product of two local gates, qubit by qubit, as the product of their
/// rotations in the magic basis. A product equal to the identity rotation can
/// be `(-I, -I)`.
impl std::ops::Mul for LocalPair {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            q0: self.q0 * rhs.q0,
            q1: self.q1 * rhs.q1,
        }
    }
}

/// The magic basis `Q`. Its columns order the canonical gate's phases as
/// `(m1, m0, -m0-m1-m2, m2)`, and local gates are real rotations in it.
pub fn magic() -> Mat4 {
    Mat4::from_fn(|r, c| qiskit_numerics::two_qubit_decompose::common::MAGIC[r][c])
}

/// The canonical gate `D(c1, c2, c3) = exp(iπ/2 (c1 XX + c2 YY + c3 ZZ))`,
/// with Weyl coordinates in `[0, 1]`.
pub fn canonical_matrix(c1: f64, c2: f64, c3: f64) -> Mat4 {
    let array = qiskit_numerics::two_qubit_decompose::common::ud(
        PI * c1 / 2.0,
        PI * c2 / 2.0,
        PI * c3 / 2.0,
    );
    Mat4::from_fn(|r, c| array[(r, c)])
}

/// The rotations around a class's canonical gate:
/// `Q† U Q = after · D(m) · before · exp(i phase)`.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    /// Local rotation after the central operation.
    pub after: Matrix4<f64>,
    /// Local rotation before the central operation.
    pub before: Matrix4<f64>,
    /// Global phase in radians.
    pub phase: f64,
}

/// Weights mixing the real and imaginary parts of the complex-symmetric
/// `Vᵀ V`. They commute, so a real combination with distinct eigenvalues
/// diagonalizes both. The first pair is Qiskit's; each later pair separates
/// eigenvalues that the ones before it merged in observed cases. The list is
/// not proven to separate every spectrum.
const MIXES: [(f64, f64); 4] = [
    (1.260_206_611_224_938_8, 0.223_178_490_467_220_27),
    (0.6, 0.8),
    (-0.28, 0.96),
    (0.96, -0.28),
];

/// A two-qubit unitary: a matrix, or the canonical gate `D(c)` of a class,
/// whose frame is the identity.
#[derive(Clone, Copy, Debug)]
#[allow(clippy::large_enum_variant)] // Inputs are mostly matrices; boxing would allocate each one.
pub enum Unitary {
    /// A matrix, checked unitary within `INPUT_ATOL` when decomposed.
    Matrix(Mat4),
    /// The canonical gate of a folded class.
    Canonical(Mono),
}

impl Unitary {
    /// The folded class.
    pub fn class(&self) -> Result<Mono, CompileError> {
        match self {
            Self::Matrix(matrix) => Ok(Mono::folded(&Spectrum::of(matrix)?.m).0),
            Self::Canonical(class) => Ok(*class),
        }
    }

    /// The folded class and the frame around its canonical gate, checked by
    /// reconstruction.
    pub fn kak(&self) -> Result<(Mono, Frame), CompileError> {
        match self {
            Self::Matrix(matrix) => Spectrum::of(matrix)?.kak(),
            Self::Canonical(class) => Ok((*class, Frame::identity())),
        }
    }
}

/// The part of a KAK that the class needs: with `V` the magic-basis matrix
/// with its global phase removed, `P` diagonalizes `Vᵀ V`, whose eigenphases
/// are the monodromy coordinates `m`; `order` arranges `P`'s columns.
struct Spectrum {
    v: Mat4,
    phase: f64,
    p: Matrix4<f64>,
    m: [f64; 3],
    order: [usize; 4],
}

impl Spectrum {
    fn of(matrix: &Mat4) -> Result<Self, CompileError> {
        // `max_error` folds with `f64::max`, which skips NaN, and the eigensolve
        // iterates until convergence, so a non-finite entry must stop here.
        let deviation = if matrix.iter().all(|z| z.re.is_finite() && z.im.is_finite()) {
            max_error(&(matrix.adjoint() * matrix - Mat4::identity()))
        } else {
            f64::NAN
        };
        if deviation.is_nan() || deviation > INPUT_ATOL {
            return Err(CompileError::NotUnitary(deviation));
        }
        let mut source = *matrix;
        if deviation >= RECONSTRUCT_TOL {
            let repaired = closest_unitary_faer(ndarray_to_faer(matrix_view(&source)))
                .map_err(|error| CompileError::Synthesis(error.to_string()))?;
            source = Mat4::from_fn(|r, c| repaired[(r, c)]);
        }
        let q = magic();
        let u = q.adjoint() * source * q;
        let phase = u.determinant().arg() / 4.0;
        let v = u * C64::from_polar(1.0, -phase);
        // With V = A Λ Pᵀ, Vᵀ V = P Λ² Pᵀ: P diagonalizes Vᵀ V, and the
        // eigenphases of Λ² are the monodromy coordinates, in turns.
        let (p, eigenvalues) = diagonalize(&(v.transpose() * v)).ok_or_else(|| {
            CompileError::Synthesis("could not diagonalize the magic-basis square".into())
        })?;
        let (m, order) = monodromy(&eigenvalues.map(|z| z.arg() / TAU));
        Ok(Self {
            v,
            phase,
            p,
            m,
            order,
        })
    }

    /// `Q† U Q = after · D(m) · before · exp(i phase)` with `after` and
    /// `before` in SO(4) and `m` folded, checked by the Frobenius error of
    /// `after` before the fold, against the input or its closest unitary.
    fn kak(&self) -> Result<(Mono, Frame), CompileError> {
        let m = self.m;
        let mut p = Matrix4::from_fn(|r, c| self.p[(r, self.order[c])]);
        if p.determinant() < 0.0 {
            p.column_mut(0).neg_mut();
        }
        let w = -m[0] - m[1] - m[2];
        let lambda = [m[1], m[0], w, m[2]].map(|turns| C64::from_polar(1.0, -PI * turns));
        // V = (V P Λ̄) Λ Pᵀ, and Λ Pᵀ is unitary, so with `after` the real part of
        // V P Λ̄ the reconstruction misses V by the norm of the imaginary part. The
        // fold below is exact and changes this only by rounding.
        let vp = self.v * p.cast::<C64>();
        let full = Matrix4::from_fn(|r, c| vp[(r, c)] * lambda[c]);
        reconstructed(full.map(|z| z.im).norm())?;
        let frame = Frame {
            after: full.map(|z| z.re),
            before: p.transpose(),
            phase: self.phase,
        };
        let (class, reflected) = Mono::folded(&m);
        Ok((class, if reflected { frame.reflected() } else { frame }))
    }
}

/// A real orthogonal `P` and the eigenvalues `d` with `Pᵀ M P = diag(d)`, off
/// the diagonal within `RECONSTRUCT_TOL`; `None` when no weight pair in
/// `MIXES` separates the eigenvalues.
fn diagonalize(m: &Mat4) -> Option<(Matrix4<f64>, [C64; 4])> {
    MIXES.iter().find_map(|&(a, b)| {
        let mixed = m.map(|z| a * z.re + b * z.im);
        let p = SymmetricEigen::new(mixed).eigenvectors;
        let d = p.transpose().cast::<C64>() * m * p.cast::<C64>();
        let diagonal = (0..4).all(|r| (0..4).all(|c| r == c || d[(r, c)].norm() < RECONSTRUCT_TOL));
        diagonal.then(|| (p, std::array::from_fn(|k| d[(k, k)])))
    })
}

impl Frame {
    /// The frame of `D(m)` around itself.
    pub fn identity() -> Self {
        Self {
            after: Matrix4::identity(),
            before: Matrix4::identity(),
            phase: 0.0,
        }
    }

    /// The same product around `D(rho(m))` instead of `D(m)`.
    /// `D(rho(m)) = i S P D(m) Pᵀ` with `P = (0 2)(1 3)` and
    /// `S = diag(1, 1, -1, -1)`, so `D(m) = i (S P) D(rho(m)) P`; both frame
    /// changes are in SO(4).
    pub fn reflected(&self) -> Self {
        const PERM: [usize; 4] = [2, 3, 0, 1];
        const SIGNS: [f64; 4] = [-1.0, -1.0, 1.0, 1.0];
        Self {
            after: Matrix4::from_fn(|r, c| self.after[(r, PERM[c])] * SIGNS[c]),
            before: Matrix4::from_fn(|r, c| self.before[(PERM[r], c)]),
            phase: self.phase + FRAC_PI_2,
        }
    }
}

/// The frame of `D(from)` around `D(to)`, for `to` the class of `from`, or
/// its reflection `rho` when `reflected`, checked by reconstruction.
pub fn relabel(from: Mono, to: Mono, reflected: bool) -> Result<Frame, CompileError> {
    let frame = if reflected {
        Frame::identity().reflected()
    } else {
        Frame::identity()
    };
    let canonical = Mat4::from_diagonal(&canonical_phases(from, 0.0).into());
    check(&canonical, to, &frame)?;
    Ok(frame)
}

/// The diagonal of `D(class) · exp(i phase)` in the magic basis.
fn canonical_phases(class: Mono, phase: f64) -> [C64; 4] {
    let w = weyl_from_monodromy(&class.0);
    std::array::from_fn(|k| {
        let angle = (0..3).map(|a| CANONICAL_SIGNS[a][k] * w[a]).sum::<f64>();
        C64::from_polar(1.0, FRAC_PI_2 * angle + phase)
    })
}

/// Fail unless `magic = after · D(class) · before · exp(i phase)` in the
/// magic basis, within `RECONSTRUCT_TOL` in the Frobenius norm.
fn check(magic: &Mat4, class: Mono, frame: &Frame) -> Result<(), CompileError> {
    let phases = canonical_phases(class, frame.phase);
    let product = Mat4::from_fn(|r, c| {
        (0..4)
            .map(|k| phases[k] * (frame.after[(r, k)] * frame.before[(k, c)]))
            .sum()
    });
    reconstructed((product - magic).norm())
}

/// Fail unless a frame reconstruction `error` is below `RECONSTRUCT_TOL`. A NaN
/// error fails.
fn reconstructed(error: f64) -> Result<(), CompileError> {
    if error < RECONSTRUCT_TOL {
        Ok(())
    } else {
        Err(CompileError::Synthesis(format!(
            "frame reconstruction error={error:.2e}"
        )))
    }
}

fn max_error(matrix: &Mat4) -> f64 {
    matrix.iter().map(|v| v.norm()).fold(0.0, f64::max)
}

fn matrix_view(matrix: &Mat4) -> ArrayView2<'_, C64> {
    ArrayView2::from_shape((4, 4).f(), matrix.as_slice()).expect("4x4 column-major matrix")
}

/// Sign table of the left quaternion factor; see [`sign`].
const LEFT_SIGNS: [[f64; 4]; 4] = [
    [1.0, 1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0, 1.0],
    [-1.0, 1.0, -1.0, 1.0],
];
/// Sign table of the right quaternion factor; see [`sign`].
const RIGHT_SIGNS: [[f64; 4]; 4] = [
    [1.0, 1.0, 1.0, 1.0],
    [1.0, -1.0, 1.0, -1.0],
    [-1.0, 1.0, 1.0, -1.0],
    [1.0, 1.0, -1.0, -1.0],
];

/// The phase of `D(c1, c2, c3) = exp(iπ/2 (c1 XX + c2 YY + c3 ZZ))` on magic
/// basis vector `k` is `π/2 Σₐ CANONICAL_SIGNS[a][k] cₐ`.
const CANONICAL_SIGNS: [[f64; 4]; 3] = [
    [1.0, 1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0, 1.0],
    [1.0, -1.0, -1.0, 1.0],
];

/// For a local gate `A ⊗ B` with unit quaternions `a`, `b` and rotation
/// `O = Q† (A ⊗ B) Q`: the coefficient `±1` of `a[k] b[l]` in `O[i][j]`, and of
/// `O[i][j]` in `4 a[k] b[l]`, where `j = i ^ k ^ l`. Every other coefficient
/// is zero.
fn sign(k: usize, l: usize, i: usize) -> f64 {
    LEFT_SIGNS[k][i] * RIGHT_SIGNS[l][i ^ k ^ l]
}

/// The single-qubit gate `w I + i (x X + y Y + z Z)` of unit quaternion `[w, x, y, z]`.
fn su2(q: Vector4<f64>) -> Mat2 {
    let [w, x, y, z]: [f64; 4] = q.into();
    Mat2::new(
        C64::new(w, z),
        C64::new(y, x),
        C64::new(-y, x),
        C64::new(w, -z),
    )
}

/// Split a local rotation into its single-qubit gates, in closed form.
///
/// In the magic basis a local gate `A ⊗ B` is a rotation `O = Q† (A ⊗ B) Q`.
/// With unit quaternions `a`, `b` of `A`, `B`, each product `a[k] b[l]` is a
/// quarter of a signed sum of four entries of `O` (see [`sign`]). Those
/// products form the rank-one matrix `a bᵀ`: its largest row gives `b`, and
/// `a = (a bᵀ) b`. The map from `O` is twice an orthogonal map, so the rank-one
/// residual measures `‖Q O Q† − A ⊗ B‖` directly; reflections and matrices
/// that are not orthogonal fail it.
pub fn factor_rotation(o: &Matrix4<f64>) -> Result<LocalPair, CompileError> {
    let ab = Matrix4::from_fn(|k, l| {
        0.25 * (0..4)
            .map(|i| sign(k, l, i) * o[(i, i ^ k ^ l)])
            .sum::<f64>()
    });
    let largest = (ab.row_iter())
        .max_by(|x, y| x.norm_squared().total_cmp(&y.norm_squared()))
        .expect("four rows");
    let b = largest.transpose().normalize();
    let a = (ab * b).normalize();
    let error = 2.0 * (ab - a * b.transpose()).norm();
    if !error.is_finite() || error >= RECONSTRUCT_TOL {
        return Err(CompileError::Synthesis(
            "matrix is not an SO(4) rotation".into(),
        ));
    }
    Ok(LocalPair {
        q1: su2(a),
        q0: su2(b),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::class::monodromy_from_weyl;

    #[test]
    fn rotation_factors_reconstruct_and_reject_reflections() {
        let axes = [
            Mat2::identity(),
            Mat2::new(crate::C0, C64::new(0.0, 1.0), C64::new(0.0, 1.0), crate::C0),
            Mat2::new(
                C64::new(0.6, 0.0),
                C64::new(0.8, 0.0),
                C64::new(-0.8, 0.0),
                C64::new(0.6, 0.0),
            ),
        ];
        let q = magic();
        for left in axes {
            for right in axes {
                let matrix = left.kronecker(&right);
                let rotation = (q.adjoint() * matrix * q).map(|v| v.re);
                let locals = factor_rotation(&rotation).expect("local rotation");
                assert!((locals.q1.kronecker(&locals.q0) - matrix).norm() < 1e-12);
            }
        }
        let mut reflection = Matrix4::identity();
        reflection[(3, 3)] = -1.0;
        assert!(factor_rotation(&reflection).is_err());
        assert!(factor_rotation(&Matrix4::zeros()).is_err());
    }

    #[test]
    fn nearby_nonlocal_classes_are_not_accepted_as_recovery_frames() {
        let target = Mono(monodromy_from_weyl(0.300_001, 0.2, 0.1));
        assert!(relabel(target, Mono(monodromy_from_weyl(0.3, 0.2, 0.1)), false).is_err());
    }
}

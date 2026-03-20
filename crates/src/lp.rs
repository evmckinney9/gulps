//! Warm-startable dual revised simplex for small dense LPs.
//!
//! Solves: minimize c'x subject to Ax ≤ b
//! where A and c are fixed, and only b changes between solves.
//!
//! The basis warm-starts across solves, making repeated solves with
//! different b vectors efficient (only a few pivots per solve).
//!
//! Reference: Nocedal & Wright, "Numerical Optimization" 2nd ed., §13.5.

use nalgebra::{DMatrix, DVector};

/// Safety cap on iterations per solve.
const MAX_PIVOTS: usize = 64;

/// Dual revised simplex solver with warm-startable basis.
pub struct DualSimplexSolver {
    /// Constraint matrix (m × n), fixed across solves.
    a: DMatrix<f64>,
    /// Objective coefficients (n), fixed across solves.
    c: DVector<f64>,
    /// Current basis: n row indices into A (warm-start state).
    basis: Vec<usize>,
    /// Feasibility tolerance.
    tol: f64,
}

impl DualSimplexSolver {
    /// Create a new solver with fixed A, c, and initial dual-feasible basis.
    pub fn new(
        a: DMatrix<f64>,
        c: DVector<f64>,
        initial_basis: Vec<usize>,
        tol: f64,
    ) -> Self {
        Self {
            a,
            c,
            basis: initial_basis,
            tol,
        }
    }

    /// Solve for a given b vector. Returns the optimal x, or None if infeasible.
    ///
    /// The basis is updated (warm-started) regardless of feasibility.
    pub fn solve(&mut self, b: &[f64]) -> Option<Vec<f64>> {
        let n = self.a.ncols();
        let m = self.a.nrows();
        let tol = self.tol;
        let mut basis = self.basis.clone();

        // B = A[basis, :] (n × n submatrix) → invert
        let mut b_inv = extract_basis(&self.a, &basis).try_inverse()?;

        for _ in 0..MAX_PIVOTS {
            // x = B⁻¹ b[basis]
            let b_basis = DVector::from_fn(n, |i, _| b[basis[i]]);
            let x = &b_inv * &b_basis;

            // Find most violated constraint: argmax(Ax - b)
            let mut max_viol = f64::NEG_INFINITY;
            let mut entering = 0;
            for i in 0..m {
                let row = self.a.row(i);
                let mut v = -b[i];
                for j in 0..n {
                    v += row[j] * x[j];
                }
                if v > max_viol {
                    max_viol = v;
                    entering = i;
                }
            }

            if max_viol <= tol {
                self.basis = basis;
                return Some(x.as_slice().to_vec());
            }

            // Dual ratio test
            let a_enter = self.a.row(entering).transpose();
            let tau = b_inv.tr_mul(&a_enter);
            let dual = -(b_inv.tr_mul(&self.c));

            let leaving = min_ratio(tau.as_slice(), dual.as_slice(), n, tol);
            if leaving < 0 {
                self.basis = basis;
                return None;
            }
            let leaving = leaving as usize;

            // Rank-1 Sherman-Morrison update of B⁻¹
            let old_row = basis[leaving];
            let delta = self.a.row(entering) - self.a.row(old_row);
            let b_inv_col = b_inv.column(leaving);
            let denom = 1.0 + (&delta * &b_inv_col)[(0, 0)];

            if denom.abs() < 1e-14 {
                // Near-degenerate pivot: recompute B⁻¹ from scratch
                basis[leaving] = entering;
                b_inv = match extract_basis(&self.a, &basis).try_inverse() {
                    Some(inv) => inv,
                    None => {
                        self.basis = basis;
                        return None;
                    }
                };
            } else {
                let col = b_inv_col.clone_owned();
                let row = &delta * &b_inv;
                // B⁻¹ -= col ⊗ row / denom
                for i in 0..n {
                    for j in 0..n {
                        b_inv[(i, j)] -= col[i] * row[(0, j)] / denom;
                    }
                }
                basis[leaving] = entering;
            }
        }

        self.basis = basis;
        None // iteration limit
    }

    /// Reset the basis to a cold-start state.
    pub fn reset_basis(&mut self, basis: Vec<usize>) {
        self.basis = basis;
    }
}

/// Extract the n×n submatrix A[basis, :].
fn extract_basis(a: &DMatrix<f64>, basis: &[usize]) -> DMatrix<f64> {
    let n = a.ncols();
    DMatrix::from_fn(basis.len(), n, |r, c| a[(basis[r], c)])
}

/// Minimum-ratio test for the dual simplex. Returns leaving index, or -1.
fn min_ratio(tau: &[f64], dual: &[f64], n: usize, tol: f64) -> i32 {
    let mut best_idx: i32 = -1;
    let mut best_ratio = f64::INFINITY;
    for k in 0..n {
        if tau[k] > tol {
            let r = dual[k] / tau[k];
            if r < best_ratio {
                best_ratio = r;
                best_idx = k as i32;
            }
        }
    }
    best_idx
}

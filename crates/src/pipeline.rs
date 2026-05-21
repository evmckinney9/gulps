//! Solve every segment, then stitch the results into circuit data.

use crate::{Mat2, Mat4};
use crate::{recover, solver};

/// Solve every segment (parallel above `min_batch_size`) and stitch. Returns
/// the fused locals and the chain's product ``T`` (phase-absorbed); the caller
/// runs `recover_local_equivalence(target, T)` per-target.
pub(crate) fn solve_segments(
    init: &Mat4,
    basis_mats: &[Mat4],
    target_mats: &[Mat4],
    target_weyl_coords: &[[f64; 3]],
    makhlin_tol: f64,
    weyl_tol: f64,
    min_batch_size: usize,
    identity_warmstart: bool,
) -> Result<(Vec<Mat2>, Vec<Mat2>, Mat4), String> {
    let n = basis_mats.len();

    let solve_one = |i: usize| {
        let seg = solver::Segment {
            prefix: if i == 0 { init } else { &target_mats[i - 1] },
            basis: &basis_mats[i],
            target: &target_mats[i],
        };
        solver::solve(seg, makhlin_tol, weyl_tol, identity_warmstart)
    };

    let results: Vec<_> = if n >= min_batch_size {
        use rayon::prelude::*;
        (0..n).into_par_iter().map(solve_one).collect()
    } else {
        (0..n).map(solve_one).collect()
    };

    for (i, &(_, _, weyl_res, makhlin_res)) in results.iter().enumerate() {
        if weyl_res > weyl_tol && makhlin_res > makhlin_tol {
            return Err(format!(
                "Segment {} synthesis failed (weyl={:.2e}, makhlin={:.2e})",
                i + 1,
                weyl_res,
                makhlin_res
            ));
        }
    }

    let u0s: Vec<Mat2> = results.iter().map(|(u0, _, _, _)| *u0).collect();
    let u1s: Vec<Mat2> = results.iter().map(|(_, u1, _, _)| *u1).collect();

    let stitch = recover::stitch_segments(init, &u0s, &u1s, basis_mats, target_weyl_coords)?;
    Ok((stitch.fused_u0s, stitch.fused_u1s, stitch.t))
}

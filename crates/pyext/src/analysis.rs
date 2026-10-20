//! Entry points for `gulps.analysis`. A region crosses as its 14 bounds.

use numpy::ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use gulps_core::class;
use gulps_core::{FORMS, Region, covers, project};

use crate::decomposer::{GulpsDecomposer, compile_error};
use crate::invariants::LocalEquivalenceClass;

/// The bounds of the region a sentence reaches.
#[pyfunction]
fn sentence_bounds(classes: Vec<LocalEquivalenceClass>) -> [f64; 14] {
    let monos: Vec<_> = classes.iter().map(|c| c.mono.0).collect();
    gulps_core::sentence_region(&monos).0
}

/// The rows of an `(N, 3)` array.
fn triples<'a>(points: &'a ArrayView2<'_, f64>) -> PyResult<impl Iterator<Item = [f64; 3]> + 'a> {
    if points.ncols() != 3 {
        return Err(PyValueError::new_err(format!(
            "expected an (N, 3) array, got {:?}",
            points.shape()
        )));
    }
    Ok(points.rows().into_iter().map(|m| [m[0], m[1], m[2]]))
}

/// Which rows of `(N, 3)` Weyl points the region with `bounds` covers, by
/// the compiler's membership test.
#[pyfunction]
fn region_contains<'py>(
    py: Python<'py>,
    bounds: [f64; 14],
    points: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let region = Region(bounds);
    let points = points.as_array();
    let inside = triples(&points)?
        .map(|weyl| covers(&region, &project(&class::monodromy_from_weyl(&weyl))))
        .collect::<Vec<_>>();
    Ok(inside.into_pyarray(py))
}

/// Rows of monodromy coordinates as Weyl coordinates.
#[pyfunction]
fn weyl_from_monodromy<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let points = points.as_array();
    let flat: Vec<f64> = triples(&points)?
        .flat_map(|m| class::weyl_from_monodromy(&m))
        .collect();
    let array = Array2::from_shape_vec((points.nrows(), 3), flat).expect("three values per row");
    Ok(array.into_pyarray(py))
}

/// A sentence's cost, gates, and region bounds.
type Sentence<'py> = (f64, Bound<'py, PyTuple>, [f64; 14]);

/// The search's `row`-th candidate sentence: its cost, gates, and region bounds.
#[pyfunction]
fn coverage_candidate<'py>(
    py: Python<'py>,
    decomposer: &GulpsDecomposer,
    row: usize,
) -> PyResult<Sentence<'py>> {
    // Release the GIL: reaching a late row can grow the search for a long time.
    let walk = decomposer.edge.binding.decomposer();
    let (slots, cost, region) = py
        .detach(|| walk.coverage_candidate(row))
        .map_err(|err| compile_error(py, err, None))?;
    Ok((cost, decomposer.gate_tuple(py, &slots)?, region.0))
}

/// The candidates the search discards as dominated before its `row`-th
/// candidate: each one's cost, gates, and region bounds.
#[pyfunction]
fn dominated_candidates<'py>(
    py: Python<'py>,
    decomposer: &GulpsDecomposer,
    row: usize,
) -> PyResult<Vec<Sentence<'py>>> {
    let walk = decomposer.edge.binding.decomposer();
    let dominated = py
        .detach(|| walk.dominated(row))
        .map_err(|err| compile_error(py, err, None))?;
    dominated
        .into_iter()
        .map(|(slots, cost, region)| Ok((cost, decomposer.gate_tuple(py, &slots)?, region.0)))
        .collect()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Each bound's subset-sum normal in monodromy coordinates and its shift:
    // a region is `normal . m >= bound + shift` over its finite bounds.
    m.add("REGION_FORMS", FORMS.to_vec())?;
    m.add_function(wrap_pyfunction!(sentence_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(region_contains, m)?)?;
    m.add_function(wrap_pyfunction!(weyl_from_monodromy, m)?)?;
    m.add_function(wrap_pyfunction!(coverage_candidate, m)?)?;
    m.add_function(wrap_pyfunction!(dominated_candidates, m)?)
}

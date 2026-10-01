//! The Rust entry points `gulps.analysis` uses: reachable-region bounds and
//! membership, the bounds' subset-sum forms, the monodromy-to-Weyl chart, and
//! coverage candidates of a decomposer. A region crosses as its 14 bounds.

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
        .map(|[c1, c2, c3]| covers(&region, &project(&class::monodromy_from_weyl(c1, c2, c3))))
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

/// The search's `row`-th candidate sentence: its cost, gates, and region bounds.
#[pyfunction]
fn coverage_candidate<'py>(
    py: Python<'py>,
    decomposer: &GulpsDecomposer,
    row: usize,
) -> PyResult<(f64, Bound<'py, PyTuple>, [f64; 14])> {
    let (slots, cost, region) = decomposer
        .edge
        .binding
        .decomposer()
        .coverage_candidate(row)
        .map_err(|err| compile_error(py, err, None))?;
    let gates = decomposer.gates.bind(py);
    let gates = slots
        .iter()
        .map(|&slot| gates.get_borrowed_item(slot).expect("each slot has a gate"));
    Ok((cost, PyTuple::new(py, gates)?, region.0))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Each bound's subset-sum normal in monodromy coordinates and its shift:
    // a region is `normal . m >= bound + shift` over its finite bounds.
    m.add("REGION_FORMS", FORMS.to_vec())?;
    m.add_function(wrap_pyfunction!(sentence_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(region_contains, m)?)?;
    m.add_function(wrap_pyfunction!(weyl_from_monodromy, m)?)?;
    m.add_function(wrap_pyfunction!(coverage_candidate, m)?)
}

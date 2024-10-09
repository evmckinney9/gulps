// crates/src/basic_functions/basic_math.rs
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(text_signature = "(a, b, /)")]
pub fn add_in_rust(a: i32, b: i32) -> PyResult<i32> {
    Ok(a + b)
}

#[pyfunction]
#[pyo3(text_signature = "(a, b, /)")]
pub fn subtract_in_rust(a: i32, b: i32) -> PyResult<i32> {
    Ok(a - b)
}

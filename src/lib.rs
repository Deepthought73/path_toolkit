use crate::path::{Path, ResamplingType};
use pyo3::prelude::*;

mod path;
mod util;

/// Useful tools for working with paths
#[pymodule]
fn path_toolkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Path>()?;
    m.add_class::<ResamplingType>()?;

    Ok(())
}

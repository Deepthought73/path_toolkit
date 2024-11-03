use crate::path2d::{ElasticBandMethod, InterpolationMethod, ResamplingMethod};
use pyo3::prelude::*;

pub mod path2d;
mod python;
mod util;

/// Useful tools for working with paths
#[pymodule]
fn path_toolkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::Path2D>()?;
    m.add_class::<ResamplingMethod>()?;
    m.add_class::<InterpolationMethod>()?;
    m.add_class::<ElasticBandMethod>()?;

    Ok(())
}

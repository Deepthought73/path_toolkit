use pyo3::{pyclass, pymethods};

#[pyclass(eq)]
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub enum ResamplingMethod {
    ByNumberPoints {
        number_points: usize,
    },
    BySamplingDistance {
        sampling_distance: f64,
        drop_last: bool,
    },
}

#[pymethods]
impl ResamplingMethod {
    #[staticmethod]
    /// by_number_points(number_points)
    ///
    /// The path will be equidistantly resampled using the given number of points.
    ///
    /// :param number_points: Number of points
    ///
    /// :type number_points: int
    ///
    /// :returns: The resampling method
    /// :rtype: ResamplingMethod
    pub fn by_number_points(number_points: usize) -> Self {
        Self::ByNumberPoints { number_points }
    }

    #[staticmethod]
    #[pyo3(signature = (sampling_distance, drop_last=true))]
    /// by_sampling_distance(sampling_distance, drop_last=True)
    ///
    /// The path will be resampled using the given sampling_distance.
    /// The distance between the last and second last point will differ from sampling distance.
    /// Setting drop_last=True will omit the last point.
    ///
    /// :param sampling_distance: Sampling distance
    /// :param drop_last: Omits the last point when true.
    ///
    /// :type sampling_distance: float
    /// :type drop_last: bool
    ///
    /// :returns: The resampling method
    /// :rtype: ResamplingMethod
    pub fn by_sampling_distance(sampling_distance: f64, drop_last: bool) -> Self {
        Self::BySamplingDistance {
            sampling_distance,
            drop_last,
        }
    }
}

#[pyclass(eq, eq_int)]
#[derive(Copy, Clone, Eq, Ord, PartialEq, PartialOrd, Debug)]
pub enum InterpolationMethod {
    Cubic,
    Linear,
}

#[pyclass(eq, eq_int)]
#[derive(Copy, Clone, Eq, Ord, PartialEq, PartialOrd, Debug)]
pub enum ElasticBandMethod {
    SquareBounds,
    OrthogonalBounds,
}

use crate::path2d;
use crate::path2d::{ElasticBandMethod, InterpolationMethod, ResamplingMethod};
use crate::util::vec2_to_pyarray2;
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods};

#[pyclass]
#[derive(Clone, Debug, Default)]
/// Path2D(points=None, x=None, y=None)
///
/// Class storing a 2D path.
///
/// :param points: List of points
/// :param x: List of x coordinates
/// :param y: List of y coordinates
///
/// :type points: list[list[float]]
/// :type x: list[float]
/// :type y: list[float]
pub struct Path2D {
    path: path2d::Path2D,
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

#[pymethods]
impl Path2D {
    #[new]
    #[pyo3(signature = (points=None, x=None, y=None))]
    pub fn new(
        points: Option<Vec<[f64; 2]>>,
        x: Option<Vec<f64>>,
        y: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        match (points, x, y) {
            (Some(points), None, None) => Ok(path2d::Path2D::from_points(points).into()),
            (None, Some(x), Some(y)) => Ok(path2d::Path2D::from_coordinates(x, y).into()),
            _ => Err(PyTypeError::new_err(
                "Create path either from points or coordinates",
            )),
        }
    }

    #[getter]
    pub fn get_points(&self) -> Vec<[f64; 2]> {
        self.path.points.clone()
    }

    #[getter]
    pub fn get_points_np<'py>(&'py self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        vec2_to_pyarray2(py, &self.path.points)
    }

    #[getter]
    pub fn get_x(&self) -> Vec<f64> {
        self.path.x.clone()
    }

    #[getter]
    pub fn get_x_np<'py>(&'py self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.path.x.to_pyarray_bound(py)
    }

    #[getter]
    pub fn get_y(&self) -> Vec<f64> {
        self.path.y.clone()
    }

    #[getter]
    pub fn get_y_np<'py>(&'py self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.path.y.to_pyarray_bound(py)
    }

    #[getter]
    pub fn get_path_length_per_point(&self) -> Vec<f64> {
        self.path.path_length_per_point().to_vec()
    }

    #[getter]
    pub fn get_path_length_per_point_np<'py>(
        &'py self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray1<f64>> {
        self.path.path_length_per_point().to_pyarray_bound(py)
    }

    #[getter]
    pub fn get_length(&self) -> f64 {
        self.path.get_length()
    }

    #[getter]
    pub fn get_orientation(&self) -> Vec<f64> {
        self.path.orientation().to_vec()
    }

    #[getter]
    pub fn get_orientation_np<'py>(&'py self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.path.orientation().to_pyarray_bound(py)
    }

    #[getter]
    pub fn get_unit_tangent_vector(&self) -> Vec<[f64; 2]> {
        self.path.unit_tangent_vector().to_vec()
    }

    #[getter]
    pub fn get_unit_tangent_vector_np<'py>(
        &'py self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray2<f64>> {
        vec2_to_pyarray2(py, self.path.unit_tangent_vector())
    }

    #[getter]
    pub fn get_curvature(&self) -> Vec<f64> {
        self.path.curvature().to_vec()
    }

    #[getter]
    pub fn get_curvature_np<'py>(&'py self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.path.curvature().to_pyarray_bound(py)
    }

    #[pyo3(signature = (max_rmse=0.15))]
    /// compute_circle_fit_curvature(max_rmse=0.15)
    ///
    /// Computes the curvature by decomposing the path into arc segments.
    ///
    /// :param max_rmse: The maximum RMSE (root mean squared error) that is not exceeded when
    ///                     fitting the arc segments
    ///
    /// :type max_rmse: float
    ///
    /// :returns: The curvature of the path
    /// :rtype: list[float]
    pub fn compute_circle_fit_curvature(&self, max_rmse: f64) -> Vec<f64> {
        self.path.compute_circle_fit_curvature(max_rmse)
    }

    #[pyo3(signature = (start, end, max_rmse=0.15))]
    /// find_circle_segments(start, end, max_rmse=0.15)
    ///
    /// Decomposes the path into its circle segments, such that the maximum
    /// RMSE (root mean squared error) is not exceeded.
    ///
    /// :param start: Index of point to start from
    /// :param end: Index of point to stop with
    /// :param max_rmse: Maximum RMSE
    ///
    /// :type start: int
    /// :type end: int
    /// :type max_rmse: float
    ///
    /// :returns: The list of arc segments
    /// :rtype: list[tuple[int, int, float]]
    pub fn find_circle_segments(
        &self,
        start: usize,
        end: usize,
        max_rmse: f64,
    ) -> Vec<(usize, usize, f64)> {
        self.path.find_circle_segments(start, end, max_rmse)
    }

    #[pyo3(signature = (resampling_method, resampling_type=InterpolationMethod::Linear, epsilon=0.01))]
    /// resampled_path(resampling_method, resampling_type=InterpolationMethod.Linear, epsilon=0.01)
    ///
    /// Resamples the path equidistantly using the given interpolation method.
    ///
    /// :param number_points: Number of points of the resampled path
    /// :param resampling_type: Method of interpolation
    /// :param epsilon: The distance within two points are considered equal
    ///
    /// :type number_points: int
    /// :type resampling_type: ResamplingType
    /// :type epsilon: float
    ///
    /// :returns: Resampled path
    /// :rtype: Path2D
    pub fn resampled_path(
        &self,
        resampling_method: ResamplingMethod,
        resampling_type: InterpolationMethod,
        epsilon: f64,
    ) -> Self {
        self.path
            .resampled_path(resampling_method, resampling_type, epsilon)
            .into()
    }

    #[pyo3(signature = (max_deviation, elastic_band_type=ElasticBandMethod::OrthogonalBounds))]
    /// smoothed_path_elastic_band(max_deviation, elastic_band_type=ElasticBandMethod.OrthogonalBounds)
    ///
    /// Smoothes the path using an algorithm from Autoware [1]. A QP has to be solved for that.
    /// CLARABEL [2] is used as the solver.
    ///
    /// [1] https://autowarefoundation.github.io/autoware.universe/refs-tags-v1.0/planning/path_smoother/docs/eb/
    /// [2] https://clarabel.org/stable/
    ///
    /// :param max_deviation: Maximum deviation from the original path
    /// :param elastic_band_type: Type of constraining the deviation to the original path
    ///
    /// :type max_deviation: float
    /// :type elastic_band_type: ElasticBandType
    ///
    /// :returns: The smoothed path
    /// :rtype: Path2D
    pub fn smoothed_path_elastic_band(
        &self,
        max_deviation: f64,
        elastic_band_type: ElasticBandMethod,
    ) -> Option<Self> {
        self.path
            .smoothed_path_elastic_band(max_deviation, elastic_band_type)
            .map(Path2D::from)
    }

    /// smoothed_path_chaikin(num_iterations)
    ///
    /// Smoothes the path using the Chaikin's path smoothing algorithm.
    ///
    /// :param num_iterations: Number of iterations used for Chaikin's path smoothing algorithm.
    ///
    /// :type num_iterations: int
    ///
    /// :returns: The smoothed path
    /// :rtype: Path2D
    pub fn smoothed_path_chaikin(&self, num_iterations: usize) -> Self {
        self.path.smoothed_path_chaikin(num_iterations).into()
    }

    #[pyo3(signature = (point, epsilon=0.01))]
    /// index_from_point(point, epsilon=0.01)
    ///
    /// Returns the index of the nearest point on the path in front of the given point.
    /// If the point outside the path, None is returned
    ///
    /// :param point: Point of interest
    /// :param epsilon: The distance within two points are considered equal
    ///
    /// :type point: list[float]
    /// :type epsilon: float
    ///
    /// :returns: The index of the nearest point
    /// :rtype: Option[int]
    pub fn index_from_point(&self, point: [f64; 2], epsilon: f64) -> Option<usize> {
        self.path.index_from_point(point, epsilon)
    }

    #[pyo3(signature = (point, epsilon=0.01))]
    /// path_length_from_point(point, epsilon=0.01)
    ///
    /// Returns the path length from the first point to the given point.
    /// If the point outside the path, None is returned
    ///
    /// :param point: Point of interest
    /// :param epsilon: The distance within two points are considered equal
    ///
    /// :type point: list[float]
    /// :type epsilon: float
    ///
    /// :returns: The path length
    /// :rtype: Option[int]
    pub fn path_length_from_point(&self, point: [f64; 2], epsilon: f64) -> Option<f64> {
        self.path.path_length_from_point(point, epsilon)
    }

    #[pyo3(signature = (start=None, end=None, epsilon=0.01))]
    /// sub_path(start=None, end=None, epsilon=0.01)
    ///
    /// Returns the sub path from start to end.
    /// If start is None, the path begins at the beginning.
    /// The same holds for end.
    /// The new path is not necessarily equidistant.
    ///
    /// :param start: Beginning of the sub path
    /// :param end: End of the sub path
    /// :param epsilon: The distance within two points are considered equal
    ///
    /// :type start: list[float]
    /// :type end: list[float]
    /// :type epsilon: float
    ///
    /// :returns: The sub path
    /// :rtype: Path2D
    pub fn sub_path(
        &self,
        start: Option<[f64; 2]>,
        end: Option<[f64; 2]>,
        epsilon: f64,
    ) -> Option<Self> {
        self.path.sub_path(start, end, epsilon).map(Path2D::from)
    }

    #[pyo3(signature = (sus_angle=2.8))]
    /// detect_corrupted_point_order(sus_angle=2.8)
    ///
    /// Returns the list of all indices where the path turns sharper than sus_angle.
    ///
    /// :param sus_angle: Every turn sharper than this angle is suspicious.
    ///
    /// :type sus_angle: float
    ///
    /// :returns: Indices of suspicious points
    /// :rtype: list[int]
    pub fn detect_corrupted_point_order(&self, sus_angle: f64) -> Vec<usize> {
        self.path.detect_corrupted_point_order(sus_angle)
    }

    #[pyo3(signature = (sus_angle=2.8))]
    /// repair_corrupted_point_order(sus_angle=2.8)
    ///
    /// Repairs corrupted point order by removing points that progress in the wrong direction.
    ///
    /// :param sus_angle: Every turn sharper than this angle is suspicious.
    ///
    /// :type sus_angle: float
    ///
    /// :returns: Repaired path
    /// :rtype: Path2D
    pub fn repair_corrupted_point_order(&self, sus_angle: f64) -> Self {
        self.path.repair_corrupted_point_order(sus_angle).into()
    }
}

impl From<path2d::Path2D> for Path2D {
    fn from(path: path2d::Path2D) -> Self {
        Path2D { path }
    }
}

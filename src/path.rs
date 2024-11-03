use crate::util::{
    compute_differences, compute_projection, linspace, make_spline, point_equals,
    taubin_circle_fit, Projection,
};
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods};
use simple_qp::constraint;
use simple_qp::expressions::quadratic_expression::QuadraticExpression;
use simple_qp::problem_variables::ProblemVariables;
use simple_qp::solver::clarabel_solver::ClarabelSolver;
use simple_qp::solver::Solver;
use splines::Interpolation;
use std::cell::OnceCell;

#[pyclass]
#[derive(Default, Debug, Clone)]
/// Path(points=None, x=None, y=None)
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
pub struct Path {
    #[pyo3(get)]
    pub points: Vec<[f64; 2]>,
    #[pyo3(get)]
    pub x: Vec<f64>,
    #[pyo3(get)]
    pub y: Vec<f64>,

    path_length_per_point: OnceCell<Vec<f64>>,
    orientation: OnceCell<Vec<f64>>,
    unit_tangent_vector: OnceCell<Vec<[f64; 2]>>,
    curvature: OnceCell<Vec<f64>>,

    associated_values: Vec<Vec<f64>>,
}

#[pyclass(eq, eq_int)]
#[derive(Copy, Clone, Eq, Ord, PartialEq, PartialOrd)]
pub enum ResamplingType {
    Cubic,
    Linear,
}

#[pymethods]
impl Path {
    #[new]
    #[pyo3(signature = (points=None, x=None, y=None))]
    pub fn new(
        points: Option<Vec<[f64; 2]>>,
        x: Option<Vec<f64>>,
        y: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        match (points, x, y) {
            (Some(points), None, None) => Ok(Self::from_points(points)),
            (None, Some(x), Some(y)) => Ok(Self::from_coordinates(x, y)),
            _ => Err(PyValueError::new_err(
                "Create path either from points or coordinates",
            )),
        }
    }

    #[getter]
    pub fn get_path_length_per_point<'py>(&'py self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice_bound(py, self.path_length_per_point())
    }

    #[getter]
    pub fn get_length(&self) -> f64 {
        let s = self.path_length_per_point();
        *s.last().unwrap_or(&0.0)
    }

    #[getter]
    pub fn get_orientation(&self) -> Vec<f64> {
        self.orientation().to_vec()
    }

    #[getter]
    pub fn get_unit_tangent_vector(&self) -> Vec<[f64; 2]> {
        self.unit_tangent_vector().to_vec()
    }

    #[getter]
    pub fn get_curvature(&self) -> Vec<f64> {
        self.curvature().to_vec()
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
        let circle_segments = self.find_circle_segments(0, self.x.len(), max_rmse);

        let mut curvature = vec![];
        for (start, end, c) in circle_segments {
            for _ in start..end {
                curvature.push(c);
            }
        }

        curvature
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
        if end - start < 3 {
            return vec![(start, end, 0.0)];
        }

        let [_, _, radius, rmse] =
            taubin_circle_fit(&self.x[start..end], &self.y[start..end]).unwrap();

        if rmse <= max_rmse {
            vec![(start, end, 1.0 / radius)]
        } else {
            let middle = start + (end - start) / 2;
            let mut left = self.find_circle_segments(start, middle, max_rmse);
            let mut right = self.find_circle_segments(middle, end, max_rmse);

            let joined_start = left.last().unwrap().0;
            let joined_end = right.first().unwrap().1;
            let [_, _, radius, rmse] = taubin_circle_fit(
                &self.x[joined_start..joined_end],
                &self.y[joined_start..joined_end],
            )
            .unwrap();
            if rmse <= max_rmse {
                left.pop();
                left.push((joined_start, joined_end, 1.0 / radius));
                left.extend(&right[1..]);
                left
            } else {
                left.append(&mut right);
                left
            }
        }
    }

    #[pyo3(signature = (number_points, resampling_type=ResamplingType::Linear))]
    /// resampled_path(number_points, resampling_type=ResamplingType.Linear)
    ///
    /// Resamples the path equidistantly using the given interpolation method.
    ///
    /// :param number_points: Number of points of the resampled path
    /// :param resampling_type: Method of interpolation
    ///
    /// :type number_points: int
    /// :type resampling_type: ResamplingType
    ///
    /// :returns: Resampled path
    /// :rtype: Path
    pub fn resampled_path(&self, number_points: usize, resampling_type: ResamplingType) -> Self {
        if self.points.is_empty() {
            return self.clone();
        }

        let s = self.path_length_per_point();
        let path_length = self.get_length();
        let s_resampled = linspace(0, path_length, number_points);

        let interpolation = match resampling_type {
            ResamplingType::Cubic => Interpolation::CatmullRom,
            ResamplingType::Linear => Interpolation::Linear,
        };
        let x_spline = make_spline(s, &self.x, interpolation);
        let y_spline = make_spline(s, &self.y, interpolation);

        let points: Vec<[f64; 2]> = s_resampled
            .iter()
            .map(|s| {
                [
                    x_spline.sample(*s).unwrap_or(0.0),
                    y_spline.sample(*s).unwrap_or(0.0),
                ]
            })
            .collect();
        let mut new_path = Self::from_points(points);

        new_path.associated_values = self
            .associated_values
            .iter()
            .map(|values| {
                let spline = make_spline(s, values, interpolation);
                s_resampled
                    .iter()
                    .map(|s| spline.sample(*s).unwrap_or(0.0))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        new_path
    }

    /// smoothed_path(max_deviation)
    ///
    /// Smoothes the path using an algorithm from Autoware [1]. A QP has to be solved for that.
    /// CLARABEL [2] is used as the solver.
    ///
    /// [1] https://autowarefoundation.github.io/autoware.universe/refs-tags-v1.0/planning/path_smoother/docs/eb/
    /// [2] https://clarabel.org/stable/
    ///
    /// :param max_deviation: Maximum deviation from the original path
    ///
    /// :type max_deviation: float
    ///
    /// :returns: The smoothed path
    /// :rtype: Path
    pub fn smoothed_path(&self, max_deviation: f64) -> Self {
        let n = self.points.len();
        let orientation = self.orientation();

        let mut prob = ProblemVariables::default();
        let xs = prob.add_vector(n, None, None);
        let ys = prob.add_vector(n, None, None);
        let deviation = prob.add_vector(n - 2, None, None);

        let mut objective = QuadraticExpression::default();
        for coords in [&xs, &ys] {
            for x in coords.windows(3) {
                objective += (x[2] - 2.0 * x[1] + x[0]).square();
            }
        }

        let mut constraints = vec![
            constraint!(xs[0] == self.x[0]),
            constraint!(ys[0] == self.y[0]),
            constraint!(xs[n - 1] == self.x[n - 1]),
            constraint!(ys[n - 1] == self.y[n - 1]),
        ];
        for i in 1..n - 1 {
            let orthogonal_vector = [
                max_deviation * orientation[i].sin(),
                -max_deviation * orientation[i].cos(),
            ];
            constraints.push(constraint!(
                self.x[i] + deviation[i - 1] * orthogonal_vector[0] == xs[i]
            ));
            constraints.push(constraint!(
                self.y[i] + deviation[i - 1] * orthogonal_vector[1] == ys[i]
            ));
            constraints.push(constraint!(deviation[i - 1] <= 1.0));
            constraints.push(constraint!(deviation[i - 1] >= -1.0));
        }

        let mut solver = ClarabelSolver::default();
        solver.settings.verbose = false;
        let solution = solver.solve(prob, objective, constraints).unwrap();
        let new_points = solution
            .eval_vec(&xs)
            .into_iter()
            .zip(solution.eval_vec(&ys))
            .map(|(x, y)| [x, y])
            .collect();
        Self::from_points(new_points)
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
        self.nearest_projection(point, epsilon).map(|(i, _)| i)
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
        let i = self.index_from_point(point, epsilon)?;
        let mut s = self.path_length_per_point()[i];
        if i + 1 < self.points.len() {
            let (a, b) = (self.points[i], self.points[i + 1]);
            let p = compute_projection(a, b, point);
            s += p.sp;
        }
        Some(s)
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
    /// :rtype: Path
    pub fn sub_path(
        &self,
        start: Option<[f64; 2]>,
        end: Option<[f64; 2]>,
        epsilon: f64,
    ) -> Option<Self> {
        let first = *self.points.first()?;
        let last = *self.points.last()?;
        let last_index = self.points.len();

        let start = start.unwrap_or(first);
        let end = end.unwrap_or(last);

        let (start_index, start_point, start_nsp) = self
            .nearest_projection(start, epsilon)
            .map(|(i, p)| (i + 1, p.middle_point, p.nsp))?;

        let (end_index, end_point, end_nsp) = self
            .nearest_projection(end, epsilon)
            .map(|(i, p)| (i + 1, p.middle_point, p.nsp))?;

        if start_index > end_index || start_index == end_index && start_nsp > end_nsp {
            None?
        }

        let mut new_points = vec![];
        let mut associated_values = vec![vec![]; self.associated_values.len()];
        if start_index > 0
            && start_index < last_index
            && !point_equals(start_point, self.points[start_index], epsilon)
        {
            new_points.push(start_point);
            for (i, associated_value) in self.associated_values.iter().enumerate() {
                associated_values[i].push(associated_value[start_index]);
            }
        }
        new_points.extend(&self.points[start_index..end_index]);
        for associated_value in self.associated_values.iter() {
            // TODO associated_values.push(associated_value[start_index]);
        }
        if new_points.is_empty() || !point_equals(end_point, *new_points.last().unwrap(), epsilon) {
            new_points.push(end_point);
            for (i, associated_value) in self.associated_values.iter().enumerate() {
                associated_values[i].push(associated_value[end_index]);
            }
        }

        let mut ret = Self::from_points(new_points);
        ret.associated_values = associated_values;
        Some(ret)
    }

    /// add_associated_values(values)
    ///
    /// Adds an associated value
    ///
    /// :param values: The values that should be added as associated to the path
    ///
    /// :type values: list[float]
    ///
    /// :returns: The handle for accessing the associated values again
    /// :rtype: int
    pub fn add_associated_values(&mut self, values: Vec<f64>) -> PyResult<usize> {
        if values.len() != self.points.len() {
            Err(PyValueError::new_err(format!(
                "The associated value has length {}, although {} was expected",
                values.len(),
                self.points.len()
            )))?
        }
        let index = self.associated_values.len();
        self.associated_values.push(values);
        Ok(index)
    }

    pub fn associated_value(&self, handle: usize) -> Option<Vec<f64>> {
        Some(self.associated_values.get(handle)?.clone())
    }

    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

impl Path {
    pub fn from_points(points: Vec<[f64; 2]>) -> Self {
        Self {
            x: points.iter().map(|it| it[0]).collect(),
            y: points.iter().map(|it| it[1]).collect(),
            points,
            ..Default::default()
        }
    }

    pub fn from_coordinates(x: Vec<f64>, y: Vec<f64>) -> Self {
        Self {
            points: x.iter().zip(y.iter()).map(|it| [*it.0, *it.1]).collect(),
            x,
            y,
            ..Default::default()
        }
    }

    fn path_length_per_point(&self) -> &[f64] {
        self.path_length_per_point.get_or_init(|| {
            let n = self.points.len();
            let mut distance = vec![0.0; n];

            for i in 1..n {
                let diff = [
                    self.points[i][0] - self.points[i - 1][0],
                    self.points[i][1] - self.points[i - 1][1],
                ];
                let norm = (diff[0] * diff[0] + diff[1] * diff[1]).sqrt();
                distance[i] = distance[i - 1] + norm;
            }

            distance
        })
    }

    fn orientation(&self) -> &[f64] {
        self.orientation.get_or_init(|| {
            let path = &self.points;
            let n = path.len();
            let mut orientation = vec![0.0; n];

            if n >= 2 {
                orientation[0] = (path[1][1] - path[0][1]).atan2(path[1][0] - path[0][0]);

                for i in 1..n - 1 {
                    let dx = path[i + 1][0] - path[i - 1][0];
                    let dy = path[i + 1][1] - path[i - 1][1];
                    orientation[i] = dy.atan2(dx);
                }

                orientation[n - 1] =
                    (path[n - 1][1] - path[n - 2][1]).atan2(path[n - 1][0] - path[n - 2][0]);
            } else if n == 1 {
                orientation[0] = 0.0;
            }

            orientation
        })
    }

    fn unit_tangent_vector(&self) -> &[[f64; 2]] {
        self.unit_tangent_vector.get_or_init(|| {
            compute_differences(&self.x)
                .into_iter()
                .zip(compute_differences(&self.y))
                .map(|(x, y)| {
                    let length = (x.powi(2) + y.powi(2)).sqrt();
                    [x / length, y / length]
                })
                .collect()
        })
    }

    fn curvature(&self) -> &[f64] {
        self.curvature.get_or_init(|| {
            let x_d = compute_differences(&self.x);
            let x_dd = compute_differences(&x_d);
            let y_d = compute_differences(&self.y);
            let y_dd = compute_differences(&y_d);

            let mut curvature = vec![];
            for i in 0..self.points.len() {
                curvature.push(
                    (x_d[i] * y_dd[i] - x_dd[i] * y_d[i])
                        / ((x_d[i].powi(2) + y_d[i].powi(2)).powf(1.5)),
                );
            }

            curvature
        })
    }

    fn nearest_projection(&self, point: [f64; 2], epsilon: f64) -> Option<(usize, Projection)> {
        let first_point = *self.points.first()?;
        let last_point = *self.points.last()?;
        if point_equals(point, first_point, epsilon) {
            Some((0, Projection::on_point(first_point)))
        } else if point_equals(point, last_point, epsilon) {
            Some((self.points.len() - 1, Projection::on_point(last_point)))
        } else {
            self.points
                .windows(2)
                .map(|ps| compute_projection(ps[0], ps[1], point))
                .enumerate()
                .filter(|(_, p)| 0.0 <= p.nsp && p.nsp < 1.0)
                .min_by(|(_, p1), (_, p2)| p1.sr.total_cmp(&p2.sr))
        }
    }
}

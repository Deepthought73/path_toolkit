mod path_error_detection;
mod path_smoothing;

use crate::util::{
    compute_differences, compute_projection, linspace, make_spline, point_equals, Projection,
};
use pyo3::pyclass;
use splines::Interpolation;
use std::cell::OnceCell;

#[derive(Clone, Default, Debug)]
pub struct Path2D {
    pub points: Vec<[f64; 2]>,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    path_length_per_point: OnceCell<Vec<f64>>,
    orientation: OnceCell<Vec<f64>>,
    unit_tangent_vector: OnceCell<Vec<[f64; 2]>>,
    curvature: OnceCell<Vec<f64>>,
}

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

impl Path2D {
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

    pub fn path_length_per_point(&self) -> &[f64] {
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

    pub fn get_length(&self) -> f64 {
        let s = self.path_length_per_point();
        *s.last().unwrap_or(&0.0)
    }

    pub fn orientation(&self) -> &[f64] {
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

    pub fn unit_tangent_vector(&self) -> &[[f64; 2]] {
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

    pub fn curvature(&self) -> &[f64] {
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

    pub fn nearest_projection(&self, point: [f64; 2], epsilon: f64) -> Option<(usize, Projection)> {
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

    pub fn compute_circle_fit_curvature(&self, max_rmse: f64) -> Vec<f64> {
        let circle_segments = self.find_circle_segments(0, self.x.len(), max_rmse);

        let mut curvature = Vec::with_capacity(self.x.len());
        for (start, end, c) in circle_segments {
            for _ in start..end {
                curvature.push(c);
            }
        }

        curvature
    }

    pub fn resampled_path(
        &self,
        resampling_method: ResamplingMethod,
        resampling_type: InterpolationMethod,
        epsilon: f64,
    ) -> Self {
        if self.points.len() <= 1 {
            return self.clone();
        }

        let interpolation = match resampling_type {
            InterpolationMethod::Cubic => Interpolation::CatmullRom,
            InterpolationMethod::Linear => Interpolation::Linear,
        };

        let s = self.path_length_per_point();
        let x_spline = make_spline(s, &self.x, interpolation);
        let y_spline = make_spline(s, &self.y, interpolation);

        let s_resampled = match resampling_method {
            ResamplingMethod::ByNumberPoints { number_points } => {
                linspace(0, self.get_length(), number_points)
            }
            ResamplingMethod::BySamplingDistance {
                sampling_distance,
                drop_last,
            } => {
                let number_points = (self.get_length() / sampling_distance).floor();
                let new_length = sampling_distance * number_points;
                let mut res = linspace(0, new_length, number_points as usize + 1);
                if !drop_last && (new_length - self.get_length()).abs() > epsilon {
                    res.push(*s.last().unwrap());
                }
                res
            }
        };

        let points: Vec<[f64; 2]> = s_resampled
            .iter()
            .map(|it| {
                [
                    x_spline.sample(*it).unwrap_or(0.0),
                    y_spline.sample(*it).unwrap_or(0.0),
                ]
            })
            .collect();

        Self::from_points(points)
    }

    pub fn index_from_point(&self, point: [f64; 2], epsilon: f64) -> Option<usize> {
        self.nearest_projection(point, epsilon).map(|(i, _)| i)
    }

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
        if start_index > 0
            && start_index < last_index
            && !point_equals(start_point, self.points[start_index], epsilon)
        {
            new_points.push(start_point);
        }
        new_points.extend(&self.points[start_index..end_index]);
        if new_points.is_empty() || !point_equals(end_point, *new_points.last()?, epsilon) {
            new_points.push(end_point);
        }

        Some(Self::from_points(new_points))
    }
}

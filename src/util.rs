use ndarray::{stack, Array1, Axis};
use ndarray_linalg::SVD;
use numpy::PyArray2;
use pyo3::{Bound, Python};
use splines::{Interpolation, Key, Spline};

pub fn compute_differences(arr: &[f64]) -> Vec<f64> {
    let n = arr.len();
    let mut grad = vec![0.0; n];

    if n >= 2 {
        for i in 1..(n - 1) {
            grad[i] = (arr[i + 1] - arr[i - 1]) / 2.0;
        }

        grad[0] = arr[1] - arr[0];
        grad[n - 1] = arr[n - 1] - arr[n - 2];
    }

    grad
}

pub fn make_spline(
    t: &[f64],
    x: &[f64],
    interpolation: Interpolation<f64, f64>,
) -> Spline<f64, f64> {
    let mut spline = Spline::from_vec(
        t.iter()
            .zip(x.iter())
            .map(|it| Key::new(*it.0, *it.1, interpolation))
            .collect(),
    );

    spline.add(Key::new(-1.0, x[0], interpolation));

    let last_t = *t.last().unwrap();
    let last_x = *x.last().unwrap();
    spline.add(Key::new(last_t + 0.01, last_x, interpolation));
    spline.add(Key::new(last_t + 0.02, last_x, interpolation));

    spline
}

pub fn taubin_circle_fit(xs: &[f64], ys: &[f64]) -> Option<[f64; 4]> {
    let mut centered_xs = Array1::from_vec(xs.to_vec());
    let mut centered_ys = Array1::from_vec(ys.to_vec());

    let centroid_x = centered_xs.mean()?;
    let centroid_y = centered_ys.mean()?;
    centered_xs -= centroid_x;
    centered_ys -= centroid_y;

    let z = &centered_xs * &centered_xs + &centered_ys * &centered_ys;

    let z_mean = z.mean()?;
    let z_mean_sqrt2 = z_mean.sqrt() * 2.0;
    let z0 = (z - z_mean) / z_mean_sqrt2;

    let zxy_matrix = stack![Axis(1), z0, centered_xs, centered_ys];
    let (_, _, vt) = zxy_matrix.svd(false, true).ok()?;

    let mut a = vt.map(|vt| vt.row(2).to_vec())?;
    a[0] /= z_mean_sqrt2;
    a.push(-z_mean * a[0]);

    let radius = (a[1].powi(2) + a[2].powi(2) - 4.0 * a[0] * a[3]).sqrt() * 0.5 / a[0].abs();

    let center_x = -0.5 * a[1] / a[0] + centroid_x;
    let center_y = -0.5 * a[2] / a[0] + centroid_y;

    let rmse = compute_rmse(xs, ys, center_x, center_y, radius);

    Some([center_x, center_y, radius, rmse])
}

fn compute_rmse(xs: &[f64], ys: &[f64], center_x: f64, center_y: f64, radius: f64) -> f64 {
    (xs.iter()
        .zip(ys.iter())
        .map(|(x, y)| (((x - center_x).powi(2) + (y - center_y).powi(2)).sqrt() - radius).powi(2))
        .sum::<f64>()
        / xs.len() as f64)
        .sqrt()
}

#[allow(unused)]
fn compute_max_deviation(xs: &[f64], ys: &[f64], center_x: f64, center_y: f64, radius: f64) -> f64 {
    xs.iter()
        .zip(ys.iter())
        .map(|(x, y)| (((x - center_x).powi(2) + (y - center_y).powi(2)).sqrt() - radius).abs())
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(0.0)
}

pub fn linspace<S, T>(start: S, end: T, length: usize) -> Vec<f64>
where
    S: Into<f64> + Copy,
    T: Into<f64> + Copy,
{
    let step: f64 = if length > 1 {
        (end.into() - start.into()) / (length as f64 - 1f64)
    } else {
        0f64
    };

    let mut v = vec![0f64; length];
    v[0] = start.into();
    v[length - 1] = end.into();

    for i in 1..length - 1 {
        v[i] = v[0] + step * (i as f64);
    }
    v
}

#[derive(Copy, Clone, Debug)]
pub struct Projection {
    /// Normalized scalar projection
    pub nsp: f64,
    /// Scalar projection
    pub sp: f64,
    /// Scalar rejection == length of the perpendicular
    pub sr: f64,
    /// Projected point on the line
    pub middle_point: [f64; 2],
}

impl Projection {
    pub fn on_point(point: [f64; 2]) -> Self {
        Self {
            nsp: 0.0,
            sp: 0.0,
            sr: 0.0,
            middle_point: point,
        }
    }
}

pub fn compute_projection(a: [f64; 2], b: [f64; 2], point: [f64; 2]) -> Projection {
    let a_b = [b[0] - a[0], b[1] - a[1]];
    let a_point = [point[0] - a[0], point[1] - a[1]];
    let a_b_length2 = a_b[0].powi(2) + a_b[1].powi(2);

    let nsp = (a_b[0] * a_point[0] + a_b[1] * a_point[1]) / a_b_length2;
    let middle_point = [a[0] + a_b[0] * nsp, a[1] + a_b[1] * nsp];
    let sr = ((middle_point[0] - point[0]).powi(2) + (middle_point[1] - point[1]).powi(2)).sqrt();

    Projection {
        nsp,
        sp: nsp * a_b_length2.sqrt(),
        sr,
        middle_point,
    }
}

pub fn point_equals(a: [f64; 2], b: [f64; 2], epsilon: f64) -> bool {
    (a[0] - b[0]).abs() <= epsilon && (a[1] - b[1]).abs() <= epsilon
}

pub fn vec2_to_pyarray2<'py>(py: Python<'py>, vec2: &[[f64; 2]]) -> Bound<'py, PyArray2<f64>> {
    PyArray2::from_vec2_bound(py, &vec2.iter().map(|it| it.to_vec()).collect::<Vec<_>>())
        // unwrap is ok here, since shape is always (n, 2) ⇒ cannot panic
        .unwrap()
}

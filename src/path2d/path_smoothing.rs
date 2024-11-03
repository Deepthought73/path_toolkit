use crate::path2d::{ElasticBandMethod, Path2D};
use crate::util::taubin_circle_fit;
use simple_qp::constraint;
use simple_qp::expressions::quadratic_expression::QuadraticExpression;
use simple_qp::problem_variables::ProblemVariables;
use simple_qp::solver::clarabel_solver::ClarabelSolver;
use simple_qp::solver::Solver;

impl Path2D {
    pub fn smoothed_path_elastic_band(
        &self,
        max_deviation: f64,
        elastic_band_type: ElasticBandMethod,
    ) -> Option<Self> {
        let n = self.points.len();
        let orientation = self.orientation();

        let mut prob = ProblemVariables::default();
        let xs = prob.add_vector(n, None, None);
        let ys = prob.add_vector(n, None, None);

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

        match elastic_band_type {
            ElasticBandMethod::SquareBounds => {
                for i in 1..n - 1 {
                    constraints.push(constraint!(
                        -max_deviation <= xs[i] - self.x[i] <= max_deviation
                    ));
                    constraints.push(constraint!(
                        -max_deviation <= ys[i] - self.y[i] <= max_deviation
                    ));
                }
            }
            ElasticBandMethod::OrthogonalBounds => {
                let deviation = prob.add_vector(n - 2, None, None);
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
            }
        }

        let mut solver = ClarabelSolver::default();

        solver.settings.verbose = false;
        solver.settings.max_iter = 1000;
        solver.settings.tol_gap_abs = 1e-4;
        solver.settings.tol_gap_rel = 1e-4;
        solver.settings.tol_feas = 1e-4;
        solver.settings.iterative_refinement_abstol = 1e-4;
        solver.settings.iterative_refinement_reltol = 1e-4;

        let solution = solver.solve(prob, objective, constraints).ok()?;
        let new_points = solution
            .eval_vec(&xs)
            .into_iter()
            .zip(solution.eval_vec(&ys))
            .map(|(x, y)| [x, y])
            .collect();
        Some(Self::from_points(new_points))
    }

    pub fn smoothed_path_chaikin(&self, num_iterations: usize) -> Self {
        if self.points.len() <= 1 {
            return self.clone();
        }

        let mut ret = self.points.clone();
        for _ in 0..num_iterations {
            let mut new_points = vec![self.points[0]];
            for ps in ret.windows(2) {
                new_points.push([
                    ps[0][0] * 0.75 + ps[1][0] * 0.25,
                    ps[0][1] * 0.75 + ps[1][1] * 0.25,
                ]);
                new_points.push([
                    ps[0][0] * 0.25 + ps[1][0] * 0.75,
                    ps[0][1] * 0.25 + ps[1][1] * 0.75,
                ]);
            }
            new_points.push(*self.points.last().unwrap());
            ret = new_points;
        }

        Self::from_points(ret)
    }

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
}

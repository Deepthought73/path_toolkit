use crate::path2d::Path2D;
use std::f64::consts::PI;

impl Path2D {
    pub fn detect_corrupted_point_order(&self, sus_angle: f64) -> Vec<usize> {
        let mut corrupted_indices = vec![];

        for (i, ps) in self.points.windows(3).enumerate() {
            let orientation1 = (ps[1][1] - ps[0][1]).atan2(ps[1][0] - ps[0][0]);
            let orientation2 = (ps[2][1] - ps[1][1]).atan2(ps[2][0] - ps[1][0]);
            let mut diff = (orientation2 - orientation1).abs();
            if diff > PI {
                diff = (2.0 * PI - diff).abs();
            }
            if diff >= sus_angle {
                corrupted_indices.push(i + 1)
            }
        }

        corrupted_indices
    }

    pub fn repair_corrupted_point_order(&self, sus_angle: f64) -> Self {
        let mut ret = self.clone();
        let mut corrupted_indices = ret.detect_corrupted_point_order(sus_angle);
        while !corrupted_indices.is_empty() {
            for i in corrupted_indices.iter().rev() {
                ret.points.remove(*i);
                ret.x.remove(*i);
                ret.y.remove(*i);
            }
            corrupted_indices = ret.detect_corrupted_point_order(sus_angle);
        }
        ret
    }
}

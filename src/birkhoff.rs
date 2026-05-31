//! Birkhoff's Ergodic Theorem: for an ergodic measure-preserving system,
//! the time average of an observable equals the space average almost everywhere.

use crate::measure::{Measure, MeasureSpace, Transformation};

/// Computes Birkhoff (time) averages and verifies convergence to space averages.
pub struct BirkhoffAverage;

impl BirkhoffAverage {
    /// Compute the time average of an observable f along an orbit starting at x.
    /// (1/N) * Σ_{k=0}^{N-1} f(T^k(x))
    pub fn time_average(
        transformation: &Transformation,
        observable: &[f64],
        start: usize,
        n_steps: usize,
    ) -> f64 {
        let mut sum = 0.0;
        let mut current = start;
        for _ in 0..n_steps {
            if current < observable.len() {
                sum += observable[current];
            }
            current = transformation.apply(current);
        }
        sum / n_steps as f64
    }

    /// Compute the space average of an observable: ∫ f dμ = Σ f(i) * μ(i).
    pub fn space_average(measure: &Measure, observable: &[f64]) -> f64 {
        measure
            .weights
            .iter()
            .zip(observable.iter())
            .map(|(w, f)| w * f)
            .sum()
    }

    /// Compute Birkhoff averages for all starting points.
    /// Returns vec of (start_point, time_average).
    pub fn all_time_averages(
        transformation: &Transformation,
        observable: &[f64],
        n_steps: usize,
    ) -> Vec<(usize, f64)> {
        let n = transformation.size();
        (0..n)
            .map(|start| {
                let avg = Self::time_average(transformation, observable, start, n_steps);
                (start, avg)
            })
            .collect()
    }

    /// Check if Birkhoff's theorem holds: time averages converge to space average.
    /// Returns (max_deviation, converged).
    pub fn verify_convergence(
        ms: &MeasureSpace,
        observable: &[f64],
        n_steps: usize,
        tolerance: f64,
    ) -> (f64, bool) {
        let space_avg = Self::space_average(&ms.measure, observable);
        let time_avgs = Self::all_time_averages(&ms.transformation, observable, n_steps);
        
        let mut max_dev = 0.0;
        let mut converged = true;
        for (_start, tavg) in &time_avgs {
            let dev = (tavg - space_avg).abs();
            if dev > max_dev {
                max_dev = dev;
            }
            if dev > tolerance {
                converged = false;
            }
        }
        (max_dev, converged)
    }

    /// Compute convergence curve: deviation from space average as function of N.
    /// Returns vec of (n_steps, max_deviation).
    pub fn convergence_curve(
        ms: &MeasureSpace,
        observable: &[f64],
        step_sizes: &[usize],
    ) -> Vec<(usize, f64)> {
        let space_avg = Self::space_average(&ms.measure, observable);
        let n = ms.transformation.size();
        
        step_sizes
            .iter()
            .map(|&n_steps| {
                let mut max_dev = 0.0;
                for start in 0..n {
                    let tavg = Self::time_average(&ms.transformation, observable, start, n_steps);
                    let dev = (tavg - space_avg).abs();
                    if dev > max_dev {
                        max_dev = dev;
                    }
                }
                (n_steps, max_dev)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_average_identity() {
        // Identity map: time average = f(start)
        let t = Transformation::identity(3);
        let f = vec![1.0, 2.0, 3.0];
        let avg = BirkhoffAverage::time_average(&t, &f, 1, 100);
        assert!((avg - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_time_average_cycle() {
        // 3-cycle: time average of f should be (f(0)+f(1)+f(2))/3 for large N
        let t = Transformation::new(vec![1, 2, 0]);
        let f = vec![1.0, 2.0, 3.0];
        let avg = BirkhoffAverage::time_average(&t, &f, 0, 3000);
        let expected = (1.0 + 2.0 + 3.0) / 3.0;
        assert!((avg - expected).abs() < 1e-3);
    }

    #[test]
    fn test_space_average() {
        let m = Measure::uniform(4);
        let f = vec![1.0, 2.0, 3.0, 4.0];
        let avg = BirkhoffAverage::space_average(&m, &f);
        assert!((avg - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_birkhoff_convergence_ergodic() {
        // Ergodic system: Birkhoff averages converge to space average
        let m = Measure::uniform(4);
        let t = Transformation::new(vec![1, 2, 3, 0]); // 4-cycle
        let ms = MeasureSpace::new(m, t);
        let f = vec![1.0, 3.0, 5.0, 7.0];
        let (max_dev, converged) = BirkhoffAverage::verify_convergence(&ms, &f, 10000, 0.01);
        assert!(converged, "max_dev = {max_dev}");
    }

    #[test]
    fn test_birkhoff_non_ergodic_doesnt_converge() {
        // Non-ergodic: time averages may not equal space average
        let t = Transformation::new(vec![1, 0, 3, 2]); // two 2-cycles
        let f = vec![1.0, 3.0, 10.0, 20.0];
        // Starting from 0: avg = (1+3)/2 = 2
        // Starting from 2: avg = (10+20)/2 = 15
        // Space average = (1+3+10+20)/4 = 8.5
        let avg0 = BirkhoffAverage::time_average(&t, &f, 0, 1000);
        let avg2 = BirkhoffAverage::time_average(&t, &f, 2, 1000);
        assert!((avg0 - 2.0).abs() < 0.1);
        assert!((avg2 - 15.0).abs() < 0.1);
        // Neither equals the space average of 8.5
        assert!((avg0 - 8.5).abs() > 1.0);
        assert!((avg2 - 8.5).abs() > 1.0);
    }

    #[test]
    fn test_convergence_curve() {
        let m = Measure::uniform(3);
        let t = Transformation::new(vec![1, 2, 0]);
        let ms = MeasureSpace::new(m, t);
        let f = vec![1.0, 2.0, 3.0];
        let curve = BirkhoffAverage::convergence_curve(&ms, &f, &[10, 100, 1000, 10000]);
        // Deviation should generally decrease
        assert!(curve.last().unwrap().1 < curve.first().unwrap().1);
    }

    #[test]
    fn test_birkhoff_weighted_measure() {
        // Uniform measure with 3-cycle
        let m = Measure::uniform(3);
        let t = Transformation::new(vec![1, 2, 0]);
        let ms = MeasureSpace::new(m.clone(), t.clone());
        let f = vec![10.0, 20.0, 30.0];
        let space_avg = BirkhoffAverage::space_average(&m, &f);
        let time_avg = BirkhoffAverage::time_average(&t, &f, 0, 6000);
        assert!((time_avg - space_avg).abs() < 0.1);
    }
}

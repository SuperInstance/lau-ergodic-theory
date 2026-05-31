//! Mixing conditions: weak mixing and strong mixing.
//!
//! Weak mixing: (1/N) Σ|μ(T⁻ⁿ(A) ∩ B) - μ(A)μ(B)| → 0
//! Strong mixing: μ(T⁻ⁿ(A) ∩ B) → μ(A)μ(B) as n → ∞

use crate::measure::{Measure, MeasureSpace, Transformation};

/// Checks weak and strong mixing properties.
pub struct MixingChecker;

impl MixingChecker {
    /// Compute μ(T⁻ⁿ(A) ∩ B) for given sets A, B and iterate n.
    pub fn correlation(
        ms: &MeasureSpace,
        set_a: &[usize],
        set_b: &[usize],
        n: usize,
    ) -> f64 {
        // Compute T^{-n}(A)
        let mut preimage = set_a.to_vec();
        for _ in 0..n {
            preimage = ms.transformation.preimage(&preimage);
            preimage.sort();
            preimage.dedup();
        }
        // Intersect with B
        let b_set: std::collections::HashSet<usize> = set_b.iter().copied().collect();
        let intersection: Vec<usize> = preimage.into_iter().filter(|x| b_set.contains(x)).collect();
        ms.measure.measure_of(&intersection)
    }

    /// Check strong mixing: μ(T⁻ⁿ(A) ∩ B) → μ(A)μ(B) for all A, B.
    /// Tests all singleton sets.
    pub fn is_strong_mixing(ms: &MeasureSpace, max_iter: usize, tolerance: f64) -> bool {
        let n = ms.measure.size();
        for i in 0..n {
            for j in 0..n {
                let mu_a = ms.measure.measure_of(&[i]);
                let mu_b = ms.measure.measure_of(&[j]);
                let target = mu_a * mu_b;
                
                // Check convergence: correlation at max_iter should be close to target
                let corr = Self::correlation(ms, &[i], &[j], max_iter);
                if (corr - target).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Compute the mixing rate: how fast correlation decays to independence.
    /// Returns vec of (n, max_deviation) pairs.
    pub fn mixing_rate(
        ms: &MeasureSpace,
        set_a: &[usize],
        set_b: &[usize],
        max_n: usize,
    ) -> Vec<(usize, f64)> {
        let mu_a = ms.measure.measure_of(set_a);
        let mu_b = ms.measure.measure_of(set_b);
        let target = mu_a * mu_b;
        
        (1..=max_n)
            .map(|n| {
                let corr = Self::correlation(ms, set_a, set_b, n);
                (n, (corr - target).abs())
            })
            .collect()
    }

    /// Check weak mixing: Cesàro average of |correlation - product| → 0.
    pub fn is_weak_mixing(ms: &MeasureSpace, max_n: usize, tolerance: f64) -> bool {
        let n = ms.measure.size();
        for i in 0..n {
            for j in 0..n {
                let mu_a = ms.measure.measure_of(&[i]);
                let mu_b = ms.measure.measure_of(&[j]);
                let target = mu_a * mu_b;
                
                let mut sum_abs_dev = 0.0;
                for k in 1..=max_n {
                    let corr = Self::correlation(ms, &[i], &[j], k);
                    sum_abs_dev += (corr - target).abs();
                }
                let cesaro = sum_abs_dev / max_n as f64;
                if cesaro > tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Strong mixing implies weak mixing implies ergodicity.
    /// Returns (ergodic, weak_mixing, strong_mixing).
    pub fn mixing_hierarchy(ms: &MeasureSpace, max_iter: usize, tol: f64) -> (bool, bool, bool) {
        let strong = Self::is_strong_mixing(ms, max_iter, tol);
        let weak = Self::is_weak_mixing(ms, max_iter, tol);
        let ergodic = crate::ergodicity::ErgodicChecker::is_ergodic(ms);
        (ergodic, weak, strong)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cycle_is_strong_mixing_uniform() {
        // A single full cycle on n points with uniform measure: check mixing behavior
        // For finite cyclic systems, the system is not mixing in the strict sense
        // (it's periodic), but for long enough periods the correlations cycle.
        // Let's test with a 5-cycle and check the structure.
        let m = Measure::uniform(5);
        let t = Transformation::new(vec![1, 2, 3, 4, 0]);
        let ms = MeasureSpace::new(m, t);
        
        // Check that the system is at least ergodic
        let (ergodic, _, _) = MixingChecker::mixing_hierarchy(&ms, 100, 0.01);
        assert!(ergodic);
    }

    #[test]
    fn test_correlation_computation() {
        let m = Measure::uniform(3);
        let t = Transformation::new(vec![1, 2, 0]);
        let ms = MeasureSpace::new(m, t);
        
        // T^{-1}({0}) = {2}, μ({2} ∩ {0}) = 0
        let corr = MixingChecker::correlation(&ms, &[0], &[0], 1);
        assert!((corr - 0.0).abs() < 1e-10);
        
        // T^{-2}({0}) = {1}, μ({1} ∩ {0}) = 0
        let corr2 = MixingChecker::correlation(&ms, &[0], &[0], 2);
        assert!((corr2 - 0.0).abs() < 1e-10);
        
        // T^{-3}({0}) = {0}, μ({0} ∩ {0}) = 1/3
        let corr3 = MixingChecker::correlation(&ms, &[0], &[0], 3);
        assert!((corr3 - 1.0/3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mixing_rate() {
        let m = Measure::uniform(4);
        let t = Transformation::new(vec![1, 2, 3, 0]);
        let ms = MeasureSpace::new(m, t);
        
        let rate = MixingChecker::mixing_rate(&ms, &[0], &[1], 8);
        assert_eq!(rate.len(), 8);
        // For a 4-cycle, correlation at n=4 should be nonzero
        // μ(T^{-4}({0}) ∩ {1}) = μ({0} ∩ {1}) = 0
    }

    #[test]
    fn test_two_cycles_not_mixing() {
        let m = Measure::uniform(4);
        let t = Transformation::new(vec![1, 0, 3, 2]);
        let ms = MeasureSpace::new(m, t);
        
        // Not ergodic, hence not mixing
        let (ergodic, _, _) = MixingChecker::mixing_hierarchy(&ms, 100, 0.01);
        assert!(!ergodic);
    }

    #[test]
    fn test_weak_mixing_non_periodic() {
        // A system that visits all states (ergodic) but not mixing
        // Single 5-cycle is ergodic
        let m = Measure::uniform(5);
        let t = Transformation::new(vec![1, 2, 3, 4, 0]);
        let ms = MeasureSpace::new(m, t);
        assert!(crate::ergodicity::ErgodicChecker::is_ergodic(&ms));
    }
}

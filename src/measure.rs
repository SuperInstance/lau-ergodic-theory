//! Measure-preserving transformations and measure spaces.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A probability measure over a finite set of states.
/// Represented as a mapping from set index to measure value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measure {
    /// Measure values for each element. Should sum to 1.0 for probability measures.
    pub weights: Vec<f64>,
}

impl Measure {
    /// Create a uniform probability measure over `n` elements.
    pub fn uniform(n: usize) -> Self {
        if n == 0 {
            return Measure { weights: vec![] };
        }
        let w = 1.0 / n as f64;
        Measure { weights: vec![w; n] }
    }

    /// Create from raw weights (will be normalized).
    pub fn from_weights(weights: Vec<f64>) -> Self {
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Measure { weights: vec![0.0; weights.len()] };
        }
        Measure {
            weights: weights.into_iter().map(|w| w / total).collect(),
        }
    }

    /// Create a Dirac (point mass) measure concentrated at index `i` over `n` elements.
    pub fn dirac(n: usize, i: usize) -> Self {
        let mut weights = vec![0.0; n];
        if i < n {
            weights[i] = 1.0;
        }
        Measure { weights }
    }

    /// Evaluate the measure on a subset (given as indices).
    pub fn measure_of(&self, subset: &[usize]) -> f64 {
        subset.iter().map(|&i| self.weights.get(i).copied().unwrap_or(0.0)).sum()
    }

    /// Total measure (should be 1.0 for probability measures).
    pub fn total(&self) -> f64 {
        self.weights.iter().sum()
    }

    /// Number of points in the space.
    pub fn size(&self) -> usize {
        self.weights.len()
    }
}

/// A deterministic transformation T: X -> X on a finite set {0, ..., n-1}.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transformation {
    /// T[i] = image of point i under the transformation.
    pub map: Vec<usize>,
}

impl Transformation {
    /// Create a new transformation from a mapping.
    pub fn new(map: Vec<usize>) -> Self {
        Transformation { map }
    }

    /// Identity transformation on `n` points.
    pub fn identity(n: usize) -> Self {
        Transformation { map: (0..n).collect() }
    }

    /// Apply the transformation to a point.
    pub fn apply(&self, x: usize) -> usize {
        self.map[x]
    }

    /// Compute the preimage T⁻¹(A) for a subset A.
    /// Returns all x such that T(x) ∈ A.
    pub fn preimage(&self, subset: &[usize]) -> Vec<usize> {
        let target_set: std::collections::HashSet<usize> = subset.iter().copied().collect();
        (0..self.map.len())
            .filter(|&i| target_set.contains(&self.map[i]))
            .collect()
    }

    /// Compute forward image T(A).
    pub fn image(&self, subset: &[usize]) -> Vec<usize> {
        let mut result: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for &i in subset {
            result.insert(self.map[i]);
        }
        let mut v: Vec<usize> = result.into_iter().collect();
        v.sort();
        v
    }

    /// Iterate the transformation: compute T^k(x).
    pub fn iterate(&self, x: usize, k: usize) -> usize {
        let mut current = x;
        for _ in 0..k {
            current = self.map[current];
        }
        current
    }

    /// Compute orbit of point x up to length max_len.
    pub fn orbit(&self, x: usize, max_len: usize) -> Vec<usize> {
        let mut orbit = Vec::with_capacity(max_len);
        let mut current = x;
        for _ in 0..max_len {
            orbit.push(current);
            current = self.map[current];
        }
        orbit
    }

    /// Number of points.
    pub fn size(&self) -> usize {
        self.map.len()
    }
}

/// A measure space (X, Σ, μ) with a transformation T.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasureSpace {
    /// The probability measure μ.
    pub measure: Measure,
    /// The transformation T.
    pub transformation: Transformation,
}

impl MeasureSpace {
    /// Create a new measure space.
    pub fn new(measure: Measure, transformation: Transformation) -> Self {
        MeasureSpace { measure, transformation }
    }

    /// Check if T is measure-preserving: μ(T⁻¹(A)) = μ(A) for all subsets A.
    /// We check all singleton sets {i} for i in {0, ..., n-1}.
    pub fn is_measure_preserving(&self) -> bool {
        let n = self.measure.size();
        for i in 0..n {
            let subset = vec![i];
            let pre = self.transformation.preimage(&subset);
            let mu_pre = self.measure.measure_of(&pre);
            let mu_a = self.measure.measure_of(&subset);
            if (mu_pre - mu_a).abs() > 1e-10 {
                return false;
            }
        }
        true
    }

    /// Verify measure-preserving with a tolerance.
    pub fn is_measure_preserving_tol(&self, tol: f64) -> bool {
        let n = self.measure.size();
        for i in 0..n {
            let subset = vec![i];
            let pre = self.transformation.preimage(&subset);
            let mu_pre = self.measure.measure_of(&pre);
            let mu_a = self.measure.measure_of(&subset);
            if (mu_pre - mu_a).abs() > tol {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_measure() {
        let m = Measure::uniform(4);
        assert!((m.total() - 1.0).abs() < 1e-10);
        assert!((m.weights[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_dirac_measure() {
        let m = Measure::dirac(5, 2);
        assert!((m.measure_of(&[2]) - 1.0).abs() < 1e-10);
        assert!((m.measure_of(&[0]) - 0.0).abs() < 1e-10);
        assert!((m.total() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_weights() {
        let m = Measure::from_weights(vec![1.0, 2.0, 3.0]);
        assert!((m.total() - 1.0).abs() < 1e-10);
        assert!((m.weights[0] - 1.0 / 6.0).abs() < 1e-10);
        assert!((m.weights[1] - 2.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_identity_preserves_measure() {
        let m = Measure::uniform(4);
        let t = Transformation::identity(4);
        let ms = MeasureSpace::new(m, t);
        assert!(ms.is_measure_preserving());
    }

    #[test]
    fn test_permutation_preserves_uniform() {
        // A permutation on {0,1,2,3} preserves uniform measure
        let m = Measure::uniform(4);
        let t = Transformation::new(vec![2, 3, 0, 1]); // swap pairs
        let ms = MeasureSpace::new(m, t);
        assert!(ms.is_measure_preserving());
    }

    #[test]
    fn test_non_preserving() {
        // Constant map doesn't preserve non-degenerate measures
        let m = Measure::uniform(4);
        let t = Transformation::new(vec![0, 0, 0, 0]); // everything maps to 0
        let ms = MeasureSpace::new(m, t);
        assert!(!ms.is_measure_preserving());
    }

    #[test]
    fn test_preimage() {
        let t = Transformation::new(vec![1, 2, 0]); // cyclic permutation
        let pre = t.preimage(&[0]);
        assert_eq!(pre, vec![2]);
    }

    #[test]
    fn test_orbit() {
        let t = Transformation::new(vec![1, 2, 0]); // 3-cycle
        let orbit = t.orbit(0, 6);
        assert_eq!(orbit, vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn test_iterate() {
        let t = Transformation::new(vec![1, 2, 0]);
        assert_eq!(t.iterate(0, 3), 0);
        assert_eq!(t.iterate(0, 1), 1);
    }
}

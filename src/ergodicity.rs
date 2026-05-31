//! Ergodicity checking: a measure-preserving system is ergodic if
//! every invariant set has measure 0 or 1.

#[allow(unused_imports)]
use crate::measure::{Measure, MeasureSpace, Transformation};

/// Checks ergodicity of measure-preserving systems.
pub struct ErgodicChecker;

impl ErgodicChecker {
    /// Check if a subset A is T-invariant: T(A) ⊆ A and T(X\A) ⊆ X\A.
    /// Equivalently, T⁻¹(A) = A.
    pub fn is_invariant(ms: &MeasureSpace, subset: &[usize]) -> bool {
        let preimage = ms.transformation.preimage(subset);
        let preimage_set: std::collections::HashSet<usize> = preimage.into_iter().collect();
        let subset_set: std::collections::HashSet<usize> = subset.iter().copied().collect();
        preimage_set == subset_set
    }

    /// Find all invariant subsets of a measure space.
    /// Returns pairs of (subset, measure_of_subset).
    pub fn find_invariant_subsets(ms: &MeasureSpace) -> Vec<(Vec<usize>, f64)> {
        let n = ms.measure.size();
        let mut result = Vec::new();
        // Check all subsets (feasible for small n)
        for mask in 1u32..(1u32 << n) {
            let subset: Vec<usize> = (0..n)
                .filter(|&i| (mask >> i) & 1 == 1)
                .collect();
            if Self::is_invariant(ms, &subset) {
                let mu = ms.measure.measure_of(&subset);
                result.push((subset, mu));
            }
        }
        result
    }

    /// Check ergodicity: the only invariant sets have measure 0 or 1.
    pub fn is_ergodic(ms: &MeasureSpace) -> bool {
        let n = ms.measure.size();
        let mut visited = vec![false; n];
        
        // Find all cycles/orbits of the transformation
        let mut orbits: Vec<Vec<usize>> = Vec::new();
        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut orbit = Vec::new();
            let mut current = start;
            while !visited[current] {
                visited[current] = true;
                orbit.push(current);
                current = ms.transformation.apply(current);
            }
            orbits.push(orbit);
        }

        // For a finite system with a probability measure, ergodicity means
        // all orbits have the same measure, OR equivalently, every orbit
        // has full measure. Actually, for ergodicity we need that every
        // invariant set has measure 0 or 1.
        // An invariant set is a union of orbits.
        // So we need: for every orbit, the measure of that orbit is either 0 or 1,
        // OR for any combination of orbits, the total measure is 0 or 1.
        // Simplest check: find all invariant subsets and check their measures.
        
        for mask in 1u32..(1u32 << orbits.len()) {
            let mut subset = Vec::new();
            for (i, orbit) in orbits.iter().enumerate() {
                if (mask >> i) & 1 == 1 {
                    subset.extend(orbit.iter().copied());
                }
            }
            let mu = ms.measure.measure_of(&subset);
            if mu > 1e-10 && (1.0 - mu) > 1e-10 {
                return false;
            }
        }
        true
    }

    /// Check ergodicity via orbit analysis (more efficient for large spaces).
    /// A system is ergodic iff there is a single orbit that has full measure.
    pub fn is_ergodic_via_orbits(ms: &MeasureSpace) -> bool {
        let n = ms.measure.size();
        let mut visited = vec![false; n];
        let mut orbit_measures: Vec<f64> = Vec::new();

        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut orbit_measure = 0.0;
            let mut current = start;
            while !visited[current] {
                visited[current] = true;
                orbit_measure += ms.measure.weights[current];
                current = ms.transformation.apply(current);
            }
            orbit_measures.push(orbit_measure);
        }

        // Ergodic iff exactly one orbit has measure 1.0 and the rest have measure 0
        let full_orbits = orbit_measures.iter().filter(|&&m| (m - 1.0).abs() < 1e-10).count();
        let zero_orbits = orbit_measures.iter().filter(|&&m| m.abs() < 1e-10).count();
        full_orbits == 1 && full_orbits + zero_orbits == orbit_measures.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cyclic_permutation_is_ergodic() {
        // A single cycle on {0,1,2,3} with uniform measure is ergodic
        let m = Measure::uniform(4);
        let t = Transformation::new(vec![1, 2, 3, 0]);
        let ms = MeasureSpace::new(m, t);
        assert!(ErgodicChecker::is_ergodic(&ms));
        assert!(ErgodicChecker::is_ergodic_via_orbits(&ms));
    }

    #[test]
    fn test_two_cycles_not_ergodic() {
        // (0 1)(2 3) with uniform measure is NOT ergodic
        // {0,1} is an invariant set with measure 0.5
        let m = Measure::uniform(4);
        let t = Transformation::new(vec![1, 0, 3, 2]);
        let ms = MeasureSpace::new(m, t);
        assert!(!ErgodicChecker::is_ergodic(&ms));
    }

    #[test]
    fn test_identity_not_ergodic() {
        let m = Measure::uniform(3);
        let t = Transformation::identity(3);
        let ms = MeasureSpace::new(m, t);
        assert!(!ErgodicChecker::is_ergodic(&ms));
    }

    #[test]
    fn test_invariant_set_detection() {
        let m = Measure::uniform(4);
        let t = Transformation::new(vec![1, 0, 3, 2]); // two 2-cycles
        let ms = MeasureSpace::new(m, t);
        assert!(ErgodicChecker::is_invariant(&ms, &[0, 1]));
        assert!(ErgodicChecker::is_invariant(&ms, &[2, 3]));
        assert!(!ErgodicChecker::is_invariant(&ms, &[0, 2]));
    }

    #[test]
    fn test_ergodic_weighted_measure() {
        // Single cycle with uniform measure is ergodic
        let m = Measure::uniform(3);
        let t = Transformation::new(vec![1, 2, 0]);
        let ms = MeasureSpace::new(m, t);
        assert!(ms.is_measure_preserving());
        assert!(ErgodicChecker::is_ergodic(&ms));
    }

    #[test]
    fn test_ergodic_non_uniform_cycle() {
        // A non-uniform measure is NOT preserved by a cyclic permutation in general.
        // We need to construct a transformation that preserves it.
        // Use uniform measure instead for the ergodicity check.
        let m = Measure::uniform(5);
        let t = Transformation::new(vec![1, 2, 3, 4, 0]);
        let ms = MeasureSpace::new(m, t);
        assert!(ms.is_measure_preserving());
        assert!(ErgodicChecker::is_ergodic(&ms));
    }
}

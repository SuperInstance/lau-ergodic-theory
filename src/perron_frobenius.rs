//! Perron-Frobenius operator (transfer operator) for computing invariant measures.

use crate::measure::Measure;
use nalgebra::DMatrix;


/// The Perron-Frobenius operator P acting on densities.
/// For a map T: ∫_A Pf dμ = ∫_{T⁻¹(A)} f dμ
/// For a finite state system, it's the transpose of the transition matrix.
pub struct PerronFrobeniusOperator;

impl PerronFrobeniusOperator {
    /// Build the Perron-Frobenius matrix for a deterministic map on {0, ..., n-1}.
    /// P[i][j] = 1 / |T⁻¹(j)| if T(i) = j, else 0.
    /// More precisely: P[i][j] counts the fraction of preimages.
    pub fn from_deterministic_map(map: &[usize]) -> DMatrix<f64> {
        let n = map.len();
        let mut pf = vec![0.0; n * n];
        for i in 0..n {
            let j = map[i];
            pf[j * n + i] += 1.0;
        }
        DMatrix::from_row_slice(n, n, &pf)
    }

    /// Build the PF operator for a Markov chain.
    /// The PF operator acts on densities: (Pf)(j) = Σᵢ P(i,j) f(i)
    /// So it's the column operator, which is P^T acting on the density vector.
    pub fn from_markov_chain(transition_matrix: &[Vec<f64>]) -> DMatrix<f64> {
        let n = transition_matrix.len();
        let m = DMatrix::from_row_iterator(
            n,
            n,
            transition_matrix.iter().flat_map(|row| row.iter().copied()),
        );
        m.transpose()
    }

    /// Build the PF operator for a stochastic map where point i maps to j with probability p(i,j).
    pub fn from_stochastic_map(probabilities: &[Vec<(usize, f64)>]) -> DMatrix<f64> {
        let n = probabilities.len();
        let mut pf = vec![0.0; n * n];
        for (i, transitions) in probabilities.iter().enumerate() {
            for &(j, p) in transitions {
                pf[j * n + i] += p;
            }
        }
        DMatrix::from_row_slice(n, n, &pf)
    }

    /// Apply the PF operator to a density (measure).
    /// Returns the density after one step.
    pub fn apply(pf: &DMatrix<f64>, density: &Measure) -> Measure {
        let _n = density.weights.len();
        let d = nalgebra::DVector::from_vec(density.weights.clone());
        let result = pf * d;
        Measure {
            weights: result.iter().copied().collect(),
        }
    }

    /// Find the invariant measure by iterating the PF operator (power method).
    pub fn find_invariant(pf: &DMatrix<f64>, n_iterations: usize, tol: f64) -> Measure {
        let n = pf.nrows();
        let mut density = vec![1.0 / n as f64; n];
        
        for _ in 0..n_iterations {
            let d = nalgebra::DVector::from_vec(density.clone());
            let result = pf * d;
            let new_density: Vec<f64> = result.iter().copied().collect();
            
            let diff: f64 = new_density
                .iter()
                .zip(density.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            
            density = new_density;
            
            // Normalize
            let total: f64 = density.iter().sum();
            if total > 0.0 {
                density = density.into_iter().map(|x| x / total).collect();
            }
            
            if diff < tol {
                break;
            }
        }
        
        Measure { weights: density }
    }

    /// Compute the spectral gap (difference between largest and second-largest eigenvalue).
    /// Related to mixing rate.
    pub fn spectral_gap(pf: &DMatrix<f64>) -> f64 {
        // For a stochastic matrix, largest eigenvalue is 1
        // The spectral gap determines the mixing rate
        let eigenvalues = pf.clone().complex_eigenvalues();
        let mut magnitudes: Vec<f64> = eigenvalues.iter().map(|z| z.norm()).collect();
        magnitudes.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        if magnitudes.len() < 2 {
            return 1.0;
        }
        
        magnitudes[0] - magnitudes[1]
    }

    /// Check if the PF operator has a unique invariant measure.
    /// (PF is ergodic if eigenvalue 1 has multiplicity 1)
    pub fn has_unique_invariant(pf: &DMatrix<f64>, tol: f64) -> bool {
        let eigenvalues = pf.clone().complex_eigenvalues();
        let unit_eigenvalues = eigenvalues
            .iter()
            .filter(|z| (z.norm() - 1.0).abs() < tol)
            .count();
        unit_eigenvalues == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pf_deterministic_cycle() {
        let pf = PerronFrobeniusOperator::from_deterministic_map(&[1, 2, 0]);
        let inv = PerronFrobeniusOperator::find_invariant(&pf, 1000, 1e-12);
        // Should be uniform for a cycle
        for w in &inv.weights {
            assert!((w - 1.0 / 3.0).abs() < 1e-6, "w = {w}");
        }
    }

    #[test]
    fn test_pf_markov_chain() {
        let pf = PerronFrobeniusOperator::from_markov_chain(&[
            vec![0.5, 0.5],
            vec![0.3, 0.7],
        ]);
        let inv = PerronFrobeniusOperator::find_invariant(&pf, 1000, 1e-12);
        // π(0)*0.5 + π(1)*0.3 = π(0) → π(1)*0.3 = π(0)*0.5 → π(0)/π(1) = 3/5
        let ratio = inv.weights[0] / inv.weights[1];
        assert!((ratio - 0.6).abs() < 1e-6, "ratio = {ratio}");
    }

    #[test]
    fn test_pf_apply_preserves_total() {
        let pf = PerronFrobeniusOperator::from_deterministic_map(&[1, 2, 0]);
        let density = Measure::uniform(3);
        let result = PerronFrobeniusOperator::apply(&pf, &density);
        assert!((result.total() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pf_identity() {
        let pf = PerronFrobeniusOperator::from_deterministic_map(&[0, 1, 2]);
        let inv = PerronFrobeniusOperator::find_invariant(&pf, 100, 1e-12);
        // Any measure is invariant for the identity; power method converges to initial
        assert!((inv.total() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_gap() {
        let pf = PerronFrobeniusOperator::from_markov_chain(&[
            vec![0.5, 0.5],
            vec![0.5, 0.5],
        ]);
        let gap = PerronFrobeniusOperator::spectral_gap(&pf);
        // For iid chain, second eigenvalue is 0
        assert!(gap > 0.5, "gap = {gap}");
    }

    #[test]
    fn test_unique_invariant() {
        let pf = PerronFrobeniusOperator::from_markov_chain(&[
            vec![0.5, 0.5],
            vec![0.5, 0.5],
        ]);
        assert!(PerronFrobeniusOperator::has_unique_invariant(&pf, 0.1));
    }

    #[test]
    fn test_stochastic_map() {
        let pf = PerronFrobeniusOperator::from_stochastic_map(&[
            vec![(0, 0.5), (1, 0.5)],
            vec![(0, 0.3), (1, 0.7)],
        ]);
        let inv = PerronFrobeniusOperator::find_invariant(&pf, 1000, 1e-12);
        assert!((inv.total() - 1.0).abs() < 1e-10);
    }
}

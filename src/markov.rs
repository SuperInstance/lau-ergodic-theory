//! Markov chains as measure-preserving systems.
//! A finite-state Markov chain with stationary distribution induces a 
//! measure-preserving transformation on the sequence space.

use crate::measure::Measure;
use crate::entropy::KolmogorovSinaiEntropy;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// A finite-state Markov chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkovChain {
    /// Number of states.
    pub n_states: usize,
    /// Transition matrix P[i][j] = P(X_{n+1} = j | X_n = i).
    pub transition_matrix: Vec<Vec<f64>>,
    /// Stationary distribution (if computed).
    pub stationary: Option<Measure>,
}

impl MarkovChain {
    /// Create a new Markov chain from a transition matrix.
    pub fn new(transition_matrix: Vec<Vec<f64>>) -> Self {
        let n = transition_matrix.len();
        MarkovChain {
            n_states: n,
            transition_matrix,
            stationary: None,
        }
    }

    /// Create a symmetric random walk on n states (ring).
    pub fn symmetric_ring(n: usize) -> Self {
        let mut p = vec![vec![0.0; n]; n];
        for i in 0..n {
            p[i][(i + 1) % n] = 0.5;
            p[i][(i + n - 1) % n] = 0.5;
        }
        MarkovChain::new(p)
    }

    /// Create a doubly stochastic Markov chain (uniform stationary).
    pub fn doubly_stochastic(transition_matrix: Vec<Vec<f64>>) -> Self {
        let mut mc = MarkovChain::new(transition_matrix);
        mc.stationary = Some(Measure::uniform(mc.n_states));
        mc
    }

    /// Compute stationary distribution by solving πP = π.
    pub fn compute_stationary(&mut self) -> &Measure {
        let n = self.n_states;
        
        // Build the system (P^T - I)π = 0 with constraint Σπᵢ = 1
        // Use power iteration as a simple approach
        let mut pi = vec![1.0 / n as f64; n];
        
        for _ in 0..10000 {
            let mut new_pi = vec![0.0; n];
            for j in 0..n {
                for i in 0..n {
                    new_pi[j] += pi[i] * self.transition_matrix[i][j];
                }
            }
            let diff: f64 = new_pi
                .iter()
                .zip(pi.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            pi = new_pi;
            if diff < 1e-14 {
                break;
            }
        }
        
        self.stationary = Some(Measure::from_weights(pi));
        self.stationary.as_ref().unwrap()
    }

    /// Get stationary distribution (computing if needed).
    pub fn stationary(&mut self) -> &Measure {
        if self.stationary.is_none() {
            self.compute_stationary();
        }
        self.stationary.as_ref().unwrap()
    }

    /// Check if the chain is irreducible (every state reachable from every other).
    pub fn is_irreducible(&self) -> bool {
        let n = self.n_states;
        // Use matrix powers to check reachability
        let mut reach = vec![vec![false; n]; n];
        for i in 0..n {
            for j in 0..n {
                reach[i][j] = self.transition_matrix[i][j] > 0.0;
            }
        }
        
        // Floyd-Warshall for transitive closure
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    reach[i][j] = reach[i][j] || (reach[i][k] && reach[k][j]);
                }
            }
        }
        
        (0..n).all(|i| (0..n).all(|j| reach[i][j]))
    }

    /// Check if the Markov chain is ergodic (irreducible and aperiodic).
    pub fn is_ergodic(&self) -> bool {
        if !self.is_irreducible() {
            return false;
        }
        // Check aperiodicity: for irreducible chains, aperiodic iff at least one diagonal entry > 0
        // More precisely, check the period of the chain
        let n = self.n_states;
        let mut period = 0usize;
        let mut current = 0usize;
        
        // Find period of state 0
        let mut visited_at: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        visited_at.insert(0, 0);
        let mut step = 0;
        loop {
            // Move to a random reachable next state (just pick first nonzero)
            let mut next = 0;
            for j in 0..n {
                if self.transition_matrix[current][j] > 0.0 {
                    next = j;
                    break;
                }
            }
            current = next;
            step += 1;
            
            if current == 0 {
                period = step;
                break;
            }
            if step > n * n {
                break;
            }
        }
        
        // Actually, let's do a proper period check
        // Use GCD of return times to state 0
        let mut return_times = Vec::new();
        // BFS from state 0
        let mut queue = vec![(0usize, 0usize)];
        let mut visited = std::collections::HashSet::new();
        while let Some((state, dist)) = queue.pop() {
            if state == 0 && dist > 0 {
                return_times.push(dist);
                if return_times.len() >= 10 {
                    break;
                }
            }
            for j in 0..n {
                if self.transition_matrix[state][j] > 0.0 {
                    let key = (j, dist + 1);
                    if !visited.contains(&key) && dist + 1 <= n * 2 {
                        visited.insert(key);
                        queue.push((j, dist + 1));
                    }
                }
            }
        }
        
        if return_times.is_empty() {
            return false;
        }
        
        let g = return_times.iter().fold(return_times[0], |a, &b| gcd(a, b));
        g == 1
    }

    /// Simulate the Markov chain for n_steps, returning the trajectory.
    pub fn simulate(&self, start: usize, n_steps: usize, rng: &mut impl FnMut() -> f64) -> Vec<usize> {
        let mut trajectory = Vec::with_capacity(n_steps);
        let mut current = start;
        for _ in 0..n_steps {
            trajectory.push(current);
            let r = rng();
            let mut cumsum = 0.0;
            for j in 0..self.n_states {
                cumsum += self.transition_matrix[current][j];
                if r < cumsum {
                    current = j;
                    break;
                }
            }
        }
        trajectory
    }

    /// Compute the entropy rate of the Markov chain.
    pub fn entropy_rate(&mut self) -> f64 {
        let stat = self.stationary().clone();
        KolmogorovSinaiEntropy::markov_entropy_rate(&self.transition_matrix, &stat)
    }

    /// Compute n-step transition probabilities using matrix powers.
    pub fn n_step_probabilities(&self, n_steps: usize) -> DMatrix<f64> {
        let m = DMatrix::from_row_iterator(
            self.n_states,
            self.n_states,
            self.transition_matrix.iter().flat_map(|row| row.iter().copied()),
        );
        let mut result = m.clone();
        for _ in 1..n_steps {
            result = &result * &m;
        }
        result
    }

    /// Convert to a deterministic measure-preserving system by using the shift map
    /// on pairs of (state, position_in_cycle). Returns a transformation on n_states points
    /// that represents the "most likely" deterministic dynamics.
    pub fn to_deterministic_map(&self) -> Vec<usize> {
        // Map each state to its most likely successor
        (0..self.n_states)
            .map(|i| {
                self.transition_matrix[i]
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(j, _)| j)
                    .unwrap_or(i)
            })
            .collect()
    }
}

fn gcd(a: usize, b: usize) -> usize {
    let mut a = a;
    let mut b = b;
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_ring_stationary() {
        let mut mc = MarkovChain::symmetric_ring(4);
        let stat = mc.stationary();
        // Should be uniform
        for w in &stat.weights {
            assert!((w - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn test_symmetric_ring_irreducible() {
        let mc = MarkovChain::symmetric_ring(4);
        assert!(mc.is_irreducible());
    }

    #[test]
    fn test_symmetric_ring_ergodic() {
        // 5-state symmetric ring: has self-loops? No, period is 2 for even, but...
        // Actually, the symmetric ring has period 2 for even n.
        // Use odd n for aperiodicity, or add a self-loop.
        let mc = MarkovChain::symmetric_ring(5);
        assert!(mc.is_irreducible());
        assert!(mc.is_ergodic()); // odd n => aperiodic for ring
    }

    #[test]
    fn test_two_state_chain() {
        // Two-state chain: 0→1, 1→0 (periodic, not ergodic)
        let mc = MarkovChain::new(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        assert!(mc.is_irreducible());
        assert!(!mc.is_ergodic()); // period 2
    }

    #[test]
    fn test_two_state_ergodic() {
        let mc = MarkovChain::new(vec![vec![0.3, 0.7], vec![0.4, 0.6]]);
        assert!(mc.is_irreducible());
        assert!(mc.is_ergodic());
    }

    #[test]
    fn test_stationary_computation() {
        let mut mc = MarkovChain::new(vec![vec![0.3, 0.7], vec![0.4, 0.6]]);
        let stat = mc.stationary();
        // π(0)*0.3 + π(1)*0.4 = π(0) → π(1)*0.4 = π(0)*0.7 → π(0)/π(1) = 4/7
        let ratio = stat.weights[0] / stat.weights[1];
        assert!((ratio - 4.0 / 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_entropy_rate() {
        let mut mc = MarkovChain::doubly_stochastic(vec![
            vec![0.5, 0.5],
            vec![0.5, 0.5],
        ]);
        let h = mc.entropy_rate();
        // Independent coin flips: entropy = ln(2)
        assert!((h - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_n_step_probabilities() {
        let mc = MarkovChain::new(vec![vec![0.5, 0.5], vec![0.5, 0.5]]);
        let p2 = mc.n_step_probabilities(2);
        // P^2 = P for this chain
        assert!((p2[(0, 0)] - 0.5).abs() < 1e-10);
        assert!((p2[(0, 1)] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_deterministic_map() {
        let mc = MarkovChain::new(vec![vec![0.1, 0.9], vec![0.8, 0.2]]);
        let map = mc.to_deterministic_map();
        assert_eq!(map[0], 1); // most likely transition from 0 is to 1
        assert_eq!(map[1], 0); // most likely transition from 1 is to 0
    }

    #[test]
    fn test_simulation() {
        let mc = MarkovChain::new(vec![vec![0.5, 0.5], vec![0.5, 0.5]]);
        let mut rng = || { 0.3 }; // deterministic "random"
        let traj = mc.simulate(0, 10, &mut rng);
        assert_eq!(traj.len(), 10);
    }
}

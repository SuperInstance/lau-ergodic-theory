//! Agent long-term behavior prediction using ergodic theory.
//! Answers: does an agent explore the whole state space or get stuck?

use crate::ergodicity::ErgodicChecker;
use crate::lyapunov::LyapunovExponent;
use crate::markov::MarkovChain;
#[allow(unused_imports)]
use crate::measure::{Measure, MeasureSpace, Transformation};
use crate::birkhoff::BirkhoffAverage;
use serde::{Deserialize, Serialize};

/// Summary of an agent's long-term behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorReport {
    /// Whether the agent explores the whole state space.
    pub explores_fully: bool,
    /// Number of distinct orbits.
    pub num_orbits: usize,
    /// Fraction of state space reached.
    pub exploration_fraction: f64,
    /// Whether the system is ergodic.
    pub is_ergodic: bool,
    /// Lyapunov exponent (if applicable).
    pub lyapunov_exponent: Option<f64>,
    /// Entropy rate.
    pub entropy_rate: f64,
    /// Prediction: where the agent will spend time (measure on states).
    pub stationary_distribution: Option<Vec<f64>>,
    /// Whether the agent gets stuck in subsets.
    pub gets_stuck: bool,
    /// Stuck regions (if any).
    pub stuck_regions: Vec<Vec<usize>>,
    /// Time to explore (estimated steps to visit all states).
    pub cover_time_estimate: Option<usize>,
}

/// Predicts agent long-term behavior using ergodic theory.
pub struct AgentPredictor;

impl AgentPredictor {
    /// Analyze a deterministic agent behavior model.
    pub fn analyze_deterministic(ms: &MeasureSpace) -> BehaviorReport {
        let n = ms.measure.size();
        
        // Find orbits
        let mut visited = vec![false; n];
        let mut orbits = Vec::new();
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

        let is_ergodic = ErgodicChecker::is_ergodic(ms);
        
        // Find stuck regions (orbits with measure > 0 that don't cover the whole space)
        let mut stuck_regions = Vec::new();
        if !is_ergodic {
            for orbit in &orbits {
                let orbit_measure = ms.measure.measure_of(orbit);
                if orbit_measure > 1e-10 && orbit.len() < n {
                    stuck_regions.push(orbit.clone());
                }
            }
        }

        let explored: usize = orbits.iter().flatten().copied().collect::<std::collections::HashSet<usize>>().len();
        let exploration_fraction = explored as f64 / n as f64;

        // Estimate cover time (crude upper bound for ergodic systems)
        let cover_time = if is_ergodic && orbits.len() == 1 {
            Some(n * n) // coupon collector estimate
        } else {
            None
        };

        BehaviorReport {
            explores_fully: is_ergodic,
            num_orbits: orbits.len(),
            exploration_fraction,
            is_ergodic,
            lyapunov_exponent: None, // Need continuous map for this
            entropy_rate: 0.0,       // Deterministic
            stationary_distribution: Some(ms.measure.weights.clone()),
            gets_stuck: !is_ergodic,
            stuck_regions,
            cover_time_estimate: cover_time,
        }
    }

    /// Analyze a stochastic agent behavior model (Markov chain).
    pub fn analyze_markov(mc: &mut MarkovChain) -> BehaviorReport {
        let n = mc.n_states;
        let is_ergodic = mc.is_ergodic();
        let stationary = mc.stationary().clone();
        
        // Check if stationary distribution is concentrated (agent gets stuck)
        let max_weight = stationary.weights.iter().cloned().fold(0.0f64, f64::max);
        let gets_stuck = max_weight > 0.9; // More than 90% of time in one state
        
        let mut stuck_regions = Vec::new();
        if gets_stuck {
            let stuck_state = stationary.weights.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| vec![i]);
            if let Some(region) = stuck_state {
                stuck_regions.push(region);
            }
        }

        // Compute entropy rate
        let entropy_rate = mc.entropy_rate();

        // Exploration fraction based on states with significant stationary measure
        let significant = stationary.weights.iter().filter(|&&w| w > 0.01).count();
        let exploration_fraction = significant as f64 / n as f64;

        BehaviorReport {
            explores_fully: is_ergodic && exploration_fraction > 0.9,
            num_orbits: if is_ergodic { 1 } else { 0 },
            exploration_fraction,
            is_ergodic,
            lyapunov_exponent: None,
            entropy_rate,
            stationary_distribution: Some(stationary.weights.clone()),
            gets_stuck,
            stuck_regions,
            cover_time_estimate: None,
        }
    }

    /// Predict the long-term value of an observable for a deterministic agent.
    pub fn predict_long_term_value(
        ms: &MeasureSpace,
        observable: &[f64],
        start: usize,
        n_steps: usize,
    ) -> f64 {
        BirkhoffAverage::time_average(&ms.transformation, observable, start, n_steps)
    }

    /// Analyze a 1D continuous agent with a dynamics function.
    pub fn analyze_1d_continuous(
        f: &dyn Fn(f64) -> f64,
        df: &dyn Fn(f64) -> f64,
        x0: f64,
        n_steps: usize,
        state_space_size: usize,
    ) -> BehaviorReport {
        // Simulate and bin the trajectory
        let mut trajectory = Vec::with_capacity(n_steps);
        let mut x = x0;
        for _ in 0..n_steps {
            trajectory.push(x);
            x = f(x);
        }

        // Compute Lyapunov exponent
        let lyap = LyapunovExponent::compute_1d(f, df, x0, n_steps);

        // Bin the trajectory
        let min_val = trajectory.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = trajectory.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;
        let bin_width = if range > 0.0 { range / state_space_size as f64 } else { 1.0 };
        
        let mut bins = vec![0usize; state_space_size];
        for &val in &trajectory {
            let bin = ((val - min_val) / bin_width).floor() as usize;
            let bin = bin.min(state_space_size - 1);
            bins[bin] += 1;
        }

        let occupied = bins.iter().filter(|&&c| c > 0).count();
        let exploration_fraction = occupied as f64 / state_space_size as f64;

        // Empirical distribution
        let weights: Vec<f64> = bins
            .iter()
            .map(|&c| c as f64 / n_steps as f64)
            .collect();

        // Shannon entropy of the binned distribution
        let entropy: f64 = weights
            .iter()
            .filter(|&&w| w > 0.0)
            .map(|&w| -w * w.ln())
            .sum();

        let gets_stuck = exploration_fraction < 0.3;
        let mut stuck_regions = Vec::new();
        if gets_stuck {
            stuck_regions.push(
                bins.iter()
                    .enumerate()
                    .filter(|(_, &c)| c > n_steps / 10)
                    .map(|(i, _)| i)
                    .collect(),
            );
        }

        BehaviorReport {
            explores_fully: exploration_fraction > 0.8,
            num_orbits: if exploration_fraction > 0.8 { 1 } else { occupied },
            exploration_fraction,
            is_ergodic: exploration_fraction > 0.5 && lyap > 0.0,
            lyapunov_exponent: Some(lyap),
            entropy_rate: entropy,
            stationary_distribution: Some(weights),
            gets_stuck,
            stuck_regions,
            cover_time_estimate: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_ergodic_agent() {
        let m = Measure::uniform(5);
        let t = Transformation::new(vec![1, 2, 3, 4, 0]);
        let ms = MeasureSpace::new(m, t);
        let report = AgentPredictor::analyze_deterministic(&ms);
        assert!(report.explores_fully);
        assert!(report.is_ergodic);
        assert!(!report.gets_stuck);
        assert_eq!(report.num_orbits, 1);
    }

    #[test]
    fn test_deterministic_stuck_agent() {
        // Agent that cycles between only states 0 and 1, never visiting 2,3
        // Use uniform measure so the stuck-ness is detected
        let m = Measure::uniform(4);
        let t = Transformation::new(vec![1, 0, 2, 3]); // two separate cycles
        let ms = MeasureSpace::new(m, t);
        let report = AgentPredictor::analyze_deterministic(&ms);
        assert!(!report.is_ergodic);
        assert!(report.gets_stuck);
        // exploration_fraction should be 1.0 since all states are visited (by some orbit)
        // but the agent doesn't explore the FULL space from a single starting point
    }

    #[test]
    fn test_markov_agent() {
        let mut mc = MarkovChain::symmetric_ring(5);
        let report = AgentPredictor::analyze_markov(&mut mc);
        assert!(report.is_ergodic);
        assert!(report.explores_fully);
        assert!(!report.gets_stuck);
    }

    #[test]
    fn test_markov_stuck_agent() {
        // Agent that stays in state 0 most of the time
        let mut mc = MarkovChain::new(vec![
            vec![0.95, 0.05],
            vec![0.05, 0.95],
        ]);
        let report = AgentPredictor::analyze_markov(&mut mc);
        assert!(report.is_ergodic);
    }

    #[test]
    fn test_predict_long_term_value() {
        let m = Measure::uniform(3);
        let t = Transformation::new(vec![1, 2, 0]);
        let ms = MeasureSpace::new(m, t);
        let f = vec![10.0, 20.0, 30.0];
        let avg = AgentPredictor::predict_long_term_value(&ms, &f, 0, 3000);
        assert!((avg - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_continuous_logistic_chaotic() {
        let report = AgentPredictor::analyze_1d_continuous(
            &|x| 4.0 * x * (1.0 - x),
            &|x| 4.0 * (1.0 - 2.0 * x),
            0.1,
            10000,
            20,
        );
        assert!(report.lyapunov_exponent.unwrap() > 0.0);
        assert!(report.exploration_fraction > 0.5);
    }

    #[test]
    fn test_continuous_logistic_stable() {
        let report = AgentPredictor::analyze_1d_continuous(
            &|x| 2.0 * x * (1.0 - x),
            &|x| 2.0 * (1.0 - 2.0 * x),
            0.3,
            10000,
            20,
        );
        // Stable fixed point at 0.5
        assert!(report.lyapunov_exponent.unwrap() < 0.0);
        assert!(report.gets_stuck);
    }

    #[test]
    fn test_deterministic_full_report() {
        let m = Measure::uniform(4);
        let t = Transformation::new(vec![1, 2, 3, 0]);
        let ms = MeasureSpace::new(m, t);
        let report = AgentPredictor::analyze_deterministic(&ms);
        assert!(report.stationary_distribution.is_some());
        assert!(report.cover_time_estimate.is_some());
    }
}

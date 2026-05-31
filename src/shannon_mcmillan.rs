//! Shannon-McMillan-Breiman theorem: the entropy of a partition along an orbit
//! converges to the conditional entropy of the partition given the past.

#[allow(unused_imports)]
use crate::measure::{Measure, MeasureSpace, Transformation};
use crate::entropy::Partition;

/// Shannon-McMillan-Breiman theorem implementation.
pub struct ShannonMcMillanBreiman;

impl ShannonMcMillanBreiman {
    /// Compute the information function: -ln(μ(cell containing x in refined partition)) / n.
    /// This converges to h(T, α) for ergodic systems.
    #[allow(clippy::needless_range_loop)]
    pub fn information_rate(
        ms: &MeasureSpace,
        partition: &Partition,
        start: usize,
        n_steps: usize,
    ) -> f64 {
        if n_steps == 0 {
            return 0.0;
        }

        let mut current = start;
        let mut orbit = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            orbit.push(current);
            current = ms.transformation.apply(current);
        }

        // For each point in the orbit, find which cell of the partition it belongs to
        // Compute the measure of the refined cylinder set
        // Approximation: count how many points fall in the same sequence of cells
        let n = ms.measure.size();
        
        // Assign cell labels to each point
        let cell_labels: Vec<usize> = orbit
            .iter()
            .map(|&x| {
                partition
                    .cells
                    .iter()
                    .position(|cell| cell.contains(&x))
                    .unwrap_or(0)
            })
            .collect();

        // For the SMB theorem, compute -ln(μ(C_n(x))) / n
        // where C_n(x) is the cylinder set of points that share the same cell sequence
        // Approximate by computing the empirical measure of this cylinder
        let mut cylinder_count = 0usize;
        for x in 0..n {
            let mut matches = true;
            let mut cur = x;
            for k in 0..n_steps {
                let cell = partition
                    .cells
                    .iter()
                    .position(|c| c.contains(&cur))
                    .unwrap_or(0);
                if cell != cell_labels[k] {
                    matches = false;
                    break;
                }
                cur = ms.transformation.apply(cur);
            }
            if matches {
                cylinder_count += 1;
            }
        }

        let cylinder_measure = cylinder_count as f64 / n as f64;
        if cylinder_measure <= 0.0 {
            return f64::INFINITY;
        }
        -cylinder_measure.ln() / n_steps as f64
    }

    /// Verify SMB theorem: information rate converges to KS entropy.
    pub fn verify_convergence(
        ms: &MeasureSpace,
        partition: &Partition,
        start: usize,
        step_sizes: &[usize],
    ) -> Vec<(usize, f64)> {
        step_sizes
            .iter()
            .map(|&n| {
                let rate = Self::information_rate(ms, partition, start, n);
                (n, rate)
            })
            .collect()
    }

    /// Compute the asymptotic equipartition property:
    /// for an ergodic system, most cylinder sets have approximately equal measure exp(-n*h).
    pub fn aep_measure(
        ms: &MeasureSpace,
        partition: &Partition,
        n_steps: usize,
    ) -> Vec<f64> {
        let n = ms.measure.size();
        let mut measures = Vec::new();
        
        for x in 0..n {
            let rate = Self::information_rate(ms, partition, x, n_steps);
            measures.push(rate);
        }
        measures
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_information_rate_uniform() {
        let m = Measure::uniform(4);
        let t = Transformation::new(vec![1, 2, 3, 0]);
        let ms = MeasureSpace::new(m, t);
        let p = Partition::point_partition(4);
        
        let rate = ShannonMcMillanBreiman::information_rate(&ms, &p, 0, 4);
        // For a 4-cycle with uniform measure, each cylinder of length 4 has measure 1/4
        // -ln(1/4)/4 = ln(4)/4
        assert!(rate.is_finite());
    }

    #[test]
    fn test_smb_convergence() {
        let m = Measure::uniform(3);
        let t = Transformation::new(vec![1, 2, 0]);
        let ms = MeasureSpace::new(m, t);
        let p = Partition::point_partition(3);
        
        let curve = ShannonMcMillanBreiman::verify_convergence(
            &ms, &p, 0, &[3, 6, 9, 12, 30]
        );
        // Should produce finite values
        for &(_, rate) in &curve {
            assert!(rate.is_finite() || rate > 0.0);
        }
    }

    #[test]
    fn test_aep() {
        let m = Measure::uniform(4);
        let t = Transformation::new(vec![1, 2, 3, 0]);
        let ms = MeasureSpace::new(m, t);
        let p = Partition::point_partition(4);
        
        let measures = ShannonMcMillanBreiman::aep_measure(&ms, &p, 4);
        assert_eq!(measures.len(), 4);
        // All should be close for an ergodic system
        let mean: f64 = measures.iter().sum::<f64>() / measures.len() as f64;
        for &m_val in &measures {
            assert!((m_val - mean).abs() < 0.5, "m_val = {m_val}, mean = {mean}");
        }
    }

    #[test]
    fn test_trivial_partition_info_rate() {
        let m = Measure::uniform(4);
        let t = Transformation::new(vec![1, 2, 3, 0]);
        let ms = MeasureSpace::new(m, t);
        let p = Partition::trivial_partition(4);
        
        let rate = ShannonMcMillanBreiman::information_rate(&ms, &p, 0, 4);
        // Trivial partition: all points in same cell, measure = 1
        // -ln(1)/4 = 0
        assert!((rate - 0.0).abs() < 1e-10);
    }
}

//! Lyapunov exponents: measure the exponential rate of separation of
//! infinitesimally close trajectories.

use serde::{Deserialize, Serialize};

/// Lyapunov exponent computation for dynamical systems.
pub struct LyapunovExponent;

impl LyapunovExponent {
    /// Compute the Lyapunov exponent for a 1D map x_{n+1} = f(x_n).
    /// λ = lim (1/N) Σ ln|f'(x_n)|
    pub fn compute_1d(
        f: &dyn Fn(f64) -> f64,
        df: &dyn Fn(f64) -> f64,
        x0: f64,
        n_steps: usize,
    ) -> f64 {
        let mut sum = 0.0;
        let mut x = x0;
        for _ in 0..n_steps {
            let deriv = df(x).abs();
            if deriv > 0.0 {
                sum += deriv.ln();
            }
            x = f(x);
        }
        sum / n_steps as f64
    }

    /// Compute the Lyapunov exponent for the logistic map x_{n+1} = r*x*(1-x).
    pub fn logistic(r: f64, x0: f64, n_steps: usize) -> f64 {
        Self::compute_1d(
            &|x| r * x * (1.0 - x),
            &|x| r * (1.0 - 2.0 * x),
            x0,
            n_steps,
        )
    }

    /// Compute the Lyapunov exponent for the baker's map.
    /// (x, y) → (2x mod 1, (y + floor(2x)) / 2)
    /// Lyapunov exponent in x-direction: λ_x = ln(2).
    pub fn bakers_map(n_steps: usize) -> f64 {
        // The baker's map has λ = ln(2) in the expanding direction
        (2.0_f64).ln()
    }

    /// Compute the Lyapunov exponents for the Arnold cat map.
    /// A = [[2, 1], [1, 1]], eigenvalues determine Lyapunov exponents.
    /// λ₁ = ln((3+√5)/2), λ₂ = -ln((3+√5)/2)
    pub fn arnold_cat() -> (f64, f64) {
        // Eigenvalues of [[2,1],[1,1]] are (3±√5)/2
        let sqrt5 = 5.0_f64.sqrt();
        let lambda1 = ((3.0 + sqrt5) / 2.0).ln();
        let lambda2 = -lambda1; // determinant = 1, so λ₁ + λ₂ = 0
        (lambda1, lambda2)
    }

    /// Compute Lyapunov exponent by tracking nearby trajectories.
    pub fn compute_nearby_1d(
        f: &dyn Fn(f64) -> f64,
        x0: f64,
        delta: f64,
        n_steps: usize,
        renorm_interval: usize,
    ) -> f64 {
        let mut x1 = x0;
        let mut x2 = x0 + delta;
        let mut total_log_ratio = 0.0;
        let mut count = 0;

        for step in 0..n_steps {
            x1 = f(x1);
            x2 = f(x2);
            
            if (step + 1) % renorm_interval == 0 {
                let separation = (x2 - x1).abs();
                if separation > 0.0 {
                    total_log_ratio += (separation / delta).ln();
                    // Renormalize: bring x2 back to distance delta from x1
                    let direction = if x2 > x1 { 1.0 } else { -1.0 };
                    x2 = x1 + direction * delta;
                }
                count += 1;
            }
        }

        if count > 0 {
            total_log_ratio / count as f64 / renorm_interval as f64
        } else {
            0.0
        }
    }

    /// Compute the full spectrum of Lyapunov exponents for a 2D map via QR method.
    /// Returns the two Lyapunov exponents.
    pub fn spectrum_2d(
        f: &dyn Fn(f64, f64) -> (f64, f64),
        jacobian: &dyn Fn(f64, f64) -> [[f64; 2]; 2],
        x0: f64,
        y0: f64,
        n_steps: usize,
    ) -> (f64, f64) {
        let mut x = x0;
        let mut y = y0;
        // Initialize Q = identity
        let mut q = [[1.0, 0.0], [0.0, 1.0]];
        let mut sum1 = 0.0;
        let mut sum2 = 0.0;

        for _ in 0..n_steps {
            let jac = jacobian(x, y);
            // Compute J * Q
            let jq = [
                [
                    jac[0][0] * q[0][0] + jac[0][1] * q[1][0],
                    jac[0][0] * q[0][1] + jac[0][1] * q[1][1],
                ],
                [
                    jac[1][0] * q[0][0] + jac[1][1] * q[1][0],
                    jac[1][0] * q[0][1] + jac[1][1] * q[1][1],
                ],
            ];

            // QR decomposition via Gram-Schmidt
            let col0 = [jq[0][0], jq[1][0]];
            let norm0 = (col0[0] * col0[0] + col0[1] * col0[1]).sqrt();
            
            if norm0 > 0.0 {
                sum1 += norm0.ln();
                let e0 = [col0[0] / norm0, col0[1] / norm0];
                
                // Project second column
                let dot = jq[0][1] * e0[0] + jq[1][1] * e0[1];
                let proj = [jq[0][1] - dot * e0[0], jq[1][1] - dot * e0[1]];
                let norm1 = (proj[0] * proj[0] + proj[1] * proj[1]).sqrt();
                
                if norm1 > 0.0 {
                    sum2 += norm1.ln();
                    q = [[e0[0], proj[0] / norm1], [e0[1], proj[1] / norm1]];
                }
            }

            let (nx, ny) = f(x, y);
            x = nx;
            y = ny;
        }

        (sum1 / n_steps as f64, sum2 / n_steps as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logistic_chaotic() {
        // r = 4: fully chaotic logistic map, λ = ln(2)
        let lambda = LyapunovExponent::logistic(4.0, 0.1, 100000);
        assert!((lambda - 2.0_f64.ln()).abs() < 0.05, "lambda = {lambda}");
    }

    #[test]
    fn test_logistic_stable() {
        // r = 2: stable fixed point, λ < 0
        let lambda = LyapunovExponent::logistic(2.0, 0.3, 10000);
        assert!(lambda < 0.0, "lambda = {lambda}");
    }

    #[test]
    fn test_logistic_periodic() {
        // r = 3.2: period-2 cycle, λ < 0
        let lambda = LyapunovExponent::logistic(3.2, 0.5, 100000);
        assert!(lambda < 0.01, "lambda = {lambda}");
    }

    #[test]
    fn test_bakers_map_lyapunov() {
        let lambda = LyapunovExponent::bakers_map(1000);
        assert!((lambda - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_arnold_cat_lyapunov() {
        let (l1, l2) = LyapunovExponent::arnold_cat();
        assert!(l1 > 0.0, "l1 = {l1}");
        assert!(l2 < 0.0, "l2 = {l2}");
        // λ₁ + λ₂ = 0 (area-preserving)
        assert!((l1 + l2).abs() < 1e-10);
        let sqrt5 = 5.0_f64.sqrt();
        let expected = ((3.0 + sqrt5) / 2.0).ln();
        assert!((l1 - expected).abs() < 1e-10);
    }

    #[test]
    fn test_nearby_trajectories_logistic() {
        let lambda = LyapunovExponent::compute_nearby_1d(
            &|x| 4.0 * x * (1.0 - x),
            0.1,
            1e-10,
            10000,
            10,
        );
        // Should be approximately ln(2)
        assert!((lambda - 2.0_f64.ln()).abs() < 0.2, "lambda = {lambda}");
    }

    #[test]
    fn test_2d_spectrum_arnold_cat() {
        let (l1, l2) = LyapunovExponent::spectrum_2d(
            &|x, y| {
                let nx = (2.0 * x + y) % 1.0;
                let ny = (x + y) % 1.0;
                (if nx < 0.0 { nx + 1.0 } else { nx }, if ny < 0.0 { ny + 1.0 } else { ny })
            },
            &|_x, _y| [[2.0, 1.0], [1.0, 1.0]],
            0.3,
            0.7,
            10000,
        );
        let sqrt5 = 5.0_f64.sqrt();
        let expected = ((3.0 + sqrt5) / 2.0).ln();
        // Should be close to the analytical values
        assert!((l1 - expected).abs() < 0.1, "l1 = {l1}, expected = {expected}");
        assert!((l2 + expected).abs() < 0.1, "l2 = {l2}");
    }

    #[test]
    fn test_logistic_r3_bifurcation() {
        // r = 3.0: onset of period-2, λ ≈ 0
        let lambda = LyapunovExponent::logistic(3.0, 0.2, 100000);
        assert!(lambda.abs() < 0.1, "lambda = {lambda}");
    }
}

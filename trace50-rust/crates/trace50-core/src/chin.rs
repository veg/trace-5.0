//! CHIN: Cluster-to-Host Incidence and Network Estimator.
//!
//! Implements:
//! - Quadratic sampling law inversion: E_obs ≈ (1/2) * n * rho * R0 = (1/2) * rho^2 * N_act * R0.
//! - Closed-form MLE for composite transmission parameter theta = rho * R0 = 2 * E_obs / n.
//! - Inferred active transmitting population scale: N_act = R0 * n^2 / (2 * E_obs) with Poisson CIs.
//! - Inferred longitudinal surveillance sampling fraction: rho = 2 * E_obs / (n * R0).
//! - Borel branching process cluster size spectrum decay test.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::math::{linregress, norm_ppf};

/// Active transmitting population and surveillance coverage estimates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivePopulationEstimate {
    pub n_act: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub relative_error: f64,
    pub r0: f64,
}

/// Borel cluster size spectrum decay test result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorelDecayResult {
    pub slope: f64,
    pub r_squared: f64,
    pub p_value: f64,
    pub n_clusters: usize,
    pub n_megaclusters: usize,
    pub max_cluster_size: usize,
    pub rejects_uniform_sampling: bool,
}

/// CHIN Macro-Epidemic Parameter Inference Engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChinEstimator {
    pub r0_default: f64,
}

impl ChinEstimator {
    pub fn new(r0_default: f64) -> Self {
        Self {
            r0_default: r0_default.max(0.1),
        }
    }

    /// Computes MLE of composite parameter theta = rho * R0 = 2 * E_obs / n.
    pub fn estimate_composite_parameter(&self, n_samples: usize, n_edges: usize) -> f64 {
        if n_samples == 0 {
            return 0.0;
        }
        (2.0 * (n_edges as f64)) / (n_samples as f64)
    }

    /// Inverts quadratic law to estimate surveillance sampling fraction rho in [0, 1].
    pub fn estimate_sampling_fraction(&self, n_samples: usize, n_edges: usize, r0_opt: Option<f64>) -> f64 {
        let r0 = r0_opt.unwrap_or(self.r0_default);
        if r0 <= 0.0 {
            return 0.0;
        }
        let theta = self.estimate_composite_parameter(n_samples, n_edges);
        (theta / r0).max(0.0).min(1.0)
    }

    /// Inverts quadratic scaling law to infer active transmitting population scale N_act:
    /// N_act = (R0 * n^2) / (2 * E_obs).
    pub fn estimate_active_population(
        &self,
        n_samples: usize,
        n_edges: usize,
        r0_opt: Option<f64>,
        confidence_level: f64,
    ) -> ActivePopulationEstimate {
        let r0 = r0_opt.unwrap_or(self.r0_default);
        if n_edges == 0 || n_samples == 0 {
            return ActivePopulationEstimate {
                n_act: f64::INFINITY,
                ci_lower: f64::INFINITY,
                ci_upper: f64::INFINITY,
                relative_error: f64::INFINITY,
                r0,
            };
        }

        let n = n_samples as f64;
        let e = n_edges as f64;
        let n_act = (r0 * n * n) / (2.0 * e);

        let alpha = 1.0 - confidence_level;
        let z = norm_ppf(1.0 - alpha / 2.0);
        let rel_err = z / e.sqrt();

        let ci_lower = n_act * (-rel_err).exp();
        let ci_upper = n_act * rel_err.exp();

        ActivePopulationEstimate {
            n_act,
            ci_lower,
            ci_upper,
            relative_error: rel_err,
            r0,
        }
    }

    /// Tests whether cluster frequencies follow subcritical Borel branching process decay.
    pub fn test_borel_branching_decay(
        &self,
        cluster_sizes: &[usize],
        max_size_fit: usize,
    ) -> BorelDecayResult {
        let valid_sizes: Vec<usize> = cluster_sizes.iter().copied().filter(|&s| s >= 2).collect();
        if valid_sizes.is_empty() {
            return BorelDecayResult {
                slope: 0.0,
                r_squared: 0.0,
                p_value: 1.0,
                n_clusters: 0,
                n_megaclusters: 0,
                max_cluster_size: 0,
                rejects_uniform_sampling: false,
            };
        }

        let max_size = valid_sizes.iter().copied().max().unwrap_or(0);
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &s in &valid_sizes {
            *counts.entry(s).or_insert(0) += 1;
        }

        let mut x_vals = Vec::new();
        let mut y_vals = Vec::new();
        let mut n_megaclusters = 0usize;

        for (&size, &cnt) in &counts {
            if size > max_size_fit {
                n_megaclusters += 1;
            }
            if size <= max_size_fit {
                x_vals.push(size as f64);
                y_vals.push((cnt as f64).ln());
            }
        }

        let (slope, r_squared, p_value) = if x_vals.len() >= 3 {
            match linregress(&x_vals, &y_vals) {
                Some(lr) => (lr.slope, lr.r_squared, lr.p_value),
                None => (0.0, 0.0, 1.0),
            }
        } else {
            (0.0, 0.0, 1.0)
        };

        // Rejection if massive single-linkage percolation mega-clusters exist (size >= 50)
        let rejects = max_size >= 50 || (n_megaclusters >= 5 && r_squared < 0.85);

        BorelDecayResult {
            slope,
            r_squared,
            p_value,
            n_clusters: valid_sizes.len(),
            n_megaclusters,
            max_cluster_size: max_size,
            rejects_uniform_sampling: rejects,
        }
    }
}

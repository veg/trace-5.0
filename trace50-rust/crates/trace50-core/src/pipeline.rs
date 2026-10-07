//! End-to-end TRACE-5.0 Molecular Transmission Surveillance Pipeline.
//!
//! Orchestrates:
//! 1. Alignment ingestion and IUPAC bitmask encoding.
//! 2. Pairwise TN93 distance matrix computation.
//! 3. Static single-linkage network benchmarking (d <= 0.015).
//! 4. DANNO Bayesian hypothesis testing, physical clock adequacy testing, and Benjamini-Hochberg FDR control.
//! 5. STEVE / AutoClock recursive spectral graph deconvolution and evolutionary velocity tracking.
//! 6. CHIN macro-epidemic quadratic scaling and active population estimation.
//! 7. Full self-contained Visualization Dossier JSON generation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::autoclock::recursive_autoclock_deconvolution;
use crate::chin::ChinEstimator;
use crate::danno::DannoEstimator;
use crate::network::extract_clusters;
use crate::tn93::Alignment;
use crate::visualization::*;

/// Configuration parameters for TRACE-5.0.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Trace50Config {
    pub mu: f64,
    pub tau_bar: f64,
    pub omega: f64,
    pub t_span: f64,
    pub alpha_adequacy: f64,
    pub fdr_threshold: f64,
    pub prior_odds: f64,
    pub max_delta_t: f64,
    pub r0: f64,
    pub min_cluster_size: usize,
    pub max_recursion_depth: usize,
    pub use_robust_bf: bool,
    pub static_distance_threshold: f64,
}

impl Default for Trace50Config {
    fn default() -> Self {
        Self {
            mu: 2.0e-3,
            tau_bar: 1.0,
            omega: 2.0,
            t_span: 20.0,
            alpha_adequacy: 0.05,
            fdr_threshold: 0.05,
            prior_odds: 0.01,
            max_delta_t: 2.0,
            r0: 1.5,
            min_cluster_size: 3,
            max_recursion_depth: 8,
            use_robust_bf: false,
            static_distance_threshold: 0.015,
        }
    }
}

/// Executes the end-to-end TRACE-5.0 surveillance pipeline.
pub fn run_trace50(
    alignment: &Alignment,
    config: &Trace50Config,
) -> Result<VisualizationDossier, String> {
    #[cfg(not(target_arch = "wasm32"))]
    let start_time = std::time::Instant::now();
    let n = alignment.num_sequences();
    let l = alignment.seq_len;

    if n < 2 {
        return Err("TRACE-5.0 requires at least 2 sequences to evaluate transmission networks".to_string());
    }

    // 1. Process dates: impute median for any NaN
    let mut valid_dates: Vec<f64> = alignment.dates.iter().copied().filter(|d| d.is_finite()).collect();
    let med_date = if !valid_dates.is_empty() {
        valid_dates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        valid_dates[valid_dates.len() / 2]
    } else {
        2020.0
    };

    let mut dates = alignment.dates.clone();
    let mut date_min = f64::INFINITY;
    let mut date_max = f64::NEG_INFINITY;
    for d in &mut dates {
        if !d.is_finite() {
            *d = med_date;
        }
        if *d < date_min { date_min = *d; }
        if *d > date_max { date_max = *d; }
    }
    let time_span_years = if date_max >= date_min { date_max - date_min } else { 0.0 };

    // 2. Compute pairwise TN93 distance matrix
    let d_matrix = alignment.compute_distance_matrix();

    // 3. Static Network Benchmark (e.g. d <= 1.5%)
    let mut static_edges = Vec::new();
    let mut degree_static = vec![0usize; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = d_matrix[i * n + j];
            if dist <= config.static_distance_threshold {
                static_edges.push((i, j, dist));
                degree_static[i] += 1;
                degree_static[j] += 1;
            }
        }
    }

    let (static_clusters, static_cluster_assign, static_giant_size) =
        extract_clusters(n, &static_edges, &dates);
    let static_clustered_count = static_cluster_assign.iter().filter(|&&c| c > 0).count();

    // 4. DANNO Bayesian Network Inference
    let mut danno = DannoEstimator::new(
        l,
        config.mu,
        config.tau_bar,
        config.omega,
        config.t_span.max(time_span_years),
        config.alpha_adequacy,
        0.065,
        0.0004,
    );

    let all_indices: Vec<usize> = (0..n).collect();
    let dyads = danno.screen_dyads(
        &all_indices,
        &dates,
        &d_matrix,
        n,
        config.max_delta_t,
        config.prior_odds,
        config.fdr_threshold,
        config.use_robust_bf,
    );

    // Partition dyads into certified transmissions vs clock violations vs uncertified
    let mut certified_edges = Vec::new();
    let mut degree_certified = vec![0usize; n];
    let mut clock_violations_count = 0usize;
    let mut dossier_edges = Vec::new();
    let mut phase_points = Vec::new();

    for d in &dyads {
        let status = if d.is_clock_violation {
            clock_violations_count += 1;
            "clock_violation".to_string()
        } else if d.is_certified {
            certified_edges.push((d.idx1, d.idx2, d.dist));
            degree_certified[d.idx1] += 1;
            degree_certified[d.idx2] += 1;
            "certified_transmission".to_string()
        } else {
            "unsupported_candidate".to_string()
        };

        phase_points.push(DossierPhasePoint {
            dist: d.dist,
            delta_t: d.delta_t,
            status: status.clone(),
        });

        dossier_edges.push(DossierEdge {
            source: alignment.headers[d.idx1].clone(),
            target: alignment.headers[d.idx2].clone(),
            source_idx: d.idx1,
            target_idx: d.idx2,
            distance: d.dist,
            delta_t_years: d.delta_t,
            k_substitutions: d.k_substitutions,
            bayes_factor: d.bayes_factor,
            pep: d.pep,
            q_value: d.q_value,
            p_adequacy: d.p_adequacy,
            status,
        });
    }

    // Extract certified clusters
    let (certified_clusters, certified_cluster_assign, certified_giant_size) =
        extract_clusters(n, &certified_edges, &dates);
    let certified_clustered_count = certified_cluster_assign.iter().filter(|&&c| c > 0).count();

    let reduction_pct = if static_giant_size > 0 {
        ((static_giant_size as f64 - certified_giant_size as f64) / (static_giant_size as f64)) * 100.0
    } else {
        0.0
    };

    // 5. STEVE / AutoClock Spectral Deconvolution
    let communities = recursive_autoclock_deconvolution(
        &all_indices,
        &dates,
        &d_matrix,
        n,
        config.min_cluster_size,
        config.max_recursion_depth,
        0,
        "root",
        0.65,
    );

    // Map community assignments to nodes
    let mut community_assignment = vec![0usize; n];
    let mut community_tier_map = HashMap::new();
    for (comm_idx, comm) in communities.iter().enumerate() {
        let comm_id = comm_idx + 1;
        community_tier_map.insert(comm_id, comm.operational_tier.clone());
        for &m in &comm.members {
            community_assignment[m] = comm_id;
        }
    }

    // 6. CHIN Macro-Epidemic Parameter Estimation
    let chin = ChinEstimator::new(config.r0);
    let e_certified = certified_edges.len();
    let theta = chin.estimate_composite_parameter(n, e_certified);
    let rho = chin.estimate_sampling_fraction(n, e_certified, Some(config.r0));
    let active_pop = chin.estimate_active_population(n, e_certified, Some(config.r0), 0.95);

    let cluster_sizes: Vec<usize> = certified_clusters.iter().map(|c| c.size).collect();
    let borel_decay = chin.test_borel_branching_decay(&cluster_sizes, 12);

    // Compute Borel spectrum histogram
    let mut size_counts: HashMap<usize, usize> = HashMap::new();
    for &s in &cluster_sizes {
        *size_counts.entry(s).or_insert(0) += 1;
    }
    let mut borel_bins: Vec<DossierBorelBin> = size_counts
        .into_iter()
        .map(|(cluster_size, count)| DossierBorelBin { cluster_size, count })
        .collect();
    borel_bins.sort_by_key(|b| b.cluster_size);

    // 7. Assemble Nodes
    let mut nodes = Vec::with_capacity(n);
    for i in 0..n {
        let comm_id = community_assignment[i];
        let tier = community_tier_map.get(&comm_id).cloned().unwrap_or_else(|| "Unassigned".to_string());

        nodes.push(DossierNode {
            id: alignment.headers[i].clone(),
            index: i,
            date: dates[i],
            degree_static: degree_static[i],
            degree_certified: degree_certified[i],
            cluster_id_static: static_cluster_assign[i],
            cluster_id_certified: certified_cluster_assign[i],
            community_id: comm_id,
            operational_tier: tier,
        });
    }

    // Assemble Dossier Clusters
    let dossier_clusters: Vec<DossierCluster> = certified_clusters
        .into_iter()
        .map(|c| DossierCluster {
            cluster_id: c.cluster_id,
            size: c.size,
            members: c.members.into_iter().map(|idx| alignment.headers[idx].clone()).collect(),
            time_span_years: c.time_span_years,
        })
        .collect();

    #[cfg(not(target_arch = "wasm32"))]
    let execution_time_seconds = start_time.elapsed().as_secs_f64();
    #[cfg(target_arch = "wasm32")]
    let execution_time_seconds = 0.0;

    Ok(VisualizationDossier {
        schema_version: "5.0.0".to_string(),
        engine: "TRACE-5.0 (Rust Kernel)".to_string(),
        metadata: DossierMetadata {
            num_sequences: n,
            seq_len: l,
            date_min,
            date_max,
            time_span_years,
            execution_time_seconds,
            parameters: DossierParameters {
                mu: config.mu,
                tau_bar: config.tau_bar,
                omega: config.omega,
                fdr_threshold: config.fdr_threshold,
                alpha_adequacy: config.alpha_adequacy,
                r0: config.r0,
            },
        },
        kpi: DossierKpi {
            total_sequences: n,
            total_pairwise_comparisons: n * (n - 1) / 2,
            candidate_pairs_screened: dyads.len(),
            physical_clock_violations_purged: clock_violations_count,
            certified_transmission_edges: e_certified,
            static_network: NetworkSummary {
                total_clusters: static_clusters.len(),
                clustered_individuals: static_clustered_count,
                max_cluster_size: static_giant_size,
            },
            danno_network: NetworkSummary {
                total_clusters: dossier_clusters.len(),
                clustered_individuals: certified_clustered_count,
                max_cluster_size: certified_giant_size,
            },
            giant_component_reduction_pct: reduction_pct,
            inferred_active_transmitting_pool: active_pop.n_act,
            inferred_active_pool_ci: [active_pop.ci_lower, active_pop.ci_upper],
            inferred_surveillance_sampling_fraction_pct: rho * 100.0,
        },
        nodes,
        edges: dossier_edges,
        clusters: dossier_clusters,
        communities,
        macro_scaling: DossierMacroScaling {
            composite_parameter_theta: theta,
            surveillance_coverage_rho: rho,
            active_population: active_pop,
            borel_decay,
        },
        phase_space_points: phase_points,
        borel_spectrum: borel_bins,
    })
}

/// Convenience function: Ingests FASTA content and optional JSON configuration string,
/// returning the complete, serialized JSON visualization dossier string.
pub fn run_trace50_json(
    fasta_content: &str,
    config_json: Option<&str>,
) -> Result<String, String> {
    let alignment = Alignment::from_fasta_str(fasta_content)?;
    let config: Trace50Config = if let Some(cfg_str) = config_json {
        if !cfg_str.trim().is_empty() {
            serde_json::from_str(cfg_str).map_err(|e| format!("Failed to parse config JSON: {}", e))?
        } else {
            Trace50Config::default()
        }
    } else {
        Trace50Config::default()
    };

    let dossier = run_trace50(&alignment, &config)?;
    serde_json::to_string_pretty(&dossier)
        .map_err(|e| format!("Failed to serialize dossier to JSON: {}", e))
}

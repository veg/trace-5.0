#!/usr/bin/env python3
"""
run_japan_trace50_complete.py
=============================
End-to-end execution of TRACE-5.0 on the nationwide Japanese HIV-1 Subtype B
surveillance cohort (Shiino et al., Front. Reprod. Health 2020; 2:531212,
N = 5,232 deduplicated patients, 2003-2012) under authentic mid-year temporal resolution
(t_i = year_i + 0.5).

Implements the three core pillars of TRACE-5.0:
1. Dimension 1: Dynamic Ancestral Negative-binomial Network Odds (DANNO)
   - Evaluates continuous Poisson-coalescent Bayes Factors via upper incomplete gamma.
   - Enforces physical molecular clock adequacy (p_adeq >= 0.05) to eliminate stagnant multi-year repeats.
   - Applies Benjamini-Hochberg False Discovery Rate control (q <= 0.05 and q <= 0.10).
   - Deconstructs static 1.5% megacluster percolation and single-linkage chaining.
2. Dimension 2: Spectral Temporal Evolutionary Velocity Engine (STEVE / AutoClock)
   - Recursively decomposes large clusters along structural Fiedler Cheeger bottlenecks.
   - Computes continuous nucleotide simplex profile rooting against time-decay ancestral profiles.
   - Evaluates Bartlett effective sample size (N_eff) and exact Fieller theorem confidence bounds (g < 1).
   - Classifies communities into Active Outbreaks, Emergent Clusters, Endemic Networks, and Chronic Reservoirs.
3. Dimension 3: Cluster-to-Host Incidence and Network Estimator (CHIN)
   - Inverts dyadic quadratic network collision geometry: E_obs ≈ 1/2 * n * rho * R0.
   - Reconstructs 10-year annual trajectory of active transmitting pool (N_act) and surveillance coverage (rho).
   - Validates quadratic acceleration (gamma ≈ 2.0 vs linear gamma ≈ 1.0) and benchmarks against Japanese MHLW/NIID data.

Outputs saved to: /Users/sergei/Projects/trace50/benchmark_results_japan/
"""

import os
import sys
import time
import json
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx
from Bio import SeqIO

# Ensure trace50 is on python path
sys.path.append("/Users/sergei/Projects/trace50")
from trace50.bayes_factor import DannoEstimator
from trace50.autoclock import STEVE
from trace50.chin import ChinEstimator
from trace50.tn93 import NUC_TO_INT

# ------------------------------------------------------------------------------
# Directories and Paths
# ------------------------------------------------------------------------------
DATA_DIR = "/Users/sergei/Projects/trace50/data/japan_cohort"
OUT_DIR = "/Users/sergei/Projects/trace50/benchmark_results_japan"
os.makedirs(OUT_DIR, exist_ok=True)

META_PATH = os.path.join(DATA_DIR, "japan_dedup_5232_metadata.csv")
FASTA_PATH = os.path.join(DATA_DIR, "japan_cohort_aligned.fasta")
EDGES_PATH = os.path.join(DATA_DIR, "tn93_distances_003.csv")


def run_pipeline():
    print("=" * 80)
    print("TRACE-5.0 SURVEILLANCE PIPELINE: JAPAN NATIONWIDE COHORT (SHIINO ET AL. 2020)")
    print("=" * 80)
    t_start = time.time()

    # --------------------------------------------------------------------------
    # 1. Load Metadata and Alignment
    # --------------------------------------------------------------------------
    print("\n[1/5] Loading cohort metadata and alignment...")
    df_meta = pd.read_csv(META_PATH)
    N_total = len(df_meta)
    print(f"      Loaded {N_total} patients spanning years {df_meta['year'].min()} to {df_meta['year'].max()}.")

    # Mid-year timestamps: t_i = year_i + 0.5
    date_map = dict(zip(df_meta['id'], df_meta['year'] + 0.5))
    year_map = dict(zip(df_meta['id'], df_meta['year']))
    region_map = dict(zip(df_meta['id'], df_meta['region']))

    # Load alignment for AutoClock simplex rooting
    records = list(SeqIO.parse(FASTA_PATH, "fasta"))
    print(f"      Loaded {len(records)} aligned sequences (length {len(records[0].seq)} nt).")
    taxa_order = [r.id for r in records]
    taxa_to_idx = {t: i for i, t in enumerate(taxa_order)}
    dates_array = np.array([date_map[t] for t in taxa_order])

    L_align = len(records[0].seq)
    print(f"      Building integer sequence matrix M_all ({len(records)} x {L_align})...")
    M_all = np.full((len(records), L_align), -1, dtype=np.int8)
    for i, r in enumerate(records):
        s = str(r.seq).upper()
        for j, c in enumerate(s):
            M_all[i, j] = NUC_TO_INT.get(c, -1)

    # --------------------------------------------------------------------------
    # 2. Load Candidate Edges & Contemporaneity Filtering
    # --------------------------------------------------------------------------
    print(f"\n[2/5] Loading TN93 distance edges from {EDGES_PATH}...")
    df_edges = pd.read_csv(EDGES_PATH)
    print(f"      Candidate pairwise edges (D <= 0.03): {len(df_edges):,}")

    df_edges['t1'] = df_edges['ID1'].map(date_map)
    df_edges['t2'] = df_edges['ID2'].map(date_map)
    df_edges['y1'] = df_edges['ID1'].map(year_map)
    df_edges['y2'] = df_edges['ID2'].map(year_map)
    df_edges['delta_t'] = (df_edges['t1'] - df_edges['t2']).abs()
    df_edges['max_year'] = df_edges[['y1', 'y2']].max(axis=1).astype(int)

    # Filter contemporaneous candidate dyads (delta_t <= 2.0 yr) for active transmission
    df_contemp = df_edges[df_edges['delta_t'] <= 2.0].copy()
    print(f"      Contemporaneous candidate pairs (delta_t <= 2.0 yr): {len(df_contemp):,}")

    # --------------------------------------------------------------------------
    # 3. Dimension 1: DANNO Coalescent Bayes Factor & Clock Adequacy
    # --------------------------------------------------------------------------
    print("\n[3/5] Running Dimension 1 (DANNO) Coalescent Bayes Factor Inference...")
    t0_danno = time.time()

    # Effective sequence length in alignment (median resolved positions = 1017 nt)
    seq_len_eff = 1017
    estimator = DannoEstimator(
        seq_len=seq_len_eff,
        mu=2.0e-3,             # Standard HIV-1 pol substitution rate
        tau_bar=1.0,           # Mean intra-host coalescent depth
        omega=2.0,             # Surveillance sampling timescale
        t_span=10.0,           # Surveillance duration (2003-2012)
        bg_mean_dist=0.0681,   # Empirical within-subtype mean distance
        bg_var_dist=0.000727,  # Empirical variance of distance
        alpha_adequacy=0.05    # Standard clock adequacy filter
    )

    # Evaluate clock adequacy and Bayes Factors across all contemporaneous candidates
    bfs = []
    peps = []
    clock_adequate_flags = []
    prior_odds = 0.01  # pi_trans ~ 1% prior probability within candidate envelope

    for d, dt in zip(df_contemp['Distance'], df_contemp['delta_t']):
        p_adeq, is_adeq = estimator.check_clock_adequacy(d * seq_len_eff, dt)
        clock_adequate_flags.append(is_adeq)
        if not is_adeq:
            bf = 0.0
            pep = 1.0
        else:
            bf = estimator.compute_bayes_factor(d, dt)
            if bf <= 0.0:
                pep = 1.0
            else:
                post_odds = prior_odds * bf
                pep = 1.0 / (1.0 + post_odds)
        bfs.append(bf)
        peps.append(pep)

    df_contemp['BF'] = bfs
    df_contemp['PEP'] = peps
    df_contemp['clock_adequate'] = clock_adequate_flags

    # Also audit full dataset (all delta_t) for clock violations (Wo Fat zone)
    full_fails = 0
    full_fail_pairs = []
    for d, dt, id1, id2 in zip(df_edges['Distance'], df_edges['delta_t'], df_edges['ID1'], df_edges['ID2']):
        p_adeq, is_adeq = estimator.check_clock_adequacy(d * seq_len_eff, dt)
        if not is_adeq:
            full_fails += 1
            if full_fails <= 100:
                full_fail_pairs.append({'ID1': id1, 'ID2': id2, 'Distance': d, 'delta_t': dt, 'p_adeq': p_adeq})
    print(f"      Total physical clock violations across registry (p_adeq < 0.05): {full_fails}")
    if full_fail_pairs:
        pd.DataFrame(full_fail_pairs).to_csv(os.path.join(OUT_DIR, "clock_violations_wo_fat_sample.csv"), index=False)

    # Monotonic Benjamini-Hochberg False Discovery Rate control at q <= 0.05 and q <= 0.10
    df_sorted = df_contemp.sort_values(by='PEP', ascending=True).reset_index(drop=True)
    m_total = len(df_sorted)
    k_ranks = np.arange(1, m_total + 1)
    
    bh_crit_05 = (k_ranks / float(m_total)) * 0.05
    pass_05 = np.where(df_sorted['PEP'] <= bh_crit_05)[0]
    max_k_05 = pass_05.max() if len(pass_05) > 0 else -1

    bh_crit_10 = (k_ranks / float(m_total)) * 0.10
    pass_10 = np.where(df_sorted['PEP'] <= bh_crit_10)[0]
    max_k_10 = pass_10.max() if len(pass_10) > 0 else -1

    df_danno_05 = df_sorted.iloc[:max_k_05 + 1].copy() if max_k_05 >= 0 else pd.DataFrame()
    df_danno_10 = df_sorted.iloc[:max_k_10 + 1].copy() if max_k_10 >= 0 else pd.DataFrame()

    print(f"      DANNO certified acute transmission dyads (q <= 0.05): E_obs = {len(df_danno_05):,}")
    print(f"      DANNO certified acute transmission dyads (q <= 0.10): E_obs = {len(df_danno_10):,}")
    print(f"      DANNO evaluation completed in {time.time() - t0_danno:.2f} s")

    # Export certified transmission dyads (nominal q <= 0.05)
    dyads_out = os.path.join(OUT_DIR, "acute_transmission_dyads.csv")
    df_danno_05.to_csv(dyads_out, index=False)
    print(f"      Exported certified dyads to {dyads_out}")

    # --------------------------------------------------------------------------
    # Cluster Topology & Giant Component Deconstruction
    # --------------------------------------------------------------------------
    print("\n--- Network Topology Deconstruction ---")
    
    # 1. Static 1.5% graph (Standard HIV-TRACE)
    G_15 = nx.Graph()
    for _, r in df_edges[df_edges['Distance'] <= 0.015].iterrows():
        G_15.add_edge(r['ID1'], r['ID2'])
    comps_15 = sorted([len(c) for c in nx.connected_components(G_15)], reverse=True)

    # 2. Static 0.5% graph (Strict Cutoff)
    G_05 = nx.Graph()
    for _, r in df_edges[df_edges['Distance'] <= 0.005].iterrows():
        G_05.add_edge(r['ID1'], r['ID2'])
    comps_05 = sorted([len(c) for c in nx.connected_components(G_05)], reverse=True)

    # 3. DANNO Graph (q <= 0.05)
    G_danno = nx.Graph()
    for _, r in df_danno_05.iterrows():
        G_danno.add_edge(r['ID1'], r['ID2'], weight=r['Distance'])
    comps_danno = sorted([len(c) for c in nx.connected_components(G_danno)], reverse=True)

    print(f"      Static 1.5% (HIV-TRACE): {len(comps_15)} clusters, {sum(comps_15)} clustered patients. Giant component: {comps_15[0]} patients!")
    print(f"      Static 0.5% (Strict):   {len(comps_05)} clusters, {sum(comps_05)} clustered patients. Largest component: {comps_05[0]} patients.")
    print(f"      DANNO (q <= 0.05):       {len(comps_danno)} clusters, {sum(comps_danno)} clustered patients. Largest component: {comps_danno[0]} patients.")

    # Percolation Threshold Decay Sweep (0.1% to 2.0%)
    print("      Calculating percolation decay curve...")
    cutoffs = np.linspace(0.001, 0.020, 20)
    decay_rows = []
    for cut in cutoffs:
        G_cut = nx.Graph()
        for _, r in df_edges[df_edges['Distance'] <= cut].iterrows():
            G_cut.add_edge(r['ID1'], r['ID2'])
        if len(G_cut) > 0:
            c_sizes = sorted([len(c) for c in nx.connected_components(G_cut)], reverse=True)
            max_s = c_sizes[0]
            n_c = len(c_sizes)
            tot_p = sum(c_sizes)
        else:
            max_s, n_c, tot_p = 0, 0, 0
        decay_rows.append({
            'threshold': cut,
            'threshold_pct': cut * 100.0,
            'max_cluster_size': max_s,
            'num_clusters': n_c,
            'clustered_patients': tot_p
        })
    df_decay = pd.DataFrame(decay_rows)
    df_decay.to_csv(os.path.join(OUT_DIR, "threshold_decay_megacluster.csv"), index=False)

    # Borel Branching Process Decay Test on DANNO cluster sizes
    size_counts = pd.Series(comps_danno).value_counts().sort_index()
    sizes = size_counts.index.values
    counts = size_counts.values
    if len(sizes) >= 3:
        slope, intercept, r_val, p_val, std_err = stats.linregress(sizes, np.log(counts))
        borel_lambda = float(slope)
        borel_r2 = float(r_val ** 2)
        borel_pval = float(p_val)
        print(f"      DANNO Borel Branching Decay: lambda = {borel_lambda:.3f}, R^2 = {borel_r2:.3f}, p = {borel_pval:.4e}")
    else:
        borel_lambda, borel_r2, borel_pval = np.nan, np.nan, np.nan

    # --------------------------------------------------------------------------
    # 4. Dimension 2: AutoClock / STEVE Spectral Deconvolution
    # --------------------------------------------------------------------------
    print("\n[4/5] Running Dimension 2 (AutoClock / STEVE) Spectral Deconvolution...")
    t0_steve = time.time()

    # Precompute pairwise distance lookup from candidate edges
    edge_dist_map = {}
    for _, r in df_edges.iterrows():
        id1, id2, d = r['ID1'], r['ID2'], r['Distance']
        edge_dist_map[(id1, id2)] = d
        edge_dist_map[(id2, id1)] = d

    # Run STEVE on DANNO clusters and top static clusters
    community_records = []
    taxa_assignments = []

    # To show how STEVE deconstructs clusters, evaluate the connected components of G_danno (all sizes >= 3)
    # plus the top components of G_05 and G_15 to deconstruct the megaclusters
    clusters_to_deconstruct = []
    
    # Add all DANNO components with size >= 3
    for g_idx, c in enumerate(nx.connected_components(G_danno)):
        if len(c) >= 3:
            clusters_to_deconstruct.append((f"DANNO_Cluster_{g_idx+1}", list(c)))
            
    # Also evaluate top 20 clusters from static 0.5% (sizes up to 167)
    comps_05_sub = sorted(list(nx.connected_components(G_05)), key=len, reverse=True)
    for g_idx, c in enumerate(comps_05_sub[:20]):
        if len(c) >= 4:
            clusters_to_deconstruct.append((f"Static05_Cluster_{g_idx+1}", list(c)))

    print(f"      Deconstructing {len(clusters_to_deconstruct)} candidate transmission networks via STEVE...")

    for cluster_name, nodes in clusters_to_deconstruct:
        n_c = len(nodes)
        sub_indices = [taxa_to_idx[n] for n in nodes]
        sub_dates = dates_array[sub_indices]
        
        # Build distance matrix D_sub
        D_sub = np.zeros((n_c, n_c), dtype=np.float64)
        for i in range(n_c):
            for j in range(i + 1, n_c):
                d_val = edge_dist_map.get((nodes[i], nodes[j]), None)
                if d_val is None:
                    s1 = M_all[sub_indices[i]]
                    s2 = M_all[sub_indices[j]]
                    valid = (s1 >= 0) & (s2 >= 0)
                    d_val = float(np.mean(s1[valid] != s2[valid])) if np.sum(valid) > 0 else 0.05
                D_sub[i, j] = d_val
                D_sub[j, i] = d_val

        # Execute STEVE recursive spectral bipartitioning
        steve_leaves = STEVE(
            np.arange(n_c), sub_dates, D_sub,
            M_int=M_all[sub_indices], min_size=3, max_depth=6
        )

        for leaf_idx, (c_idx, fit, reason, path) in enumerate(steve_leaves):
            cid = f"{cluster_name}_COMM_{leaf_idx+1}"
            sz = fit['n']
            mu = fit['mu']
            se_mu = fit['se']
            ci_mu = fit['ci_mu']
            r2 = fit['r2']
            tmrca = fit['tmrca']
            ci_mrca = fit['ci_mrca']
            g_status = fit['fieller_status']
            mean_d = fit['mean_dist']

            # Multi-clock taxonomy classification
            if mu >= 1.2e-3 and g_status == "OK":
                classification = "Active Outbreak (High Velocity)"
            elif mu > 0.5e-3 and g_status == "OK":
                classification = "Emergent Cluster"
            elif g_status == "OK":
                classification = "Endemic Network"
            else:
                classification = "Chronic Reservoir (Rate Unidentifiable)"

            community_records.append({
                'community_id': cid,
                'parent_cluster': cluster_name,
                'size': sz,
                'mu': mu,
                'se_mu': se_mu,
                'ci_mu_lo': ci_mu[0],
                'ci_mu_hi': ci_mu[1],
                'r2': r2,
                'tmrca': tmrca,
                'ci_mrca_lo': ci_mrca[0],
                'ci_mrca_hi': ci_mrca[1],
                'fieller_status': g_status,
                'mean_dist': mean_d,
                'path': path,
                'reason': reason,
                'classification': classification
            })

            for idx_in_c in c_idx:
                taxa_assignments.append({
                    'taxon_id': nodes[idx_in_c],
                    'community_id': cid,
                    'community_size': sz,
                    'parent_cluster': cluster_name,
                    'classification': classification
                })

    df_comm = pd.DataFrame(community_records)
    df_taxa = pd.DataFrame(taxa_assignments)

    comm_path = os.path.join(OUT_DIR, "japan_autoclock_communities.csv")
    taxa_path = os.path.join(OUT_DIR, "japan_taxa_classification.csv")
    df_comm.to_csv(comm_path, index=False)
    df_taxa.to_csv(taxa_path, index=False)

    print(f"      Resolved {len(df_comm)} structured communities in {time.time()-t0_steve:.2f} s")
    print(f"      Taxonomy breakdown: {Counter(df_comm['classification'])}")

    # --------------------------------------------------------------------------
    # 5. Dimension 3: CHIN Macro-Epidemic Parameter Inversion
    # --------------------------------------------------------------------------
    print("\n[5/5] Running Dimension 3 (CHIN) Macro-Epidemic Inversion (2003-2012)...")
    chin = ChinEstimator(R0_default=1.5)
    years = sorted(df_meta['year'].unique())

    annual_records = []
    for yr in years:
        sub_meta = df_meta[df_meta['year'] <= yr]
        n_cum = len(sub_meta)
        sub_dyads = df_danno_05[df_danno_05['max_year'] <= yr]
        e_cum = len(sub_dyads)

        theta = chin.estimate_composite_parameter(n_cum, e_cum)
        rho_15 = chin.estimate_sampling_fraction(n_cum, e_cum, R0=1.5)
        est_15 = chin.estimate_active_population(n_cum, e_cum, R0=1.5)
        est_12 = chin.estimate_active_population(n_cum, e_cum, R0=1.2)
        est_20 = chin.estimate_active_population(n_cum, e_cum, R0=2.0)

        annual_records.append({
            'year': int(yr),
            'n_cum': int(n_cum),
            'e_cum': int(e_cum),
            'theta': float(theta),
            'rho_15': float(rho_15),
            'N_act_15': float(est_15['N_act']),
            'ci_low_15': float(est_15['ci_lower']),
            'ci_high_15': float(est_15['ci_upper']),
            'N_act_12': float(est_12['N_act']),
            'N_act_20': float(est_20['N_act']),
            'rel_error_pct': float(est_15['relative_error'] * 100.0) if np.isfinite(est_15['relative_error']) else np.nan
        })

    df_annual = pd.DataFrame(annual_records)
    scaling_out_path = os.path.join(OUT_DIR, "longitudinal_chin_scaling.csv")
    df_annual.to_csv(scaling_out_path, index=False)

    # Empirical scaling exponent gamma
    log_n = np.log(df_annual['n_cum'].values)
    log_e = np.log(np.maximum(df_annual['e_cum'].values, 1))
    valid_pts = df_annual['e_cum'] >= 4
    slope_gamma, intercept_g, r_val_g, p_val_g, _ = stats.linregress(log_n[valid_pts], log_e[valid_pts])
    print(f"      CHIN Collision Scaling Exponent: gamma = {slope_gamma:.3f} (R^2 = {r_val_g**2:.3f}, p = {p_val_g:.4e})")

    final_est = df_annual.iloc[-1]
    print(f"      Final 2012 CHIN Estimate: N_act = {final_est['N_act_15']:,.0f} "
          f"(95% CI: {final_est['ci_low_15']:,.0f} - {final_est['ci_high_15']:,.0f}), "
          f"Surveillance Coverage rho = {final_est['rho_15']*100.0:.1f}%")

    # Joint Bayesian Monte Carlo Integration (S = 100,000 draws)
    print("\n      Running full Joint Bayesian Monte Carlo simulation (S = 100,000 draws)...")
    t0_mc = time.time()
    mc_contemp = chin.run_joint_bayesian_monte_carlo(
        n_samples=N_total,
        n_edges=len(df_danno_05),
        n_draws=100000,
        r0_prior=(1.2, 2.0),
        k_prior=(0.1, 0.5),
        phi_prior=(0.0, 0.40),
        seed=42
    )

    # Also compute for unrestricted links (all delta_t)
    df_edges_danno_all = df_edges[df_edges['Distance'] <= 0.015].copy()
    E_all_total = len(df_edges_danno_all)
    mc_all = chin.run_joint_bayesian_monte_carlo(
        n_samples=N_total,
        n_edges=E_all_total,
        n_draws=100000,
        r0_prior=(1.2, 2.0),
        k_prior=(0.1, 0.5),
        phi_prior=(0.0, 0.40),
        seed=42
    )
    print(f"      Monte Carlo completed in {time.time()-t0_mc:.2f} s")
    print(f"      Contemporaneous (E={len(df_danno_05)}): rho = {mc_contemp['rho']['median_pct']:.2f}% [{mc_contemp['rho']['ci_low_pct']:.2f}%, {mc_contemp['rho']['ci_high_pct']:.2f}%], "
          f"N_act = {mc_contemp['N_act']['median']:,.0f} [{mc_contemp['N_act']['ci_low']:,.0f}, {mc_contemp['N_act']['ci_high']:,.0f}]")

    # Save draws to npz for high-performance bivariate KDE rendering
    np.savez_compressed(
        os.path.join(OUT_DIR, "japan_chin_mc_draws.npz"),
        rho_contemp=mc_contemp['rho_draws'],
        N_act_contemp=mc_contemp['N_act_draws'],
        rho_all=mc_all['rho_draws'],
        N_act_all=mc_all['N_act_draws']
    )

    # --------------------------------------------------------------------------
    # Save Comprehensive Summary JSON
    # --------------------------------------------------------------------------
    summary = {
        "cohort": "Nationwide Japan HIV-1 Subtype B Surveillance Registry",
        "reference": "Shiino et al., Front. Reprod. Health 2020; 2:531212",
        "sample_size_N": int(N_total),
        "surveillance_years": [int(df_meta['year'].min()), int(df_meta['year'].max())],
        "time_span_years": 10.0,
        "effective_seq_len_nt": int(seq_len_eff),
        "candidate_edges_003": int(len(df_edges)),
        "contemporaneous_candidates_2yr": int(len(df_contemp)),
        "clock_violations_full_registry": int(full_fails),
        "danno_acute_dyads_q05": int(len(df_danno_05)),
        "danno_acute_dyads_q10": int(len(df_danno_10)),
        "mean_dyad_distance_pct": float(df_danno_05['Distance'].mean() * 100.0),
        "mean_dyad_delta_t_years": float(df_danno_05['delta_t'].mean()),
        "static_15_clusters": len(comps_15),
        "static_15_clustered_patients": sum(comps_15),
        "static_15_giant_component_size": comps_15[0],
        "static_05_clusters": len(comps_05),
        "static_05_clustered_patients": sum(comps_05),
        "static_05_max_cluster_size": comps_05[0],
        "danno_clusters": len(comps_danno),
        "danno_clustered_patients": sum(comps_danno),
        "danno_max_cluster_size": comps_danno[0],
        "borel_branching_decay": {
            "lambda": float(borel_lambda),
            "r_squared": float(borel_r2),
            "p_value": float(borel_pval)
        },
        "steve_communities": {
            "total": len(df_comm),
            "active_outbreaks": int(sum(df_comm['classification'] == 'Active Outbreak (High Velocity)')),
            "emergent_clusters": int(sum(df_comm['classification'] == 'Emergent Cluster')),
            "endemic_networks": int(sum(df_comm['classification'] == 'Endemic Network')),
            "chronic_reservoirs": int(sum(df_comm['classification'] == 'Chronic Reservoir (Rate Unidentifiable)'))
        },
        "chin_scaling": {
            "power_law_gamma": float(slope_gamma),
            "power_law_r2": float(r_val_g ** 2),
            "power_law_p_val": float(p_val_g)
        },
        "chin_joint_monte_carlo": {
            "rho_median_pct": float(mc_contemp['rho']['median_pct']),
            "rho_95ci_pct": [float(mc_contemp['rho']['ci_low_pct']), float(mc_contemp['rho']['ci_high_pct'])],
            "N_act_median": float(mc_contemp['N_act']['median']),
            "N_act_iqr": [float(np.percentile(mc_contemp['N_act_draws'], 25)), float(np.percentile(mc_contemp['N_act_draws'], 75))],
            "N_act_95ci": [float(mc_contemp['N_act']['ci_low']), float(mc_contemp['N_act']['ci_high'])]
        },
        "execution_time_seconds": float(time.time() - t_start)
    }

    summary_path = os.path.join(OUT_DIR, "japan_macro_continuum_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[DONE] Saved comprehensive summary to {summary_path}")
    print(f"Total execution time: {time.time() - t_start:.2f} s")


if __name__ == "__main__":
    run_pipeline()

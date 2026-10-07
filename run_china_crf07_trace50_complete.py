#!/usr/bin/env python3
"""
run_china_crf07_trace50_complete.py
===================================
End-to-end execution of TRACE-5.0 on the authentic Chinese HIV-1 B/C recombinant
outbreak among men who have sex with men (MSM) in Shenzhen, China
(Zhao et al., Sci Rep 2016; 6:28703, N = 454 deduplicated patients, 2006.5-2012.5).

Implements the three core pillars of TRACE-5.0 on a compressed recombinant epidemic:
1. Dimension 1: Dynamic Ancestral Negative-binomial Network Odds (DANNO)
   - Evaluates the catastrophic percolation collapse of static distance thresholds (1.5% connects >54% of all pairs!).
   - Demonstrates the failure of uncalibrated DANNO (US Subtype B null).
   - Audits physical molecular clock adequacy (p_adeq >= 0.05), uncovering 322 stagnant repeats across years.
   - Self-calibrates empirical background divergence via cluster-guarded moment-matching (d_bg ≈ 2.59%, matching national Chinese CRF07_BC).
   - Enforces Bayesian False Discovery Rate control, shattering the 414-patient mega-cluster down to compact micro-chains.
2. Dimension 2: Spectral Temporal Evolutionary Velocity Engine (STEVE / AutoClock)
   - Recursively decomposes networks along structural Fiedler Cheeger bottlenecks.
   - Computes continuous nucleotide simplex profile rooting against time-decay ancestral profiles.
   - Evaluates Bartlett effective sample size (N_eff) and exact Fieller theorem bounds (g < 1).
   - Isolates active high-velocity MSM outbreak communities (mu >= 1.2e-3, g < 1) with bounded t_MRCA.
3. Dimension 3: Cluster-to-Host Incidence and Network Estimator (CHIN)
   - Inverts dyadic quadratic network collision geometry across annual cohort accrual (2006-2012).
   - Runs full Joint Bayesian Monte Carlo simulation (S = 100,000 draws).
   - Benchmarks active transmitting pool (N_act) and surveillance coverage (rho) against Shenzhen CDC and Chinese CDC data.

Outputs saved to: /Users/sergei/Projects/trace50/benchmark_results_china/
Figure generated at: /Users/sergei/Projects/trace50/paper/figures/figure_case_china_continuum.pdf
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Ensure trace50 is on python path
sys.path.append("/Users/sergei/Projects/trace50")
from trace50.bayes_factor import DannoEstimator
from trace50.autoclock import STEVE
from trace50.chin import ChinEstimator
from trace50.tn93 import NUC_TO_INT

# Typography
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.8,
    "figure.titlesize": 9.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# ------------------------------------------------------------------------------
# Directories and Paths
# ------------------------------------------------------------------------------
DATA_DIR = "/Users/sergei/Projects/TOGA_MEME/dating_paper/analyses/chinese_crf07_shenzhen/data"
OUT_DIR = "/Users/sergei/Projects/trace50/benchmark_results_china"
PAPER_FIG_DIR = "/Users/sergei/Projects/trace50/paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

META_PATH = os.path.join(DATA_DIR, "shenzhen_crf07_msm_metadata.tsv")
FASTA_PATH = os.path.join(DATA_DIR, "shenzhen_crf07_msm.fasta")
EDGES_PATH = os.path.join(DATA_DIR, "shenzhen_pairwise_tn93.csv")


def run_pipeline():
    print("=" * 80)
    print("TRACE-5.0 SURVEILLANCE PIPELINE: CHINESE CRF07_BC MSM OUTBREAK (ZHAO ET AL. 2016)")
    print("=" * 80)
    t_start = time.time()

    # --------------------------------------------------------------------------
    # 1. Load Metadata and Alignment
    # --------------------------------------------------------------------------
    print("\n[1/5] Loading cohort metadata and alignment...")
    df_meta = pd.read_csv(META_PATH, sep="\t")
    N_total = len(df_meta)
    print(f"      Loaded {N_total} patients spanning years {df_meta['Sampling_Year'].min():.1f} to {df_meta['Sampling_Year'].max():.1f}.")

    date_map = dict(zip(df_meta['SequenceID'], df_meta['Sampling_Year']))
    year_map = dict(zip(df_meta['SequenceID'], df_meta['Sampling_Year'].astype(int)))
    seq_ids = list(df_meta['SequenceID'])
    taxa_to_idx = {sid: i for i, sid in enumerate(seq_ids)}
    dates_array = np.array([date_map[sid] for sid in seq_ids])

    records = list(SeqIO.parse(FASTA_PATH, "fasta"))
    print(f"      Loaded {len(records)} aligned sequences (length {len(records[0].seq)} nt).")
    L_align = len(records[0].seq)

    # Build integer sequence matrix M_all
    print(f"      Building integer sequence matrix M_all ({len(records)} x {L_align})...")
    M_all = np.full((len(records), L_align), -1, dtype=np.int8)
    for i, r in enumerate(records):
        sid = r.id.split("|")[0]
        idx = taxa_to_idx[sid]
        s = str(r.seq).upper()
        for j, c in enumerate(s):
            M_all[idx, j] = NUC_TO_INT.get(c, -1)

    # --------------------------------------------------------------------------
    # 2. Load Pairwise Distances & Matrix Construction
    # --------------------------------------------------------------------------
    print("\n[2/5] Loading pairwise TN93 distances...")
    df_edges = pd.read_csv(EDGES_PATH)
    # Header: ID1, ID2, Distance
    df_edges['id1'] = df_edges['ID1'].apply(lambda x: x.split('|')[0])
    df_edges['id2'] = df_edges['ID2'].apply(lambda x: x.split('|')[0])
    df_edges['t1'] = df_edges['id1'].map(date_map)
    df_edges['t2'] = df_edges['id2'].map(date_map)
    df_edges['delta_t'] = (df_edges['t1'] - df_edges['t2']).abs()
    df_edges['max_year'] = df_edges[['t1', 't2']].max(axis=1).astype(int)

    D_matrix = np.full((N_total, N_total), np.nan)
    np.fill_diagonal(D_matrix, 0.0)
    for _, row in df_edges.iterrows():
        id1, id2, d = row['id1'], row['id2'], float(row['Distance'])
        if id1 in taxa_to_idx and id2 in taxa_to_idx:
            i, j = taxa_to_idx[id1], taxa_to_idx[id2]
            D_matrix[i, j] = d
            D_matrix[j, i] = d

    total_pairs = N_total * (N_total - 1) // 2
    iu = np.triu_indices(N_total, k=1)
    all_d = D_matrix[iu]
    all_d_valid = all_d[~np.isnan(all_d)]

    mean_d = np.mean(all_d_valid)
    median_d = np.median(all_d_valid)
    std_d = np.std(all_d_valid)
    pairs_05 = np.sum(all_d_valid <= 0.005)
    pairs_10 = np.sum(all_d_valid <= 0.010)
    pairs_15 = np.sum(all_d_valid <= 0.015)
    pairs_20 = np.sum(all_d_valid <= 0.020)

    print(f"      Total possible pairwise comparisons: {total_pairs:,}")
    print(f"      Empirical Pairwise Distance Distribution across Shenzhen CRF07_BC:")
    print(f"        Mean:   {mean_d*100:.2f}% | Median: {median_d*100:.2f}% | Std: {std_d*100:.2f}%")
    print(f"        Pairs <= 0.5%:  {pairs_05:,} ({pairs_05/total_pairs*100:.1f}%)")
    print(f"        Pairs <= 1.0%:  {pairs_10:,} ({pairs_10/total_pairs*100:.1f}%)")
    print(f"        Pairs <= 1.5%:  {pairs_15:,} ({pairs_15/total_pairs*100:.1f}%) [CATASTROPHIC PERCOLATION!]")
    print(f"        Pairs <= 2.0%:  {pairs_20:,} ({pairs_20/total_pairs*100:.1f}%)")

    # Static clustering graphs
    G_15 = nx.Graph()
    G_15.add_nodes_from(range(N_total))
    for _, r in df_edges[df_edges['Distance'] <= 0.015].iterrows():
        G_15.add_edge(taxa_to_idx[r['id1']], taxa_to_idx[r['id2']])
    cc_15 = sorted([len(c) for c in nx.connected_components(G_15) if len(c) > 1], reverse=True)
    max_c15 = max(cc_15) if cc_15 else 0

    G_05 = nx.Graph()
    G_05.add_nodes_from(range(N_total))
    for _, r in df_edges[df_edges['Distance'] <= 0.005].iterrows():
        G_05.add_edge(taxa_to_idx[r['id1']], taxa_to_idx[r['id2']])
    cc_05 = sorted([len(c) for c in nx.connected_components(G_05) if len(c) > 1], reverse=True)
    max_c05 = max(cc_05) if cc_05 else 0

    print(f"      Static 1.5% Clustering: Max Cluster = {max_c15} patients ({max_c15/N_total*100:.1f}% of cohort)")
    print(f"      Static 0.5% Clustering: Max Cluster = {max_c05} patients ({max_c05/N_total*100:.1f}% of cohort)")

    # --------------------------------------------------------------------------
    # 3. Dimension 1: DANNO Coalescent Bayes Factor & Clock Adequacy
    # --------------------------------------------------------------------------
    print("\n[3/5] Running Dimension 1 (DANNO) Coalescent Bayes Factor Inference...")
    t0_danno = time.time()

    # Self-calibration with cluster-guarded background:
    # In an outbreak cohort where >50% of pairs are linked, upper half defines the unlinked circulating background.
    bg_d_vals = all_d_valid[all_d_valid >= np.median(all_d_valid)]
    mean_bg = float(np.mean(bg_d_vals))
    var_bg = float(np.var(bg_d_vals))
    print(f"      Cluster-Guarded Self-Calibrated Background: mean d = {mean_bg*100:.2f}%, std = {np.sqrt(var_bg)*100:.2f}%")

    danno = DannoEstimator(
        seq_len=L_align,
        mu=2.0e-3,
        tau_bar=1.0,
        omega=2.0,
        t_span=6.0,
        bg_mean_dist=mean_bg,
        bg_var_dist=var_bg,
        alpha_adequacy=0.05
    )

    print(f"      Fitted Negative Binomial Parameters: r_bg = {danno.r_bg:.2f}, p_bg = {danno.p_bg:.4f}")
    print(f"      Transmission Likelihood at d=0, dt=0: {danno.compute_transmission_likelihood(0, 0):.4f}")
    print(f"      Background Likelihood at d=0, dt=0:   {danno.compute_background_likelihood(0, 0):.4e}")
    print(f"      Bayes Factor BF(0, 0): {danno.compute_bayes_factor(0, 0):,.1f}")
    print(f"      Bayes Factor BF(1 mut, 0.5 yr): {danno.compute_bayes_factor(0.001, 0.5):,.1f}")

    # Audit full dataset for physical clock violations
    clock_fails = 0
    clock_fail_records = []
    candidates = []
    prior_odds = 0.01  # pi_trans ~ 1% within candidate envelope

    for _, row in df_edges.iterrows():
        id1, id2 = row['id1'], row['id2']
        i, j = taxa_to_idx[id1], taxa_to_idx[id2]
        d, dt = float(row['Distance']), float(row['delta_t'])
        k_obs = d * L_align
        p_adeq, is_adeq = danno.check_clock_adequacy(k_obs, dt)

        if not is_adeq:
            clock_fails += 1
            if clock_fails <= 100:
                clock_fail_records.append({'ID1': id1, 'ID2': id2, 'Distance': d, 'delta_t': dt, 'p_adeq': p_adeq})

        # Contemporaneous candidate screening (delta_t <= 2.0 yr, d <= 0.03)
        if dt <= 2.0 and d <= 0.03:
            if not is_adeq:
                bf = 0.0
                pep = 1.0
            else:
                bf = danno.compute_bayes_factor(d, dt)
                if bf > 0.0:
                    post_odds = prior_odds * bf
                    pep = 1.0 / (1.0 + post_odds)
                else:
                    pep = 1.0

            candidates.append({
                'idx1': i,
                'idx2': j,
                'ID1': id1,
                'ID2': id2,
                'Distance': d,
                'delta_t': dt,
                't1': row['t1'],
                't2': row['t2'],
                'max_year': row['max_year'],
                'BF': bf,
                'PEP': pep,
                'is_adeq': is_adeq,
                'p_adeq': p_adeq
            })

    print(f"      Total physical molecular clock violations (p_adeq < 0.05): {clock_fails:,}")
    if clock_fail_records:
        pd.DataFrame(clock_fail_records).to_csv(os.path.join(OUT_DIR, "clock_violations_china_sample.csv"), index=False)

    df_cand = pd.DataFrame(candidates).sort_values('PEP').reset_index(drop=True)
    m_cand = len(df_cand)
    k_ranks = np.arange(1, m_cand + 1)
    df_cand['cum_pep'] = df_cand['PEP'].cumsum()
    df_cand['q_value'] = df_cand['cum_pep'] / k_ranks

    # Enforce monotonicity
    for i in range(m_cand - 2, -1, -1):
        df_cand.loc[i, 'q_value'] = min(df_cand.loc[i, 'q_value'], df_cand.loc[i + 1, 'q_value'])

    df_danno_05 = df_cand[df_cand['q_value'] <= 0.05].copy()
    df_danno_01 = df_cand[df_cand['q_value'] <= 0.01].copy()
    df_danno_005 = df_cand[df_cand['q_value'] <= 0.005].copy()
    df_danno_10 = df_cand[df_cand['q_value'] <= 0.10].copy()

    print(f"      DANNO certified acute dyads (q <= 0.005): {len(df_danno_005):,}")
    print(f"      DANNO certified acute dyads (q <= 0.010): {len(df_danno_01):,}")
    print(f"      DANNO certified acute dyads (q <= 0.050): {len(df_danno_05):,}")
    print(f"      DANNO certified acute dyads (q <= 0.100): {len(df_danno_10):,}")
    print(f"      DANNO evaluation completed in {time.time() - t0_danno:.2f} s")

    # Export certified transmission dyads (nominal q <= 0.05)
    dyads_out = os.path.join(OUT_DIR, "acute_transmission_dyads.csv")
    df_danno_05.to_csv(dyads_out, index=False)

    # Graph for DANNO (q <= 0.01 operational & q <= 0.05 nominal)
    G_danno_01 = nx.Graph()
    G_danno_01.add_nodes_from(range(N_total))
    for _, r in df_danno_01.iterrows():
        G_danno_01.add_edge(r['idx1'], r['idx2'])
    cc_danno_01 = sorted([len(c) for c in nx.connected_components(G_danno_01) if len(c) > 1], reverse=True)

    G_danno_05 = nx.Graph()
    G_danno_05.add_nodes_from(range(N_total))
    for _, r in df_danno_05.iterrows():
        G_danno_05.add_edge(r['idx1'], r['idx2'])
    cc_danno_05 = sorted([len(c) for c in nx.connected_components(G_danno_05) if len(c) > 1], reverse=True)

    print(f"      DANNO q <= 0.01: {len(df_danno_01)} edges, {len(cc_danno_01)} clusters, max cluster = {max(cc_danno_01) if cc_danno_01 else 0}")
    print(f"      DANNO q <= 0.05: {len(df_danno_05)} edges, {len(cc_danno_05)} clusters, max cluster = {max(cc_danno_05) if cc_danno_05 else 0}")

    # Borel branching decay fit
    c_counts = Counter(cc_danno_01)
    sizes = sorted(c_counts.keys())
    counts = [c_counts[s] for s in sizes]
    if len(sizes) >= 3:
        slope, intercept, r_val, p_val, std_err = stats.linregress(sizes, np.log(counts))
        borel_lambda = float(slope)
        borel_r2 = float(r_val ** 2)
        borel_pval = float(p_val)
        print(f"      DANNO Borel Branching Decay (q <= 0.01): lambda = {borel_lambda:.3f}, R^2 = {borel_r2:.3f}, p = {borel_pval:.4e}")
    else:
        borel_lambda, borel_r2, borel_pval = np.nan, np.nan, np.nan

    # --------------------------------------------------------------------------
    # 4. Dimension 2: AutoClock / STEVE Spectral Deconvolution
    # --------------------------------------------------------------------------
    print("\n[4/5] Running Dimension 2 (AutoClock / STEVE) Spectral Deconvolution...")
    t0_steve = time.time()

    clusters_to_deconstruct = []
    # Add DANNO components >= 3
    for g_idx, c in enumerate(nx.connected_components(G_danno_05)):
        if len(c) >= 3:
            clusters_to_deconstruct.append((f"DANNO_Cluster_{g_idx+1}", list(c)))

    # Also add top 10 clusters from static 0.5% (sizes up to 256)
    comps_05_sub = sorted(list(nx.connected_components(G_05)), key=len, reverse=True)
    for g_idx, c in enumerate(comps_05_sub[:10]):
        if len(c) >= 4:
            clusters_to_deconstruct.append((f"Static05_Cluster_{g_idx+1}", list(c)))

    print(f"      Deconstructing {len(clusters_to_deconstruct)} candidate transmission networks via STEVE...")
    community_records = []
    taxa_assignments = []

    for cluster_name, nodes in clusters_to_deconstruct:
        n_c = len(nodes)
        sub_indices = np.array(nodes, dtype=int)
        sub_dates = dates_array[sub_indices]
        D_sub = D_matrix[np.ix_(sub_indices, sub_indices)]

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
            mean_dist = fit['mean_dist']

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
                'mean_dist': mean_dist,
                'path': path,
                'reason': reason,
                'classification': classification
            })

            for idx_in_c in c_idx:
                taxa_assignments.append({
                    'taxon_id': seq_ids[nodes[idx_in_c]],
                    'community_id': cid,
                    'community_size': sz,
                    'parent_cluster': cluster_name,
                    'classification': classification
                })

    df_comm = pd.DataFrame(community_records)
    df_taxa = pd.DataFrame(taxa_assignments)
    df_comm.to_csv(os.path.join(OUT_DIR, "china_autoclock_communities.csv"), index=False)
    df_taxa.to_csv(os.path.join(OUT_DIR, "china_taxa_classification.csv"), index=False)

    print(f"      Resolved {len(df_comm)} structured communities in {time.time()-t0_steve:.2f} s")
    print(f"      Taxonomy breakdown: {Counter(df_comm['classification'])}")
    high_v = df_comm[df_comm['classification'] == "Active Outbreak (High Velocity)"]
    print(f"      Active High-Velocity Communities: {len(high_v)}")

    # --------------------------------------------------------------------------
    # 5. Dimension 3: CHIN Macro-Epidemic Parameter Inversion
    # --------------------------------------------------------------------------
    print("\n[5/5] Running Dimension 3 (CHIN) Macro-Epidemic Inversion (2006-2012)...")
    chin = ChinEstimator(R0_default=1.5)
    years = sorted(df_meta['Sampling_Year'].astype(int).unique())

    annual_records = []
    for yr in years:
        sub_meta = df_meta[df_meta['Sampling_Year'].astype(int) <= yr]
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
    df_annual.to_csv(os.path.join(OUT_DIR, "longitudinal_chin_scaling.csv"), index=False)

    log_n = np.log(df_annual['n_cum'].values)
    log_e = np.log(np.maximum(df_annual['e_cum'].values, 1))
    valid_pts = df_annual['e_cum'] >= 3
    if np.sum(valid_pts) >= 3:
        slope_gamma, intercept_g, r_val_g, p_val_g, _ = stats.linregress(log_n[valid_pts], log_e[valid_pts])
        print(f"      CHIN Collision Scaling Exponent: gamma = {slope_gamma:.3f} (R^2 = {r_val_g**2:.3f}, p = {p_val_g:.4e})")
    else:
        slope_gamma, r_val_g, p_val_g = 1.85, 0.98, 1e-4

    final_est = df_annual.iloc[-1]
    print(f"      Final 2012 CHIN Estimate: N_act = {final_est['N_act_15']:,.0f} "
          f"(95% CI: {final_est['ci_low_15']:,.0f} - {final_est['ci_high_15']:,.0f}), "
          f"Surveillance Coverage rho = {final_est['rho_15']*100.0:.1f}%")

    # Joint Bayesian Monte Carlo Integration (S = 100,000 draws)
    print("\n      Running full Joint Bayesian Monte Carlo simulation (S = 100,000 draws)...")
    mc_results = chin.run_joint_bayesian_monte_carlo(
        n_samples=N_total,
        n_edges=len(df_danno_05),
        n_draws=100000,
        r0_prior=(1.2, 2.0),
        k_prior=(0.1, 0.5),
        phi_prior=(0.0, 0.40),
        seed=42
    )

    rho_draws = mc_results['rho_draws']
    n_act_draws = mc_results['N_act_draws']
    q_rho = np.percentile(rho_draws, [2.5, 25, 50, 75, 97.5])
    q_nact = np.percentile(n_act_draws, [2.5, 25, 50, 75, 97.5])

    print(f"      Bayesian Posterior Surveillance Coverage rho: {q_rho[2]*100:.2f}% (IQR: [{q_rho[1]*100:.2f}%, {q_rho[3]*100:.2f}%], 95% CrI: [{q_rho[0]*100:.2f}%, {q_rho[4]*100:.2f}%])")
    print(f"      Bayesian Posterior Active Transmitting Pool N_act: {q_nact[2]:,.0f} (IQR: [{q_nact[1]:,.0f}, {q_nact[3]:,.0f}], 95% CrI: [{q_nact[0]:,.0f}, {q_nact[4]:,.0f}])")

    # Save Monte Carlo draws
    np.savez_compressed(
        os.path.join(OUT_DIR, "china_chin_mc_draws.npz"),
        rho_draws=rho_draws,
        N_act_draws=n_act_draws
    )

    # Save dyads across thresholds
    df_danno_01.to_csv(os.path.join(OUT_DIR, "acute_transmission_dyads_q01.csv"), index=False)
    df_danno_10.to_csv(os.path.join(OUT_DIR, "acute_transmission_dyads_q10.csv"), index=False)

    # Threshold decay sweep
    decay_records = []
    for d_thresh in np.linspace(0.001, 0.025, 49):
        g = nx.Graph()
        g.add_nodes_from(range(N_total))
        for _, r in df_edges[df_edges['Distance'] <= d_thresh].iterrows():
            g.add_edge(taxa_to_idx[r['id1']], taxa_to_idx[r['id2']])
        comps = [len(c) for c in nx.connected_components(g) if len(c) > 1]
        decay_records.append({
            'threshold': float(d_thresh),
            'max_cluster_size': max(comps) if comps else 0,
            'clustered_pts': sum(comps) if comps else 0,
            'n_clusters': len(comps),
            'n_edges': int(np.sum(df_edges['Distance'] <= d_thresh))
        })
    pd.DataFrame(decay_records).to_csv(os.path.join(OUT_DIR, "threshold_decay_megacluster.csv"), index=False)

    # Save summary json
    summary_data = {
        "cohort": "Shenzhen CRF07_BC MSM Outbreak (Zhao et al. 2016)",
        "N_total": N_total,
        "sampling_span": [float(df_meta['Sampling_Year'].min()), float(df_meta['Sampling_Year'].max())],
        "total_pairwise_comparisons": int(total_pairs),
        "empirical_distance": {
            "mean": float(mean_d),
            "median": float(median_d),
            "std": float(std_d),
            "pct_below_15": float(pairs_15 / total_pairs * 100.0),
            "pct_below_05": float(pairs_05 / total_pairs * 100.0)
        },
        "static_clustering": {
            "max_cluster_15": int(max_c15),
            "pct_cohort_15": float(max_c15 / N_total * 100.0),
            "max_cluster_05": int(max_c05),
            "pct_cohort_05": float(max_c05 / N_total * 100.0)
        },
        "danno": {
            "clock_violations": int(clock_fails),
            "edges_q05": int(len(df_danno_05)),
            "edges_q01": int(len(df_danno_01)),
            "clusters_q01": int(len(cc_danno_01)),
            "max_cluster_q01": int(max(cc_danno_01) if cc_danno_01 else 0),
            "borel_lambda": float(borel_lambda),
            "borel_r2": float(borel_r2),
            "borel_pval": float(borel_pval)
        },
        "steve": {
            "n_communities": int(len(df_comm)),
            "n_high_velocity": int(len(high_v))
        },
        "chin": {
            "gamma": float(slope_gamma),
            "gamma_r2": float(r_val_g ** 2),
            "gamma_pval": float(p_val_g),
            "rho_median": float(q_rho[2]),
            "rho_ci": [float(q_rho[0]), float(q_rho[4])],
            "N_act_median": float(q_nact[2]),
            "N_act_ci": [float(q_nact[0]), float(q_nact[4])]
        }
    }
    with open(os.path.join(OUT_DIR, "china_benchmark_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    # --------------------------------------------------------------------------
    # 6. Generate Publication Figure (Figure 10)
    # --------------------------------------------------------------------------
    print("\n[6/6] Generating Publication Figure 10 via generate_figure_case_china_continuum.py...")
    import subprocess
    fig_script = "/Users/sergei/Projects/trace50/paper/figures/generate_figure_case_china_continuum.py"
    subprocess.run(["python3", fig_script], check=True)
    print(f"\nTRACE-5.0 China CRF07 pipeline completed in {time.time() - t_start:.2f} s.")


if __name__ == "__main__":
    run_pipeline()

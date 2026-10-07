"""
Command-Line Interface and Pipeline Orchestrator for TRACE-5.0.

Usage:
    trace50 --fasta alignment.fasta --meta metadata.csv --out-dir results/
    trace50 --fasta alignment.fasta --meta metadata.csv --reference-panel subtype_b --out-dir results/
"""

import os
import sys
import argparse
import json
import re
import numpy as np
import pandas as pd

from .tn93 import parse_fasta, encode_alignment, compute_tn93_distance_matrix
from .bayes_factor import CoalescentBayesFactor
from .autoclock import recursive_autoclock_deconvolution
from .taxonomy import LineageTier, classify_subcommunity
from .reference import (
    load_reference_panel,
    merge_cohort_with_reference,
    test_metric_interleaving
)


def parse_date(date_str):
    """Parses decimal dates or standard date strings into decimal years."""
    if pd.isna(date_str):
        return np.nan
    s = str(date_str).strip()
    try:
        val = float(s)
        if 1950.0 <= val <= 2035.0:
            return val
    except ValueError:
        pass

    # Look for 4-digit year
    m = re.search(r'\b(19\d{2}|20\d{2})\b', s)
    if m:
        return float(m.group(1)) + 0.5
    return np.nan


def run_pipeline(
    fasta_path,
    meta_path=None,
    reference_panel=None,
    reference_meta=None,
    wedge_dyads=True,
    dist_matrix_path=None,
    id_col="id",
    date_col="date",
    out_dir="trace50_results",
    fdr_threshold=0.05,
    min_size=3,
    max_depth=8,
    mu_acute=2.0e-3,
    omega_acute=2.0
):
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== TRACE-5.0 Molecular Transmission Surveillance Engine ===")
    print(f"Loading local alignment: {fasta_path}")

    cohort_records = parse_fasta(fasta_path)
    if not cohort_records:
        print("Error: No sequences parsed from FASTA.", file=sys.stderr)
        return 1

    N_loc = len(cohort_records)
    print(f"Local cohort loaded: N = {N_loc} sequences.")

    # Match cohort metadata dates
    cohort_dates = np.full(N_loc, np.nan, dtype=np.float64)
    cohort_meta_map = {}
    if meta_path and os.path.exists(meta_path):
        sep = '\t' if meta_path.endswith('.tsv') else ','
        df_meta = pd.read_csv(meta_path, sep=sep)
        df_meta[id_col] = df_meta[id_col].astype(str)
        date_map = {}
        for _, row in df_meta.iterrows():
            tid = str(row[id_col]).strip()
            date_map[tid] = parse_date(row[date_col])
            cohort_meta_map[tid] = {
                "country": str(row.get("country", "Local")),
                "subtype": str(row.get("subtype", "Local"))
            }

        for i, (h, _) in enumerate(cohort_records):
            h_clean = h.strip()
            if h_clean in date_map:
                cohort_dates[i] = date_map[h_clean]
            else:
                token = h_clean.split('|')[0].split('_')[0].split('.')[0]
                cohort_dates[i] = date_map.get(token, np.nan)
    else:
        for i, (h, _) in enumerate(cohort_records):
            cohort_dates[i] = parse_date(h)

    # Impute missing cohort dates if any
    n_cohort_dated = int(np.sum(~np.isnan(cohort_dates)))
    print(f"Cohort temporal calibration: {n_cohort_dated}/{N_loc} isolates possess valid collection dates.")
    if n_cohort_dated < 3:
        print("Warning: Insufficient cohort collection dates for clock velocity tracking. Imputing median.", file=sys.stderr)
        med_date = 2020.0
    else:
        med_date = float(np.nanmedian(cohort_dates))
    cohort_dates[np.isnan(cohort_dates)] = med_date

    # Load Reference Panel if requested (Architecture A)
    has_reference = (reference_panel is not None and str(reference_panel).strip().lower() not in ("none", "false", ""))
    if has_reference:
        print(f"Loading external reference panel: '{reference_panel}'...")
        panel_records, panel_meta_map, panel_desc = load_reference_panel(reference_panel, reference_meta)
        print(f"Reference panel loaded: {len(panel_records)} anchors ({panel_desc}).")

        # Merge cohort and reference anchors
        headers, M_int, M_bits, dates, is_anchor, meta_list = merge_cohort_with_reference(
            cohort_records=cohort_records,
            cohort_dates=cohort_dates,
            cohort_meta_map=cohort_meta_map,
            panel_records=panel_records,
            panel_meta_map=panel_meta_map
        )
    else:
        headers, M_int, M_bits = encode_alignment(cohort_records)
        dates = cohort_dates
        is_anchor = np.zeros(N_loc, dtype=bool)
        meta_list = [{"id": h, "date": dates[i], "is_anchor": False} for i, (h, _) in enumerate(cohort_records)]

    N, L = M_int.shape
    N_anc = int(np.sum(is_anchor))
    print(f"Joint Sequence Manifold: {N_loc} local isolates + {N_anc} reference anchors = {N} total taxa across L = {L} sites.")

    t_span = float(np.max(dates) - np.min(dates))
    max_date = float(np.max(dates[~is_anchor])) if N_loc > 0 else float(np.max(dates))

    # 1. Compute or Load Pairwise Distance Matrix
    if dist_matrix_path and os.path.exists(dist_matrix_path):
        print(f"Loading precomputed pairwise distance matrix from {dist_matrix_path}...", flush=True)
        D = np.load(dist_matrix_path)
        if D.shape != (N, N):
            raise ValueError(f"Distance matrix shape {D.shape} does not match combined sequences ({N}, {N})")
    else:
        print("Computing pairwise TN93 distance matrix...", flush=True)
        D = compute_tn93_distance_matrix(M_bits)

    # 2. Stage 1: Coalescent Negative Binomial Bayes Factor for Contemporaneous Dyads
    print("Evaluating Coalescent Negative Binomial Bayes Factors for acute dyads...", flush=True)
    bf_engine = CoalescentBayesFactor(
        seq_len=L,
        mu_acute=mu_acute,
        omega_acute=omega_acute,
        t_span=max(t_span, 5.0)
    )

    # Principle 1: Screen dyads strictly among local cohort isolates
    local_indices = np.where(~is_anchor)[0]
    significant_dyads = bf_engine.screen_dyads(
        indices=local_indices,
        dates=dates,
        D_matrix=D,
        max_delta_t=2.0,
        fdr_threshold=fdr_threshold
    )

    # Principle 4: Metric Interleaving Test (The External Wedge for Dyads)
    verified_dyads = []
    wedged_dyads = []
    if has_reference and wedge_dyads:
        for d in significant_dyads:
            is_wedged, wedging_anchors = test_metric_interleaving(d["idx1"], d["idx2"], D, is_anchor)
            if is_wedged:
                d["wedging_anchor"] = headers[wedging_anchors[0]]
                d["num_wedging_anchors"] = len(wedging_anchors)
                wedged_dyads.append(d)
            else:
                verified_dyads.append(d)
        print(f"Stage 1 Complete: Discovered {len(verified_dyads)} genuine acute dyads (FDR q <= {fdr_threshold}).")
        if len(wedged_dyads) > 0:
            print(f"  * Dissolved {len(wedged_dyads)} spurious dyads wedged by background reference anchors.")
    else:
        verified_dyads = significant_dyads
        print(f"Stage 1 Complete: Discovered {len(verified_dyads)} acute transmission edges (FDR q <= {fdr_threshold}).")

    # Export acute dyads
    dyads_df = pd.DataFrame([
        {
            "taxon_1": headers[d["idx1"]],
            "taxon_2": headers[d["idx2"]],
            "tn93_dist": round(d["dist"], 5),
            "delta_t_years": round(d["delta_t"], 3),
            "bayes_factor": round(d["bayes_factor"], 2),
            "q_value": round(d["q_value"], 5)
        }
        for d in verified_dyads
    ])
    dyads_csv = os.path.join(out_dir, "acute_transmission_dyads.csv")
    dyads_df.to_csv(dyads_csv, index=False)

    if len(wedged_dyads) > 0:
        wedged_df = pd.DataFrame([
            {
                "taxon_1": headers[d["idx1"]],
                "taxon_2": headers[d["idx2"]],
                "tn93_dist": round(d["dist"], 5),
                "delta_t_years": round(d["delta_t"], 3),
                "bayes_factor": round(d["bayes_factor"], 2),
                "primary_wedging_anchor": d["wedging_anchor"],
                "total_wedging_anchors": d["num_wedging_anchors"]
            }
            for d in wedged_dyads
        ])
        wedged_csv = os.path.join(out_dir, "wedged_background_dyads.csv")
        wedged_df.to_csv(wedged_csv, index=False)

    # 3. Stage 2: Recursive AutoClock Deconvolution
    print("Running Recursive AutoClock spectral bisection and simplex profile rooting...", flush=True)
    all_indices = np.arange(N)
    leaf_clusters = recursive_autoclock_deconvolution(
        sub_idx=all_indices,
        dates=dates,
        D_matrix=D,
        M_int=M_int,
        is_anchor=is_anchor,
        min_size=min_size,
        max_depth=max_depth
    )
    print(f"Stage 2 Complete: AutoClock resolved {len(leaf_clusters)} structural communities.")

    # 4. Stage 3: Operational Taxonomy & Prioritization
    cluster_records = []
    patient_assignments = []

    for c_id, (c_idx, fit_dict, stop_reason, path) in enumerate(leaf_clusters):
        tier, priority, rationale = classify_subcommunity(fit_dict, max_surveillance_date=max_date)

        sub_local = [idx for idx in c_idx if not is_anchor[idx]]
        sub_anc = [idx for idx in c_idx if is_anchor[idx]]
        local_taxa = [headers[idx] for idx in sub_local]
        anc_taxa = [headers[idx] for idx in sub_anc]

        cluster_records.append({
            "cluster_id": c_id + 1,
            "path": path,
            "total_size": fit_dict["n"],
            "local_taxa_count": len(sub_local),
            "anchor_taxa_count": len(sub_anc),
            "is_hybrid": fit_dict.get("is_hybrid", False),
            "effective_size_Neff": round(fit_dict["n_eff"], 2),
            "clock_rate_mu": fit_dict["mu"],
            "r_squared": round(fit_dict["r2"], 4),
            "t_mrca": round(fit_dict["tmrca"], 2),
            "ci_mrca_low": round(fit_dict["ci_mrca"][0], 2) if np.isfinite(fit_dict["ci_mrca"][0]) else None,
            "ci_mrca_high": round(fit_dict["ci_mrca"][1], 2) if np.isfinite(fit_dict["ci_mrca"][1]) else None,
            "fieller_status": fit_dict["fieller_status"],
            "mean_pairwise_dist": round(fit_dict["mean_dist"], 5),
            "time_span_years": round(fit_dict["span"], 2),
            "operational_tier": tier,
            "public_health_priority": priority,
            "rationale": rationale,
            "local_taxa": ",".join(local_taxa),
            "anchor_taxa": ",".join(anc_taxa)
        })

        for tid in local_taxa:
            patient_assignments.append({
                "patient_id": tid,
                "cluster_id": c_id + 1,
                "operational_tier": tier,
                "public_health_priority": priority,
                "cluster_clock_velocity": fit_dict["mu"],
                "cluster_linearity_R2": round(fit_dict["r2"], 4),
                "cluster_t_mrca": round(fit_dict["tmrca"], 2),
                "is_wedged_exogenous": fit_dict.get("is_hybrid", False)
            })

    clusters_df = pd.DataFrame(cluster_records)
    clusters_csv = os.path.join(out_dir, "autoclock_communities.csv")
    clusters_df.to_csv(clusters_csv, index=False)

    patient_df = pd.DataFrame(patient_assignments)
    patient_csv = os.path.join(out_dir, "patient_surveillance_assignments.csv")
    patient_df.to_csv(patient_csv, index=False)

    # Summary Report JSON
    summary_report = {
        "dataset": {
            "fasta_path": fasta_path,
            "local_sequences": N_loc,
            "reference_anchors": N_anc,
            "total_sequences": N,
            "num_sequences": N,
            "num_sites": L,
            "time_span_years": round(t_span, 2)
        },
        "stage1_acute_dyads": {
            "num_edges": len(verified_dyads),
            "num_verified_edges": len(verified_dyads),
            "num_wedged_edges": len(wedged_dyads),
            "fdr_threshold": fdr_threshold
        },
        "stage2_autoclock": {
            "num_communities": len(leaf_clusters),
            "tier_counts": clusters_df["operational_tier"].value_counts().to_dict()
        }
    }
    report_json = os.path.join(out_dir, "trace50_summary.json")
    with open(report_json, 'w') as f:
        json.dump(summary_report, f, indent=2)

    print("\n--- Operational Summary by Tier ---")
    tier_counts = clusters_df["operational_tier"].value_counts()
    for tier, count in tier_counts.items():
        print(f"  * {tier}: {count} communities")

    print(f"\nResults successfully written to: {out_dir}")
    print(f"  - Acute Dyads: {dyads_csv}")
    if len(wedged_dyads) > 0:
        print(f"  - Wedged Spurious Dyads: {os.path.join(out_dir, 'wedged_background_dyads.csv')}")
    print(f"  - AutoClock Communities: {clusters_csv}")
    print(f"  - Patient Surveillance Action Plan: {patient_csv}")
    print(f"  - Machine-readable Summary: {report_json}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="TRACE-5.0: Dynamic Molecular Surveillance of Viral Transmission Networks"
    )
    parser.add_argument("--fasta", required=True, help="Input FASTA alignment path")
    parser.add_argument("--meta", help="Metadata CSV/TSV path with collection dates")
    parser.add_argument(
        "--reference-panel",
        default=None,
        help="Optional reference panel: 'subtype_b', 'multisubtype', or path to FASTA (Architecture A)"
    )
    parser.add_argument("--reference-meta", default=None, help="Optional metadata CSV for custom reference panel")
    parser.add_argument("--no-wedge-dyads", action="store_true", help="Disable metric interleaving test for acute dyads")
    parser.add_argument("--dist-matrix", help="Optional path to precomputed .npy pairwise distance matrix")
    parser.add_argument("--id-col", default="id", help="Taxon identifier column in metadata (default: id)")
    parser.add_argument("--date-col", default="date", help="Date column in metadata (default: date)")
    parser.add_argument("--out-dir", default="trace50_results", help="Output directory path (default: trace50_results)")
    parser.add_argument("--fdr", type=float, default=0.05, help="FDR threshold for acute dyads (default: 0.05)")
    parser.add_argument("--min-size", type=int, default=3, help="Minimum leaf cluster size (default: 3)")
    parser.add_argument("--max-depth", type=int, default=8, help="Maximum recursion depth for AutoClock (default: 8)")
    parser.add_argument("--mu-acute", type=float, default=2.0e-3, help="Intra-host/acute substitution rate (default: 2.0e-3)")

    args = parser.parse_args()
    ret = run_pipeline(
        fasta_path=args.fasta,
        meta_path=args.meta,
        reference_panel=args.reference_panel,
        reference_meta=args.reference_meta,
        wedge_dyads=(not args.no_wedge_dyads),
        dist_matrix_path=args.dist_matrix,
        id_col=args.id_col,
        date_col=args.date_col,
        out_dir=args.out_dir,
        fdr_threshold=args.fdr,
        min_size=args.min_size,
        max_depth=args.max_depth,
        mu_acute=args.mu_acute
    )
    sys.exit(ret)


if __name__ == "__main__":
    main()

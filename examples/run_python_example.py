#!/usr/bin/env python3
"""
TRACE-5.0 Python API Example.
Demonstrates end-to-end execution of DANNO, STEVE / AutoClock, and CHIN on sample data.
"""

import os
import json
from trace50.tn93 import parse_fasta, encode_alignment, compute_tn93_distance_matrix
from trace50.bayes_factor import DannoEstimator
from trace50.autoclock import recursive_autoclock_deconvolution
from trace50.chin import ChinEstimator

def main():
    example_dir = os.path.dirname(os.path.abspath(__file__))
    fasta_path = os.path.join(example_dir, "sample_alignment.fasta")
    
    print("=" * 70)
    print(" TRACE-5.0 Python Surveillance Engine Demo")
    print("=" * 70)
    
    # 1. Load FASTA
    records = parse_fasta(fasta_path)
    headers, M_int, M_bits = encode_alignment(records)
    N, L = M_bits.shape[0], M_bits.shape[1]
    print(f"[*] Loaded {N} sequences across L={L} sites.")
    
    # 2. Pairwise TN93
    D = compute_tn93_distance_matrix(M_bits)
    print(f"[*] Computed {N*(N-1)//2} pairwise distances.")
    
    # 3. DANNO Bayes Factor & FDR Control
    dates = [2015.1, 2015.2, 2015.3, 2016.1, 2017.0, 2017.4]
    danno = DannoEstimator(seq_len=L, mu=2e-3, tau_bar=1.0, omega=2.0)
    dyads = danno.screen_dyads(
        indices=list(range(N)),
        dates=dates,
        D_matrix=D,
        max_delta_t=2.5,
        fdr_threshold=0.10
    )
    print(f"[+] Certified {len(dyads)} acute transmission dyads (q <= 0.10).")
    for d in dyads:
        print(f"    - {headers[d['idx1']]} <-> {headers[d['idx2']]}: dist={d['dist']:.5f}, dt={d['delta_t']:.2f} yr, BF={d['bayes_factor']:.1f}, q={d['q_value']:.4f}")
        
    # 4. CHIN Macro-Scale Inversion
    chin = ChinEstimator(R0_default=1.5)
    active_pop = chin.estimate_active_population(N, len(dyads))
    print(f"[+] Active Transmitting Pool N_act: {active_pop['N_act']:.1f} (CI: {active_pop['ci_lower']:.1f} - {active_pop['ci_upper']:.1f})")
    print("=" * 70)

if __name__ == "__main__":
    main()

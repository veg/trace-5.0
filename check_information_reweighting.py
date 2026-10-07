import os, sys
import numpy as np
import pandas as pd
from Bio import SeqIO

print("[1/4] Loading Tennessee alignment and computing site-specific nucleotide background frequencies...")
fasta_file = "/Users/sergei/Projects/trace50/data/tennessee_cohort/tennessee_aligned.fasta"
records = list(SeqIO.parse(fasta_file, "fasta"))
N = len(records)
L = len(records[0].seq)
print(f"  N = {N:,} sequences, L = {L:,} alignment columns.")

# Convert to 2D numpy array of characters (uppercase)
# Filter sequences to clean IDs
seq_ids = [r.id.split('|')[0] for r in records]
id_to_idx = {sid: i for i, sid in enumerate(seq_ids)}

# Build matrix of characters
char_mat = np.array([list(str(r.seq).upper()) for r in records], dtype='U1')

# Calculate empirical nucleotide frequencies per position
# Only consider standard bases A, C, G, T
bases = ['A', 'C', 'G', 'T']
freq_mat = np.zeros((L, 4)) # L x 4

for l in range(L):
    col = char_mat[:, l]
    counts = [np.sum(col == b) for b in bases]
    tot = sum(counts)
    if tot > 0:
        freq_mat[l, :] = [c / tot for c in counts]
    else:
        freq_mat[l, :] = 0.25

# Add small pseudocount to avoid division by zero (min frequency 1 / (2*N))
min_freq = 1.0 / (2.0 * N)
freq_mat = np.clip(freq_mat, min_freq, 1.0)
# Re-normalize
freq_mat = freq_mat / freq_mat.sum(axis=1, keepdims=True)

base_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

print("[2/4] Loading candidate dyads and metadata...")
df_dist = pd.read_csv("/Users/sergei/Projects/trace50/data/tennessee_cohort/tn93_distances_003.csv")
df_dist['id1'] = df_dist['ID1'].apply(lambda x: str(x).split('|')[0])
df_dist['id2'] = df_dist['ID2'].apply(lambda x: str(x).split('|')[0])

df_meta = pd.read_csv("/Users/sergei/Projects/trace50/data/tennessee_cohort/tennessee_metadata.csv")
date_map = dict(zip(df_meta['accession'], df_meta['year']))
df_dist['t1'] = df_dist['id1'].map(date_map)
df_dist['t2'] = df_dist['id2'].map(date_map)
df_dist['delta_t'] = (df_dist['t1'] - df_dist['t2']).abs()

df_cand = df_dist.dropna(subset=['delta_t']).copy()

# Certified acute dyads
df_acute = pd.read_csv("/Users/sergei/Projects/trace50/benchmark_results_tennessee/acute_transmission_dyads.csv")
acute_pairs = set(tuple(sorted([str(r['acc1']).split('|')[0], str(r['acc2']).split('|')[0]])) for _, r in df_acute.iterrows())

print(f"  Evaluating {len(df_cand):,} candidate pairs (including {len(acute_pairs)} acute dyads)...")

print("[3/4] Computing Site-Specific Information Score vs. Unweighted Distance...")
# For each candidate pair, compare unweighted distance to Information-Theoretic Evidence
# Information score:
# For positions where both have resolved base:
# If identical base b: log(1 / f_l(b))
# If different base (a != b): log( (2*mu*delta_t + 1e-4) / (f_l(a) * f_l(b)) )
# Notice: If a rare mutation is shared (f_l(b) = 0.001), score is +6.9 nats.
# If a common base is shared (f_l(b) = 0.98), score is +0.02 nats!

results = []
# Evaluate on candidate pairs with d <= 0.015 (the 1.5% network)
sub_cand = df_cand[df_cand['Distance'] <= 0.015].copy()

mu = 2.0e-3 # per site per year

for _, r in sub_cand.iterrows():
    p1, p2 = r['id1'], r['id2']
    if p1 not in id_to_idx or p2 not in id_to_idx:
        continue
    i1, i2 = id_to_idx[p1], id_to_idx[p2]
    d = r['Distance']
    dt = r['delta_t']
    pair_key = tuple(sorted([p1, p2]))
    is_acute = pair_key in acute_pairs
    is_clock_viol = (d <= 0.005) and (dt >= 4.0)
    
    s1 = char_mat[i1, :]
    s2 = char_mat[i2, :]
    
    # Mask of mutually resolved standard bases
    mask = np.isin(s1, bases) & np.isin(s2, bases)
    resolved_len = np.sum(mask)
    if resolved_len == 0:
        continue
        
    s1_res = s1[mask]
    s2_res = s2[mask]
    freqs_res = freq_mat[mask, :]
    
    # Indices
    b1_indices = np.array([base_idx[b] for b in s1_res])
    b2_indices = np.array([base_idx[b] for b in s2_res])
    
    identical_mask = (b1_indices == b2_indices)
    diff_mask = ~identical_mask
    
    # Frequencies of shared bases
    shared_freqs = freqs_res[np.arange(len(s1_res))[identical_mask], b1_indices[identical_mask]]
    
    # Count how many shared bases are rare (e.g. background frequency < 5% or < 1%)
    n_shared_rare_5pct = np.sum(shared_freqs < 0.05)
    n_shared_rare_1pct = np.sum(shared_freqs < 0.01)
    n_shared_rare_01pct = np.sum(shared_freqs < 0.001)
    
    # Log-likelihood ratio contribution
    # Shared: log( 1 / f_l(b) )
    log_shared = np.sum(-np.log(shared_freqs))
    
    # Mismatches:
    f1_diff = freqs_res[np.arange(len(s1_res))[diff_mask], b1_indices[diff_mask]]
    f2_diff = freqs_res[np.arange(len(s1_res))[diff_mask], b2_indices[diff_mask]]
    # Probability under H_trans for a mismatch: roughly 2 * mu * max(dt, 0.5) / 3
    p_trans_mut = (2.0 * mu * max(dt, 0.5) + 0.001) / 3.0
    log_diff = np.sum(np.log(p_trans_mut) - np.log(f1_diff * f2_diff))
    
    total_info_score = log_shared + log_diff
    
    results.append({
        'id1': p1, 'id2': p2, 'd': d, 'delta_t': dt,
        'is_acute': is_acute, 'is_clock_viol': is_clock_viol,
        'n_diff': np.sum(diff_mask),
        'n_shared_rare_5pct': n_shared_rare_5pct,
        'n_shared_rare_1pct': n_shared_rare_1pct,
        'n_shared_rare_01pct': n_shared_rare_01pct,
        'info_score': total_info_score,
        'mean_shared_entropy': np.mean(-np.log2(shared_freqs)) if len(shared_freqs) > 0 else 0
    })

df_info = pd.DataFrame(results)

print(f"\n[4/4] RESULTS: Information-Theoretic Resolution across {len(df_info):,} pairs (d <= 1.5%):")
print("="*80)

# Compare Acute Dyads vs. Near-Boundary vs. Clock Violations
print("COMPARISON ACROSS EPIDEMIOLOGICAL CLASSES:")
print("="*80)
for cat_name, sub in [
    ("DANNO-Certified Acute Transmission Dyads", df_info[df_info['is_acute']]),
    ("Wo-Fat Stagnant Clock Violations (dt >= 4 yr)", df_info[df_info['is_clock_viol']]),
    ("Near-Boundary Pairs (d in [0.8%, 1.5%], dt <= 2 yr)", df_info[(df_info['d'] >= 0.008) & (df_info['d'] <= 0.015) & (df_info['delta_t'] <= 2.0)]),
    ("Long-Delay Pairs (d in [0.8%, 1.5%], dt >= 4 yr)", df_info[(df_info['d'] >= 0.008) & (df_info['d'] <= 0.015) & (df_info['delta_t'] >= 4.0)])
]:
    n = len(sub)
    if n == 0: continue
    print(f"\n{cat_name} (N = {n:,}):")
    print(f"  Mean Distance d:               {sub['d'].mean()*100:.2f}%")
    print(f"  Mean Delta T:                  {sub['delta_t'].mean():.2f} years")
    print(f"  Mean Shared Rare Bases (<5%):  {sub['n_shared_rare_5pct'].mean():.1f} positions")
    print(f"  Mean Shared Rare Bases (<1%):  {sub['n_shared_rare_1pct'].mean():.1f} positions")
    print(f"  Mean Shared Rare Bases (<0.1%):{sub['n_shared_rare_01pct'].mean():.2f} positions")
    print(f"  Mean Information Score (nats): {sub['info_score'].mean():.1f}")
    print(f"  Information Score IQR:         [{sub['info_score'].quantile(0.25):.1f}, {sub['info_score'].quantile(0.75):.1f}]")

print("\n" + "="*80)
print("DEMONSTRATION OF DIVERGENCE BETWEEN DISTANCE AND INFORMATION:")
print("="*80)
# Look at pairs with the EXACT SAME DISTANCE d = 1.0% (+- 0.1%)
sub_same_d = df_info[(df_info['d'] >= 0.009) & (df_info['d'] <= 0.011)].copy()
print(f"Pairs with identical distance d ~ 1.0% (N = {len(sub_same_d):,}):")
print(f"  Information Score range: {sub_same_d['info_score'].min():.1f} to {sub_same_d['info_score'].max():.1f} nats!")
print(f"  Shared rare (<1%) bases range: {sub_same_d['n_shared_rare_1pct'].min()} to {sub_same_d['n_shared_rare_1pct'].max()} positions!")

top_info = sub_same_d.sort_values('info_score', ascending=False).head(3)
bot_info = sub_same_d.sort_values('info_score', ascending=True).head(3)

print("\nTop 3 Pairs at d ~ 1.0% (HIGH transmission evidence due to shared rare variants):")
for _, r in top_info.iterrows():
    print(f"  Pair {r['id1']} - {r['id2']}: d = {r['d']*100:.2f}%, dt = {r['delta_t']:.1f} yr | Info Score: {r['info_score']:.1f} | Shared Rare (<1%): {r['n_shared_rare_1pct']}")

print("\nBottom 3 Pairs at d ~ 1.0% (LOW transmission evidence; spurious similarity on common bases):")
for _, r in bot_info.iterrows():
    print(f"  Pair {r['id1']} - {r['id2']}: d = {r['d']*100:.2f}%, dt = {r['delta_t']:.1f} yr | Info Score: {r['info_score']:.1f} | Shared Rare (<1%): {r['n_shared_rare_1pct']}")


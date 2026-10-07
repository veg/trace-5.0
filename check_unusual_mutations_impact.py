import os, sys, re
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

print("[1/5] Loading HIVDB facts reference tables...")
facts_dir = "/Users/sergei/Projects/trace50/data/hivdb_facts"

# 1. AAPCNT Subtype B
df_aapcnt = pd.read_csv(os.path.join(facts_dir, "rx-all_subtype-B.csv"))
# Filter for PR and RT
df_aapcnt = df_aapcnt[df_aapcnt['gene'].isin(['PR', 'RT'])].copy()
# Create lookup: (gene, pos, aa) -> {'percent': float, 'isUnusual': bool}
freq_map = {}
unusual_set = set()
for _, r in df_aapcnt.iterrows():
    g = r['gene']
    p = int(r['position'])
    aa = r['aa']
    pcnt = float(r['percent'])
    is_unusual = bool(r['isUnusual'])
    freq_map[(g, p, aa)] = pcnt
    if is_unusual or pcnt < 0.0005:  # < 0.05% prevalence
        unusual_set.add((g, p, aa))

print(f"  Loaded {len(freq_map):,} AA prevalence entries. Found {len(unusual_set):,} unusual/rare alleles in PR/RT.")

# 2. APOBEC signature mutations
df_apobec = pd.read_csv(os.path.join(facts_dir, "apobecs.csv"))
df_apobec = df_apobec[df_apobec['gene'].isin(['PR', 'RT'])].copy()
apobec_set = set()
for _, r in df_apobec.iterrows():
    apobec_set.add((r['gene'], int(r['position']), r['aa']))
print(f"  Loaded {len(apobec_set)} signature APOBEC mutations in PR/RT.")

# 3. DRMs
df_drms = pd.read_csv(os.path.join(facts_dir, "drms_hiv1.csv"))
df_drms = df_drms[df_drms['gene'].isin(['PR', 'RT'])].copy()
drm_set = set()
for _, r in df_drms.iterrows():
    drm_set.add((r['gene'], int(r['position']), r['aa']))
print(f"  Loaded {len(drm_set)} DRM mutations in PR/RT.")

# Standard consensus wild-type for Subtype B (most common AA per position in HIVDB)
wt_map = {}
for g in ['PR', 'RT']:
    sub = df_aapcnt[df_aapcnt['gene'] == g]
    for pos, grp in sub.groupby('position'):
        top_aa = grp.sort_values('percent', ascending=False).iloc[0]['aa']
        wt_map[(g, int(pos))] = top_aa

print("[2/5] Translating Tennessee cohort sequences...")
fasta_file = "/Users/sergei/Projects/trace50/data/tennessee_cohort/tennessee_aligned.fasta"
records = list(SeqIO.parse(fasta_file, "fasta"))
print(f"  Loaded {len(records):,} sequences.")

# Translate each sequence into PR and RT
seq_mutations = {} # id -> list of non-WT mutations: (gene, pos, aa, type)
for r in records:
    seq_id = r.id.split('|')[0]
    s_nt = str(r.seq).upper().replace('-', 'N')
    
    # PR is 1..297 nt (99 codons)
    pr_nt = s_nt[:297]
    # RT is 298..1497 nt (400 codons)
    rt_nt = s_nt[297:297 + 1200]
    
    muts = []
    # Translate PR
    for codon_idx in range(99):
        c_nt = pr_nt[codon_idx*3 : codon_idx*3 + 3]
        if 'N' in c_nt or len(c_nt) < 3:
            continue
        aa = str(Seq(c_nt).translate())
        pos = codon_idx + 1
        wt = wt_map.get(('PR', pos), None)
        if wt and aa != wt:
            # Classify mutation
            is_drm = ('PR', pos, aa) in drm_set
            is_apobec = ('PR', pos, aa) in apobec_set
            is_unusual = ('PR', pos, aa) in unusual_set
            freq = freq_map.get(('PR', pos, aa), 1e-5)
            muts.append(('PR', pos, aa, is_drm, is_apobec, is_unusual, freq))
            
    # Translate RT
    for codon_idx in range(len(rt_nt) // 3):
        c_nt = rt_nt[codon_idx*3 : codon_idx*3 + 3]
        if 'N' in c_nt or len(c_nt) < 3:
            continue
        aa = str(Seq(c_nt).translate())
        pos = codon_idx + 1
        wt = wt_map.get(('RT', pos), None)
        if wt and aa != wt:
            is_drm = ('RT', pos, aa) in drm_set
            is_apobec = ('RT', pos, aa) in apobec_set
            is_unusual = ('RT', pos, aa) in unusual_set
            freq = freq_map.get(('RT', pos, aa), 1e-5)
            muts.append(('RT', pos, aa, is_drm, is_apobec, is_unusual, freq))
            
    seq_mutations[seq_id] = muts

print(f"  Translated and called mutations across {len(seq_mutations):,} isolates.")

print("[3/5] Loading candidate dyads and classifications...")
df_dist = pd.read_csv("/Users/sergei/Projects/trace50/data/tennessee_cohort/tn93_distances_003.csv")
df_dist['id1'] = df_dist['ID1'].apply(lambda x: str(x).split('|')[0])
df_dist['id2'] = df_dist['ID2'].apply(lambda x: str(x).split('|')[0])

# Load metadata for sampling dates
df_meta = pd.read_csv("/Users/sergei/Projects/trace50/data/tennessee_cohort/tennessee_metadata.csv")
date_map = dict(zip(df_meta['accession'], df_meta['year']))
df_dist['t1'] = df_dist['id1'].map(date_map)
df_dist['t2'] = df_dist['id2'].map(date_map)
df_dist['delta_t'] = (df_dist['t1'] - df_dist['t2']).abs()

# Filter candidate pairs with valid dates
df_cand = df_dist.dropna(subset=['delta_t']).copy()
print(f"  Total candidate pairs with dates: {len(df_cand):,}")

# Load certified acute transmission dyads
df_acute = pd.read_csv("/Users/sergei/Projects/trace50/benchmark_results_tennessee/acute_transmission_dyads.csv")
acute_pairs = set()
for _, r in df_acute.iterrows():
    p1 = str(r['acc1']).split('|')[0]
    p2 = str(r['acc2']).split('|')[0]
    acute_pairs.add(tuple(sorted([p1, p2])))

print(f"  Loaded {len(acute_pairs)} DANNO-certified acute dyads.")

print("[4/5] Computing shared mutation profiles for candidate pairs...")
results = []
for idx, r in df_cand.iterrows():
    p1, p2 = r['id1'], r['id2']
    pair_key = tuple(sorted([p1, p2]))
    d = r['Distance']
    dt = r['delta_t']
    is_acute = pair_key in acute_pairs
    is_clock_viol = (d <= 0.005) and (dt >= 4.0) # Wo Fat zone
    
    m1 = seq_mutations.get(p1, [])
    m2 = seq_mutations.get(p2, [])
    
    # Dict of (gene, pos) -> (aa, is_drm, is_apobec, is_unusual, freq)
    d1 = {(m[0], m[1]): m[2:] for m in m1}
    d2 = {(m[0], m[1]): m[2:] for m in m2}
    
    shared_muts = []
    shared_drms = 0
    shared_apobec = 0
    shared_unusual = 0
    shared_common = 0
    
    # Information-theoretic log likelihood ratio contribution from shared mutations:
    # LR_site = P(shared | H_trans) / P(shared | H_null) ~ 1 / freq
    log_info_score = 0.0
    
    for loc in set(d1.keys()) & set(d2.keys()):
        aa1, drm1, apo1, unu1, freq1 = d1[loc]
        aa2, drm2, apo2, unu2, freq2 = d2[loc]
        if aa1 == aa2:
            shared_muts.append((loc[0], loc[1], aa1, unu1, drm1, apo1, freq1))
            if drm1: shared_drms += 1
            if apo1: shared_apobec += 1
            if unu1: shared_unusual += 1
            else: shared_common += 1
            
            # If it's a DRM, frequency under ART is elevated (e.g. 0.30 instead of rare)
            effective_null_freq = 0.30 if drm1 else max(freq1, 1e-4)
            log_info_score += np.log(1.0 / effective_null_freq)
            
    results.append({
        'id1': p1, 'id2': p2, 'd': d, 'delta_t': dt,
        'is_acute': is_acute, 'is_clock_viol': is_clock_viol,
        'n_shared_muts': len(shared_muts),
        'n_shared_unusual': shared_unusual,
        'n_shared_drms': shared_drms,
        'n_shared_apobec': shared_apobec,
        'n_shared_common': shared_common,
        'info_score': log_info_score
    })

df_res = pd.DataFrame(results)

print("\n[5/5] ANALYSIS OF FINDINGS:")
print("="*80)
print("1. SHARED UNUSUAL MUTATIONS BY EPIDEMIOLOGICAL CATEGORY:")
print("="*80)

for cat_name, sub in [
    ("DANNO Acute Transmission Dyads (q <= 0.05)", df_res[df_res['is_acute']]),
    ("Wo-Fat Clock Violations (d <= 0.5%, dt >= 4 yr)", df_res[df_res['is_clock_viol']]),
    ("Near-Boundary Candidates (d in [0.5%, 1.5%], dt <= 2 yr)", df_res[(df_res['d'] > 0.005) & (df_res['d'] <= 0.015) & (df_res['delta_t'] <= 2.0)]),
    ("Distant Candidates (d > 2.0%)", df_res[df_res['d'] > 0.020])
]:
    n = len(sub)
    if n == 0: continue
    pct_has_unusual = (sub['n_shared_unusual'] >= 1).mean() * 100.0
    mean_unusual = sub['n_shared_unusual'].mean()
    pct_has_drm = (sub['n_shared_drms'] >= 1).mean() * 100.0
    mean_info = sub['info_score'].mean()
    print(f"\nCategory: {cat_name} (N = {n:,})")
    print(f"  Pairs sharing >= 1 UNUSUAL mutation: {pct_has_unusual:.1f}%")
    print(f"  Mean shared unusual mutations:       {mean_unusual:.2f}")
    print(f"  Pairs sharing >= 1 DRM:              {pct_has_drm:.1f}%")
    print(f"  Mean Information Score (log LR):     {mean_info:.2f}")

print("\n" + "="*80)
print("2. THE 'SMOKING GUN': CANDIDATES DISCARDED BY STATIC 0.5% THAT CARRY RARE MUTATIONS:")
print("="*80)
# Look at pairs with d in [0.005, 0.015] (i.e. 0.5% to 1.5%) with delta_t in [0.5, 3.0] yr
df_rescue = df_res[(df_res['d'] > 0.005) & (df_res['d'] <= 0.015) & (df_res['delta_t'] >= 0.5) & (df_res['delta_t'] <= 3.0)].copy()
df_rescue_strong = df_rescue[df_rescue['n_shared_unusual'] >= 2].sort_values('n_shared_unusual', ascending=False)

print(f"Total pairs in [0.5%, 1.5%] with 0.5-3.0 yr delay: {len(df_rescue):,}")
print(f"Pairs sharing >= 1 unusual mutation: {(df_rescue['n_shared_unusual'] >= 1).sum():,} ({(df_rescue['n_shared_unusual'] >= 1).mean()*100:.1f}%)")
print(f"Pairs sharing >= 2 unusual mutations: {len(df_rescue_strong):,} ({len(df_rescue_strong)/len(df_rescue)*100:.1f}%)")

print("\nTop 5 'Smoking Gun' Rescued Pairs (d > 0.5%, discarded by strict static cutoff, but sharing multiple rare mutations):")
for _, r in df_rescue_strong.head(5).iterrows():
    print(f"  Pair {r['id1']} - {r['id2']}: d = {r['d']*100:.2f}%, Delta T = {r['delta_t']:.2f} yr | Shared Unusual: {r['n_shared_unusual']} | Info Score: {r['info_score']:.1f}")

print("\n" + "="*80)
print("3. THE 'HOMOPLASY / SPURIOUS' LINKS: PAIRS LINKED MAINLY BY DRMS OR COMMON MUTATIONS:")
print("="*80)
# Pairs with d <= 0.015 but ZERO shared unusual mutations and >= 1 shared DRM
df_spurious = df_res[(df_res['d'] <= 0.015) & (df_res['n_shared_unusual'] == 0) & (df_res['n_shared_drms'] >= 1) & (df_res['delta_t'] >= 2.0)]
print(f"Pairs with d <= 1.5%, Delta T >= 2 yr, sharing DRMs but ZERO unusual mutations: {len(df_spurious):,}")
if len(df_spurious) > 0:
    for _, r in df_spurious.head(5).iterrows():
        print(f"  Pair {r['id1']} - {r['id2']}: d = {r['d']*100:.2f}%, Delta T = {r['delta_t']:.2f} yr | Shared DRMs: {r['n_shared_drms']} | Shared Unusual: 0 | Info Score: {r['info_score']:.1f}")


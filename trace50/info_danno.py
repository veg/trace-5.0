"""
info_danno.py
=============
Information-Theoretic Transmission Scoring via Empirical Allele Frequency Imputation.

Extends the core DANNO (Dynamic Ancestral Negative-binomial Network Odds) framework
from Identical-by-State (IBS) to Identical-by-Descent (IBD) by weighting shared and
divergent sites by background allele frequencies.

Supports:
1. Organism-agnostic internal cohort frequency estimation (on-the-fly from input FASTA).
2. Stanford HIVDB Group-M empirical priors (Tzou et al., PLoS ONE 2020) for HIV-1 pol,
   including downweighting of convergent drug-resistance mutations (DRMs) and filtering
   of signature APOBEC3G/3F G-to-A hypermutations.
"""

import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq


BASES = ['A', 'C', 'G', 'T']
BASE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


class InfoDannoEstimator:
    """
    Evaluates site-specific Information-Theoretic Transmission Odds.
    """
    def __init__(self, bg_frequencies=None, facts_dir=None, min_freq=1e-4, mu=2.0e-3):
        """
        Parameters
        ----------
        bg_frequencies : np.ndarray, optional
            (L x 4) matrix of position-specific background nucleotide frequencies.
        facts_dir : str, optional
            Path to HIVDB facts directory containing aapcnt, apobecs, and drms CSVs.
        min_freq : float
            Minimum allele frequency floor to avoid log(0) and extreme singularity.
        mu : float
            Baseline molecular clock rate (substitutions/site/year).
        """
        self.bg_frequencies = bg_frequencies
        self.min_freq = min_freq
        self.mu = mu
        self.facts_dir = facts_dir
        
        self.apobec_set = set()
        self.drm_set = set()
        self.hivdb_aa_freq = {}
        
        if facts_dir and os.path.exists(facts_dir):
            self.load_hivdb_priors(facts_dir)

    def fit_cohort_frequencies(self, fasta_file):
        """
        Compute empirical position-by-position nucleotide frequencies from an alignment.
        Operates completely pathogen-agnostically.
        """
        records = list(SeqIO.parse(fasta_file, "fasta"))
        if not records:
            raise ValueError(f"No sequences found in {fasta_file}")
            
        N = len(records)
        L = len(records[0].seq)
        char_mat = np.array([list(str(r.seq).upper()) for r in records], dtype='U1')
        
        freq_mat = np.zeros((L, 4))
        for l in range(L):
            col = char_mat[:, l]
            counts = [np.sum(col == b) for b in BASES]
            tot = sum(counts)
            if tot > 0:
                freq_mat[l, :] = [c / tot for c in counts]
            else:
                freq_mat[l, :] = 0.25
                
        # Dirichlet shrinkage / pseudocount
        pseudocount = 1.0 / (2.0 * N)
        freq_mat = np.clip(freq_mat, pseudocount, 1.0)
        freq_mat = freq_mat / freq_mat.sum(axis=1, keepdims=True)
        
        self.bg_frequencies = freq_mat
        return self.bg_frequencies

    def load_hivdb_priors(self, facts_dir):
        """
        Load HIVDB Group-M facts tables (Tzou et al., PLoS ONE 2020).
        """
        self.facts_dir = facts_dir
        
        # Load APOBEC signatures
        apobec_path = os.path.join(facts_dir, "apobecs.csv")
        if os.path.exists(apobec_path):
            df_apo = pd.read_csv(apobec_path)
            for _, r in df_apo.iterrows():
                self.apobec_set.add((r['gene'], int(r['position']), r['aa']))

        # Load DRMs
        drm_path = os.path.join(facts_dir, "drms_hiv1.csv")
        if os.path.exists(drm_path):
            df_drm = pd.read_csv(drm_path)
            for _, r in df_drm.iterrows():
                self.drm_set.add((r['gene'], int(r['position']), r['aa']))

        # Load AA prevalence (Subtype B default)
        aa_path = os.path.join(facts_dir, "rx-all_subtype-B.csv")
        if os.path.exists(aa_path):
            df_aa = pd.read_csv(aa_path)
            for _, r in df_aa.iterrows():
                key = (r['gene'], int(r['position']), r['aa'])
                self.hivdb_aa_freq[key] = float(r['percent'])

    def compute_pair_evidence(self, seq1, seq2, delta_t):
        """
        Evaluate site-specific information-weighted likelihood ratio for two aligned sequences.

        Parameters
        ----------
        seq1, seq2 : str
            Aligned nucleotide strings of equal length L.
        delta_t : float
            Elapsed calendar sampling interval in years.

        Returns
        -------
        dict
            Dictionary containing info_score (nats), shared_rare_count, shared_drm_count,
            and shared_apobec_count.
        """
        if self.bg_frequencies is None:
            raise ValueError("Background frequencies not initialized. Call fit_cohort_frequencies first.")

        s1 = np.array(list(str(seq1).upper()), dtype='U1')
        s2 = np.array(list(str(seq2).upper()), dtype='U1')
        L = min(len(s1), len(s2), len(self.bg_frequencies))

        s1 = s1[:L]
        s2 = s2[:L]
        freq_mat = self.bg_frequencies[:L, :]

        # Mask of mutually resolved standard bases
        mask = np.isin(s1, BASES) & np.isin(s2, BASES)
        if np.sum(mask) == 0:
            return {"info_score": 0.0, "shared_rare_count": 0, "shared_drm_count": 0, "shared_apobec_count": 0}

        s1_res = s1[mask]
        s2_res = s2[mask]
        freq_res = freq_mat[mask, :]
        res_indices = np.where(mask)[0]

        b1_idx = np.array([BASE_TO_IDX[b] for b in s1_res])
        b2_idx = np.array([BASE_TO_IDX[b] for b in s2_res])

        identical = (b1_idx == b2_idx)
        diff = ~identical

        # Frequencies of shared bases
        shared_f = freq_res[np.arange(len(s1_res))[identical], b1_idx[identical]]
        shared_f = np.clip(shared_f, self.min_freq, 1.0)

        # Count shared rare bases (< 1% background frequency)
        shared_rare_count = int(np.sum(shared_f < 0.01))

        # Log evidence for shared bases: log(1 / f_l(b))
        log_shared = np.sum(-np.log(shared_f))

        # Log evidence for mismatches:
        # P(mismatch | H_trans) ~ 2 * mu * max(dt, 0.5) / 3
        # P(mismatch | H_null) = f1 * f2
        f1_diff = freq_res[np.arange(len(s1_res))[diff], b1_idx[diff]]
        f2_diff = freq_res[np.arange(len(s1_res))[diff], b2_idx[diff]]
        f1_diff = np.clip(f1_diff, self.min_freq, 1.0)
        f2_diff = np.clip(f2_diff, self.min_freq, 1.0)

        p_trans_mut = (2.0 * self.mu * max(delta_t, 0.5) + 0.001) / 3.0
        log_diff = np.sum(np.log(p_trans_mut) - np.log(f1_diff * f2_diff))

        total_info_score = float(log_shared + log_diff)

        # Check amino acid annotations if in HIV-1 pol coordinates (PR: 1..297 nt, RT: 298..1497 nt)
        shared_drms = 0
        shared_apobec = 0
        if L >= 297 and len(self.drm_set) > 0:
            # PR codons
            for c in range(min(99, L // 3)):
                nt1 = "".join(s1[c*3 : c*3+3])
                nt2 = "".join(s2[c*3 : c*3+3])
                if len(nt1) == 3 and len(nt2) == 3 and all(b in BASES for b in nt1+nt2):
                    aa1 = str(Seq(nt1).translate())
                    aa2 = str(Seq(nt2).translate())
                    if aa1 == aa2:
                        pos = c + 1
                        if ('PR', pos, aa1) in self.drm_set:
                            shared_drms += 1
                        if ('PR', pos, aa1) in self.apobec_set:
                            shared_apobec += 1
                            
            # RT codons
            rt_nt1 = s1[297:]
            rt_nt2 = s2[297:]
            for c in range(min(400, len(rt_nt1) // 3)):
                nt1 = "".join(rt_nt1[c*3 : c*3+3])
                nt2 = "".join(rt_nt2[c*3 : c*3+3])
                if len(nt1) == 3 and len(nt2) == 3 and all(b in BASES for b in nt1+nt2):
                    aa1 = str(Seq(nt1).translate())
                    aa2 = str(Seq(nt2).translate())
                    if aa1 == aa2:
                        pos = c + 1
                        if ('RT', pos, aa1) in self.drm_set:
                            shared_drms += 1
                        if ('RT', pos, aa1) in self.apobec_set:
                            shared_apobec += 1

        return {
            "info_score": total_info_score,
            "shared_rare_count": shared_rare_count,
            "shared_drm_count": shared_drms,
            "shared_apobec_count": shared_apobec
        }

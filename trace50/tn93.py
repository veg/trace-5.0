"""
Tamura-Nei 93 (TN93) Distance Calculations for TRACE-5.0.

Provides:
- Fast vectorized pairwise TN93 nucleotide substitution distances.
- IUPAC nucleotide ambiguity resolution.
- Continuous simplex TN93 distance against probabilistic ancestral profiles.
- Logarithmic saturation boundaries and Hamming fallback safeguards.
"""

import numpy as np

# Nucleotide mapping
NUC_TO_INT = {
    'A': 0, 'C': 1, 'G': 2, 'T': 3,
    'R': 4, 'Y': 5, 'S': 6, 'W': 7, 'K': 8, 'M': 9,
    'B': 10, 'D': 11, 'H': 12, 'V': 13, 'N': 14,
    '-': -1, '?': -1
}

# Bitmask representations for IUPAC codes (A=1, C=2, G=4, T=8)
IUPAC_MASKS = {
    'A': 0b0001, 'C': 0b0010, 'G': 0b0100, 'T': 0b1000,
    'R': 0b0101,  # A or G
    'Y': 0b1010,  # C or T
    'S': 0b0110,  # C or G
    'W': 0b1001,  # A or T
    'K': 0b1100,  # G or T
    'M': 0b0011,  # A or C
    'B': 0b1110,  # C, G, or T
    'D': 0b1101,  # A, G, or T
    'H': 0b1011,  # A, C, or T
    'V': 0b0111,  # A, C, or G
    'N': 0b1111,  # Any
    '-': 0b0000,
    '?': 0b0000,
}

PURINE_MASK = 0b0101      # A or G
PYRIMIDINE_MASK = 0b1010  # C or T


def parse_fasta(filepath):
    """
    Parses a FASTA file into a list of (header, sequence) tuples.
    """
    records = []
    with open(filepath, 'r') as f:
        header, seq_chunks = None, []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    records.append((header, "".join(seq_chunks).upper()))
                header = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            records.append((header, "".join(seq_chunks).upper()))
    return records


def encode_alignment(records):
    """
    Encodes a list of (header, sequence) records into integer and bitmask matrices.

    Returns:
        headers (list of str)
        M_int (np.ndarray of shape (N, L), int8: 0=A, 1=C, 2=G, 3=T, -1=other)
        M_bits (np.ndarray of shape (N, L), uint8: IUPAC bitmask)
    """
    if not records:
        return [], np.zeros((0, 0), dtype=np.int8), np.zeros((0, 0), dtype=np.uint8)

    headers = [r[0] for r in records]
    seqs = [r[1] for r in records]
    N = len(seqs)
    L = len(seqs[0])

    M_int = np.full((N, L), -1, dtype=np.int8)
    M_bits = np.zeros((N, L), dtype=np.uint8)

    for i, seq in enumerate(seqs):
        for j, char in enumerate(seq[:L]):
            M_int[i, j] = NUC_TO_INT.get(char, -1)
            M_bits[i, j] = IUPAC_MASKS.get(char, 0)

    return headers, M_int, M_bits


def compute_pairwise_tn93(seq_a_bits, seq_b_bits):
    """
    Computes analytical TN93 distance between two IUPAC bitmask-encoded sequences.
    Resolves ambiguities to minimize distance (matching characters share bits).
    """
    # Valid overlap: both must not be gaps
    valid = (seq_a_bits > 0) & (seq_b_bits > 0)
    L_valid = np.sum(valid)
    if L_valid == 0:
        return 0.0

    a = seq_a_bits[valid]
    b = seq_b_bits[valid]

    # Shared characters: if (a & b) > 0, divergence is 0
    diff = (a & b) == 0
    if not np.any(diff):
        return 0.0

    a_diff = a[diff]
    b_diff = b[diff]

    # Transitions vs transversions
    # Purine transitions: (A & G)
    p1_mask = ((a_diff & PURINE_MASK) > 0) & ((b_diff & PURINE_MASK) > 0)
    # Pyrimidine transitions: (C & T)
    p2_mask = ((a_diff & PYRIMIDINE_MASK) > 0) & ((b_diff & PYRIMIDINE_MASK) > 0)
    # Transversions: purine to pyrimidine or vice versa
    q_mask = ~p1_mask & ~p2_mask

    P1 = np.sum(p1_mask) / L_valid
    P2 = np.sum(p2_mask) / L_valid
    Q = np.sum(q_mask) / L_valid
    p_ham = P1 + P2 + Q

    # Empirical nucleotide frequencies across valid positions
    a_and_b = a | b
    cnt_A = np.sum((a_and_b & 0b0001) > 0)
    cnt_C = np.sum((a_and_b & 0b0010) > 0)
    cnt_G = np.sum((a_and_b & 0b0100) > 0)
    cnt_T = np.sum((a_and_b & 0b1000) > 0)
    tot = max(cnt_A + cnt_C + cnt_G + cnt_T, 1)

    f_A = max(cnt_A / tot, 1e-4)
    f_C = max(cnt_C / tot, 1e-4)
    f_G = max(cnt_G / tot, 1e-4)
    f_T = max(cnt_T / tot, 1e-4)

    f_R = f_A + f_G
    f_Y = f_C + f_T

    k1 = 2.0 * f_A * f_G / f_R
    k2 = 2.0 * f_C * f_T / f_Y
    k3 = 2.0 * (f_R * f_Y - (f_A * f_G * f_Y / f_R) - (f_C * f_T * f_R / f_Y))

    arg1 = 1.0 - (f_R / (2.0 * f_A * f_G)) * P1 - (Q / (2.0 * f_R))
    arg2 = 1.0 - (f_Y / (2.0 * f_C * f_T)) * P2 - (Q / (2.0 * f_Y))
    arg3 = 1.0 - (Q / (2.0 * f_R * f_Y))

    # Boundary check: if log argument <= 0, clamp to Hamming distance
    if arg1 <= 1e-6 or arg2 <= 1e-6 or arg3 <= 1e-6:
        return float(p_ham)

    d = -k1 * np.log(arg1) - k2 * np.log(arg2) - k3 * np.log(arg3)
    if not np.isfinite(d) or d < 0:
        return float(p_ham)
    return float(d)


def compute_tn93_distance_matrix(M_bits):
    """
    Computes the full pairwise TN93 distance matrix across all sequences.
    """
    N = M_bits.shape[0]
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            dist = compute_pairwise_tn93(M_bits[i], M_bits[j])
            D[i, j] = dist
            D[j, i] = dist
    return D


def compute_simplex_tn93_divergences(M_int, times, gamma=None):
    """
    Computes root-to-tip divergences from a continuous nucleotide probability simplex profile.

    Ancestral state at site l is the continuous probability vector Q_l \\in \\Delta^4:
        Q_l(b) = \\frac{\\sum_i w_i(\\gamma) \\mathbb{I}(s_{i, l} == b)}{\\sum_i w_i(\\gamma)}
    with exponential time-decay weights:
        w_i(\\gamma) = \\exp\\left(-\\gamma \\frac{t_i - t_{\\min}}{\\bar{t} - t_{\\min}}\\right).

    Continuous transition rates:
        P_{1, i} = \\frac{1}{L} \\sum_l (s_{i, l, A} Q_{l, G} + s_{i, l, G} Q_{l, A})
        P_{2, i} = \\frac{1}{L} \\sum_l (s_{i, l, C} Q_{l, T} + s_{i, l, T} Q_{l, C})
        Q_{\\text{tv}, i} = \\frac{1}{L} \\sum_l [(s_{i,l,A} + s_{i,l,G})(Q_{l,C} + Q_{l,T}) + (s_{i,l,C} + s_{i,l,T})(Q_{l,A} + Q_{l,G})]

    Returns:
        distances (np.ndarray of shape (N,))
        optimal_gamma (float)
    """
    N, L = M_int.shape
    if N == 0:
        return np.zeros(0, dtype=np.float64), 0.0

    t_min = float(np.min(times))
    t_span = float(np.max(times) - t_min)
    t_denom = max(float(np.mean(times) - t_min), 1e-4)

    is_A = (M_int == 0)
    is_C = (M_int == 1)
    is_G = (M_int == 2)
    is_T = (M_int == 3)
    valid_mask = (M_int >= 0)
    L_valid = np.maximum(np.sum(valid_mask, axis=1), 1)

    def _eval(g):
        weights = np.exp(-g * (times - t_min) / t_denom)
        w_sum = np.sum(weights)
        w = weights / w_sum if w_sum > 0 else np.ones(N) / N

        # Construct continuous simplex profile Q of shape (L, 4)
        Q = np.zeros((L, 4), dtype=np.float64)
        for b in range(4):
            Q[:, b] = np.sum((M_int == b) * w[:, None], axis=0)
        q_sum = np.sum(Q, axis=1, keepdims=True)
        zero_pos = (q_sum.squeeze() <= 0)
        Q[~zero_pos, :] /= q_sum[~zero_pos]
        Q[zero_pos, :] = 0.25

        # Analytical continuous expectations
        P1 = np.sum(is_A * Q[None, :, 2] + is_G * Q[None, :, 0], axis=1) / L_valid
        P2 = np.sum(is_C * Q[None, :, 3] + is_T * Q[None, :, 1], axis=1) / L_valid
        Q_tv = np.sum(
            (is_A | is_G) * (Q[None, :, 1] + Q[None, :, 3]) +
            (is_C | is_T) * (Q[None, :, 0] + Q[None, :, 2]),
            axis=1
        ) / L_valid

        # Pooled base frequencies
        mean_Q = np.mean(Q, axis=0)
        f_A = np.maximum(0.5 * (mean_Q[0] + np.sum(is_A, axis=1) / L_valid), 1e-4)
        f_C = np.maximum(0.5 * (mean_Q[1] + np.sum(is_C, axis=1) / L_valid), 1e-4)
        f_G = np.maximum(0.5 * (mean_Q[2] + np.sum(is_G, axis=1) / L_valid), 1e-4)
        f_T = np.maximum(0.5 * (mean_Q[3] + np.sum(is_T, axis=1) / L_valid), 1e-4)
        f_R = f_A + f_G
        f_Y = f_C + f_T

        arg1 = 1.0 - (f_R / (2.0 * f_A * f_G)) * P1 - (Q_tv / (2.0 * f_R))
        arg2 = 1.0 - (f_Y / (2.0 * f_C * f_T)) * P2 - (Q_tv / (2.0 * f_Y))
        arg3 = 1.0 - (Q_tv / (2.0 * f_R * f_Y))

        k1 = 2.0 * f_A * f_G / f_R
        k2 = 2.0 * f_C * f_T / f_Y
        k3 = 2.0 * (f_R * f_Y - (f_A * f_G * f_Y / f_R) - (f_C * f_T * f_R / f_Y))

        p_ham = P1 + P2 + Q_tv
        valid_log = (arg1 > 1e-6) & (arg2 > 1e-6) & (arg3 > 1e-6)
        d = np.zeros(N, dtype=np.float64)
        d[~valid_log] = p_ham[~valid_log]
        d[valid_log] = (
            -k1[valid_log] * np.log(arg1[valid_log])
            - k2[valid_log] * np.log(arg2[valid_log])
            - k3[valid_log] * np.log(arg3[valid_log])
        )
        invalid_d = (~np.isfinite(d)) | (d < 0)
        d[invalid_d] = p_ham[invalid_d]
        return d, g

    if gamma is not None:
        return _eval(float(gamma))

    # Grid search over candidate gamma decay parameters
    if t_span > 0 and N >= 4:
        cand_gammas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        best_r = -1e9
        best_d = None
        best_g = cand_gammas[2]
        var_t = np.var(times)

        for g_cand in cand_gammas:
            d_c, _ = _eval(g_cand)
            var_d = np.var(d_c)
            if var_t > 0 and var_d > 0:
                r = float(np.cov(times, d_c)[0, 1] / np.sqrt(var_t * var_d))
            else:
                r = -1.0
            if r > best_r:
                best_r = r
                best_g = g_cand
                best_d = d_c

        return (best_d if best_d is not None else _eval(best_g)[0]), best_g
    else:
        return _eval(1.0)

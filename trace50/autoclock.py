"""
Recursive AutoClock Engine for TRACE-5.0.

Implements:
- Normalized Graph Laplacian spectral partitioning with local adaptive bandwidth.
- Newman-Girvan modularity optimization across Fiedler Cheeger cuts.
- Decoupled structural bisection (independent of parent clock linearity).
- Continuous nucleotide simplex profile rooting with time-decay weighting.
- Bartlett effective sample size (N_eff) phylogenetic covariance penalty.
- Fieller's theorem exact confidence set inversion for evolutionary clock velocity and t_MRCA.
"""

import numpy as np
from scipy import stats
from scipy.linalg import eigh

from .tn93 import compute_simplex_tn93_divergences


def compute_fieller_mrca(mu, d0, se_mu, se_d0, cov_mud0, t_ref, df, alpha=0.05):
    """
    Computes exact Fieller theorem confidence intervals for t_MRCA = t_ref - (d0 / mu).

    Let theta = d0 / mu. Ratio of two correlated Gaussian estimators.
    Fieller's statistic g = (t_crit * se_mu / mu)^2.
    When g < 1, roots form a closed finite interval.
    When g >= 1, the rate is unidentifiable, yielding unbounded confidence rays.
    """
    if df < 1 or abs(mu) < 1e-12:
        return [np.nan, np.nan], "UNIDENTIFIABLE_RATE"

    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    g = (t_crit * se_mu / mu) ** 2

    if g >= 1.0:
        return [-np.inf, np.inf], "RATE_UNIDENTIFIABLE_G_GE_1"

    # Fieller's quadratic solution for theta = d0 / mu
    denom = 1.0 - g
    center = (d0 / mu) - (g * cov_mud0) / (mu * (se_mu ** 2))
    discriminant = (
        ((d0 / mu) - (g * cov_mud0) / (mu * (se_mu ** 2))) ** 2
        - (1.0 - g) * ((d0 ** 2 - (t_crit ** 2) * (se_d0 ** 2)) / (mu ** 2))
    )

    if discriminant < 0:
        return [-np.inf, np.inf], "COMPLEX_ROOTS"

    delta = np.sqrt(discriminant)
    theta_lo = (center - delta) / denom
    theta_hi = (center + delta) / denom

    # t_MRCA = t_ref - theta
    ci_lo = float(t_ref - theta_hi)
    ci_hi = float(t_ref - theta_lo)

    return [ci_lo, ci_hi], "OK"


def compute_bartlett_neff(D_sub, decay_param=100.0):
    """
    Computes Bartlett effective sample size N_eff = N^2 / (1^T C(lambda) 1)
    under spatial/phylogenetic exponential covariance C(lambda)_{ij} = exp(-lambda * D_{ij}).

    Args:
        D_sub: Pairwise distance matrix of shape (N, N)
        decay_param: Decay rate in substitutions^-1 (default: 100.0, corresponding to
                     a correlation length scale of ~0.01 substitutions/site).
    """
    N = D_sub.shape[0]
    if N <= 2:
        return float(N)

    lam = float(decay_param)
    C = np.exp(-lam * D_sub)
    denom = np.sum(C)
    if denom <= 0:
        return float(N)

    n_eff = (float(N) ** 2) / float(denom)
    return float(max(1.0, min(float(N), n_eff)))


def fit_clock_community(sub_idx, dates, D_matrix, M_int=None, is_anchor=None, alpha=0.05):
    """
    Fits root-to-tip linear regression with continuous simplex profile rooting and Fieller bounds.
    Tracks local vs anchor composition when is_anchor mask is supplied.
    """
    n = len(sub_idx)
    sub_dates = dates[sub_idx]
    D_sub = D_matrix[np.ix_(sub_idx, sub_idx)]

    if is_anchor is not None:
        sub_is_anchor = is_anchor[sub_idx]
        n_anc = int(np.sum(sub_is_anchor))
        n_loc = int(np.sum(~sub_is_anchor))
        sub_anc = [int(sub_idx[i]) for i in range(n) if sub_is_anchor[i]]
        sub_loc = [int(sub_idx[i]) for i in range(n) if not sub_is_anchor[i]]
        is_hybrid = bool(n_anc > 0 and n_loc > 0)
    else:
        n_anc = 0
        n_loc = n
        sub_anc = []
        sub_loc = [int(i) for i in sub_idx]
        is_hybrid = False

    if n < 3:
        t_val = float(sub_dates.min()) if n > 0 else 0.0
        return {
            "n": n, "n_local": n_loc, "n_anchor": n_anc, "is_hybrid": is_hybrid,
            "anchor_indices": sub_anc, "local_indices": sub_loc,
            "n_eff": float(n), "mu": 0.0, "se": 0.0, "ci_mu": [0.0, 0.0], "r2": 0.0, "rss": 0.0,
            "tmrca": t_val, "ci_mrca": [t_val, t_val], "fieller_status": "TOO_SMALL",
            "mean_dist": 0.0, "span": 0.0, "gamma": 0.0
        }

    nonzero = D_sub[D_sub > 0]
    mean_d = float(nonzero.mean()) if len(nonzero) > 0 else 0.0
    span = float(sub_dates.max() - sub_dates.min())

    # Root distances via continuous simplex profile or earliest isolate
    if M_int is not None:
        root_dists, eff_gamma = compute_simplex_tn93_divergences(M_int[sub_idx], sub_dates)
    else:
        earliest_idx = int(np.argmin(sub_dates))
        root_dists = D_sub[earliest_idx]
        eff_gamma = 0.0

    t_mean = float(np.mean(sub_dates))
    d_mean = float(np.mean(root_dists))
    s_tt = float(np.sum((sub_dates - t_mean) ** 2))
    s_td = float(np.sum((sub_dates - t_mean) * (root_dists - d_mean)))
    s_dd = float(np.sum((root_dists - d_mean) ** 2))

    n_eff = compute_bartlett_neff(D_sub)
    df_eff = max(1.0, n_eff - 2.0)

    if s_tt < 1e-12 or s_dd < 1e-12:
        t_val = float(sub_dates.min())
        return {
            "n": n, "n_local": n_loc, "n_anchor": n_anc, "is_hybrid": is_hybrid,
            "anchor_indices": sub_anc, "local_indices": sub_loc,
            "n_eff": n_eff, "mu": 0.0, "se": 0.0, "ci_mu": [0.0, 0.0], "r2": 0.0, "rss": s_dd,
            "tmrca": t_val, "ci_mrca": [t_val, t_val], "fieller_status": "ZERO_VARIANCE",
            "mean_dist": mean_d, "span": span, "gamma": eff_gamma
        }

    mu = s_td / s_tt
    rss = max(0.0, s_dd - mu * s_td)
    r2 = max(0.0, min(1.0, (s_td ** 2) / (s_tt * s_dd)))

    sig2 = rss / df_eff
    se_mu = np.sqrt(sig2 / s_tt)
    se_d0 = np.sqrt(sig2 / n_eff)
    cov_mud0 = 0.0  # Zero covariance because centered at t_mean

    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df_eff))
    ci_mu = [float(mu - t_crit * se_mu), float(mu + t_crit * se_mu)]

    ci, fieller_status = compute_fieller_mrca(
        mu=mu, d0=d_mean, se_mu=se_mu, se_d0=se_d0, cov_mud0=cov_mud0,
        t_ref=t_mean, df=df_eff, alpha=alpha
    )
    tmrca = float(t_mean - d_mean / mu) if abs(mu) > 1e-12 else float(sub_dates.min())

    return {
        "n": n, "n_local": n_loc, "n_anchor": n_anc, "is_hybrid": is_hybrid,
        "anchor_indices": sub_anc, "local_indices": sub_loc,
        "n_eff": n_eff, "mu": float(mu), "se": float(se_mu), "ci_mu": ci_mu, "r2": float(r2),
        "rss": float(rss), "tmrca": float(tmrca), "ci_mrca": ci,
        "fieller_status": fieller_status, "mean_dist": mean_d,
        "span": span, "gamma": float(eff_gamma)
    }


def optimize_fiedler_modularity(W, fiedler, min_size=3):
    """
    Sweeps candidate split thresholds along the sorted Fiedler vector to maximize
    the Newman-Girvan modularity Q.

    Returns:
        best_c0 (np.ndarray of indices)
        best_c1 (np.ndarray of indices)
        best_modularity (float)
        modularity_gain (float)
    """
    N = W.shape[0]
    if N < 2 * min_size:
        return np.arange(N), np.array([], dtype=np.int64), 0.0, 0.0

    degrees = np.sum(W, axis=1)
    two_m = float(np.sum(degrees))
    if two_m <= 1e-12:
        return np.arange(N), np.array([], dtype=np.int64), 0.0, 0.0

    # Unsplit base modularity: single community
    # Q_base = (two_m - two_m^2 / two_m) / two_m = 0.0
    q_base = 0.0

    sorted_indices = np.argsort(fiedler)

    best_q = -1e9
    best_split_k = -1

    # Cumulative sums along sorted order for fast modularity evaluation
    deg_sorted = degrees[sorted_indices]
    W_sorted = W[np.ix_(sorted_indices, sorted_indices)]

    # Iterate candidate cut points
    for k in range(min_size, N - min_size + 1):
        # Community 0 is indices [0:k], Community 1 is [k:N]
        k0_deg = np.sum(deg_sorted[:k])
        k1_deg = np.sum(deg_sorted[k:])

        # Internal edges: sum of upper-left block and lower-right block
        a00 = np.sum(W_sorted[:k, :k])
        a11 = np.sum(W_sorted[k:, k:])

        # Modularity Q = 1/(2m) * [ (a00 - k0^2/(2m)) + (a11 - k1^2/(2m)) ]
        q = (1.0 / two_m) * (
            (a00 - (k0_deg ** 2) / two_m) +
            (a11 - (k1_deg ** 2) / two_m)
        )

        if q > best_q:
            best_q = q
            best_split_k = k

    if best_split_k < 0:
        return np.arange(N), np.array([], dtype=np.int64), 0.0, 0.0

    c0 = sorted_indices[:best_split_k]
    c1 = sorted_indices[best_split_k:]
    delta_q = best_q - q_base

    return c0, c1, float(best_q), float(delta_q)


def recursive_autoclock_deconvolution(
    sub_idx, dates, D_matrix, M_int=None, is_anchor=None,
    min_size=4, max_depth=10, depth=0, path="root",
    conductance_cut=0.65
):
    """
    Recursively partitions sequence populations along structural graph spectral bottlenecks.

    Decoupled Stopping Rules:
    - Bisection proceeds if: N >= 2 * min_size, depth < max_depth, delta_Q > 0, and conductance <= cut.
    - Leaf communities are then dated and classified via continuous profile rooting.
    """
    n = len(sub_idx)
    fit_summary = fit_clock_community(sub_idx, dates, D_matrix, M_int=M_int, is_anchor=is_anchor)

    if depth >= max_depth or n < 2 * min_size:
        return [(sub_idx, fit_summary, "leaf_size_floor", path)]

    D_sub = D_matrix[np.ix_(sub_idx, sub_idx)]
    nonzero = D_sub[D_sub > 0]
    if len(nonzero) == 0:
        return [(sub_idx, fit_summary, "identical_submanifold", path)]

    # Local adaptive bandwidth: 15th percentile with 0.005 floor
    sigma = max(float(np.percentile(nonzero, 15)), 0.005)
    W = np.exp(-(D_sub ** 2) / (2.0 * sigma ** 2))
    np.fill_diagonal(W, 0.0)

    degrees = np.sum(W, axis=1)
    if np.any(degrees <= 1e-12):
        return [(sub_idx, fit_summary, "disconnected_graph", path)]

    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    L_sym = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

    try:
        evals, evecs = eigh(L_sym)
    except Exception:
        return [(sub_idx, fit_summary, "eigen_failure", path)]

    if len(evals) < 2:
        return [(sub_idx, fit_summary, "trivial_spectrum", path)]

    lambda_2 = float(evals[1])
    fiedler = evecs[:, 1]

    # Newman-Girvan modularity optimization along Fiedler cut
    c0, c1, best_q, delta_q = optimize_fiedler_modularity(W, fiedler, min_size=min_size)

    if len(c0) < min_size or len(c1) < min_size:
        return [(sub_idx, fit_summary, "modularity_size_floor", path)]

    # Cheeger conductance of the cut
    cut_weight = float(np.sum(W[np.ix_(c0, c1)]))
    vol_c0 = float(np.sum(degrees[c0]))
    vol_c1 = float(np.sum(degrees[c1]))
    conductance = cut_weight / max(min(vol_c0, vol_c1), 1e-12)

    # Decoupled structural stopping condition:
    # If modularity does not increase or conductance is too high, preserve community as cohesive unit
    if delta_q <= 0.0 or conductance > conductance_cut:
        return [(sub_idx, fit_summary, f"cohesive_bottleneck_h{conductance:.2f}", path)]

    # Structural partition accepted: recurse down both branches
    left_leaves = recursive_autoclock_deconvolution(
        sub_idx[c0], dates, D_matrix, M_int=M_int, is_anchor=is_anchor,
        min_size=min_size, max_depth=max_depth, depth=depth + 1, path=f"{path}.0",
        conductance_cut=conductance_cut
    )
    right_leaves = recursive_autoclock_deconvolution(
        sub_idx[c1], dates, D_matrix, M_int=M_int, is_anchor=is_anchor,
        min_size=min_size, max_depth=max_depth, depth=depth + 1, path=f"{path}.1",
        conductance_cut=conductance_cut
    )

    return left_leaves + right_leaves


# Canonical aliases: STEVE (Spectral Temporal Evolutionary Velocity Engine)
STEVE = recursive_autoclock_deconvolution
steve_deconvolution = recursive_autoclock_deconvolution

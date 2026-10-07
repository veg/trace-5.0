"""
trace50.simulator: Epidemiological Transmission & Surveillance Data Simulator.

Generates physical, forward coalescent-transmission dyads and surveillance data pathologies
to establish the operating envelope of DANNO (Dynamic Ancestral Negative-binomial Network Odds)
against static distance thresholds.
"""

import numpy as np
from scipy import stats
import pandas as pd


def simulate_transmission_dyad(
    donor_stage="acute",
    sampling_lag="concurrent",
    L=1000,
    mu=2.0e-3,
    tau_bar=1.0
):
    """
    Simulates an authentic transmission dyad under Kingman's within-host coalescent
    and the molecular clock substitution process.

    Parameters:
        donor_stage (str): 'acute' (T_donor in [0.05, 0.5] yr) or 'chronic' (T_donor in [1.0, 5.0] yr).
        sampling_lag (str): 'concurrent' (dt in [0.0, 0.25] yr) or 'delayed' (dt in [1.0, 4.0] yr).
        L (int): Alignment length (nucleotides).
        mu (float): Retroviral substitution rate (substitutions/site/year).
        tau_bar (float): Mean intra-host coalescent depth (years).

    Returns:
        dict: Physical transmission parameters and simulated sequence divergence.
    """
    if donor_stage == "acute":
        t_donor = np.random.uniform(0.05, 0.50)
    elif donor_stage == "chronic":
        t_donor = np.random.uniform(1.00, 5.00)
    else:
        t_donor = float(donor_stage)

    # Within-host coalescent time strictly bounded by donor duration of infection
    tau = min(np.random.exponential(tau_bar), t_donor)

    if sampling_lag == "concurrent":
        dt = np.random.uniform(0.0, 0.25)
    elif sampling_lag == "delayed":
        dt = np.random.uniform(1.0, 4.00)
    else:
        dt = float(sampling_lag)

    # Total evolutionary branch length connecting the two sampled isolates
    branch_len = dt + 2.0 * tau
    k = int(np.random.poisson(mu * L * branch_len))
    dist = float(k / L)

    return {
        "k": k,
        "dist": dist,
        "dt": float(dt),
        "tau": float(tau),
        "t_donor": float(t_donor),
        "stage": donor_stage,
        "lag": sampling_lag,
        "true_link": True,
        "is_pathology": False
    }


def simulate_unlinked_pair(L=1000, t_span=20.0, r_bg=5.0, p_bg=0.95):
    """
    Simulates an unlinked background pair drawn independently from the circulating epidemic.
    """
    k = int(stats.nbinom.rvs(r_bg, 1.0 - p_bg))
    # Exact triangular distribution for difference of two independent uniforms on [0, T_span]
    dt = float(stats.triang.rvs(c=0, scale=t_span))
    dist = float(k / L)
    return {
        "k": k,
        "dist": dist,
        "dt": dt,
        "tau": np.nan,
        "t_donor": np.nan,
        "stage": "background",
        "lag": "unlinked",
        "true_link": False,
        "is_pathology": False
    }


def simulate_wo_fat_pathology(L=1000, t_span=20.0, error_rate=0.0005):
    """
    Simulates a 'Wo Fat' surveillance data pathology:
    Identical or near-identical sequences (k in {0, 1}) spanning multi-year intervals,
    modeling laboratory cross-contamination (e.g. HXB2), specimen intake duplicates,
    or unevolved archival proviral latency rebound.
    """
    k = int(np.random.poisson(L * error_rate))
    # Sampled across multi-year surveillance intervals (1.0 to T_span years)
    dt = float(np.random.uniform(1.0, min(t_span, 8.0)))
    dist = float(k / L)
    return {
        "k": k,
        "dist": dist,
        "dt": dt,
        "tau": 0.0,
        "t_donor": 0.0,
        "stage": "pathology",
        "lag": "stagnant",
        "true_link": False,
        "is_pathology": True
    }


def generate_benchmark_suite(
    n_per_class=1000,
    L=1000,
    mu=2.0e-3,
    tau_bar=1.0,
    t_span=20.0,
    r_bg=5.0,
    p_bg=0.95,
    seed=42
):
    """
    Generates a comprehensive benchmark cohort spanning all four transmission regimes,
    unlinked background controls, and 'Wo Fat' surveillance data pathologies.
    """
    np.random.seed(seed)
    records = []

    # Scenario 1: Acute Concurrent
    for _ in range(n_per_class):
        rec = simulate_transmission_dyad("acute", "concurrent", L, mu, tau_bar)
        rec["scenario"] = "Scenario 1: Acute Concurrent"
        records.append(rec)

    # Scenario 2: Acute Delayed
    for _ in range(n_per_class):
        rec = simulate_transmission_dyad("acute", "delayed", L, mu, tau_bar)
        rec["scenario"] = "Scenario 2: Acute Delayed"
        records.append(rec)

    # Scenario 3: Chronic Concurrent
    for _ in range(n_per_class):
        rec = simulate_transmission_dyad("chronic", "concurrent", L, mu, tau_bar)
        rec["scenario"] = "Scenario 3: Chronic Concurrent"
        records.append(rec)

    # Scenario 4: Chronic Delayed
    for _ in range(n_per_class):
        rec = simulate_transmission_dyad("chronic", "delayed", L, mu, tau_bar)
        rec["scenario"] = "Scenario 4: Chronic Delayed"
        records.append(rec)

    # Unlinked Background
    for _ in range(n_per_class):
        rec = simulate_unlinked_pair(L, t_span, r_bg, p_bg)
        rec["scenario"] = "Unlinked Background"
        records.append(rec)

    # Wo Fat Data Pathologies
    for _ in range(n_per_class):
        rec = simulate_wo_fat_pathology(L, t_span)
        rec["scenario"] = "Wo Fat Pathology (Stagnant)"
        records.append(rec)

    return pd.DataFrame(records)


def compute_operating_envelope_grid(
    estimator,
    max_dist=0.035,
    max_dt=5.0,
    n_dist=120,
    n_dt=120,
    prior_odds=0.01
):
    """
    Computes a fine 2D evaluation grid in (distance, delta_t) space,
    returning Bayes Factors, link probabilities, and physical adequacy contours.
    """
    d_vals = np.linspace(0.0, max_dist, n_dist)
    dt_vals = np.linspace(0.0, max_dt, n_dt)
    D_grid, T_grid = np.meshgrid(d_vals, dt_vals)

    bf_grid = np.zeros_like(D_grid)
    prob_grid = np.zeros_like(D_grid)
    adequate_grid = np.zeros_like(D_grid, dtype=bool)

    for i in range(n_dt):
        dt = dt_vals[i]
        for j in range(n_dist):
            d = d_vals[j]
            k = int(round(d * estimator.seq_len))
            _, is_adeq = estimator.check_clock_adequacy(k, dt)
            bf = estimator.compute_bayes_factor(d, dt)
            prob = estimator.compute_link_probability(d, dt, prior_odds)

            bf_grid[i, j] = bf
            prob_grid[i, j] = prob
            adequate_grid[i, j] = is_adeq

    return D_grid, T_grid, bf_grid, prob_grid, adequate_grid

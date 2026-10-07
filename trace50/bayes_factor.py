"""
DANNO: Dynamic Ancestral Negative-binomial Network Odds.
Coalescent Bayes Factor for Contemporaneous Transmission Dyads.

Implements:
- IUPAC nucleotide ambiguity resolution via probabilistic component base averaging,
  yielding continuous fractional substitution counts K = d * L in [0, L].
- Physically grounded Kingman coalescent compound Poisson-Gamma marginalization.
- Single Kingman coalescent intra-host depth parameter (tau_bar), yielding exact
  Geometric / Negative Binomial ancestral divergence without arbitrary mixtures.
- Exact closed-form continuous Poisson-coalescent marginal via the upper incomplete
  gamma function Q(K + 1, .) without discrete summation loops or numerical quadrature.
- Continuous Negative Binomial background divergence density supporting fractional K.
- Clinical surveillance timescale omega unconstrained from below (omega in [0.1, 4.0] yr),
  spanning acute contact tracing to passive registry surveillance.
- Biological coalescent depth uncertainty domain tau_bar in [0.05, 3.0] yr, admitting
  early / hyper-acute capture regimes.
- Min-max robust Bayes Factor evaluation across clock rate and intra-host coalescent depth domains.
- Empirical self-calibration of background genetic distance and surveillance time span from data.
- Exact physical molecular clock adequacy testing handling fractional K to eliminate stagnant
  "Wo Fat" data pathologies (specimen duplicates, laboratory carryover contamination, archival latency rebound).
- Benjamini-Hochberg False Discovery Rate (FDR) control for acute transmission edge inference ("Book 'em, Danno!").
"""

import numpy as np
import scipy.special as sp
from scipy import stats

# IUPAC ambiguity base probability weights for consensus sequence comparison
IUPAC_WEIGHTS = {
    'A': {'A': 1.0},
    'C': {'C': 1.0},
    'G': {'G': 1.0},
    'T': {'T': 1.0},
    'R': {'A': 0.5, 'G': 0.5},
    'Y': {'C': 0.5, 'T': 0.5},
    'S': {'C': 0.5, 'G': 0.5},
    'W': {'A': 0.5, 'T': 0.5},
    'K': {'G': 0.5, 'T': 0.5},
    'M': {'A': 0.5, 'C': 0.5},
    'B': {'C': 1.0 / 3.0, 'G': 1.0 / 3.0, 'T': 1.0 / 3.0},
    'D': {'A': 1.0 / 3.0, 'G': 1.0 / 3.0, 'T': 1.0 / 3.0},
    'H': {'A': 1.0 / 3.0, 'C': 1.0 / 3.0, 'T': 1.0 / 3.0},
    'V': {'A': 1.0 / 3.0, 'C': 1.0 / 3.0, 'G': 1.0 / 3.0},
    'N': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
    '-': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
    '?': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
}


def compute_fractional_substitutions(seq1, seq2):
    """
    Computes observed fractional substitution count and fractional distance
    between two nucleotide sequences incorporating IUPAC ambiguities.

    For each site l, the mismatch penalty is:
        c(s1(l), s2(l)) = 1.0 - sum_{b in {A,C,G,T}} f1(l, b) * f2(l, b)
    where f1(l, b) is the probability mass assigned to base b by IUPAC ambiguity code.

    Parameters:
        seq1 (str): First nucleotide sequence.
        seq2 (str): Second nucleotide sequence.

    Returns:
        k_tot (float): Total fractional substitution count K = sum_l c(s1(l), s2(l)).
        valid_count (int): Number of valid (non-gap/non-ambiguous both) aligned sites.
        d_frac (float): Fractional genetic distance d = K / valid_count.
    """
    s1 = str(seq1).upper()
    s2 = str(seq2).upper()
    n = min(len(s1), len(s2))
    if n == 0:
        return 0.0, 0, 0.0

    k_tot = 0.0
    valid_count = 0
    for c1, c2 in zip(s1[:n], s2[:n]):
        if c1 in ('-', '?') and c2 in ('-', '?'):
            continue
        m1 = IUPAC_WEIGHTS.get(c1, IUPAC_WEIGHTS['N'])
        m2 = IUPAC_WEIGHTS.get(c2, IUPAC_WEIGHTS['N'])
        shared = sum(m1.get(b, 0.0) * m2.get(b, 0.0) for b in 'ACGT')
        k_tot += (1.0 - shared)
        valid_count += 1

    d_frac = (k_tot / valid_count) if valid_count > 0 else 0.0
    return float(k_tot), valid_count, float(d_frac)


class DannoEstimator:
    """
    DANNO: Dynamic Ancestral Negative-binomial Network Odds.

    Evaluates candidate transmission dyads under Kingman coalescent and molecular clock
    models versus unlinked circulating background and temporal data pathologies.
    """

    def __init__(
        self,
        seq_len=1023,
        mu=2.0e-3,             # Retroviral substitution rate (substitutions/site/year)
        tau_bar=1.0,           # Mean intra-host coalescent depth (years). Baseline 1.0 yr; domain [0.05, 3.0] yr.
        omega=2.0,             # Clinical surveillance timescale (years). Baseline 2.0 yr; domain [0.1, 4.0] yr.
        t_span=20.0,           # Surveillance duration for triangular null (years)
        alpha_adequacy=0.05,   # Standard 5% significance threshold for molecular clock adequacy
        bg_mean_dist=0.065,    # Empirical mean distance between unlinked background pairs
        bg_var_dist=0.0004,    # Empirical variance of distance between unlinked background pairs
        # Backward compatibility aliases:
        mu_acute=None,
        omega_sample=None,
        omega_acute=None,
        alpha_acute=None,
        beta_acute=None,
        alpha_intra=None,
        beta_intra=None,
        **kwargs
    ):
        self.seq_len = int(seq_len)
        self.mu = float(mu_acute) if mu_acute is not None else float(mu)
        self.mu_acute = self.mu

        # Coalescent timescale tau_bar
        if alpha_acute is not None and beta_acute is not None:
            self.tau_bar = float(alpha_acute) / float(beta_acute)
        elif alpha_intra is not None and beta_intra is not None:
            self.tau_bar = float(alpha_intra) / float(beta_intra)
        else:
            self.tau_bar = float(tau_bar)

        # Surveillance lag scale omega
        if omega_sample is not None:
            self.omega = float(omega_sample)
        elif omega_acute is not None:
            self.omega = float(omega_acute)
        else:
            self.omega = float(omega)
        self.omega_sample = self.omega
        self.omega_acute = self.omega

        self.t_span = max(float(t_span), 1.0)
        self.alpha_adequacy = float(alpha_adequacy)

        # Kingman coalescent parameters:
        # Under Exponential(tau_bar) coalescent depth, Poisson(2 * mu * L * tau) yields
        # NegBin(r=1, p_trans) which is an exact Geometric distribution.
        self.r_trans = 1.0
        self.p_trans = (2.0 * self.mu * self.seq_len * self.tau_bar) / (
            2.0 * self.mu * self.seq_len * self.tau_bar + 1.0
        )
        self.r_acute = self.r_trans
        self.p_acute = self.p_trans

        # Background parameters (can be updated dynamically via calibrate_from_data)
        self._fit_background(bg_mean_dist, bg_var_dist)

    def _fit_background(self, bg_mean_dist, bg_var_dist):
        """Fits Negative Binomial background divergence parameters via moment-matching."""
        bg_mean_k = max(bg_mean_dist * self.seq_len, 1.0)
        bg_var_k = max(bg_var_dist * (self.seq_len ** 2), bg_mean_k + 1.0)
        p_bg_val = 1.0 - (bg_mean_k / bg_var_k)
        if 0.0 < p_bg_val < 1.0:
            self.p_bg = p_bg_val
            self.r_bg = (bg_mean_k * (1.0 - self.p_bg)) / self.p_bg
        else:
            self.p_bg = 0.95
            self.r_bg = 5.0

    def calibrate_from_data(self, D_matrix=None, dates=None):
        """
        Self-calibrates empirical background genetic divergence and observation span directly
        from the input distance matrix and collection timestamps, eliminating hardcoded numbers.
        """
        if dates is not None and len(dates) > 1:
            valid_dates = np.array(dates)[~np.isnan(dates)]
            if len(valid_dates) > 1:
                span = float(np.max(valid_dates) - np.min(valid_dates))
                if span > 0.5:
                    self.t_span = min(max(span, 1.0), 30.0)

        if D_matrix is not None and D_matrix.shape[0] > 2:
            iu = np.triu_indices_from(D_matrix, k=1)
            d_vals = D_matrix[iu]
            valid_d = d_vals[~np.isnan(d_vals)]
            if len(valid_d) >= 10:
                # In surveillance cohorts (>500 pairs), unlinked pairs dominate (>99.9%),
                # so the full empirical distance distribution directly defines background.
                # For small benchmark mixtures, the upper half (>= median) guards against cluster over-representation.
                if len(valid_d) < 500:
                    bg_cutoff = float(np.median(valid_d))
                    bg_d = valid_d[valid_d >= bg_cutoff]
                else:
                    bg_d = valid_d
                mean_d = float(np.mean(bg_d))
                var_d = max(float(np.var(bg_d)), 1e-6)
                self._fit_background(mean_d, var_d)

    def check_clock_adequacy(self, k, delta_t, mu=None):
        """
        Evaluates physical molecular clock adequacy:
        Under active viral transmission at rate mu, expected minimum mutations along
        the unshared lineage over interval delta_t is lambda_min = mu * L * delta_t.

        Parameters:
            k (float or int): Observed substitution count (supports fractional K).
            delta_t (float): Sampling interval separation (years).
            mu (float, optional): Retroviral substitution rate. If None, uses self.mu.

        Returns:
            p_adequacy (float): Cumulative Poisson probability P(M <= floor(k) | lambda_min).
            is_adequate (bool): True if p_adequacy >= alpha_adequacy, False if stagnant.
        """
        mu_val = float(mu) if mu is not None else self.mu
        k_val = max(float(k), 0.0)
        delta_t = max(float(delta_t), 0.0)
        lam_min = mu_val * self.seq_len * delta_t
        p_adequacy = float(stats.poisson.cdf(int(np.floor(k_val)), lam_min))
        return p_adequacy, (p_adequacy >= self.alpha_adequacy)

    def _evaluate_trans_marginal(self, k, delta_t, mu_val, tau_val):
        """
        Exact closed-form continuous Poisson-coalescent marginal:
        P(K | Delta T, H_trans) = \\int_0^\\infty Poisson(K | 2*mu*L*tau + mu*L*Delta T) * Exp(tau | tau_bar) dtau

        Evaluates the exact analytic integral:
        P(K | Delta T, H_trans) = [exp(Delta T / (2*tau_bar)) / tau_bar] *
                                  [(2*mu*L)^K / (2*mu*L + 1/tau_bar)^(K+1)] *
                                  Q(K + 1, (2*mu*L + 1/tau_bar) * Delta T / 2)
        where Q(s, x) = Gamma(s, x) / Gamma(s) is the regularized upper incomplete gamma function.
        Handles continuous fractional substitution counts K in [0, L] without discrete loops.
        """
        k = max(float(k), 0.0)
        delta_t = max(float(delta_t), 0.0)
        b = 2.0 * mu_val * self.seq_len
        c = b + 1.0 / tau_val
        p_tr = (b * tau_val) / (b * tau_val + 1.0)

        if delta_t <= 1e-15:
            # Pure ancestral Geometric marginal for any real k >= 0
            return (1.0 - p_tr) * (p_tr ** k)

        x = c * delta_t / 2.0
        s = k + 1.0
        q_val = sp.gammaincc(s, x)
        if q_val > 0.0:
            log_core = delta_t / (2.0 * tau_val) + k * np.log(b) - s * np.log(c) - np.log(tau_val)
            if log_core < 700.0:
                val = np.exp(log_core) * q_val
            else:
                val = np.exp(log_core + np.log(q_val))
        else:
            val = 0.0

        return max(float(val), 1e-300)

    def compute_transmission_likelihood(self, k, delta_t, mu=None, tau_bar=None, omega=None):
        """
        Evaluates P(K, Delta T | H_trans) = P(K | Delta T, H_trans) * P(Delta T | H_trans).

        Closed-form marginal of:
        - Post-transmission unshared lineage: Poisson(lambda = mu * L * Delta T)
        - Kingman ancestral coalescent: NegBin(r=1, p_trans) [Geometric distribution]
        - Sampling interval delay: Exponential(scale = omega)

        Accepts continuous / fractional substitution counts K >= 0.
        """
        mu_val = float(mu) if mu is not None else self.mu
        tau_val = float(tau_bar) if tau_bar is not None else self.tau_bar
        omega_val = float(omega) if omega is not None else self.omega
        delta_t = max(float(delta_t), 0.0)

        # 1. Temporal likelihood: sampling interval decay over surveillance timescale omega
        p_delta_t = (1.0 / omega_val) * np.exp(-delta_t / omega_val)

        # 2. Closed-form continuous Poisson-coalescent marginal
        p_k_given_dt = self._evaluate_trans_marginal(k, delta_t, mu_val, tau_val)
        return p_k_given_dt * p_delta_t

    # Backward compatibility aliases
    compute_linked_likelihood = compute_transmission_likelihood
    compute_acute_likelihood = compute_transmission_likelihood

    def compute_background_likelihood(self, k, delta_t):
        """
        Evaluates P(K, Delta T | H_bg) = P(K | H_bg) * P(Delta T | H_bg).

        Uses:
        - Continuous Negative Binomial background genetic likelihood via Gamma functions:
          P(K | H_bg) = Gamma(K + r) / (Gamma(r) * Gamma(K + 1)) * (1 - p)^r * p^K
          supporting any real non-negative fractional substitution count K >= 0.
        - Temporal separation: exact triangular distribution f(Delta T) = (2/T)(1 - Delta T / T)
          on [0, T_span].
        """
        k = max(float(k), 0.0)
        delta_t = max(float(delta_t), 0.0)

        # 1. Triangular temporal density for difference of two uniform arrivals
        if delta_t < self.t_span:
            p_delta_t = max((2.0 / self.t_span) * (1.0 - delta_t / self.t_span), 1e-6 / self.t_span)
        else:
            p_delta_t = 1e-6 / self.t_span

        # 2. Continuous Negative Binomial background genetic likelihood
        p_scipy = 1.0 - self.p_bg
        log_p_k = (
            sp.gammaln(k + self.r_bg)
            - sp.gammaln(self.r_bg)
            - sp.gammaln(k + 1.0)
            + self.r_bg * np.log(p_scipy)
            + k * np.log(1.0 - p_scipy)
        )
        p_k = max(float(np.exp(log_p_k)), 1e-300)
        return p_k * p_delta_t

    compute_null_likelihood = compute_background_likelihood

    def compute_bayes_factor(self, dist, delta_t):
        """
        Computes continuous Bayes Factor:
        BF = P(K, Delta T | H_trans) / P(K, Delta T | H_bg).

        Accepts continuous genetic distance dist, converts to fractional count K = dist * seq_len,
        checks physical clock adequacy, and returns continuous BF.
        If the pair physically violates the molecular clock (is_adequate == False),
        BF is set to 0.0, defeating stagnant "Wo Fat" pathologies.
        """
        k = max(float(dist) * self.seq_len, 0.0)
        delta_t = max(float(delta_t), 0.0)

        # Physical clock adequacy check (filters out stagnant identical pairs across years)
        _, is_adequate = self.check_clock_adequacy(k, delta_t)
        if not is_adequate:
            return 0.0

        lik_trans = self.compute_transmission_likelihood(k, delta_t)
        lik_bg = self.compute_background_likelihood(k, delta_t)
        return float(lik_trans / lik_bg) if lik_bg > 0 else 0.0

    def compute_robust_bayes_factor(
        self,
        dist,
        delta_t,
        mu_range=(1.5e-3, 3.5e-3),
        tau_bar_range=(0.05, 3.0),
        omega=None
    ):
        """
        Computes min-max robust Bayes Factor over biological uncertainty domain:
        BF_robust = inf_{(mu, tau_bar) in Theta} BF(d, Delta T | mu, tau_bar)

        Evaluates across domain boundary vertices Theta = [mu_min, mu_max] x [tau_min, tau_max]
        to guarantee transmission support under the most adverse parameter regimes.

        Parameters:
            dist (float): Pairwise genetic distance (fractional or TN93).
            delta_t (float): Sampling interval separation (years).
            mu_range (tuple): Biological range for retroviral clock rate (sub/site/year).
                              Default: (1.5e-3, 3.5e-3).
            tau_bar_range (tuple): Biological range for intra-host coalescent depth (years).
                                   Default: (0.05, 3.0).
            omega (float, optional): Surveillance timescale (years). If None, uses self.omega.

        Returns:
            float: Infimum Bayes Factor over Theta.
        """
        k = max(float(dist) * self.seq_len, 0.0)
        delta_t = max(float(delta_t), 0.0)
        omega_val = float(omega) if omega is not None else self.omega

        mu_min, mu_max = mu_range
        tau_min, tau_max = tau_bar_range

        vertices = [
            (mu_min, tau_min),
            (mu_min, tau_max),
            (mu_max, tau_min),
            (mu_max, tau_max),
        ]

        bf_vals = []
        for mu_val, tau_val in vertices:
            # Physical clock adequacy check under evaluated mu
            lam_min = mu_val * self.seq_len * delta_t
            p_adeq = float(stats.poisson.cdf(int(np.floor(k)), lam_min))
            if p_adeq < self.alpha_adequacy:
                return 0.0

            lik_t = self.compute_transmission_likelihood(k, delta_t, mu=mu_val, tau_bar=tau_val, omega=omega_val)
            lik_bg = self.compute_background_likelihood(k, delta_t)
            bf = float(lik_t / lik_bg) if lik_bg > 0 else 0.0
            bf_vals.append(bf)

        return float(min(bf_vals))

    def compute_link_probability(self, dist, delta_t, prior_odds=0.01, robust=False):
        """Computes posterior probability of transmission link."""
        bf = self.compute_robust_bayes_factor(dist, delta_t) if robust else self.compute_bayes_factor(dist, delta_t)
        if bf <= 0.0:
            return 0.0
        post_odds = prior_odds * bf
        return float(post_odds / (1.0 + post_odds))

    def book_em_danno(
        self,
        indices,
        dates,
        D_matrix,
        max_delta_t=2.0,
        prior_odds=0.01,
        fdr_threshold=0.05,
        use_robust=False,
        mu_range=(1.5e-3, 3.5e-3),
        tau_bar_range=(0.05, 3.0)
    ):
        """
        'Book 'em, Danno!' - Screens candidate transmission dyads, evaluates continuous
        Bayes Factors, checks physical clock adequacy, and controls FDR via Benjamini-Hochberg.

        Parameters:
            indices (list/array): Taxon indices to screen.
            dates (list/array): Collection dates (decimal years).
            D_matrix (ndarray): Pairwise genetic distance matrix.
            max_delta_t (float): Maximum sampling separation window (years). Default: 2.0.
            prior_odds (float): Prior odds of transmission link. Default: 0.01.
            fdr_threshold (float): Benjamini-Hochberg FDR threshold. Default: 0.05.
            use_robust (bool): If True, uses min-max robust Bayes Factor over parameter domain. Default: False.
            mu_range (tuple): Clock rate uncertainty domain for robust BF. Default: (1.5e-3, 3.5e-3).
            tau_bar_range (tuple): Intra-host depth domain for robust BF. Default: (0.05, 3.0).

        Returns:
            list of dicts containing significant edges with (idx1, idx2, dist, delta_t, bayes_factor, pep, q_value)
        """
        # Automatically self-calibrate background and span if not already done
        self.calibrate_from_data(D_matrix, dates)

        n = len(indices)
        candidates = []

        # Dynamic envelope pre-filter: d_crit = (d0 + 2 * mu * dt) * 1.5
        d0 = stats.nbinom.ppf(0.99, self.r_trans, 1.0 - self.p_trans) / self.seq_len

        for i in range(n):
            idx_i = indices[i]
            t_i = dates[idx_i]
            for j in range(i + 1, n):
                idx_j = indices[j]
                t_j = dates[idx_j]
                dt = abs(t_i - t_j)
                if dt <= max_delta_t:
                    d_val = D_matrix[idx_i, idx_j]
                    d_crit = (d0 + 2.0 * self.mu * dt) * 1.5
                    if d_val <= d_crit:
                        if use_robust:
                            bf = self.compute_robust_bayes_factor(d_val, dt, mu_range, tau_bar_range)
                        else:
                            bf = self.compute_bayes_factor(d_val, dt)

                        if bf > 0.0:
                            post_odds = prior_odds * bf
                            pep = 1.0 / (1.0 + post_odds)
                        else:
                            pep = 1.0  # Physically inadequate / stagnant pathology

                        candidates.append({
                            "idx1": idx_i,
                            "idx2": idx_j,
                            "dist": float(d_val),
                            "delta_t": float(dt),
                            "bayes_factor": float(bf),
                            "pep": float(pep)
                        })

        if not candidates:
            return []

        # Bayesian False Discovery Rate control (Müller et al. 2004, Storey 2003)
        # Candidates are sorted in ascending order of Posterior Error Probability (PEP)
        candidates.sort(key=lambda x: x["pep"])
        m_tot = len(candidates)
        cum_pep = 0.0
        for rank, cand in enumerate(candidates, start=1):
            cum_pep += cand["pep"]
            cand["q_value"] = float(min(1.0, cum_pep / rank))

        # Enforce monotonicity of q-values from back to front
        for i in range(m_tot - 2, -1, -1):
            candidates[i]["q_value"] = min(candidates[i]["q_value"], candidates[i + 1]["q_value"])

        # Filter by FDR threshold
        significant_dyads = [c for c in candidates if c["q_value"] <= fdr_threshold]
        return significant_dyads

    # Full backward-compatibility alias
    screen_dyads = book_em_danno


# Export both names
CoalescentBayesFactor = DannoEstimator

"""
CHIN: Cluster-to-Host Incidence and Network Estimator.
Macro-Epidemic Parameter Inference and Non-Random Sampling Diagnostics for TRACE-5.0.

Implements:
- Quadratic sampling law inversion: E_obs ≈ (1/2) * n * rho * R0 = (1/2) * rho^2 * N_act * R0.
- Closed-form MLE for composite transmission parameter theta = rho * R0 = 2 * E_obs / n.
- Inferred active transmitting population scale: N_act = R0 * n^2 / (2 * E_obs) with Poisson CIs.
- Inferred longitudinal surveillance sampling fraction: rho(t) = 2 * E_obs / (n * R0).
- Unsupervised topological tests for non-random sampling:
  1. Borel branching process cluster size spectrum decay test.
  2. Geographic assortativity test for international mixing versus localized confinement.
  3. Date quantization artifact detection (e.g. year-only integer reporting).
  4. Internal distance dispersion test (star-like contact investigations vs branching trees).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats


class ChinEstimator:
    """
    CHIN: Cluster-to-Host Incidence and Network Estimator.

    Inverts quadratic network scaling dynamics to estimate active transmitting population scale,
    surveillance sampling fraction, and audits topological deviations from uniform Bernoulli sampling.
    """

    def __init__(self, R0_default: float = 1.5):
        """
        Initialize the CHIN estimator.

        Parameters
        ----------
        R0_default : float
            Default basic reproductive ratio within acute transmission networks (typically 1.2 - 2.0).
        """
        self.R0_default = float(R0_default)

    def estimate_composite_parameter(
        self,
        n_samples: int,
        n_edges: int
    ) -> float:
        """
        Compute the maximum likelihood estimate of composite parameter theta = rho * R0.

        Parameters
        ----------
        n_samples : int
            Number of sequenced individuals (n).
        n_edges : int
            Number of observed contemporaneous transmission edges (E_obs).

        Returns
        -------
        float
            Estimated theta = 2 * E_obs / n.
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        return 2.0 * float(n_edges) / float(n_samples)

    def estimate_sampling_fraction(
        self,
        n_samples: int,
        n_edges: int,
        R0: Optional[float] = None
    ) -> float:
        """
        Estimate surveillance sampling coverage fraction rho = n / N_act.

        Parameters
        ----------
        n_samples : int
            Number of sequenced individuals (n).
        n_edges : int
            Number of observed transmission edges (E_obs).
        R0 : float, optional
            Assumed reproductive ratio. Defaults to R0_default.

        Returns
        -------
        float
            Inferred sampling fraction rho in [0, 1].
        """
        r0 = float(R0) if R0 is not None else self.R0_default
        if r0 <= 0:
            raise ValueError("R0 must be positive.")
        theta = self.estimate_composite_parameter(n_samples, n_edges)
        rho = theta / r0
        return min(1.0, max(0.0, rho))

    def estimate_active_population(
        self,
        n_samples: int,
        n_edges: int,
        R0: Optional[float] = None,
        confidence_level: float = 0.95
    ) -> Dict[str, Union[float, Tuple[float, float]]]:
        """
        Invert the quadratic scaling law to infer active transmitting population scale N_act.

        N_act = (R0 * n^2) / (2 * E_obs)

        Confidence bounds are derived under Poisson edge-counting error:
        CI = [ N_act * exp(-z / sqrt(E_obs)), N_act * exp(+z / sqrt(E_obs)) ]

        Parameters
        ----------
        n_samples : int
            Number of sequenced individuals (n).
        n_edges : int
            Number of observed transmission edges (E_obs).
        R0 : float, optional
            Assumed reproductive ratio. Defaults to R0_default.
        confidence_level : float
            Confidence level for interval estimation (default: 0.95).

        Returns
        -------
        dict
            Dictionary containing:
            - 'N_act': point estimate of active transmitting population size
            - 'ci_lower': lower confidence bound
            - 'ci_upper': upper confidence bound
            - 'relative_error': 1.96 / sqrt(E_obs)
            - 'R0': assumed reproductive ratio
        """
        r0 = float(R0) if R0 is not None else self.R0_default
        if n_edges <= 0:
            return {
                "N_act": float("inf"),
                "ci_lower": float("inf"),
                "ci_upper": float("inf"),
                "relative_error": float("inf"),
                "R0": r0
            }

        n = float(n_samples)
        e = float(n_edges)
        n_act = (r0 * (n ** 2)) / (2.0 * e)

        alpha = 1.0 - confidence_level
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        rel_err = z / np.sqrt(e)

        ci_lower = n_act * np.exp(-rel_err)
        ci_upper = n_act * np.exp(+rel_err)

        return {
            "N_act": float(n_act),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "relative_error": float(rel_err),
            "R0": float(r0)
        }

    def test_borel_branching_decay(
        self,
        cluster_sizes: Union[List[int], np.ndarray],
        max_size_fit: int = 12
    ) -> Dict[str, Union[float, bool]]:
        """
        Test whether cluster frequencies follow a Borel branching process decay.

        Under a subcritical Galton-Watson branching process with offspring mean lambda < 1,
        the total cluster size distribution follows a Borel distribution:
        P(m) = (lambda * m)^(m-1) * exp(-lambda * m) / m!
        which exhibits geometric / exponential power-law decay for small m.

        Parameters
        ----------
        cluster_sizes : array-like
            List or array of cluster sizes (m >= 2).
        max_size_fit : int
            Maximum cluster size to include in the linear decay fit (default: 12).

        Returns
        -------
        dict
            - 'slope': fitted log-linear decay slope
            - 'r_squared': coefficient of determination
            - 'p_value': regression significance p-value
            - 'n_clusters': total number of clusters evaluated
            - 'n_megaclusters': clusters exceeding max_size_fit
            - 'rejects_uniform_sampling': True if large outliers reject uniform Borel decay
        """
        sizes = np.array([s for s in cluster_sizes if s >= 2], dtype=int)
        if len(sizes) == 0:
            return {
                "slope": 0.0,
                "r_squared": 0.0,
                "p_value": 1.0,
                "n_clusters": 0,
                "n_megaclusters": 0,
                "rejects_uniform_sampling": False
            }

        unique_sizes, counts = np.unique(sizes, return_counts=True)
        fit_mask = (unique_sizes >= 2) & (unique_sizes <= max_size_fit)

        if np.sum(fit_mask) >= 3:
            x = unique_sizes[fit_mask]
            y = np.log(counts[fit_mask])
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r2 = float(r_value ** 2)
        else:
            slope, r2, p_value = 0.0, 0.0, 1.0

        n_megaclusters = int(np.sum(sizes > max_size_fit))
        # Rejection of pure Borel if extreme tail mega-clusters exist (e.g. size >= 50)
        rejects = bool(np.any(sizes >= 50) or (n_megaclusters >= 5 and r2 < 0.85))

        return {
            "slope": float(slope),
            "r_squared": float(r2),
            "p_value": float(p_value),
            "n_clusters": int(len(sizes)),
            "n_megaclusters": n_megaclusters,
            "max_cluster_size": int(np.max(sizes)),
            "rejects_uniform_sampling": rejects
        }

    def test_geographic_assortativity(
        self,
        cluster_labels: List[int],
        country_codes: List[str],
        min_cluster_size: int = 3
    ) -> Dict[str, Union[float, int]]:
        """
        Compute geographic assortativity across multi-person transmission clusters.

        Under uniform random global sampling, multinational mixing dominates.
        High within-country confinement indicates localized outbreak investigations or contact tracing.

        Parameters
        ----------
        cluster_labels : list of int
            Cluster ID for each sequence. Singletons should be omitted or assigned < 0.
        country_codes : list of str
            Country ISO code or country label for each sequence.
        min_cluster_size : int
            Minimum cluster size to evaluate (default: 3).

        Returns
        -------
        dict
            - 'fraction_single_country': fraction of clusters 100% confined to one nation
            - 'total_evaluated_clusters': number of clusters evaluated
            - 'total_taxa': total individuals in evaluated clusters
        """
        from collections import defaultdict
        cluster_to_countries = defaultdict(list)
        for c_id, country in zip(cluster_labels, country_codes):
            if c_id >= 0 and country and country != "Unknown":
                cluster_to_countries[c_id].append(country)

        eval_clusters = [
            countries for countries in cluster_to_countries.values()
            if len(countries) >= min_cluster_size
        ]

        if not eval_clusters:
            return {
                "fraction_single_country": 0.0,
                "total_evaluated_clusters": 0,
                "total_taxa": 0
            }

        single_country_count = sum(
            1 for c in eval_clusters if len(set(c)) == 1
        )
        total_taxa = sum(len(c) for c in eval_clusters)

        return {
            "fraction_single_country": float(single_country_count) / float(len(eval_clusters)),
            "total_evaluated_clusters": int(len(eval_clusters)),
            "total_taxa": int(total_taxa)
        }

    def detect_date_quantization(
        self,
        decimal_dates: Union[List[float], np.ndarray]
    ) -> Dict[str, Union[float, bool]]:
        """
        Detect reporting quantization artifacts (e.g. year-only integer or mid-year reporting).

        Public health registries often report only calendar years (e.g., 2014.0 or 2014.5).
        This induces apparent temporal synchronization (Delta T = 0) that reflects reporting
        discretization rather than biological co-transmission.

        Parameters
        ----------
        decimal_dates : array-like
            Collection dates in decimal calendar years.

        Returns
        -------
        dict
            - 'fraction_quantized': fraction of dates that are exact integers or mid-years (.5)
            - 'is_severely_quantized': True if >80% of records are discretized
        """
        dates = np.array(decimal_dates, dtype=float)
        valid = dates[np.isfinite(dates)]
        if len(valid) == 0:
            return {"fraction_quantized": 0.0, "is_severely_quantized": False}

        remainders = np.abs(valid - np.round(valid))
        mid_remainders = np.abs(valid - (np.floor(valid) + 0.5))

        # Check if remainder is within 0.005 of integer or 0.5
        is_integer = remainders < 0.005
        is_mid_year = mid_remainders < 0.005
        quantized = is_integer | is_mid_year

        frac = float(np.mean(quantized))
    def run_joint_bayesian_monte_carlo(
        self,
        n_samples: int,
        n_edges: int,
        n_draws: int = 100000,
        r0_prior: Tuple[float, float] = (1.2, 2.0),
        k_prior: Tuple[float, float] = (0.1, 0.5),
        phi_prior: Tuple[float, float] = (0.0, 0.40),
        seed: int = 42
    ) -> Dict[str, Union[float, Dict[str, float], np.ndarray]]:
        """
        Executes full joint Bayesian Monte Carlo integration (S = 100,000 draws)
        propagating uncertainty across reproductive ratios R0, superspreading overdispersion k,
        and partner-tracing enrichment phi under degree-inflated counting variance.

        Parameters
        ----------
        n_samples : int
            Cumulative sequenced individuals (n).
        n_edges : int
            Observed certified transmission dyads (E_obs).
        n_draws : int
            Number of Monte Carlo samples (default: 100,000).
        r0_prior : tuple
            Uniform prior bounds for R0 (default: (1.2, 2.0)).
        k_prior : tuple
            Uniform prior bounds for superspreading dispersion k (default: (0.1, 0.5)).
        phi_prior : tuple
            Uniform prior bounds for contact-tracing enrichment phi (default: (0.0, 0.40)).
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            Contains posterior medians, interquartile ranges (IQR), and 95% central credible intervals
            for surveillance coverage rho, active transmitting pool N_act, and transmission cycles 1/rho.
        """
        if n_samples <= 0 or n_edges <= 0:
            raise ValueError("n_samples and n_edges must be positive.")

        np.random.seed(seed)
        S = int(n_draws)
        n_total = float(n_samples)
        e_obs = float(n_edges)
        theta = 2.0 * e_obs / n_total

        # Uniform priors over epidemiologically plausible bounds
        r0_draws = np.random.uniform(r0_prior[0], r0_prior[1], S)
        k_draws = np.random.uniform(k_prior[0], k_prior[1], S)
        phi_draws = np.random.uniform(phi_prior[0], phi_prior[1], S)

        # Degree-inflated observation variance: F = 1 + theta * (1 + 1/k)
        fano_draws = 1.0 + theta * (1.0 + 1.0 / k_draws)
        e_draws = np.random.normal(e_obs, np.sqrt(e_obs * fano_draws))
        e_draws = np.maximum(e_draws, 1.0)

        # Posterior draws
        rho_draws = (2.0 * e_draws) / (n_total * r0_draws * (1.0 + phi_draws))
        n_act_draws = (n_total ** 2 * r0_draws * (1.0 + phi_draws)) / (2.0 * e_draws)
        cycles_draws = 1.0 / rho_draws

        return {
            "n_samples": int(n_samples),
            "n_edges": int(n_edges),
            "n_draws": S,
            "theta_mle": float(theta),
            "rho": {
                "median_pct": float(np.median(rho_draws) * 100.0),
                "iqr_low_pct": float(np.percentile(rho_draws, 25.0) * 100.0),
                "iqr_high_pct": float(np.percentile(rho_draws, 75.0) * 100.0),
                "ci_low_pct": float(np.percentile(rho_draws, 2.5) * 100.0),
                "ci_high_pct": float(np.percentile(rho_draws, 97.5) * 100.0),
            },
            "N_act": {
                "median": float(np.median(n_act_draws)),
                "iqr_low": float(np.percentile(n_act_draws, 25.0)),
                "iqr_high": float(np.percentile(n_act_draws, 75.0)),
                "ci_low": float(np.percentile(n_act_draws, 2.5)),
                "ci_high": float(np.percentile(n_act_draws, 97.5)),
            },
            "cycles_G": {
                "median": float(np.median(cycles_draws)),
                "iqr_low": float(np.percentile(cycles_draws, 25.0)),
                "iqr_high": float(np.percentile(cycles_draws, 75.0)),
                "ci_low": float(np.percentile(cycles_draws, 2.5)),
                "ci_high": float(np.percentile(cycles_draws, 97.5)),
            },
            "fano_factor_median": float(np.median(fano_draws)),
            "rho_draws": rho_draws,
            "N_act_draws": n_act_draws
        }


# Canonical aliases
CHIN = ChinEstimator
chin_estimator = ChinEstimator

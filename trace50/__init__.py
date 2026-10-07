"""
TRACE-5.0: Dynamic Molecular Surveillance of HIV-1 Transmission Chains at Registry Scale.
"""

__version__ = "5.0.0"

from .tn93 import (
    parse_fasta,
    encode_alignment,
    compute_pairwise_tn93,
    compute_tn93_distance_matrix,
    compute_simplex_tn93_divergences
)
from .bayes_factor import (
    CoalescentBayesFactor,
    DannoEstimator,
    compute_fractional_substitutions
)
from .info_danno import InfoDannoEstimator
from .landmark import select_maxmin_landmarks, nystrom_spectral_embedding
from .autoclock import (
    fit_clock_community,
    optimize_fiedler_modularity,
    recursive_autoclock_deconvolution,
    STEVE,
    steve_deconvolution,
    compute_fieller_mrca,
    compute_bartlett_neff
)
from .taxonomy import LineageTier, classify_subcommunity
from .chin import ChinEstimator, CHIN, chin_estimator
from .simulator import (
    simulate_transmission_dyad,
    simulate_unlinked_pair,
    simulate_wo_fat_pathology,
    generate_benchmark_suite,
    compute_operating_envelope_grid
)

__all__ = [
    "parse_fasta",
    "encode_alignment",
    "compute_pairwise_tn93",
    "compute_tn93_distance_matrix",
    "compute_simplex_tn93_divergences",
    "CoalescentBayesFactor",
    "DannoEstimator",
    "InfoDannoEstimator",
    "compute_fractional_substitutions",
    "select_maxmin_landmarks",
    "nystrom_spectral_embedding",
    "fit_clock_community",
    "optimize_fiedler_modularity",
    "recursive_autoclock_deconvolution",
    "STEVE",
    "steve_deconvolution",
    "compute_fieller_mrca",
    "compute_bartlett_neff",
    "LineageTier",
    "classify_subcommunity",
    "ChinEstimator",
    "CHIN",
    "chin_estimator",
    "simulate_transmission_dyad",
    "simulate_unlinked_pair",
    "simulate_wo_fat_pathology",
    "generate_benchmark_suite",
    "compute_operating_envelope_grid"
]

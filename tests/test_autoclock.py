import unittest
import numpy as np
from trace50.autoclock import (
    optimize_fiedler_modularity,
    recursive_autoclock_deconvolution,
    compute_bartlett_neff,
    compute_fieller_mrca,
    fit_clock_community
)

class TestAutoClock(unittest.TestCase):
    def test_modularity_dumbbell_split(self):
        # 10 taxa: 0..4 in cluster A, 5..9 in cluster B
        # High within-cluster affinity (W=1.0), low between-cluster affinity (W=0.01)
        W = np.full((10, 10), 0.01)
        W[:5, :5] = 1.0
        W[5:, 5:] = 1.0
        np.fill_diagonal(W, 0.0)

        # Simulated Fiedler vector separating the two halves
        fiedler = np.array([-0.3] * 5 + [0.3] * 5)
        c0, c1, best_q, delta_q = optimize_fiedler_modularity(W, fiedler, min_size=3)

        self.assertEqual(len(c0), 5)
        self.assertEqual(len(c1), 5)
        self.assertGreater(best_q, 0.35)
        self.assertGreater(delta_q, 0.0)
        self.assertTrue(set(c0) == set(range(5)) or set(c0) == set(range(5, 10)))

    def test_decoupled_stopping_rule_splits_composite(self):
        # Two distinct clades with flat individual clocks or unrelated dates
        # Parent clock has zero correlation (R2 ~ 0), but strong topological bottleneck
        N = 12
        D = np.full((N, N), 0.08)
        # Sub-cluster A: 0..5, internal dist = 0.005
        D[:6, :6] = 0.005
        # Sub-cluster B: 6..11, internal dist = 0.005
        D[6:, 6:] = 0.005
        np.fill_diagonal(D, 0.0)

        # Completely unclocked/random dates
        dates = np.array([2020.0, 2020.5, 2021.0, 2021.5, 2022.0, 2022.5] * 2)
        indices = np.arange(N)

        # Deconvolution must successfully bisect this into the two true clades
        leaves = recursive_autoclock_deconvolution(
            sub_idx=indices,
            dates=dates,
            D_matrix=D,
            min_size=3,
            max_depth=4
        )
        self.assertEqual(len(leaves), 2)
        leaf_sizes = sorted([len(leaf[0]) for leaf in leaves])
        self.assertEqual(leaf_sizes, [6, 6])

    def test_bartlett_neff_penalization(self):
        # Identical or near-identical sequences should have Neff < N
        D = np.full((10, 10), 0.0001)
        np.fill_diagonal(D, 0.0)
        neff = compute_bartlett_neff(D)
        self.assertLess(neff, 2.0)

        # Highly divergent sequences should have Neff close to N
        D_div = np.full((10, 10), 0.15)
        np.fill_diagonal(D_div, 0.0)
        neff_div = compute_bartlett_neff(D_div)
        self.assertGreater(neff_div, 7.0)

    def test_fieller_confidence_inversion(self):
        # Strong clock signal: mu = 2e-3, se = 2e-4 (t_slope = 10 >> t_crit)
        ci_strong, status_strong = compute_fieller_mrca(
            mu=2.0e-3, d0=0.01, se_mu=2.0e-4, se_d0=1.0e-3, cov_mud0=0.0,
            t_ref=2020.0, df=20, alpha=0.05
        )
        self.assertEqual(status_strong, "OK")
        self.assertTrue(np.isfinite(ci_strong[0]))
        self.assertTrue(np.isfinite(ci_strong[1]))
        self.assertLess(ci_strong[0], 2020.0)
        self.assertGreater(ci_strong[1], ci_strong[0])

        # Weak/flat clock signal: mu = 1e-5, se = 1e-3 (g >= 1)
        ci_weak, status_weak = compute_fieller_mrca(
            mu=1.0e-5, d0=0.01, se_mu=1.0e-3, se_d0=1.0e-3, cov_mud0=0.0,
            t_ref=2020.0, df=20, alpha=0.05
        )
        self.assertEqual(status_weak, "RATE_UNIDENTIFIABLE_G_GE_1")
        self.assertEqual(ci_weak, [-np.inf, np.inf])

if __name__ == "__main__":
    unittest.main()

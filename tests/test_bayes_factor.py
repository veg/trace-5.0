import unittest
import numpy as np
from trace50.bayes_factor import (
    CoalescentBayesFactor,
    DannoEstimator,
    compute_fractional_substitutions
)

class TestBayesFactor(unittest.TestCase):
    def setUp(self):
        self.bf_engine = CoalescentBayesFactor(
            seq_len=1000,
            mu_acute=2.0e-3,
            omega_sample=2.0,
            t_span=20.0
        )

    def test_identical_near_simultaneous_pair(self):
        # Identical sequences (dist = 0), sampled 1 month apart (dt = 0.08 yr)
        # Should have extremely strong support for H_linked (BF >> 100)
        bf = self.bf_engine.compute_bayes_factor(dist=0.0, delta_t=0.08)
        self.assertGreater(bf, 500.0)

    def test_biological_improbability_delayed_identical(self):
        # Identical sequences sampled across multi-year intervals:
        # Near simultaneous (dt=0.08y) -> BF > 500 (active transmission)
        # Delayed 2y -> BF < 15
        # Delayed 3y -> BF < 1.0 (favors null / data pathology)
        # Delayed 5y -> BF < 0.01 (strongly rejected)
        # Delayed 8y -> BF < 1e-5 (complete rejection)
        bf_near = self.bf_engine.compute_bayes_factor(dist=0.0, delta_t=0.08)
        bf_2y = self.bf_engine.compute_bayes_factor(dist=0.0, delta_t=2.0)
        bf_3y = self.bf_engine.compute_bayes_factor(dist=0.0, delta_t=3.0)
        bf_5y = self.bf_engine.compute_bayes_factor(dist=0.0, delta_t=5.0)
        bf_8y = self.bf_engine.compute_bayes_factor(dist=0.0, delta_t=8.0)

        self.assertGreater(bf_near, 500.0)
        self.assertLess(bf_2y, 15.0)
        self.assertLess(bf_3y, 1.0)
        self.assertLess(bf_5y, 0.01)
        self.assertLess(bf_8y, 1e-5)

    def test_delayed_acute_pair_preserved(self):
        # Acute pair delayed by 2 years: accumulates divergence (k=12 -> dist=0.012)
        # Model preserves the link without penalizing natural post-transmission drift
        bf = self.bf_engine.compute_bayes_factor(dist=0.012, delta_t=2.0)
        self.assertGreater(bf, 1000.0)
        prior_odds = 0.01
        prob = (prior_odds * bf) / (1.0 + prior_odds * bf)
        self.assertGreater(prob, 0.90)

    def test_chronic_donor_intrahost_diversity_preserved(self):
        # Chronic donor with pre-transmission intra-host diversity (dist=0.015, dt=0.1)
        # Mixture coalescent accommodates the pre-transmission depth
        bf = self.bf_engine.compute_bayes_factor(dist=0.015, delta_t=0.1)
        self.assertGreater(bf, 200.0)
        prior_odds = 0.01
        prob = (prior_odds * bf) / (1.0 + prior_odds * bf)
        self.assertGreater(prob, 0.70)

    def test_divergent_distant_pair(self):
        # Divergent sequences (dist = 0.065), sampled 8 years apart
        # Strongly favors background null (BF << 1)
        bf = self.bf_engine.compute_bayes_factor(dist=0.065, delta_t=8.0)
        self.assertLess(bf, 5e-4)

    def test_monotonicity_in_divergence_tail(self):
        # In the post-pathology divergence tail (dist >= 0.005), BF monotonically decreases
        bf1 = self.bf_engine.compute_bayes_factor(dist=0.005, delta_t=0.2)
        bf2 = self.bf_engine.compute_bayes_factor(dist=0.015, delta_t=0.2)
        bf3 = self.bf_engine.compute_bayes_factor(dist=0.030, delta_t=0.2)
        bf4 = self.bf_engine.compute_bayes_factor(dist=0.050, delta_t=0.2)
        self.assertGreater(bf1, bf2)
        self.assertGreater(bf2, bf3)
        self.assertGreater(bf3, bf4)

    def test_monotonicity_with_time_separation(self):
        # As delta_t increases for fixed acute distance (dist=0.002), BF monotonically decreases
        bf1 = self.bf_engine.compute_bayes_factor(dist=0.002, delta_t=0.1)
        bf2 = self.bf_engine.compute_bayes_factor(dist=0.002, delta_t=1.0)
        bf3 = self.bf_engine.compute_bayes_factor(dist=0.002, delta_t=3.0)
        self.assertGreater(bf1, bf2)
        self.assertGreater(bf2, bf3)

    def test_fdr_screening(self):
        # 4 taxa: pair (0, 1) is acute dyad; (2, 3) is acute dyad; across is background
        D = np.array([
            [0.0, 0.001, 0.060, 0.062],
            [0.001, 0.0, 0.059, 0.061],
            [0.060, 0.059, 0.0, 0.001],
            [0.062, 0.061, 0.001, 0.0]
        ])
        dates = np.array([2023.0, 2023.1, 2023.2, 2023.3])
        indices = np.arange(4)

        dyads = self.bf_engine.screen_dyads(
            indices=indices,
            dates=dates,
            D_matrix=D,
            max_delta_t=2.0,
            fdr_threshold=0.05
        )
        self.assertEqual(len(dyads), 2)
        pairs = {(d['idx1'], d['idx2']) for d in dyads}
        self.assertIn((0, 1), pairs)
        self.assertIn((2, 3), pairs)

    def test_fdr_screening_pathology_rejection(self):
        # Taxa 0 & 1: genuine acute dyad (dist=0.002, dt=0.1)
        # Taxa 2 & 3: identical pathology across 4 years (dist=0.0, dt=4.0)
        D = np.array([
            [0.0, 0.002, 0.060, 0.060],
            [0.002, 0.0, 0.060, 0.060],
            [0.060, 0.060, 0.0, 0.000],
            [0.060, 0.060, 0.000, 0.0]
        ])
        dates = np.array([2023.0, 2023.1, 2019.0, 2023.0])
        indices = np.arange(4)

        dyads = self.bf_engine.screen_dyads(
            indices=indices,
            dates=dates,
            D_matrix=D,
            max_delta_t=5.0,
            fdr_threshold=0.05
        )
        self.assertEqual(len(dyads), 1)
        self.assertEqual(dyads[0]['idx1'], 0)
        self.assertEqual(dyads[0]['idx2'], 1)

    def test_fractional_substitutions_iupac(self):
        # Mismatch penalty: c(s1, s2) = 1.0 - sum_b f1(b) * f2(b)
        # 'A' vs 'R' (R = 0.5 A + 0.5 G): penalty = 1.0 - 0.5 = 0.5
        k, valid, d = compute_fractional_substitutions('A', 'R')
        self.assertEqual(k, 0.5)
        self.assertEqual(valid, 1)
        self.assertEqual(d, 0.5)

        # 'R' vs 'R': penalty = 1.0 - (0.25 + 0.25) = 0.5
        k, valid, d = compute_fractional_substitutions('R', 'R')
        self.assertEqual(k, 0.5)
        self.assertEqual(valid, 1)

        # 'R' vs 'S' (R = A/G, S = C/G): shared G = 0.25 -> penalty = 0.75
        k, valid, d = compute_fractional_substitutions('R', 'S')
        self.assertEqual(k, 0.75)
        self.assertEqual(valid, 1)

        # Gaps skipped when both uninformative
        k, valid, d = compute_fractional_substitutions('ACGT-', 'ACGT-')
        self.assertEqual(k, 0.0)
        self.assertEqual(valid, 4)

    def test_fractional_substitutions_continuous_monotonicity(self):
        # Tests continuous Poisson-coalescent marginal for non-integer substitution counts
        k_vals = [0.0, 0.5, 1.0, 1.5, 2.0]
        bfs = [self.bf_engine.compute_bayes_factor(dist=k / 1000.0, delta_t=0.2) for k in k_vals]
        for i in range(len(bfs) - 1):
            self.assertGreater(bfs[i], bfs[i + 1])

    def test_early_capture_regime(self):
        # Early capture public health regime: tau_bar = 0.05 yr, omega = 0.1 yr
        danno_early = DannoEstimator(seq_len=1000, tau_bar=0.05, omega=0.1)
        bf_early = danno_early.compute_bayes_factor(dist=0.0, delta_t=0.02)
        self.assertGreater(bf_early, 1e9)

    def test_minmax_robust_bayes_factor(self):
        # Min-max robust Bayes Factor over Theta = [1.5e-3, 3.5e-3] x [0.05, 3.0]
        bf_base = self.bf_engine.compute_bayes_factor(dist=0.002, delta_t=0.2)
        bf_robust = self.bf_engine.compute_robust_bayes_factor(dist=0.002, delta_t=0.2)
        self.assertGreater(bf_robust, 1000.0)
        self.assertLessEqual(bf_robust, bf_base)

    def test_robust_fdr_screening(self):
        # Screens using min-max robust BF in book_em_danno
        D = np.array([
            [0.0, 0.001, 0.060, 0.062],
            [0.001, 0.0, 0.059, 0.061],
            [0.060, 0.059, 0.0, 0.001],
            [0.062, 0.061, 0.001, 0.0]
        ])
        dates = np.array([2023.0, 2023.1, 2023.2, 2023.3])
        indices = np.arange(4)

        dyads = self.bf_engine.book_em_danno(
            indices=indices,
            dates=dates,
            D_matrix=D,
            max_delta_t=2.0,
            fdr_threshold=0.05,
            use_robust=True
        )
        self.assertEqual(len(dyads), 2)
        pairs = {(d['idx1'], d['idx2']) for d in dyads}
        self.assertIn((0, 1), pairs)
        self.assertIn((2, 3), pairs)

    def test_clock_adequacy_fractional_k(self):
        # Under Delta T = 4.0 yr, lambda_min = 2e-3 * 1000 * 4.0 = 8.0
        # Fractional k = 0.5 substitutions: floor(0.5) = 0
        # Poisson CDF(0, 8.0) = exp(-8) = 0.000335 < 0.05 -> rejected
        p_val, is_adeq = self.bf_engine.check_clock_adequacy(k=0.5, delta_t=4.0)
        self.assertFalse(is_adeq)
        self.assertLess(p_val, 0.05)


if __name__ == "__main__":
    unittest.main()

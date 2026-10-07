import unittest
import numpy as np
from trace50.simulator import (
    simulate_transmission_dyad,
    simulate_unlinked_pair,
    simulate_wo_fat_pathology,
    generate_benchmark_suite,
    compute_operating_envelope_grid
)
from trace50.bayes_factor import DannoEstimator


class TestSimulator(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.estimator = DannoEstimator(seq_len=1000)

    def test_simulate_transmission_dyad(self):
        # Acute concurrent
        rec_acute = simulate_transmission_dyad("acute", "concurrent", L=1000)
        self.assertTrue(rec_acute["true_link"])
        self.assertFalse(rec_acute["is_pathology"])
        self.assertLessEqual(rec_acute["dt"], 0.25)
        self.assertLessEqual(rec_acute["t_donor"], 0.50)

        # Chronic delayed
        rec_chronic = simulate_transmission_dyad("chronic", "delayed", L=1000)
        self.assertTrue(rec_chronic["true_link"])
        self.assertGreaterEqual(rec_chronic["dt"], 1.0)
        self.assertGreaterEqual(rec_chronic["t_donor"], 1.0)

    def test_simulate_unlinked_pair(self):
        rec_bg = simulate_unlinked_pair(L=1000, t_span=20.0)
        self.assertFalse(rec_bg["true_link"])
        self.assertFalse(rec_bg["is_pathology"])
        self.assertGreaterEqual(rec_bg["dist"], 0.01)

    def test_simulate_wo_fat_pathology(self):
        rec_wf = simulate_wo_fat_pathology(L=1000, t_span=20.0)
        self.assertFalse(rec_wf["true_link"])
        self.assertTrue(rec_wf["is_pathology"])
        self.assertIn(rec_wf["k"], [0, 1, 2])
        self.assertGreaterEqual(rec_wf["dt"], 1.0)

    def test_generate_benchmark_suite(self):
        df = generate_benchmark_suite(n_per_class=50, L=1000)
        self.assertEqual(len(df), 300) # 6 classes * 50
        self.assertIn("Scenario 1: Acute Concurrent", set(df["scenario"]))
        self.assertIn("Wo Fat Pathology (Stagnant)", set(df["scenario"]))

    def test_compute_operating_envelope_grid(self):
        D_grid, T_grid, bf_grid, prob_grid, adeq_grid = compute_operating_envelope_grid(
            self.estimator, max_dist=0.03, max_dt=4.0, n_dist=20, n_dt=20
        )
        self.assertEqual(D_grid.shape, (20, 20))
        self.assertEqual(prob_grid.shape, (20, 20))
        # Top right (high dist, long time) should have low probability
        self.assertLess(prob_grid[-1, -1], 0.20)
        # Near simultaneous acute (low dist, small dt) should have high probability
        self.assertGreater(prob_grid[0, 1], 0.80)


if __name__ == "__main__":
    unittest.main()

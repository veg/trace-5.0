import unittest
import numpy as np
from trace50.tn93 import (
    compute_pairwise_tn93,
    encode_alignment,
    compute_simplex_tn93_divergences
)

class TestTN93(unittest.TestCase):
    def test_identical_sequences(self):
        records = [
            ("seq1", "ACGTACGTACGT"),
            ("seq2", "ACGTACGTACGT")
        ]
        _, _, M_bits = encode_alignment(records)
        d = compute_pairwise_tn93(M_bits[0], M_bits[1])
        self.assertAlmostEqual(d, 0.0, places=6)

    def test_known_transition_distance(self):
        # 1 transition in 100 bp => P1 = 0.01, Q = 0
        s1 = "A" * 50 + "C" * 50
        s2 = "A" * 49 + "G" + "C" * 50
        records = [("s1", s1), ("s2", s2)]
        _, _, M_bits = encode_alignment(records)
        d = compute_pairwise_tn93(M_bits[0], M_bits[1])
        self.assertGreater(d, 0.009)
        self.assertLess(d, 0.015)

    def test_ambiguity_resolution(self):
        # 'R' can be A or G; when compared to 'A', distance should be 0
        records = [
            ("s1", "ACGTACGT"),
            ("s2", "RCGTACGT")
        ]
        _, _, M_bits = encode_alignment(records)
        d = compute_pairwise_tn93(M_bits[0], M_bits[1])
        self.assertAlmostEqual(d, 0.0, places=6)

    def test_simplex_continuous_distance(self):
        # 4 sequences over 100 sites, sampled at different times
        np.random.seed(42)
        records = [
            ("t1", "ACGT" * 25),
            ("t2", "ACGT" * 25),
            ("t3", "ACGT" * 24 + "GCGT"),
            ("t4", "ACGT" * 23 + "GGGT")
        ]
        _, M_int, _ = encode_alignment(records)
        times = np.array([2020.0, 2021.0, 2022.0, 2023.0])
        dists, gamma = compute_simplex_tn93_divergences(M_int, times)
        self.assertEqual(len(dists), 4)
        self.assertTrue(np.all(dists >= 0.0))
        # Earliest sequences should be closer to root profile
        self.assertLessEqual(dists[0], dists[3])

if __name__ == "__main__":
    unittest.main()

import unittest
import os
import shutil
import tempfile
import json
import numpy as np
import pandas as pd

from trace50.reference import (
    load_reference_panel,
    merge_cohort_with_reference,
    test_metric_interleaving,
    resolve_reference_panel
)
from trace50.cli import run_pipeline


class TestReferenceArchitectureA(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.fasta_path = os.path.join(self.test_dir, "local_cohort.fasta")
        self.meta_path = os.path.join(self.test_dir, "local_meta.csv")
        self.out_dir = os.path.join(self.test_dir, "results")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_preset_resolution(self):
        fpath, mpath, desc = resolve_reference_panel("subtype_b")
        self.assertTrue(os.path.exists(fpath))
        self.assertTrue(os.path.exists(mpath))
        self.assertIn("Subtype B", desc)

        fpath_m, mpath_m, desc_m = resolve_reference_panel("multisubtype")
        self.assertTrue(os.path.exists(fpath_m))
        self.assertTrue(os.path.exists(mpath_m))
        self.assertIn("global anchors", desc_m)

    def test_load_builtin_reference_panel(self):
        records, meta_map, desc = load_reference_panel("subtype_b")
        self.assertEqual(len(records), 215)
        self.assertIn("B.US.1979", records[0][0])
        self.assertTrue(meta_map[records[0][0]]["is_anchor"])
        self.assertAlmostEqual(meta_map[records[0][0]]["date"], 1979.5, places=1)

    def test_metric_interleaving_severing(self):
        # 3 taxa: local 0, local 1, anchor 2
        # d(0, 1) = 0.015
        # d(0, 2) = 0.008 (anchor is closer to 0 than 1 is)
        # d(1, 2) = 0.009 (anchor is closer to 1 than 0 is)
        D = np.array([
            [0.000, 0.015, 0.008],
            [0.015, 0.000, 0.009],
            [0.008, 0.009, 0.000]
        ], dtype=float)
        is_anchor = np.array([False, False, True])

        is_wedged, wedging = test_metric_interleaving(0, 1, D, is_anchor)
        self.assertTrue(is_wedged)
        self.assertEqual(wedging, [2])

        # Another pair: local 0 and local 1 are very close (0.002), anchor is far (0.050)
        D_unwedged = np.array([
            [0.000, 0.002, 0.050],
            [0.002, 0.000, 0.050],
            [0.050, 0.050, 0.000]
        ], dtype=float)
        is_wedged2, wedging2 = test_metric_interleaving(0, 1, D_unwedged, is_anchor)
        self.assertFalse(is_wedged2)
        self.assertEqual(len(wedging2), 0)

    def test_pipeline_with_reference_panel(self):
        # Create a small cohort of 6 taxa
        # Taxa 1-3: Active local cluster (2022-2024, rate ~ 1e-3)
        # Taxa 4-6: Distant lineage co-clustering with historical Subtype B
        np.random.seed(101)
        bases = np.array(['A', 'C', 'G', 'T'])
        root_seq = "".join(np.random.choice(bases, size=1617))

        seqs = {}
        meta_rows = []

        # Local cluster
        c_seq = list(root_seq)
        for i, yr in enumerate([2021.5, 2022.5, 2023.5]):
            tid = f"LOCAL_OUTBREAK_{i+1}"
            if i > 0:
                c_seq[10 + i] = 'T' if c_seq[10 + i] != 'T' else 'C'
            seqs[tid] = "".join(c_seq)
            meta_rows.append({"id": tid, "date": yr})

        # Distant local cases
        d_seq = list(root_seq)
        for j in range(80):
            d_seq[j] = 'G' if d_seq[j] != 'G' else 'A'
        for i, yr in enumerate([2018.0, 2019.0, 2020.0]):
            tid = f"LOCAL_DISTANT_{i+1}"
            seqs[tid] = "".join(d_seq)
            meta_rows.append({"id": tid, "date": yr})

        with open(self.fasta_path, 'w') as f:
            for tid, seq in seqs.items():
                f.write(f">{tid}\n{seq}\n")

        df_meta = pd.DataFrame(meta_rows)
        df_meta.to_csv(self.meta_path, index=False)

        ret = run_pipeline(
            fasta_path=self.fasta_path,
            meta_path=self.meta_path,
            reference_panel="subtype_b",
            out_dir=self.out_dir,
            fdr_threshold=0.10,
            min_size=3
        )
        self.assertEqual(ret, 0)

        # Verify output files
        comm_csv = os.path.join(self.out_dir, "autoclock_communities.csv")
        pat_csv = os.path.join(self.out_dir, "patient_surveillance_assignments.csv")
        json_file = os.path.join(self.out_dir, "trace50_summary.json")

        self.assertTrue(os.path.exists(comm_csv))
        self.assertTrue(os.path.exists(pat_csv))
        self.assertTrue(os.path.exists(json_file))

        df_comm = pd.read_csv(comm_csv)
        df_pat = pd.read_csv(pat_csv)
        self.assertEqual(len(df_pat), 6)  # Exactly the 6 local patients mapped!

        with open(json_file) as f:
            data = json.load(f)

        self.assertEqual(data["dataset"]["local_sequences"], 6)
        self.assertEqual(data["dataset"]["reference_anchors"], 215)
        self.assertEqual(data["dataset"]["total_sequences"], 221)


if __name__ == "__main__":
    unittest.main()

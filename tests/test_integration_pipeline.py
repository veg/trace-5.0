import unittest
import os
import shutil
import tempfile
import json
import numpy as np
import pandas as pd
from trace50.cli import run_pipeline

class TestIntegrationPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.fasta_path = os.path.join(self.test_dir, "test_alignment.fasta")
        self.meta_path = os.path.join(self.test_dir, "test_metadata.csv")
        self.out_dir = os.path.join(self.test_dir, "results")

        # Create a synthetic alignment of 12 taxa over 1000 bp
        # Base genome:
        np.random.seed(42)
        bases = np.array(['A', 'C', 'G', 'T'])
        root_seq = "".join(np.random.choice(bases, size=1000))

        # Taxa 1-4: Active Outbreak (accumulating 1 mutation per year)
        # Dates: 2021.0, 2022.0, 2023.0, 2024.0 (rate ~ 1.0e-3)
        seqs = {}
        meta_rows = []

        # Outbreak clade
        ob_seq = list(root_seq)
        for i, yr in enumerate([2021.0, 2022.0, 2023.0, 2024.0]):
            tid = f"OB_{i+1}"
            if i > 0:
                # mutate 1 site
                mut_site = 100 + i
                ob_seq[mut_site] = 'G' if ob_seq[mut_site] != 'G' else 'A'
            seqs[tid] = "".join(ob_seq)
            meta_rows.append({"id": tid, "date": yr})

        # Taxa 5-8: Distant Chronic Clade (starts with 50 mutations away from root)
        chronic_base = list(root_seq)
        for s in range(50):
            chronic_base[s] = 'T' if chronic_base[s] != 'T' else 'C'

        for i, yr in enumerate([2014.0, 2017.0, 2020.0, 2023.0]):
            tid = f"CHRONIC_{i+1}"
            seqs[tid] = "".join(chronic_base)  # completely stagnant
            meta_rows.append({"id": tid, "date": yr})

        # Write FASTA
        with open(self.fasta_path, 'w') as f:
            for tid, seq in seqs.items():
                f.write(f">{tid}\n{seq}\n")

        # Write Metadata
        df_meta = pd.DataFrame(meta_rows)
        df_meta.to_csv(self.meta_path, index=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_full_pipeline_run(self):
        ret = run_pipeline(
            fasta_path=self.fasta_path,
            meta_path=self.meta_path,
            out_dir=self.out_dir,
            fdr_threshold=0.10,
            min_size=3
        )
        self.assertEqual(ret, 0)

        # Check outputs exist
        dyads_csv = os.path.join(self.out_dir, "acute_transmission_dyads.csv")
        clusters_csv = os.path.join(self.out_dir, "autoclock_communities.csv")
        summary_json = os.path.join(self.out_dir, "trace50_summary.json")

        self.assertTrue(os.path.exists(dyads_csv))
        self.assertTrue(os.path.exists(clusters_csv))
        self.assertTrue(os.path.exists(summary_json))

        # Check clusters
        df_clusters = pd.read_csv(clusters_csv)
        self.assertGreaterEqual(len(df_clusters), 2)

        # Check JSON
        with open(summary_json) as f:
            data = json.load(f)
        self.assertEqual(data["dataset"]["num_sequences"], 8)
        self.assertEqual(data["dataset"]["num_sites"], 1000)

if __name__ == "__main__":
    unittest.main()

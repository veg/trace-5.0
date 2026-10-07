"""
Reference Scaffolding and Topological Wedging Module for TRACE-5.0 (Architecture A).

Implements:
- Standardized, homoplasy-hygienic reference panel ingestion (Subtype B and Multi-subtype).
- Asymmetric role tracking: Dirichlet landmark boundary conditions (is_anchor indicator).
- Joint sequence manifold construction and length alignment harmonization.
- Metric interleaving test for acute dyads:
  Testing whether a candidate local transmission pair (u, v) is wedged by an external
  ancestral reference sequence z such that max(d(u, z), d(v, z)) <= d(u, v).
- Multi-decade temporal baseline scaffolding for narrow cohort observation windows.
"""

import os
import re
import numpy as np
import pandas as pd

from .tn93 import parse_fasta, NUC_TO_INT, IUPAC_MASKS


PANEL_PRESETS = {
    "subtype_b": {
        "fasta": "hiv1_subtype_b_pr_rt_reference_panel.fasta",
        "meta": "hiv1_subtype_b_pr_rt_reference_panel_metadata.csv",
        "description": "215 curated HIV-1 Subtype B anchors (1979-2021, 43 countries, 0 SDRMs)"
    },
    "b": {
        "fasta": "hiv1_subtype_b_pr_rt_reference_panel.fasta",
        "meta": "hiv1_subtype_b_pr_rt_reference_panel_metadata.csv",
        "description": "215 curated HIV-1 Subtype B anchors (1979-2021, 43 countries, 0 SDRMs)"
    },
    "hiv1_b": {
        "fasta": "hiv1_subtype_b_pr_rt_reference_panel.fasta",
        "meta": "hiv1_subtype_b_pr_rt_reference_panel_metadata.csv",
        "description": "215 curated HIV-1 Subtype B anchors (1979-2021, 43 countries, 0 SDRMs)"
    },
    "multisubtype": {
        "fasta": "hiv1_multisubtype_pr_rt_reference_panel.fasta",
        "meta": "hiv1_multisubtype_pr_rt_reference_panel_metadata.csv",
        "description": "375 curated global anchors across Subtypes A, B, C, D, CRF01_AE, CRF02_AG (1983-2020)"
    },
    "multi": {
        "fasta": "hiv1_multisubtype_pr_rt_reference_panel.fasta",
        "meta": "hiv1_multisubtype_pr_rt_reference_panel_metadata.csv",
        "description": "375 curated global anchors across Subtypes A, B, C, D, CRF01_AE, CRF02_AG (1983-2020)"
    },
    "global": {
        "fasta": "hiv1_multisubtype_pr_rt_reference_panel.fasta",
        "meta": "hiv1_multisubtype_pr_rt_reference_panel_metadata.csv",
        "description": "375 curated global anchors across Subtypes A, B, C, D, CRF01_AE, CRF02_AG (1983-2020)"
    }
}


def get_data_dir():
    """Returns absolute path to trace50 bundled data directory."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(pkg_dir, "data", "reference_panels")
    return data_dir


def parse_reference_date(val):
    """Parses decimal dates from reference panel metadata."""
    if pd.isna(val):
        return np.nan
    try:
        f = float(val)
        if 1950.0 <= f <= 2035.0:
            return f
    except ValueError:
        pass
    m = re.search(r'\b(19\d{2}|20\d{2})\b', str(val))
    if m:
        return float(m.group(1)) + 0.5
    return np.nan


def resolve_reference_panel(panel_arg, meta_arg=None):
    """
    Resolves panel argument to fasta and metadata paths.
    
    Args:
        panel_arg (str): Preset name ('subtype_b', 'multisubtype') or path to fasta file.
        meta_arg (str, optional): Path to metadata CSV if custom fasta.
        
    Returns:
        fasta_path (str), meta_path (str or None), description (str)
    """
    data_dir = get_data_dir()
    clean_arg = str(panel_arg).strip().lower()

    if clean_arg in PANEL_PRESETS:
        info = PANEL_PRESETS[clean_arg]
        fpath = os.path.join(data_dir, info["fasta"])
        mpath = os.path.join(data_dir, info["meta"])
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Bundled reference panel FASTA not found at {fpath}")
        return fpath, mpath, info["description"]

    # User-specified path
    if os.path.exists(panel_arg):
        fasta_path = os.path.abspath(panel_arg)
        meta_path = os.path.abspath(meta_arg) if (meta_arg and os.path.exists(meta_arg)) else None
        return fasta_path, meta_path, f"Custom reference panel from {panel_arg}"

    valid_keys = ", ".join(PANEL_PRESETS.keys())
    raise ValueError(
        f"Unknown reference panel '{panel_arg}'. Must be a file path or one of presets: {valid_keys}"
    )


def load_reference_panel(panel_arg, meta_arg=None):
    """
    Loads reference panel sequences and metadata.
    
    Returns:
        panel_records: list of (header, seq)
        panel_meta_map: dict of header -> dict(date, country, subtype, era, is_anchor=True)
    """
    fasta_path, meta_path, desc = resolve_reference_panel(panel_arg, meta_arg)
    records = parse_fasta(fasta_path)
    if not records:
        raise ValueError(f"No sequences found in reference panel at {fasta_path}")

    meta_map = {}
    if meta_path and os.path.exists(meta_path):
        df = pd.read_csv(meta_path)
        id_col = None
        for cand in ["taxon_id", "accession", "id", "header", "sequence_name"]:
            if cand in df.columns:
                id_col = cand
                break
        if id_col is None:
            id_col = df.columns[0]

        date_col = None
        for cand in ["date_decimal", "date", "collection_date", "year"]:
            if cand in df.columns:
                date_col = cand
                break

        for _, row in df.iterrows():
            tid = str(row[id_col]).strip()
            d_val = parse_reference_date(row[date_col]) if date_col else np.nan
            meta_map[tid] = {
                "date": d_val,
                "country": str(row.get("country", row.get("region", "Unknown"))),
                "subtype": str(row.get("subtype", "Unknown")),
                "era": str(row.get("clinical_era", row.get("era", "Unknown"))),
                "is_anchor": True
            }

    # Fill metadata from headers if not in CSV
    clean_records = []
    for h, seq in records:
        clean_records.append((h, seq))
        if h not in meta_map:
            # Try parsing date from header e.g. B.US.1979...
            d_est = np.nan
            m = re.search(r'\.(19\d{2}|20\d{2})\.', h)
            if m:
                d_est = float(m.group(1)) + 0.5
            meta_map[h] = {
                "date": d_est,
                "country": "Reference",
                "subtype": "Reference",
                "era": "Archival",
                "is_anchor": True
            }

    return clean_records, meta_map, desc


def merge_cohort_with_reference(
    cohort_records,
    cohort_dates,
    cohort_meta_map=None,
    panel_records=None,
    panel_meta_map=None
):
    """
    Merges local cohort sequences with reference panel anchors.
    Harmonizes sequence lengths via gap padding if necessary.
    
    Args:
        cohort_records: list of (header, seq)
        cohort_dates: np.ndarray or list of float
        cohort_meta_map: dict of header -> dict
        panel_records: list of (header, seq) or None
        panel_meta_map: dict of header -> dict or None
        
    Returns:
        all_headers: list of str
        M_int: np.ndarray of shape (N, L), int8
        M_bits: np.ndarray of shape (N, L), uint8
        all_dates: np.ndarray of shape (N,), float64
        is_anchor: np.ndarray of shape (N,), bool
        meta_list: list of dicts
    """
    if panel_records is None:
        panel_records = []
        panel_meta_map = {}

    all_records = list(cohort_records) + list(panel_records)
    N_loc = len(cohort_records)
    N_anc = len(panel_records)
    N_tot = N_loc + N_anc

    is_anchor = np.zeros(N_tot, dtype=bool)
    is_anchor[N_loc:] = True

    # Harmonize lengths
    lengths = [len(r[1]) for r in all_records]
    max_L = max(lengths) if lengths else 0

    all_headers = [r[0] for r in all_records]
    all_dates = np.full(N_tot, np.nan, dtype=np.float64)

    # Fill cohort dates
    for i in range(N_loc):
        if i < len(cohort_dates):
            all_dates[i] = cohort_dates[i]

    # Fill anchor dates
    for j, (h, _) in enumerate(panel_records):
        idx = N_loc + j
        if panel_meta_map and h in panel_meta_map:
            all_dates[idx] = panel_meta_map[h].get("date", np.nan)

    # Encode matrices with uniform length
    M_int = np.full((N_tot, max_L), -1, dtype=np.int8)
    M_bits = np.zeros((N_tot, max_L), dtype=np.uint8)

    for i, (_, seq) in enumerate(all_records):
        L_seq = len(seq)
        for j, char in enumerate(seq):
            M_int[i, j] = NUC_TO_INT.get(char, -1)
            M_bits[i, j] = IUPAC_MASKS.get(char, 0)
        # Any remaining columns (if L_seq < max_L) remain -1 and 0 (gaps)

    # Compile metadata list
    meta_list = []
    for i in range(N_loc):
        h = all_headers[i]
        c_meta = cohort_meta_map.get(h, {}) if cohort_meta_map else {}
        meta_list.append({
            "id": h,
            "date": all_dates[i],
            "is_anchor": False,
            "country": c_meta.get("country", "Local"),
            "subtype": c_meta.get("subtype", "Local"),
            "era": "Cohort"
        })

    for j, (h, _) in enumerate(panel_records):
        idx = N_loc + j
        p_meta = panel_meta_map.get(h, {}) if panel_meta_map else {}
        meta_list.append({
            "id": h,
            "date": all_dates[idx],
            "is_anchor": True,
            "country": p_meta.get("country", "Reference"),
            "subtype": p_meta.get("subtype", "Reference"),
            "era": p_meta.get("era", "Archival")
        })

    return all_headers, M_int, M_bits, all_dates, is_anchor, meta_list


def test_metric_interleaving(idx_u, idx_v, D_matrix, is_anchor_mask):
    """
    Evaluates Principle 4 (Metric Interleaving Test) for an apparent local pair (u, v).
    
    An external reference anchor z wedges a local pair (u, v) if:
        max(D[u, z], D[v, z]) <= D[u, v]
        
    In other words, z is genetically closer to both u and v than u and v are to each other.
    When this holds, the edge (u, v) is severed as a background ancestral artifact.
    
    Args:
        idx_u (int): Index of first local taxon
        idx_v (int): Index of second local taxon
        D_matrix (np.ndarray): Full pairwise distance matrix
        is_anchor_mask (np.ndarray): Boolean array indicating reference anchors
        
    Returns:
        is_wedged (bool): True if wedged by at least one reference anchor
        wedging_anchors (list of int): Indices of wedging reference anchors
    """
    d_uv = float(D_matrix[idx_u, idx_v])
    anchor_indices = np.where(is_anchor_mask)[0]
    if len(anchor_indices) == 0:
        return False, []

    d_u_anchors = D_matrix[idx_u, anchor_indices]
    d_v_anchors = D_matrix[idx_v, anchor_indices]

    # Condition: both distances <= d_uv
    is_closer = (d_u_anchors <= d_uv) & (d_v_anchors <= d_uv)
    wedging = anchor_indices[is_closer]

    if len(wedging) > 0:
        return True, list(wedging)
    return False, []

test_metric_interleaving.__test__ = False
check_metric_interleaving = test_metric_interleaving

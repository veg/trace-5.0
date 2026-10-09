"""
Operational Lineage Taxonomy and Public Health Prioritization for TRACE-5.0.

Provides an objective, non-stigmatizing classification of molecular transmission
structures based on evolutionary velocity, clock linearity, and topological centrality.
"""


class LineageTier:
    ACTIVE_OUTBREAK = "Active Outbreak Chain"
    EMERGENT_CLUSTER = "Emergent Seed Cluster"
    ACUTE_DYAD = "Contemporaneous Acute Dyad"
    STATIONARY_CHRONIC = "Stationary Chronic Compartment"
    ARCHIVAL_CENTROID = "Archival Centroid Node"
    ENDEMIC_EXTINCT = "Endemic Extinct Lineage"
    EXOGENOUS_INTRO = "Wedged Circulating Lineage (Exogenous Introduction)"
    REFERENCE_BACKBONE = "Reference Backbone Anchor"


def classify_subcommunity(fit_dict, max_surveillance_date=None, degree=0):
    """
    Maps an AutoClock subcommunity into the operational seven-tier taxonomy.

    Args:
        fit_dict: Summary dictionary from fit_clock_community()
        max_surveillance_date: Latest collection timestamp in dataset
        degree: Degree of community or node in distance network

    Returns:
        tier (str): Category from LineageTier
        priority (str): Clinical / Public health action recommendation
        rationale (str): Non-stigmatizing scientific justification
    """
    n = fit_dict.get("n", 0)
    n_local = fit_dict.get("n_local", n)
    n_anchor = fit_dict.get("n_anchor", 0)
    is_hybrid = fit_dict.get("is_hybrid", False)

    # 1. Pure reference anchors (no local cohort cases)
    if n_local == 0:
        return (
            LineageTier.REFERENCE_BACKBONE,
            "Archival Calibration Baseline",
            "Contains strictly external reference anchors; excluded from local field surveillance."
        )

    # 2. Hybrid community: Local cases wedged alongside reference anchors
    if is_hybrid or n_anchor > 0:
        return (
            LineageTier.EXOGENOUS_INTRO,
            "Regional Epidemiological Cross-Referencing",
            f"Local cohort cases (N={n_local}) co-cluster with {n_anchor} global reference anchors, "
            "confirming this sublineage represents broad circulating background or external introduction "
            "rather than an isolated local outbreak."
        )

    mu = fit_dict.get("mu", 0.0)
    r2 = fit_dict.get("r2", 0.0)
    tmrca = fit_dict.get("tmrca", 0.0)
    span = fit_dict.get("span", 0.0)
    ci_hi = fit_dict.get("ci_mrca", [0.0, 0.0])[1]

    t_horizon = max_surveillance_date if max_surveillance_date is not None else tmrca

    if n_local == 1:
        return (
            "Singleton Node",
            "Individual Case Surveillance",
            "Isolated singleton isolate without identified transmission links."
        )

    if n_local == 2:
        return (
            LineageTier.ACUTE_DYAD,
            "Targeted Clinical Engagement",
            "Micro-chain or dyad without sufficient taxa for clock regression."
        )

    # Archival Centroid Nodes: central high-degree nodes spanning multi-year windows with stagnant rates
    # Note: Explicitly non-stigmatizing. Reflects ancestral consensus proximity, NOT transmission culpability.
    if degree >= 10 and mu <= 3.0e-4 and span >= 5.0:
        return (
            LineageTier.ARCHIVAL_CENTROID,
            "Routine Clinical Care / Resistance Monitoring",
            "Centroid node proximity reflects slow substitution or ancestral consensus preservation, "
            "not behavioral superspreading."
        )

    mean_dist = fit_dict.get("mean_dist", 0.0)
    fieller_status = fit_dict.get("fieller_status", "OK")

    # Active Outbreak Chains: high velocity, linearity, recent emergence, validated Fieller clock
    is_recent = (tmrca >= t_horizon - 6.0) or (ci_hi >= t_horizon - 4.0)
    is_fieller_ok = (fieller_status == "OK")

    if is_recent and is_fieller_ok and mu >= 1.0e-3 and r2 >= 0.15:
        return (
            LineageTier.ACTIVE_OUTBREAK,
            "Immediate Public Health Field Prioritization",
            f"Elevated clock velocity (mu={mu:.2e}) and high linearity (R2={r2:.2f}) indicate active, "
            "uninterrupted forward transmission."
        )

    # Emergent Seed Clusters: intermediate velocity, recent emergence
    if is_recent and is_fieller_ok and mu >= 5.0e-4:
        return (
            LineageTier.EMERGENT_CLUSTER,
            "Enhanced Molecular Surveillance",
            f"Moderate clock progression (mu={mu:.2e}) warrants prospective monitoring."
        )

    # Endemic Extinct Lineages: non-positive rate or lack of recent activity
    if mu <= 0.0 or (span > 5.0 and r2 < 0.02):
        return (
            LineageTier.ENDEMIC_EXTINCT,
            "Archival Surveillance Registry",
            "Non-progressive clock indicates historical transmission cessation."
        )

    # Stationary Chronic Compartments: stagnant substitution, low linearity
    if mu < 5.0e-4 and r2 < 0.10:
        return (
            LineageTier.STATIONARY_CHRONIC,
            "Standard Outpatient HIV Care",
            "Slow evolutionary accumulation characteristic of long-standing intra-host carriage."
        )

    return (
        LineageTier.STATIONARY_CHRONIC,
        "Routine Surveillance Monitoring",
        f"Equivocal clock dynamics (mu={mu:.2e}, R2={r2:.2f}); classified as stationary compartment."
    )

import json
import os

json_path = "/Users/sergei/Projects/trace50/prospective_validation_results/multi_cohort_prospective_validation_summary.json"
with open(json_path) as f:
    records = json.load(f)

pct_sym = r"\%"
eol = r"\\"

lines = []
lines.append(r"\begin{table*}[t]")
lines.append(r"\centering")
lines.append(r"\footnotesize")
lines.append(r"\caption{\textbf{Standardized prospective transmission surveillance validation across four diverse molecular HIV-1 cohorts.} Prospective case accrual was evaluated following the protocol of Wertheim et al. across Washington, DC ($N=1{,}658$), Shenzhen, China ($N=454$), Middle Tennessee ($N=2{,}915$), and nationwide surveillance in Japan ($N=5{,}232$). Across all four distinct epidemic regimes, \STEVE evolutionary velocity and \DANNO coalescent screening consistently outpace static distance cutoffs ($d \le 1.5\%$) and historical growth metrics (Wertheim $\Delta N / \sqrt{N} \ge 0.70$), dramatically improving precision at the critical 12-month public health planning window and maximizing resource-normalized case yield per targeted patient.}")
lines.append(r"\label{tab:multi_cohort_prospective_validation}")
lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}")
lines.append(r"\toprule")
lines.append(r"\textbf{Surveillance Strategy} & \textbf{Clusters} & \textbf{Targeted} & \textbf{PPV} & \textbf{PPV} & \textbf{Cases} & \textbf{Case Yield} & \textbf{False} \\")
lines.append(r"& ($N_{\mathrm{cl}}$) & \textbf{Patients} & (Full) & (12-Mo) & \textbf{Linked} & (Per Patient) & \textbf{Alarms} \\")

cohorts = [
    ("Washington, DC ($N=1{,}658$, Subtypes B & C)", r"\textbf{Washington, DC ($N=1{,}658$, Subtypes B \& C)}"),
    ("Shenzhen, China ($N=454$, CRF07_BC)", r"\textbf{Shenzhen, China ($N=454$, CRF07\_BC)}"),
    ("Middle Tennessee ($N=2{,}915$, Subtype B)", r"\textbf{Middle Tennessee ($N=2{,}915$, Subtype B)}"),
    ("Japan Nationwide NIID ($N=5{,}232$, Subtype B)", r"\textbf{Japan Nationwide NIID ($N=5{,}232$, Subtype B)}")
]

method_format = {
    "Static TN93 <= 1.5% (All >= 2)": r"Static TN93 ($d \le 1.5\%$, all clusters)",
    "Static TN93 <= 1.5% (Wertheim >= 0.70)": r"Static TN93 ($d \le 1.5\%$, Wertheim $\ge 0.70$)",
    "Static TN93 <= 1.0% (All >= 2)": r"Static TN93 ($d \le 1.0\%$, all clusters)",
    "Static TN93 <= 0.5% (All >= 2)": r"Static TN93 ($d \le 0.5\%$, all clusters)",
    "DANNO (q <= 0.001, dt <= 2 yr)": r"\DANNO ($q \le 0.001, \Delta T \le 2\text{ yr}$)",
    "DANNO (q <= 0.005, dt <= 2 yr)": r"\DANNO ($q \le 0.005, \Delta T \le 2\text{ yr}$)",
    "DANNO (q <= 0.01, dt <= 2 yr)": r"\DANNO ($q \le 0.01, \Delta T \le 2\text{ yr}$)",
    "DANNO (q <= 0.05, dt <= 2 yr)": r"\DANNO ($q \le 0.05, \Delta T \le 2\text{ yr}$)",
    "DANNO (q <= 0.10, dt <= 2 yr)": r"\DANNO ($q \le 0.10, \Delta T \le 2\text{ yr}$)",
    "AutoClock STEVE (Active & Emergent Prioritized)": r"AutoClock \STEVE (Active \& Emergent Prioritized)",
    "AutoClock STEVE (Stationary Chronic / Dormant)": r"AutoClock \STEVE (Stationary Chronic / Dormant)"
}

by_cohort = {}
for r in records:
    c = r["Cohort"]
    by_cohort.setdefault(c, []).append(r)

for c_key, c_header in cohorts:
    if c_key not in by_cohort:
        continue
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{8}{l}{" + c_header + r"} \\")
    lines.append(r"\midrule")
    for r in by_cohort[c_key]:
        m_raw = r["Method"]
        m_latex = method_format.get(m_raw, m_raw.replace("&", r"\&").replace("%", r"\%"))
        cl = int(r["Prioritized_Clusters"])
        pts = int(r["Targeted_Patients"])
        ppv_f = f"{r['PPV_Full_Pct']:.1f}" + pct_sym
        ppv_12 = f"{r['PPV_12mo_Pct']:.1f}" + pct_sym
        cases = int(r["Prospective_Cases_Linked"])
        yld = f"{r['Yield_Per_Patient']:.3f}"
        false_al = int(r["False_Alarm_Clusters"])
        row = f"{m_latex} & {cl} & {pts} & {ppv_f} & {ppv_12} & {cases} & {yld} & {false_al} " + eol
        lines.append(row)

lines.append(r"\bottomrule")
lines.append(r"\end{tabular*}")
lines.append(r"\end{table*}")

out_file = "/Users/sergei/Projects/trace50/paper/table_multi_cohort_prospective_validation.tex"
with open(out_file, "w") as f:
    f.write("\n".join(lines) + "\n")
print("Wrote properly formatted table to:", out_file)

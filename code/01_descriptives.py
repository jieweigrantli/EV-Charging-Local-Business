"""
01_descriptives.py
------------------
Descriptive statistics and the data behind the descriptive figures.

Outputs (outputs/tables and outputs/source_data):
  * supplementary_table_1_summary_stats.csv/.tex   (Supplementary Table 1)
  * table_1_spending_by_store.csv/.tex              (Table 1)
  * fig1a_visits_by_store.csv, fig1b_transactions_by_store.csv  (Figure 1)
  * fig2_amenity_counts.csv                          (Figure 2)
  * fig4a_visit_share_by_reason.csv, fig4b_transactions_by_reason.csv (Figure 4)
  * suppfig6_demographic_comparison.csv              (Supplementary Figure 6)
"""
import numpy as np
import pandas as pd

from common import (AMENITY_VARS, DATA_DIR, LABELS, REASON_ORDER, REASON_VARS, SD_DIR,
                    STORE_TYPES, TAB_DIR, TARGET, fmt_num, load_data, transactions_long,
                    write_csv, write_tex)

df = load_data()
N = len(df)

# --------------------------------------------------------------------------- #
# Supplementary Table 1: descriptive statistics (N = 1,586)
# --------------------------------------------------------------------------- #
unknown_col_map = {
    "charge_duration": "charge_duration_unknown",
    "wait_time": "wait_time_unknown",
}
summary_spec = [
    ("Demographics", [
        ("Household Adults (\\#)", "hh_adults"),
        ("Household Children (\\#)", "hh_children"),
        ("Household Income (k\\$)", "income_continuous_number"),
        ("Age (Years)", "age_continuous_number"),
    ]),
    ("Session \\& Vehicle", [
        ("Session Cost (\\$)", "session_cost"),
        ("Charge Duration (h)", "charge_duration"),
        ("Wait Time (h)", "wait_time"),
        ("Battery Capacity (kWh)", "battery_capacity"),
        ("Distance from Home (mi)", "distance_from_home"),
    ]),
    ("Amenities (250m radius, \\#)", [
        ("Department Store", "250m_department_store"),
        ("Entertainment", "250m_entertainment"),
        ("Grocery", "250m_grocery"),
        ("Medical", "250m_medical"),
        ("Park", "250m_park"),
        ("Religion", "250m_religion"),
        ("Restaurant", "250m_restaurant"),
        ("Restroom (Standalone)", "250m_standalone_bathroom"),
        ("Restroom (Co-located)", "250m_colocated_bathroom"),
        ("Retail", "250m_retail"),
        ("Work", "250m_work"),
    ]),
]
rows = []
for section, items in summary_spec:
    for label, col in items:
        vals = pd.to_numeric(df[col], errors="coerce")
        if col in unknown_col_map:  # "I don't remember" answers were imputed -> treat as missing
            unc = pd.to_numeric(df[unknown_col_map[col]], errors="coerce")
            vals = vals[(unc != 1)]
        vals = vals.dropna()
        if col in ("income_continuous_number", "age_continuous_number"):
            vals = vals[vals != 0]   # 0 encodes "not reported"
        rows.append({
            "section": section, "variable": label, "column": col, "N": int(len(vals)),
            "mean": vals.mean(), "sd": vals.std(ddof=0), "min": vals.min(),
            "q1": vals.quantile(0.25), "median": vals.median(), "q3": vals.quantile(0.75), "max": vals.max(),
        })
summary = pd.DataFrame(rows)
write_csv(summary, TAB_DIR / "supplementary_table_1_summary_stats.csv")

lines = []
for section in summary["section"].unique():
    lines.append(f"\\multicolumn{{9}}{{l}}{{\\textbf{{{section}}}}} \\\\")
    for _, r in summary[summary["section"] == section].iterrows():
        lines.append(f"{r['variable']} & {r['N']} & {r['mean']:.3f} & {r['sd']:.3f} & {r['min']:.3f} & "
                     f"{r['q1']:.3f} & {r['median']:.3f} & {r['q3']:.3f} & {r['max']:.3f} \\\\")
    lines.append("\\addlinespace\n\\midrule\n\\addlinespace")
write_tex("supplementary_table_1_summary_stats_body.tex", "\n".join(lines[:-1]) + "\n")

# --------------------------------------------------------------------------- #
# Transactions by store category: Table 1 and Figure 1
# --------------------------------------------------------------------------- #
tx = transactions_long(df)
tx_amt = tx.dropna(subset=["amount"]).copy()   # transactions with a reported amount (incl. $0)

tab1 = (tx_amt.groupby("store_type")["amount"]
        .agg(n="size", mean="mean", min="min", q1=lambda s: s.quantile(0.25), median="median",
             q3=lambda s: s.quantile(0.75), max="max")
        .sort_values("n", ascending=False).reset_index())
write_csv(tab1, TAB_DIR / "table_1_spending_by_store.csv")
short = {"Café/Restaurant/Bar": "Café/Rest./Bar", "Grocery Store": "Grocery Store",
         "Retail/Clothing Store": "Retail/Clothing", "Other": "Other",
         "Department/Big Box Store": "Dept./Big Box", "Cinema/Theatre/Entertainment": "Entertainment",
         "Place of Work/Study": "Place of Work", "Medical/Dental Facility": "Medical/Dental"}
body = "\n".join(
    f"{short[r.store_type]} & {r.n} & {r['mean']:.2f} & {r['min']:.2f} & {r.q1:.2f} & {r['median']:.2f} & {r.q3:.2f} & {r['max']:.2f} \\\\"
    for _, r in tab1.iterrows())
n_tx = int(tab1["n"].sum())
n_sessions_tx = int(tx_amt["response_id"].nunique())
write_tex("table_1_spending_by_store_body.tex", body + "\n")
print(f"[table 1] {n_tx} transactions with a reported amount from {n_sessions_tx} sessions")

# Figure 1a: visits per category (= transactions with a reported amount, as in the original figure)
fig1a = tab1[["store_type", "n"]].rename(columns={"n": "visits"})
write_csv(fig1a, SD_DIR / "fig1a_visits_by_store.csv")
# Figure 1b: every individual transaction amount (plotted as box + dots)
fig1b = tx_amt[["store_type", "store_slot", "amount"]].sort_values(["store_type", "amount"])
write_csv(fig1b, SD_DIR / "fig1b_transactions_by_store.csv")

# --------------------------------------------------------------------------- #
# Figure 2: amenity counts within 250 m of each reported charging location
# --------------------------------------------------------------------------- #
amen = df[["response_id"] + AMENITY_VARS].dropna(subset=AMENITY_VARS)
amen_long = amen.melt(id_vars="response_id", var_name="amenity", value_name="count")
amen_long["amenity"] = amen_long["amenity"].str.replace("250m_", "", regex=False)
# discrete distribution: frequency of each integer count per amenity type
fig2 = (amen_long.groupby(["amenity", "count"]).size().rename("n_locations").reset_index())
fig2_stats = (amen_long.groupby("amenity")["count"]
              .agg(n="size", mean="mean", median="median", q1=lambda s: s.quantile(0.25),
                   q3=lambda s: s.quantile(0.75), max="max").reset_index()
              .sort_values(["median", "mean"], ascending=False))
write_csv(fig2, SD_DIR / "fig2_amenity_count_distribution.csv")
write_csv(fig2_stats, SD_DIR / "fig2_amenity_count_summary.csv")
print(f"[figure 2] {amen.shape[0]} locations with amenity data")

# --------------------------------------------------------------------------- #
# Figure 4: visit share and transaction amounts by reason for charging
# --------------------------------------------------------------------------- #
rows = []
for reason in REASON_ORDER:
    rcol = REASON_VARS[reason]
    in_group = df[rcol] == 1
    n_group = int(in_group.sum())
    tx_r = tx[tx["response_id"].isin(df.loc[in_group, "response_id"])]
    for key, display in STORE_TYPES.items():
        visits = int((tx_r["store_key"] == key).sum())
        p = visits / n_group
        rows.append({"reason": reason, "n_respondents": n_group, "store_type": display,
                     "visits": visits, "percent": 100 * p,
                     "se_percent": 100 * np.sqrt(p * (1 - p) / n_group)})
fig4a = pd.DataFrame(rows)
write_csv(fig4a, SD_DIR / "fig4a_visit_share_by_reason.csv")

rows = []
for reason in REASON_ORDER:
    rcol = REASON_VARS[reason]
    ids = df.loc[df[rcol] == 1, "response_id"]
    sub = tx_amt[tx_amt["response_id"].isin(ids)]
    for _, r in sub.iterrows():
        rows.append({"reason": reason, "store_type": r["store_type"], "store_slot": r["store_slot"],
                     "amount": r["amount"]})
fig4b = pd.DataFrame(rows)
write_csv(fig4b, SD_DIR / "fig4b_transactions_by_reason.csv")
n4b = fig4b.groupby("reason").size().reindex(REASON_ORDER)
n4a = fig4a.groupby("reason")["n_respondents"].first().reindex(REASON_ORDER)
print("[figure 4] panel a respondents per reason:", n4a.to_dict())
print("[figure 4] panel b transactions with amount per reason:", n4b.to_dict())
# Why the n differ: panel a counts respondents in each group; panel b counts store
# visits with a reported dollar amount.  Visits without an amount are listed here.
visits_total = fig4a.groupby("reason")["visits"].sum().reindex(REASON_ORDER)
pd.DataFrame({"reason": REASON_ORDER, "respondents_in_group": n4a.values,
              "store_visits_reported": visits_total.values,
              "visits_with_reported_amount": n4b.values,
              "visits_without_reported_amount": visits_total.values - n4b.values}
             ).to_csv(TAB_DIR / "fig4_sample_size_reconciliation.csv", index=False)

# --------------------------------------------------------------------------- #
# Supplementary Figure 6: sample demographics vs. EV-owner benchmarks
# --------------------------------------------------------------------------- #
INCOME_ORDER = ["<$50k", "$50k-$99k", "$100k-$149k", "$150k-$199k", "$200k+"]
AGE_ORDER = ["18-34", "35-44", "45-54", "55-64", "65-74", "75+"]
EDU_ORDER = ["<High school", "High school/GED", "Some college/Associate", "Bachelor's", "Graduate/professional"]


def _pct(values, cats):
    values = pd.Series(values).dropna()
    counts = pd.Categorical(values, categories=cats).value_counts().reindex(cats, fill_value=0)
    return counts / counts.sum() * 100, int(counts.sum())


income = pd.cut(pd.to_numeric(df["income_continuous_number"]).replace(0, np.nan),
                bins=[0, 50, 100, 150, 200, np.inf], labels=INCOME_ORDER, right=False)
age = pd.cut(pd.to_numeric(df["age_continuous_number"]).replace(0, np.nan),
             bins=[18, 35, 45, 55, 65, 75, np.inf], labels=AGE_ORDER, right=False)
rows = []
metrics = {"Household income": (income, INCOME_ORDER), "Age": (age, AGE_ORDER)}
if "resp_education" in df.columns:
    # The survey instrument offers no "some college/associate" option, hence 0 % by design.
    edu = df["resp_education"].map({
        "Some high school": "<High school", "High School Graduate or GED": "High school/GED",
        "College Graduate": "Bachelor's", "Masters, Doctorate, or Professional Degree": "Graduate/professional"})
    metrics["Education"] = (edu, EDU_ORDER)
for metric, (vals, cats) in metrics.items():
    pct, n = _pct(vals, cats)
    for cat, v in pct.items():
        rows.append({"Metric": metric, "Category": cat, "Source": "Survey sample", "Percent": v, "n": n})
survey = pd.DataFrame(rows)
bench = pd.read_csv(DATA_DIR / "external" / "ev_owner_demographic_benchmarks.csv")
bench = bench.melt(id_vars=["Metric", "Category"], value_vars=["California EV Owners", "National EV Owners"],
                   var_name="Source", value_name="Percent")
bench["Source"] = bench["Source"].map({"California EV Owners": "California EV owners",
                                       "National EV Owners": "National EV owners"})
demo = pd.concat([survey, bench], ignore_index=True)
write_csv(demo, SD_DIR / "suppfig6_demographic_comparison.csv")
print("[done] 01_descriptives.py")

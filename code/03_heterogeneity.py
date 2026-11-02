"""
03_heterogeneity.py
-------------------
Extensive-margin logit stratified by the reported reason for charging
(Table 3 and Supplementary Table 3) and Wald tests of coefficient equality
across the three groups.

Within each reason-for-charging group the extensive-margin logit is re-estimated;
equality of coefficients across groups is tested in a fully interacted pooled logit.

Outputs:
  outputs/tables/table_3_trip_purpose.{csv,tex}
  outputs/tables/supplementary_table_3_trip_purpose_full.{csv,tex}
  outputs/tables/wald_equality_tests.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit

from common import (LABELS, LOG_DIR, PREDICTORS, REASON_ORDER, REASON_VARS, TAB_DIR, TARGET,
                    available_predictors, coef_table, cox_snell_r2, fmt_coef, fmt_p, load_data,
                    write_csv, write_tex)

df = load_data()
predictors = available_predictors(df)

# Grouped regressions: within each reason group, drop the reason dummies and the
# gig-economy dummy (quasi-complete separation in sub-groups); missing predictor
# values are set to 0 (the centred mean).
GROUPED_EXCL = {"veh_use_gigecon"} | set(REASON_VARS.values())
group_cols = [c for c in predictors if c not in GROUPED_EXCL]

results, tables = {}, []
log = open(LOG_DIR / "03_grouped_logit_summary.txt", "w")
for reason in REASON_ORDER:
    rcol = REASON_VARS[reason]
    sub = df[df[rcol] == 1].copy()
    Xg = sub[group_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    Xg = Xg.loc[:, Xg.nunique() > 1]
    y = (pd.to_numeric(sub[TARGET], errors="coerce") > 0).astype(int)
    m = sm.GLM(y, sm.add_constant(Xg, has_constant="add"), family=sm.families.Binomial()).fit(cov_type="HC3")
    results[reason] = m
    t = coef_table(m, reason)
    t["N"] = int(m.nobs)
    t["pseudo_R2_cox_snell"] = cox_snell_r2(m)
    tables.append(t)
    log.write(f"=== {reason} (N = {int(m.nobs)}) ===\n{m.summary()}\n\n")
    print(f"[grouped] {reason}: N = {int(m.nobs)}, Cox-Snell pseudo-R2 = {cox_snell_r2(m):.3f}")
log.close()
grouped = pd.concat(tables, ignore_index=True)

# --------------------------------------------------------------------------- #
# Wald tests of coefficient equality across groups: fully interacted pooled
# logit (opportunity = reference; each session assigned to one group in the
# order opportunity > long trip > routine).
# --------------------------------------------------------------------------- #
def assign_group(r):
    if r["charge_session_reason_opportunity"] == 1:
        return 0
    if r["charge_session_reason_long_trip"] == 1:
        return 1
    if r["charge_session_reason_routine"] == 1:
        return 2
    return np.nan


pooled = df.copy()
pooled["_group"] = pooled.apply(assign_group, axis=1)
pooled = pooled.dropna(subset=["_group"] + group_cols)
# outcome defined exactly as in the stratified models: reported amount > 0
pooled["spend_any"] = (pd.to_numeric(pooled[TARGET], errors="coerce") > 0).astype(int)
pooled["_long"] = (pooled["_group"] == 1).astype(int)
pooled["_routine"] = (pooled["_group"] == 2).astype(int)
cols = {}
for c in group_cols:
    cols[c] = pooled[c].values
    cols[f"{c}_long"] = (pooled[c] * pooled["_long"]).values
    cols[f"{c}_routine"] = (pooled[c] * pooled["_routine"]).values
Xw = sm.add_constant(pd.DataFrame(cols, index=pooled.index))
mw = Logit(pooled["spend_any"], Xw).fit(disp=0, method="newton", maxiter=200)
assert mw.mle_retvals["converged"], "pooled interacted logit did not converge"
wald = []
for c in group_cols:
    R = np.zeros((2, len(mw.params)))
    R[0, mw.params.index.get_loc(f"{c}_long")] = 1
    R[1, mw.params.index.get_loc(f"{c}_routine")] = 1
    w = mw.wald_test(R, scalar=True)
    wald.append({"term": c, "label": LABELS.get(c, c), "wald_chi2": float(w.statistic), "df": 2,
                 "p_two_sided": float(w.pvalue)})
wald = pd.DataFrame(wald)
wald["N_pooled"] = int(mw.nobs)
wald["outcome"] = "store_total_spent_calc > 0 (as in the stratified models)"
write_csv(wald, TAB_DIR / "wald_equality_tests.csv")
print(f"[wald] pooled fully interacted logit, N = {int(mw.nobs)}, converged = {mw.mle_retvals['converged']}")

# Sensitivity: the same Wald tests with the survey yes/no item ("did you spend
# money?") as outcome.
pooled2 = df.copy()
pooled2["_group"] = pooled2.apply(assign_group, axis=1)
pooled2 = pooled2.dropna(subset=["_group", "store_total_spent_anything"] + group_cols)
pooled2["spend_any"] = (pooled2["store_total_spent_anything"] == 1).astype(int)
pooled2["_long"] = (pooled2["_group"] == 1).astype(int)
pooled2["_routine"] = (pooled2["_group"] == 2).astype(int)
cols2 = {}
for c in group_cols:
    cols2[c] = pooled2[c].values
    cols2[f"{c}_long"] = (pooled2[c] * pooled2["_long"]).values
    cols2[f"{c}_routine"] = (pooled2[c] * pooled2["_routine"]).values
mw2 = Logit(pooled2["spend_any"], sm.add_constant(pd.DataFrame(cols2, index=pooled2.index))).fit(disp=0, method="newton", maxiter=200)
wald2 = []
for c in group_cols:
    R = np.zeros((2, len(mw2.params)))
    R[0, mw2.params.index.get_loc(f"{c}_long")] = 1
    R[1, mw2.params.index.get_loc(f"{c}_routine")] = 1
    w = mw2.wald_test(R, scalar=True)
    wald2.append({"term": c, "label": LABELS.get(c, c), "wald_chi2": float(w.statistic), "df": 2,
                  "p_two_sided": float(w.pvalue), "N_pooled": int(mw2.nobs),
                  "outcome": "store_total_spent_anything (survey yes/no item)"})
write_csv(pd.DataFrame(wald2), TAB_DIR / "wald_equality_tests_sensitivity_survey_item_outcome.csv")

# --------------------------------------------------------------------------- #
# Table 3 (condensed) and Supplementary Table 3 (full)
# --------------------------------------------------------------------------- #
G = {r: grouped[grouped["model"] == r].set_index("term") for r in REASON_ORDER}
W = wald.set_index("term")


def cell(r, term):
    t = G[r]
    if term not in t.index:
        return "--", "--"
    return fmt_coef(t.loc[term, "coef"], t.loc[term, "se"], t.loc[term, "p_two_sided"]), fmt_p(t.loc[term, "p_two_sided"])


def row3(term, label=None, indent=False, with_wald=False):
    lab = label or LABELS.get(term, term)
    if indent:
        lab = f"\\hspace{{3mm}} \\textit{{{lab}}}"
    cells = []
    for r in REASON_ORDER:
        cells += list(cell(r, term))
    s = f"{lab} & " + " & ".join(cells)
    if with_wald:
        s += " & " + (fmt_p(W.loc[term, "p_two_sided"]) if term in W.index else "--")
    return s + " \\\\"


t3_panels = [
    ("Panel A: Driver Demographics", [
        ("gender_female", None, False), ("hh_adults", None, False), ("hh_children", None, False),
        ("age_centered", "Driver Age (Years)$^{a}$", False), ("age_continuous_exist", "Age Reported (Dummy)", True)]),
    ("Panel B: Vehicle \\& Session Characteristics", [
        ("charge_duration", None, False), ("battery_capacity_centered", "Battery Capacity (kWh)$^{a}$", False),
        ("battery_capacity_exist", "Battery Reported (Dummy)", True),
        ("distance_from_home_log1p_centered", "Dist. from Home (log)$^{a}$", False),
        ("distance_from_home_exist", "Dist. Reported (Dummy)", True),
        ("session_cost_log1p_centered", "Session Cost (log)", False)]),
    ("Panel C: Commercial Environment (within 250m)", [
        ("250m_grocery", "Grocery (Count)", False), ("250m_medical", None, False)]),
]
lines = []
for title, items in t3_panels:
    lines.append(f"\\multicolumn{{7}}{{l}}{{\\textbf{{{title}}}}} \\\\[0.2em]")
    lines += [row3(*r) for r in items]
    lines.append("\\midrule")
lines += ["\\textbf{Controls Included:} & & & & & & \\\\[0.2em]",
          "Demographic Controls & \\multicolumn{2}{c}{Yes} & \\multicolumn{2}{c}{Yes} & \\multicolumn{2}{c}{Yes} \\\\",
          "Session and Vehicle Controls & \\multicolumn{2}{c}{Yes} & \\multicolumn{2}{c}{Yes} & \\multicolumn{2}{c}{Yes} \\\\",
          "Additional Amenity Controls & \\multicolumn{2}{c}{Yes} & \\multicolumn{2}{c}{Yes} & \\multicolumn{2}{c}{Yes} \\\\",
          "\\midrule",
          "Observations & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{int(results[r].nobs)}}}" for r in REASON_ORDER) + " \\\\",
          "Pseudo $R^2$ & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{cox_snell_r2(results[r]):.3f}}}" for r in REASON_ORDER) + " \\\\",
          row3("const", "Constant")]
write_tex("table_3_trip_purpose_body.tex", "\n".join(lines) + "\n")
t3_terms = [t for _, items in t3_panels for t, _, _ in items] + ["const"]
write_csv(grouped[grouped["term"].isin(t3_terms)], TAB_DIR / "table_3_trip_purpose.csv")

s3_panels = [
    ("Panel A: Race \\& Ethnicity", [("race_asian", "Asian (Dummy)", False), ("race_black", "Black (Dummy)", False),
                                     ("race_hispanic", "Hispanic (Dummy)", False)]),
    ("Panel B: Other Demographics", [
        ("gender_female", None, False), ("edu_college", "College Degree (Dummy)", False),
        ("edu_grad", "Graduate Degree (Dummy)", False), ("hh_adults", None, False), ("hh_children", None, False),
        ("age_centered", "Driver Age (Years)$^{a}$", False), ("age_continuous_exist", "Age Reported (Dummy)", True),
        ("log_income_centered", "Income (log)$^{a}$", False), ("log_income_continuous_exist", "Income Reported (Dummy)", True)]),
    ("Panel C: Session \\& Vehicle Characteristics", [
        ("charge_duration", None, False), ("wait_time", None, False),
        ("veh_use_commuting", "Vehicle Use: Commuting (Dummy)", False), ("veh_use_private", "Vehicle Use: Private (Dummy)", False),
        ("veh_use_business", "Vehicle Use: Business (Dummy)", False),
        ("battery_capacity_centered", "Battery Capacity (kWh)$^{a}$", False), ("battery_capacity_exist", "Battery Reported (Dummy)", True),
        ("distance_from_home_log1p_centered", "Dist. from Home (log)$^{a}$", False), ("distance_from_home_exist", "Dist. Reported (Dummy)", True),
        ("session_cost_log1p_centered", "Session Cost (log)", False)]),
    ("Panel D: Commercial Environment (within 250m)", [
        ("250m_department_store", None, False), ("250m_entertainment", None, False), ("250m_grocery", "Grocery (Count)", False),
        ("250m_medical", None, False), ("250m_park", None, False), ("250m_religion", "Religion (Count)", False),
        ("250m_restaurant", None, False), ("250m_standalone_bathroom", "Standalone Bathroom (Count)", False),
        ("250m_colocated_bathroom", "Colocated Bathroom (Count)", False), ("250m_retail", None, False),
        ("250m_work", "Work (Count)", False)]),
]
lines = []
for title, items in s3_panels:
    lines.append(f"\\multicolumn{{8}}{{l}}{{\\textbf{{{title}}}}} \\\\[0.2em]")
    lines += [row3(*r, with_wald=True) for r in items]
    lines.append("\\midrule")
lines += [row3("const", "Constant", with_wald=True).replace("& -- \\\\", "& --- \\\\"),
          "\\midrule",
          "Observations & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{int(results[r].nobs)}}}" for r in REASON_ORDER) + " & --- \\\\",
          "Pseudo $R^2$ & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{cox_snell_r2(results[r]):.3f}}}" for r in REASON_ORDER) + " & --- \\\\"]
write_tex("supplementary_table_3_trip_purpose_full_body.tex", "\n".join(lines) + "\n")
full = grouped.drop(columns=["model"]).rename(columns={"N": "N"})
full.insert(0, "group", grouped["model"])
write_csv(full, TAB_DIR / "supplementary_table_3_trip_purpose_full.csv")

# p-values quoted in the text
quoted = {}
for r in REASON_ORDER:
    for term in ["charge_duration", "battery_capacity_centered", "session_cost_log1p_centered",
                 "distance_from_home_log1p_centered", "hh_adults", "hh_children", "250m_medical", "age_centered",
                 "gender_female", "250m_grocery", "veh_use_commuting"]:
        if term in G[r].index:
            quoted[f"{r}|{term}"] = {"coef": float(G[r].loc[term, "coef"]), "p": float(G[r].loc[term, "p_two_sided"])}
pd.DataFrame(quoted).T.to_csv(TAB_DIR / "table_3_quoted_pvalues.csv")
print("[done] 03_heterogeneity.py")

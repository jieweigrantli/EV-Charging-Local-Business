"""
02_two_part_model.py
--------------------
Main two-part hurdle model (Table 2, Supplementary Table 2), variance inflation
factors (Supplementary Table 4), the amenity-specification robustness checks
(Supplementary Data 1) and the predicted-probability curves of Figure 3.

Outputs:
  outputs/tables/table_2_main_model.{csv,tex}
  outputs/tables/supplementary_table_2_full_model.{csv,tex}
  outputs/tables/supplementary_table_4_vif.{csv,tex}
  outputs/tables/robustness_amenity_specifications.csv (+ xlsx in supplementary_data)
  outputs/source_data/fig3*_marginal_*.csv
"""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from common import (AMENITY_VARS, LABELS, LOG_DIR, PREDICTORS, SD_DIR, SUPP_DIR, TAB_DIR, TARGET,
                    X_AXIS_LABELS, available_predictors, coef_table, cox_snell_r2, design_matrix,
                    fit_two_part, fmt_coef, fmt_num, fmt_p, load_data, mcfadden_r2, reference_row,
                    write_csv, write_tex)

df = load_data()
predictors = available_predictors(df)
X = design_matrix(df, predictors)
res = fit_two_part(df, X)
glm, ols = res.glm, res.ols
print(f"[model] extensive margin: N = {int(glm.nobs)}, log-lik = {glm.llf:.2f}, "
      f"Cox-Snell pseudo-R2 = {cox_snell_r2(glm):.3f}, McFadden = {mcfadden_r2(glm):.3f}")
print(f"[model] intensive margin: N = {int(ols.nobs)}, log-lik = {ols.llf:.2f}, R2 = {ols.rsquared:.3f}, "
      f"adj. R2 = {ols.rsquared_adj:.3f}")

with open(LOG_DIR / "02_main_model_summary.txt", "w") as fh:
    fh.write("Extensive margin: binomial GLM (logit), HC3 robust SE\n")
    fh.write(str(glm.summary()) + "\n\n")
    fh.write("Intensive margin: OLS on log(1 + spend) | spend > 0, HC3 robust SE\n")
    fh.write(str(ols.summary()) + "\n")

tab_glm = coef_table(glm, "extensive_logit")
tab_ols = coef_table(ols, "intensive_ols")
full = tab_glm.merge(tab_ols, on=["term", "label"], suffixes=("_logit", "_ols")).drop(
    columns=["model_logit", "model_ols"])
write_csv(full, TAB_DIR / "supplementary_table_2_full_model.csv")

# --------------------------------------------------------------------------- #
# LaTeX: Table 2 (condensed) and Supplementary Table 2 (full)
# --------------------------------------------------------------------------- #
g = tab_glm.set_index("term")
o = tab_ols.set_index("term")


def row(term, label=None, indent=False):
    lab = label or LABELS.get(term, term)
    if indent:
        lab = f"\\hspace{{3mm}} \\textit{{{lab}}}"
    if term not in g.index:
        return f"{lab} & -- & -- & -- & -- \\\\"
    return (f"{lab} & {fmt_coef(g.loc[term,'coef'], g.loc[term,'se'], g.loc[term,'p_two_sided'])} & "
            f"{fmt_p(g.loc[term,'p_two_sided'])} & "
            f"{fmt_coef(o.loc[term,'coef'], o.loc[term,'se'], o.loc[term,'p_two_sided'])} & "
            f"{fmt_p(o.loc[term,'p_two_sided'])} \\\\")


panelA = [("gender_female", None, False), ("hh_children", None, False),
          ("log_income_centered", "Household Income (log)$^{a}$", False),
          ("log_income_continuous_exist", "Income Reported (Dummy)", True)]
panelB = [("charge_duration", None, False),
          ("distance_from_home_log1p_centered", "Distance from Home (log)$^{a}$", False),
          ("distance_from_home_exist", "Distance Reported (Dummy)", True),
          ("session_cost_log1p_centered", "Session Cost (log)", False),
          ("battery_capacity_centered", "Battery Capacity (kWh)$^{a}$", False),
          ("battery_capacity_exist", "Battery Reported (Dummy)", True),
          ("charge_session_reason_long_trip", None, False),
          ("charge_session_reason_opportunity", None, False)]
panelC = [("250m_department_store", None, False), ("250m_entertainment", None, False),
          ("250m_grocery", None, False), ("250m_medical", None, False), ("250m_restaurant", None, False),
          ("250m_retail", None, False), ("250m_standalone_bathroom", None, False)]
lines = ["\\multicolumn{5}{l}{\\textbf{Panel A: Driver Demographics}} \\\\[0.2em]"]
lines += [row(*r) for r in panelA]
lines += ["\\midrule", "\\multicolumn{5}{l}{\\textbf{Panel B: Vehicle and Session Contexts}} \\\\[0.2em]"]
lines += [row(*r) for r in panelB]
lines += ["\\midrule", "\\multicolumn{5}{l}{\\textbf{Panel C: Commercial Environment (within 250m)}} \\\\[0.2em]"]
lines += [row(*r) for r in panelC]
lines += ["\\midrule", "\\textbf{Controls Included:} & & & & \\\\[0.2em]",
          "Demographic Controls & \\multicolumn{2}{c}{Yes} & \\multicolumn{2}{c}{Yes} \\\\",
          "Session and Vehicle Controls & \\multicolumn{2}{c}{Yes} & \\multicolumn{2}{c}{Yes} \\\\",
          "Additional Amenity Controls & \\multicolumn{2}{c}{Yes} & \\multicolumn{2}{c}{Yes} \\\\",
          "\\midrule",
          f"Observations & \\multicolumn{{2}}{{c}}{{{int(glm.nobs):,}}} & \\multicolumn{{2}}{{c}}{{{int(ols.nobs):,}}} \\\\",
          f"Pseudo $R^{{2}}$ / $R^{{2}}$ & \\multicolumn{{2}}{{c}}{{{cox_snell_r2(glm):.3f}}} & \\multicolumn{{2}}{{c}}{{{ols.rsquared:.3f}}} \\\\",
          row("const", "Constant")]
write_tex("table_2_main_model_body.tex", "\n".join(lines) + "\n")
write_csv(full[full["term"].isin([t for t, _, _ in panelA + panelB + panelC] + ["const"])],
          TAB_DIR / "table_2_main_model.csv")


def row_full(term, label=None, indent=False):
    lab = label or LABELS.get(term, term)
    if indent:
        lab = f"\\hspace{{3mm}} \\textit{{{lab}}}"
    if term not in g.index:
        return f"{lab} & -- & -- & -- & -- & -- & -- \\\\"
    gc, oc = g.loc[term], o.loc[term]
    return (f"{lab} & {fmt_num(gc['coef'])}{'' if pd.isna(gc['p_two_sided']) else ''} & ({gc['se']:.3f}) & {fmt_p(gc['p_two_sided'])} & "
            f"{fmt_num(oc['coef'])} & ({oc['se']:.3f}) & {fmt_p(oc['p_two_sided'])} \\\\")


full_panels = [
    ("Panel A: Session \\& Vehicle Characteristics", [
        ("charge_duration", None, False), ("wait_time", None, False),
        ("battery_capacity_centered", "Battery Capacity (kWh, centered)", False),
        ("battery_capacity_exist", "Capacity Reported (Dummy)", True),
        ("distance_from_home_log1p_centered", "Distance from Home (log, centered)", False),
        ("distance_from_home_exist", "Distance Reported (Dummy)", True),
        ("session_cost_log1p_centered", "Session Cost (log, centered)", False),
        ("charge_session_reason_long_trip", None, False), ("charge_session_reason_opportunity", None, False),
        ("veh_use_commuting", None, False), ("veh_use_private", None, False),
        ("veh_use_business", None, False), ("veh_use_gigecon", None, False)]),
    ("Panel B: Commercial Environment (within 250m)", [
        ("250m_department_store", None, False), ("250m_entertainment", None, False), ("250m_grocery", None, False),
        ("250m_medical", None, False), ("250m_restaurant", None, False), ("250m_retail", None, False),
        ("250m_standalone_bathroom", None, False), ("250m_colocated_bathroom", None, False),
        ("250m_park", None, False), ("250m_religion", "Religion (Count)", False), ("250m_work", "Work (Count)", False)]),
    ("Panel C: Driver Demographics", [
        ("gender_female", None, False), ("age_centered", None, False),
        ("age_continuous_exist", "Age Reported (Dummy)", True),
        ("race_asian", None, False), ("race_black", None, False), ("race_hispanic", None, False),
        ("edu_college", "Education: College", False), ("edu_grad", "Education: Graduate", False),
        ("hh_adults", None, False), ("hh_children", None, False),
        ("log_income_centered", None, False), ("log_income_continuous_exist", "Income Reported (Dummy)", True)]),
]
lines = []
for title, items in full_panels:
    lines.append(f"\\multicolumn{{7}}{{l}}{{\\textbf{{{title}}}}} \\\\[0.2em]")
    lines += [row_full(*r) for r in items]
    lines.append("\\midrule")
lines += [f"Observations & \\multicolumn{{3}}{{c|}}{{{int(glm.nobs):,}}} & \\multicolumn{{3}}{{c}}{{{int(ols.nobs):,}}} \\\\",
          f"Pseudo $R^{{2}}$ / $R^{{2}}$ & \\multicolumn{{3}}{{c|}}{{{cox_snell_r2(glm):.3f}}} & \\multicolumn{{3}}{{c}}{{{ols.rsquared:.3f}}} \\\\",
          f"Log-Likelihood & \\multicolumn{{3}}{{c|}}{{{glm.llf:.2f}}} & \\multicolumn{{3}}{{c}}{{{ols.llf:.2f}}} \\\\",
          row_full("const", "Constant")]
write_tex("supplementary_table_2_full_model_body.tex", "\n".join(lines) + "\n")

# --------------------------------------------------------------------------- #
# Variance inflation factors (Supplementary Table 4)
# --------------------------------------------------------------------------- #
Xv = sm.add_constant(X.dropna(), has_constant="add")
vif = pd.DataFrame({"term": Xv.columns,
                    "VIF": [variance_inflation_factor(Xv.values, i) for i in range(Xv.shape[1])]})
vif = vif[vif["term"] != "const"].sort_values("VIF", ascending=False).reset_index(drop=True)
vif["label"] = vif["term"].map(LABELS)
write_csv(vif[["term", "label", "VIF"]], TAB_DIR / "supplementary_table_4_vif.csv")
write_tex("supplementary_table_4_vif_body.tex",
          "\n".join(f"{r.label} & {r.VIF:.2f} \\\\" for r in vif.itertuples()) + "\n")

# --------------------------------------------------------------------------- #
# Robustness: amenity variable specifications (Supplementary Data 1)
#   (1) raw counts  (2) log(count+1)  (3) existence dummies
#   (4) counts + dummies  (5) log counts + dummies
# --------------------------------------------------------------------------- #
base = [c for c in predictors if not c.startswith("250m_")]
amen = [c for c in predictors if c.startswith("250m_")]
counts = X[amen]
logs = np.log1p(counts).add_suffix("_log1p")
exist = (counts > 0).astype(float).where(counts.notna()).add_suffix("_exist")
specs = {
    "(1) Count": pd.concat([X[base], counts], axis=1),
    "(2) Log count": pd.concat([X[base], logs], axis=1),
    "(3) Existence dummy": pd.concat([X[base], exist], axis=1),
    "(4) Count + existence": pd.concat([X[base], counts, exist], axis=1),
    "(5) Log count + existence": pd.concat([X[base], logs, exist], axis=1),
}
rob_rows, rob_fit = [], []
for name, Xs in specs.items():
    r = fit_two_part(df, Xs)
    for part, m in (("Extensive margin (logit)", r.glm), ("Intensive margin (OLS)", r.ols)):
        t = coef_table(m)
        t.insert(0, "specification", name)
        t.insert(1, "part", part)
        rob_rows.append(t)
    rob_fit.append({"specification": name,
                    "logit_N": int(r.glm.nobs), "logit_pseudo_R2_cox_snell": cox_snell_r2(r.glm),
                    "logit_loglik": r.glm.llf, "ols_N": int(r.ols.nobs), "ols_R2": r.ols.rsquared,
                    "ols_adj_R2": r.ols.rsquared_adj, "ols_loglik": r.ols.llf})
rob = pd.concat(rob_rows, ignore_index=True)
amen_terms = [c for c in rob["term"].unique() if c.startswith("250m_")]
rob["amenity_term"] = rob["term"].isin(amen_terms)
rob_fit = pd.DataFrame(rob_fit)
write_csv(rob, TAB_DIR / "robustness_amenity_specifications_coefficients.csv")
write_csv(rob_fit, TAB_DIR / "robustness_amenity_specifications_fit.csv")

# Wide table in the layout of the former Supplementary Table (amenity terms only)
def _base_name(t):
    return t.replace("_log1p", "").replace("_exist", "")
wide_rows = []
for a in amen:
    for kind, suffix in (("count / log count", ""), ("existence dummy", "_exist")):
        rec = {"amenity": LABELS[a].replace(" (Count)", ""), "term": kind}
        for part, short in (("Extensive margin (logit)", "logit"), ("Intensive margin (OLS)", "ols")):
            for name in specs:
                sub = rob[(rob["specification"] == name) & (rob["part"] == part)]
                if suffix == "":
                    cand = sub[sub["term"].isin([a, a + "_log1p"])]
                else:
                    cand = sub[sub["term"] == a + "_exist"]
                if len(cand):
                    c = cand.iloc[0]
                    rec[f"{short} {name} coef"] = c["coef"]
                    rec[f"{short} {name} SE"] = c["se"]
                    rec[f"{short} {name} P"] = c["p_two_sided"]
                else:
                    rec[f"{short} {name} coef"] = np.nan
                    rec[f"{short} {name} SE"] = np.nan
                    rec[f"{short} {name} P"] = np.nan
        wide_rows.append(rec)
rob_wide = pd.DataFrame(wide_rows)
with pd.ExcelWriter(SUPP_DIR / "Supplementary_Data_1_robustness_amenity_specifications.xlsx") as xw:
    readme = pd.DataFrame({"Supplementary Data 1": [
        "Robustness checks evaluating the sensitivity of ancillary spending to amenity variable specifications.",
        "Two-part hurdle model re-estimated under five specifications of the 250 m amenity variables:",
        "(1) raw counts; (2) log-transformed counts, ln(count + 1); (3) binary existence dummies;",
        "(4) raw counts and existence dummies; (5) log-transformed counts and existence dummies.",
        "All non-amenity controls of the main model are included in every specification.",
        "Extensive margin: binomial GLM (logit link) for the probability of any spending, N = 1,360 sessions.",
        "Intensive margin: OLS on ln(1 + spend) conditional on positive spending, N = 540 sessions.",
        "Standard errors are heteroscedasticity-robust (HC3); P values are from two-sided Wald tests with no adjustment for multiple comparisons.",
        "Caution: in specifications (4) and (5) the standalone-restroom count/log-count term is identified by a single location (the only one with two standalone restrooms; leverage = 1), so its HC3 standard error and P value are numerically unstable and should not be interpreted.",
        "Sheet 'amenity_terms_wide': coefficients, SEs and P values of the amenity terms in the layout of the former supplementary table.",
        "Sheet 'all_coefficients_long': every coefficient of every specification.",
        "Sheet 'model_fit': sample sizes and fit statistics.",
    ]})
    readme.to_excel(xw, sheet_name="README", index=False)
    rob_wide.to_excel(xw, sheet_name="amenity_terms_wide", index=False)
    rob.drop(columns=["model"]).to_excel(xw, sheet_name="all_coefficients_long", index=False)
    rob_fit.to_excel(xw, sheet_name="model_fit", index=False)
print("[xlsx] outputs/supplementary_data/Supplementary_Data_1_robustness_amenity_specifications.xlsx")

# --------------------------------------------------------------------------- #
# Figure 3: predicted probability of spending for significant covariates
# --------------------------------------------------------------------------- #
pvals = glm.pvalues.drop("const")
signif = pvals[pvals < 0.05].index.tolist()
print(f"[figure 3] covariates with P < 0.05 in the extensive margin: {signif}")
Xg = res.X_glm.drop(columns="const")
INTEGER_RANGE_VARS = {"hh_children": None, "250m_medical": (0, 17)}
panel_specs = []
for var in signif:
    x = Xg[var]
    if var in INTEGER_RANGE_VARS:
        lo, hi = INTEGER_RANGE_VARS[var] or (int(x.min()), int(x.max()))
        grid = np.arange(lo, hi + 1, dtype=float)
        kind = "discrete"
    elif x.nunique() < 10:
        grid = np.sort(x.unique())
        kind = "binary" if len(grid) == 2 else "discrete"
    else:
        grid = np.linspace(x.min(), x.max(), 200)
        kind = "continuous"
    ref = reference_row(Xg, exclude=var)
    newX = pd.DataFrame([ref] * len(grid))
    newX[var] = grid
    newX = sm.add_constant(newX[Xg.columns], has_constant="add")[res.X_glm.columns]
    pred = glm.get_prediction(newX)
    lin = newX.values @ glm.params.values
    se_lin = np.sqrt(np.einsum("ij,jk,ik->i", newX.values, glm.cov_params().values, newX.values))
    prob = 1 / (1 + np.exp(-lin))
    lo_ci = 1 / (1 + np.exp(-(lin + norm.ppf(0.025) * se_lin)))
    hi_ci = 1 / (1 + np.exp(-(lin + norm.ppf(0.975) * se_lin)))
    out = pd.DataFrame({"x": grid, "predicted_probability": prob, "ci_low_95": lo_ci, "ci_high_95": hi_ci})
    # sample distribution of the covariate (for the grey background)
    if kind == "continuous":
        hist, edges = np.histogram(x, bins=40)
        dist = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "n_sessions": hist})
    else:
        vc = x.round().value_counts().sort_index()
        dist = pd.DataFrame({"x": vc.index.values, "n_sessions": vc.values})
    safe = var.replace("250m_", "X250m_")
    write_csv(out, SD_DIR / f"fig3_marginal_{safe}.csv")
    write_csv(dist, SD_DIR / f"fig3_distribution_{safe}.csv")
    panel_specs.append({"var": var, "kind": kind, "xlabel": X_AXIS_LABELS.get(var, LABELS.get(var, var)),
                        "p_value": float(pvals[var]), "n": int(glm.nobs)})
json.dump(panel_specs, open(SD_DIR / "fig3_panels.json", "w"), indent=2)

# Headline quantities quoted in the text (predicted probabilities at specific values)
ref = reference_row(Xg, exclude="charge_duration")
def _p_at(var, val):
    r = reference_row(Xg, exclude=var); r[var] = val
    nx = sm.add_constant(pd.DataFrame([r])[Xg.columns], has_constant="add")[res.X_glm.columns]
    return float(glm.predict(nx)[0])
headline = {
    "p_spend_30min": _p_at("charge_duration", 0.5), "p_spend_60min": _p_at("charge_duration", 1.0),
    "p_spend_battery_mean": _p_at("battery_capacity_centered", 0.0),
    "p_spend_battery_plus25": _p_at("battery_capacity_centered", 25.0),
    "p_long_trip_0": _p_at("charge_session_reason_long_trip", 0), "p_long_trip_1": _p_at("charge_session_reason_long_trip", 1),
    "p_opportunity_0": _p_at("charge_session_reason_opportunity", 0), "p_opportunity_1": _p_at("charge_session_reason_opportunity", 1),
    "p_children_0": _p_at("hh_children", 0), "p_children_1": _p_at("hh_children", 1),
    "p_medical_0": _p_at("250m_medical", 0), "p_medical_1": _p_at("250m_medical", 1),
    "p_entertainment_0": _p_at("250m_entertainment", 0), "p_entertainment_1": _p_at("250m_entertainment", 1),
    "p_grocery_0": _p_at("250m_grocery", 0), "p_grocery_1": _p_at("250m_grocery", 1),
    "p_standalone_0": _p_at("250m_standalone_bathroom", 0), "p_standalone_1": _p_at("250m_standalone_bathroom", 1),
    "ols_pct_long_trip": 100 * (np.exp(ols.params["charge_session_reason_long_trip"]) - 1),
    "ols_pct_female": 100 * (np.exp(ols.params["gender_female"]) - 1),
    "ols_pct_battery_25kwh": 100 * (np.exp(25 * ols.params["battery_capacity_centered"]) - 1),
    "ols_elasticity_income_10pct": 100 * (1.1 ** ols.params["log_income_centered"] - 1),
}
json.dump(headline, open(LOG_DIR / "02_headline_quantities.json", "w"), indent=2)
print(json.dumps(headline, indent=2))
print("[done] 02_two_part_model.py")

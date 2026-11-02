"""
08_source_data.py
-----------------
Assembles the journal deliverables from the CSV outputs of scripts 01-07:

  outputs/source_data/Source_Data.xlsx
      one sheet per figure panel (Fig1a, Fig1b, Fig2, Fig3a-f, Fig4a, Fig4b,
      SuppFig1a-c, SuppFig2a-c, SuppFig3, SuppFig5a-c, SuppFig6a-c) holding
      exactly the values that are plotted.
  outputs/supplementary_data/Supplementary_Data_2_model_outputs.xlsx
      every table that appears in the main text or the Supplementary
      Information (Tables 1-3, Supplementary Tables 1-4), the Wald tests and
      the counterfactual scenarios, with full-precision coefficients and
      two-sided P values.
Also copies the figure PDFs into manuscript/Figures.
"""
import json
import shutil

import pandas as pd

from common import FIG_DIR, ROOT, SD_DIR, SUPP_DIR, TAB_DIR

# --------------------------------------------------------------------------- #
# Source Data (figures)
# --------------------------------------------------------------------------- #
sheets = {}
sheets["Fig1a"] = pd.read_csv(SD_DIR / "fig1a_visits_by_store.csv")
sheets["Fig1b"] = pd.read_csv(SD_DIR / "fig1b_transactions_by_store.csv")
fig2 = pd.read_csv(SD_DIR / "fig2_amenity_count_distribution.csv")
fig2s = pd.read_csv(SD_DIR / "fig2_amenity_count_summary.csv")
sheets["Fig2"] = fig2
sheets["Fig2_summary"] = fig2s
panels = json.load(open(SD_DIR / "fig3_panels.json"))
order = ["battery_capacity_centered", "charge_duration", "hh_children", "250m_medical",
         "charge_session_reason_long_trip", "charge_session_reason_opportunity", "250m_standalone_bathroom"]
letters = "abcdefg"
for letter, var in zip(letters, order):
    safe = var.replace("250m_", "X250m_")
    m = pd.read_csv(SD_DIR / f"fig3_marginal_{safe}.csv")
    d = pd.read_csv(SD_DIR / f"fig3_distribution_{safe}.csv")
    m.insert(0, "covariate", var)
    d.insert(0, "covariate", var)
    name = f"Fig3{letter}" if letter != "g" else "Fig3_extra_standalone"
    sheets[name] = m
    sheets[name + "_dist"] = d
sheets["Fig4a"] = pd.read_csv(SD_DIR / "fig4a_visit_share_by_reason.csv")
sheets["Fig4b"] = pd.read_csv(SD_DIR / "fig4b_transactions_by_reason.csv")
s1 = pd.read_csv(SD_DIR / "suppfig1_stage1_convergence.csv")
sheets["SuppFig1a"] = s1[["sample_size", "n_fits", "mean_mcfadden_pseudo_r2"]]
sheets["SuppFig1b"] = s1[["sample_size", "n_fits", "mean_coef_charge_duration", "sd_coef_charge_duration"]]
sheets["SuppFig1c"] = s1[["sample_size", "n_fits", "mean_coef_long_trip", "sd_coef_long_trip"]]
s2 = pd.read_csv(SD_DIR / "suppfig2_stage2_convergence.csv")
sheets["SuppFig2a"] = s2[["sample_size", "n_fits", "mean_adj_r2"]]
sheets["SuppFig2b"] = s2[["sample_size", "n_fits", "mean_coef_charge_duration", "sd_coef_charge_duration"]]
sheets["SuppFig2c"] = s2[["sample_size", "n_fits", "mean_coef_battery_capacity", "sd_coef_battery_capacity"]]
if (SD_DIR / "suppfig3_charger_coordinates.csv").exists():
    sheets["SuppFig3"] = pd.read_csv(SD_DIR / "suppfig3_charger_coordinates.csv")
cf = pd.read_csv(SD_DIR / "suppfig5_counterfactual.csv")
d = cf[cf["scenario_type"] == "delta_minutes"]
sheets["SuppFig5a"] = d[["scenario", "delta_minutes", "avg_probability", "pct_change_probability"]]
sheets["SuppFig5b"] = d[["scenario", "delta_minutes", "avg_expected_spend", "pct_change_expected_spend"]]
sheets["SuppFig5c"] = d[["scenario", "delta_minutes", "avg_expected_spend", "abs_change_expected_spend_usd"]]
demo = pd.read_csv(SD_DIR / "suppfig6_demographic_comparison.csv")
for letter, metric in zip("abc", ["Household income", "Education", "Age"]):
    sheets[f"SuppFig6{letter}"] = demo[demo["Metric"] == metric]

readme = pd.DataFrame({"Sheet": list(sheets), "Content": [
    "Figure 1a: number of reported transactions per amenity type (N = 1,586 sessions).",
    "Figure 1b: every individual reported transaction amount (US$) by amenity type; plotted as box, dots and mean.",
    "Figure 2: number of charging locations (n = 1,429) with each integer count of amenities within 250 m, by amenity type.",
    "Figure 2: summary statistics (median, quartiles) drawn on Figure 2.",
] + [x for letter, var in zip(letters, order) for x in (
    f"Figure 3{letter if letter != 'g' else ' (optional panel)'}: predicted probability of spending and 95% CI along {var} (n = 1,360 sessions).",
    f"Figure 3{letter if letter != 'g' else ' (optional panel)'}: sample distribution of {var} shown as grey bars.")] + [
    "Figure 4a: percentage of respondents in each charging-reason group who visited each amenity type, with standard error of the proportion.",
    "Figure 4b: every transaction amount (US$) with a reported value, by amenity type and charging reason.",
    "Supplementary Figure 1a: mean McFadden pseudo-R2 of the stage-1 logit across 200 random sub-samples per size.",
    "Supplementary Figure 1b: mean charge-duration coefficient (stage 1).",
    "Supplementary Figure 1c: mean long-trip coefficient (stage 1).",
    "Supplementary Figure 2a: mean adjusted R2 of the stage-2 OLS across 200 random sub-samples per size.",
    "Supplementary Figure 2b: mean charge-duration coefficient (stage 2).",
    "Supplementary Figure 2c: mean battery-capacity coefficient (stage 2).",
] + (["Supplementary Figure 3: coordinates of the chargers nearest to the reported charging locations (model sample, n = 1,360)."] if "SuppFig3" in sheets else []) + [
    "Supplementary Figure 5a: change in average predicted probability of spending for each charge-duration shift.",
    "Supplementary Figure 5b: percentage change in average expected ancillary spend.",
    "Supplementary Figure 5c: change in average expected ancillary spend in US$ per session.",
    "Supplementary Figure 6a: household income distribution of the survey sample and EV-owner benchmarks (%).",
    "Supplementary Figure 6b: education distribution (%).",
    "Supplementary Figure 6c: age distribution (%).",
]})
with pd.ExcelWriter(SD_DIR / "Source_Data.xlsx") as xw:
    readme.to_excel(xw, sheet_name="README", index=False)
    for name, df in sheets.items():
        df.to_excel(xw, sheet_name=name[:31], index=False)
print("[xlsx] outputs/source_data/Source_Data.xlsx")

# --------------------------------------------------------------------------- #
# Supplementary Data 2: all tables
# --------------------------------------------------------------------------- #
tables = {
    "Table1_spending_by_store": TAB_DIR / "table_1_spending_by_store.csv",
    "Table2_SuppTable2_main_model": TAB_DIR / "supplementary_table_2_full_model.csv",
    "Table3_SuppTable3_by_reason": TAB_DIR / "supplementary_table_3_trip_purpose_full.csv",
    "SuppTable3_Wald_tests": TAB_DIR / "wald_equality_tests.csv",
    "SuppTable3_Wald_survey_item": TAB_DIR / "wald_equality_tests_sensitivity_survey_item_outcome.csv",
    "SuppTable1_summary_stats": TAB_DIR / "supplementary_table_1_summary_stats.csv",
    "SuppTable4_VIF": TAB_DIR / "supplementary_table_4_vif.csv",
    "Counterfactual_scenarios": TAB_DIR / "counterfactual_charge_duration.csv",
    "Fig4_sample_sizes": TAB_DIR / "fig4_sample_size_reconciliation.csv",
}
notes = pd.DataFrame({"Supplementary Data 2": [
    "Full-precision outputs of every table in the main text and Supplementary Information.",
    "Models: two-part hurdle model. Extensive margin = binomial GLM (logit link) for any ancillary spending;",
    "intensive margin = OLS on ln(1 + spend) conditional on positive spending. Standard errors are HC3 robust;",
    "P values are two-sided Wald tests with no adjustment for multiple comparisons.",
    "Table2_SuppTable2_main_model: columns with suffix _logit refer to the extensive margin, _ols to the intensive margin.",
    "Table3_SuppTable3_by_reason: extensive-margin logit estimated separately within each charging-reason group.",
    "SuppTable3_Wald_tests: joint Wald test (2 d.f.) that a coefficient is equal across the three groups, from a fully interacted pooled logit.",
    "Counterfactual_scenarios: recycled predictions for shifts and caps of charge duration (Supplementary Figure 5).",
]})
with pd.ExcelWriter(SUPP_DIR / "Supplementary_Data_2_model_outputs.xlsx") as xw:
    notes.to_excel(xw, sheet_name="README", index=False)
    for name, p in tables.items():
        pd.read_csv(p).to_excel(xw, sheet_name=name[:31], index=False)
print("[xlsx] outputs/supplementary_data/Supplementary_Data_2_model_outputs.xlsx")

# --------------------------------------------------------------------------- #
# Copy figures into the manuscript folder
# --------------------------------------------------------------------------- #
dst = ROOT / "manuscript" / "Figures"
dst.mkdir(parents=True, exist_ok=True)
for p in sorted(FIG_DIR.glob("*.pdf")):
    shutil.copy(p, dst / p.name)
print(f"[copy] {len(list(FIG_DIR.glob('*.pdf')))} figure PDFs -> manuscript/Figures/")
print("[done] 08_source_data.py")

"""
04_counterfactual.py
--------------------
Counterfactual sensitivity of ancillary spending to charge duration
(Supplementary Figure 5 and the numbers quoted in the Results).

Each session's observed charge duration is shifted by -20 ... +20 minutes
(5-minute steps, bounded at zero) or fixed at 10, 12.5 and 15 minutes.  For
every scenario the expected unconditional spend is

    E[spend] = Pr(spend > 0)  x  max(0, exp(X'gamma) * s - 1),

where s is Duan's smearing factor, mean(exp(residuals)) of the intensive-margin
OLS model.  Predictions are averaged over the extensive-margin estimation
sample (recycled predictions).

Outputs: outputs/tables/counterfactual_charge_duration.csv and
         outputs/source_data/suppfig5_counterfactual.csv
"""
import numpy as np
import pandas as pd

from common import SD_DIR, TAB_DIR, available_predictors, design_matrix, fit_two_part, load_data, write_csv

df = load_data()
predictors = available_predictors(df)
res = fit_two_part(df, design_matrix(df, predictors))
glm, ols = res.glm, res.ols
smearing = float(np.mean(np.exp(ols.resid)))
base = res.X_glm.copy()   # N = 1,360 sessions incl. constant


def expected_spend(Xn):
    p_any = np.asarray(glm.predict(Xn))
    log_spend = np.asarray(ols.predict(Xn[ols.model.exog_names]))
    cond = np.maximum(0.0, np.exp(log_spend) * smearing - 1.0)
    return pd.DataFrame({"p_any": p_any, "cond_spend": cond, "expected_spend": p_any * cond})


baseline = expected_spend(base).mean()
rows = []
for delta in range(-20, 25, 5):
    cf = base.copy()
    cf["charge_duration"] = np.maximum(0.0, cf["charge_duration"] + delta / 60.0)
    avg = expected_spend(cf).mean()
    rows.append({"scenario_type": "delta_minutes", "scenario": f"{delta:+d} minutes", "delta_minutes": delta,
                 "charge_duration_minutes": np.nan, **avg.rename({"p_any": "avg_probability",
                 "cond_spend": "avg_conditional_spend", "expected_spend": "avg_expected_spend"}).to_dict()})
for minutes in (10, 12.5, 15):
    cf = base.copy()
    cf["charge_duration"] = minutes / 60.0
    avg = expected_spend(cf).mean()
    rows.append({"scenario_type": "fixed_minutes", "scenario": f"{minutes:g} minutes (cap)", "delta_minutes": np.nan,
                 "charge_duration_minutes": minutes, **avg.rename({"p_any": "avg_probability",
                 "cond_spend": "avg_conditional_spend", "expected_spend": "avg_expected_spend"}).to_dict()})
out = pd.DataFrame(rows)
out["abs_change_expected_spend_usd"] = out["avg_expected_spend"] - baseline["expected_spend"]
out["pct_change_probability"] = 100 * (out["avg_probability"] / baseline["p_any"] - 1)
out["pct_change_expected_spend"] = 100 * (out["avg_expected_spend"] / baseline["expected_spend"] - 1)
out.attrs["smearing"] = smearing
write_csv(out, TAB_DIR / "counterfactual_charge_duration.csv")
write_csv(out, SD_DIR / "suppfig5_counterfactual.csv")
print(f"[counterfactual] N = {int(glm.nobs)}; smearing factor = {smearing:.4f}; "
      f"baseline P(spend) = {baseline['p_any']:.4f}; baseline expected spend = ${baseline['expected_spend']:.2f}")
with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
    print(out[["scenario", "avg_probability", "avg_expected_spend", "abs_change_expected_spend_usd",
               "pct_change_probability", "pct_change_expected_spend"]])
print("[done] 04_counterfactual.py")

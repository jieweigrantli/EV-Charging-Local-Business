"""
05_convergence.py
-----------------
Monte Carlo parameter-convergence analysis (Supplementary Figures 1 and 2).

For increasing sub-sample sizes, the extensive-margin logit (Stage 1) and the
intensive-margin OLS (Stage 2) are re-fitted on 200 random sub-samples; the
mean fit statistic and the mean of two key coefficients are tracked.

Outputs: outputs/source_data/suppfig1_stage1_convergence.csv,
         outputs/source_data/suppfig2_stage2_convergence.csv
"""
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import SD_DIR, available_predictors, design_matrix, fit_two_part, load_data, write_csv

ITERATIONS = 200
SEED = 42
rng = np.random.default_rng(SEED)

df = load_data()
predictors = available_predictors(df)
res = fit_two_part(df, design_matrix(df, predictors))
X1, y1 = res.X_glm.values, res.y_glm.values
X2, y2 = res.X_ols.values, res.y_ols.values
names1, names2 = list(res.X_glm.columns), list(res.X_ols.columns)

t0 = time.time()
# Stage 1 ---------------------------------------------------------------------
sizes1 = np.arange(200, (len(y1) // 10) * 10 + 1, 50)
rows = []
i_cd, i_lt = names1.index("charge_duration"), names1.index("charge_session_reason_long_trip")
for n in sizes1:
    r2s, c1s, c2s = [], [], []
    for _ in range(ITERATIONS):
        idx = rng.choice(len(y1), size=n, replace=False)
        try:
            m = sm.GLM(y1[idx], X1[idx], family=sm.families.Binomial()).fit()
        except Exception:
            continue
        # drop sub-samples with (quasi-)complete separation, where the MLE diverges
        if not m.converged or np.abs(m.params).max() > 50:
            continue
        r2s.append(1 - m.deviance / m.null_deviance)
        c1s.append(m.params[i_cd]); c2s.append(m.params[i_lt])
    rows.append({"sample_size": int(n), "n_fits": len(r2s), "mean_mcfadden_pseudo_r2": np.mean(r2s),
                 "mean_coef_charge_duration": np.mean(c1s), "mean_coef_long_trip": np.mean(c2s),
                 "sd_coef_charge_duration": np.std(c1s), "sd_coef_long_trip": np.std(c2s)})
    print(f"[stage 1] n = {n}: pseudo-R2 = {np.mean(r2s):.4f}, charge_duration = {np.mean(c1s):.4f}")
s1 = pd.DataFrame(rows)
write_csv(s1, SD_DIR / "suppfig1_stage1_convergence.csv")

# Stage 2 ---------------------------------------------------------------------
sizes2 = np.arange(100, (len(y2) // 10) * 10 + 1, 20)
rows = []
j_cd, j_bc = names2.index("charge_duration"), names2.index("battery_capacity_centered")
for n in sizes2:
    r2s, c1s, c2s = [], [], []
    for _ in range(ITERATIONS):
        idx = rng.choice(len(y2), size=n, replace=False)
        m = sm.OLS(y2[idx], X2[idx]).fit()
        r2s.append(m.rsquared_adj)
        c1s.append(m.params[j_cd]); c2s.append(m.params[j_bc])
    rows.append({"sample_size": int(n), "n_fits": len(r2s), "mean_adj_r2": np.mean(r2s),
                 "mean_coef_charge_duration": np.mean(c1s), "mean_coef_battery_capacity": np.mean(c2s),
                 "sd_coef_charge_duration": np.std(c1s), "sd_coef_battery_capacity": np.std(c2s)})
s2 = pd.DataFrame(rows)
write_csv(s2, SD_DIR / "suppfig2_stage2_convergence.csv")
print(f"[done] 05_convergence.py ({time.time() - t0:.0f} s)")

# Electric vehicle charging supports local business spending beyond charging revenue

Replication code, Source Data and Supplementary Data for

> Li, J. G., Hardman, S. & Jenn, A. *Electric vehicle charging supports local business spending beyond charging revenue.* Communications Sustainability (2026). Manuscript COMMSSUSTAIN-26-0205A.

The study links a 2025 survey of 1,586 self-reported DC fast charging (DCFC) sessions in the United States and Canada with the commercial amenities within 250 m of each charging location, and estimates a two-part hurdle model of ancillary spending (whether anything was spent, and how much).

* Code archive (Zenodo): https://doi.org/10.5281/zenodo.XXXXXXX
* Data (Dryad): https://doi.org/10.5061/dryad.XXXXXXX
* GitHub: https://github.com/jieweigrantli/EV-Charging-Local-Business

## What is reproduced

| Output | Script | Files |
|---|---|---|
| Table 1 (spending by amenity type), Supplementary Table 1 (descriptives) | `01_descriptives.py` | `outputs/tables/table_1_*`, `supplementary_table_1_*` |
| Table 2 / Supplementary Table 2 (two-part hurdle model), Supplementary Table 4 (VIF), Supplementary Data 1 (amenity-specification robustness) | `02_two_part_model.py` | `outputs/tables/table_2_*`, `supplementary_table_2_*`, `supplementary_table_4_*`, `outputs/supplementary_data/Supplementary_Data_1_*.xlsx` |
| Table 3 / Supplementary Table 3 (logit by reason for charging) and Wald tests of coefficient equality | `03_heterogeneity.py` | `outputs/tables/table_3_*`, `supplementary_table_3_*`, `wald_equality_tests*.csv` |
| Charge-duration counterfactuals (Results text, Supplementary Fig. 5) | `04_counterfactual.py` | `outputs/tables/counterfactual_charge_duration.csv` |
| Monte Carlo convergence analysis (Supplementary Figs. 1-2) | `05_convergence.py` | `outputs/source_data/suppfig1_*`, `suppfig2_*` |
| Figures 1-4, Supplementary Figs. 1, 2, 5, 6 | `06_figures.py` | `outputs/figures/*.pdf` (+ `.png`) |
| Supplementary Fig. 3 (map of charging locations) | `07_map.py` | `outputs/figures/Supplementary_Figure_3.pdf` |
| Source Data workbook (one sheet per figure panel), Supplementary Data 2 (all model outputs) | `08_source_data.py` | `outputs/source_data/Source_Data.xlsx`, `outputs/supplementary_data/Supplementary_Data_2_model_outputs.xlsx` |
| Independent R re-estimation of the two-part model with `sandwich` HC3 errors | `R/crosscheck_two_part_model.R` | `outputs/tables/r_crosscheck_two_part_model.csv` |

Supplementary Fig. 4 (amenities around one charger in Davis, CA) is an illustration made with the Google Places API and is kept as a static image in `manuscript/image/`.

## Quick start

```bash
git clone https://github.com/jieweigrantli/EV-Charging-Local-Business.git
cd EV-Charging-Local-Business
# 1. get the data: download the analytic dataset from Dryad (https://doi.org/10.5061/dryad.XXXXXXX)
#    and save it as data/filtered_dataset.csv
# 2. environment
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 3. run everything
python run_all.py              # ~3-4 min; add --quick to skip the Monte Carlo step
Rscript code/R/crosscheck_two_part_model.R   # optional (needs R >= 4.1, sandwich, lmtest)
```

Every script can also be run on its own from the `code/` folder (`python code/02_two_part_model.py`); each one reads the dataset directly and writes to `outputs/`. The scripts are numbered in the order used by `run_all.py` (06-08 depend on the CSVs written by 01-05).

Tested with Python 3.11.1 on macOS 14; package versions are pinned in `requirements.txt` (pandas 2.2.3, NumPy 1.26.4, SciPy 1.14.1, statsmodels 0.14.4, matplotlib 3.9.2, cmcrameri 1.8, geopandas 1.0.1). The R cross-check was run with R 4.5.0, sandwich 3.1-1 and lmtest 0.9-40.

## Data

No survey data are distributed with this code. The analytic dataset (one row per reported DC fast charging session, N = 1,586, with a variable dictionary) is deposited in Dryad (https://doi.org/10.5061/dryad.XXXXXXX); download it and save it as `data/filtered_dataset.csv`. The scripts read that file directly and perform no further cleaning. See [`data/README.md`](data/README.md) for the variable dictionary.

The Dryad file omits the race and education variables to prevent re-identification. The pipeline detects this, drops the five race/education predictors from every model and prints a warning; sample sizes are unchanged (N = 1,360 / 540) but coefficients differ slightly from the published values. The published estimates were obtained with the restricted file that includes those variables, which is available from the corresponding author (jwgli@ucdavis.edu) under a data use agreement; placing it at the same path reproduces the paper exactly. A different path can be given with `EVCLB_DATA=/path/to/file.csv`.

Two small, non-personal inputs are included in the repository: `data/external/ev_owner_demographic_benchmarks.csv` (percentages of EV owners by income, education and age, derived from the public NHTS 2022 and 2024 California Vehicle Survey microdata; used for Supplementary Fig. 6) and the Natural Earth boundaries, which `07_map.py` downloads on first use. `data/charger_coordinates.csv` (the public charger nearest to each marked charging location, unlinked to responses) is also included; it is the Source Data of Supplementary Fig. 3, and `07_map.py` skips the map if it is removed.

## Model summary

* **Extensive margin**: binomial GLM (logit link) for `store_total_spent_calc > 0`, N = 1,360 sessions with complete covariates. 36 predictors (session, vehicle, 250 m amenity counts, demographics, reporting dummies for imputed covariates).
* **Intensive margin**: OLS on ln(1 + spend) for the 540 sessions with positive spending, same predictors.
* Standard errors: HC3 (heteroscedasticity robust); P values: two-sided Wald tests, no multiple-comparison adjustment. Pseudo-R2 reported in the tables is Cox-Snell.
* **Stratified models** (Table 3): the extensive-margin logit re-estimated within each reason-for-charging group (missing predictors set to zero, gig-economy dummy dropped), plus joint Wald tests (2 d.f.) from a fully interacted pooled logit.
* **Counterfactuals**: recycled predictions with Duan's smearing retransformation.

## Repository layout

```
code/                 numbered replication scripts + common.py (configuration, model helpers)
code/R/               R cross-check of the two-part model
data/                 analytic datasets, variable dictionary, charger coordinates, benchmark percentages
outputs/figures       Figure_1-4 and Supplementary_Figure_1-6 (PDF + PNG)
outputs/tables        CSV and LaTeX table bodies for every table in the paper
outputs/source_data   per-panel CSVs and Source_Data.xlsx
outputs/supplementary_data  Supplementary_Data_1 (robustness) and _2 (all model outputs)
outputs/logs          model summaries and headline quantities quoted in the text
manuscript/           LaTeX source of the main text (main.tex), SI (si.tex), bibliography and figures
```

## Differences from the originally submitted outputs

Running the clean pipeline on the analytic dataset reproduces the two-part model, the stratified models, the VIFs and the counterfactual scenarios exactly. Three items differ and are documented here:

1. **Table 1 and Figure 1** are now computed on the same N = 1,586 analytic sample as everything else (896 transactions with a reported amount). The submitted version had been tabulated on an earlier, larger extract (1,066 transactions).
2. **Wald tests in Supplementary Table 3** are now computed with a Newton-Raphson fit of the pooled interacted logit, which converges; the L-BFGS fit in the original script stopped before convergence, so a few P values move by up to a few hundredths.
3. **Robustness specifications (4) and (5)**: the standalone-restroom terms are identified by a single location with two standalone restrooms (leverage ~1), so their HC3 standard errors are numerically unstable and differ between statsmodels versions. Coefficients are unaffected.

## License

Code: MIT (see `LICENSE`). The datasets are released under the terms stated in the Dryad deposit.

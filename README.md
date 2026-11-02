# Electric vehicle charging supports local business spending beyond charging revenue

Replication code, Source Data and Supplementary Data for

> Li, J. G., Hardman, S. & Jenn, A. *Electric vehicle charging supports local business spending beyond charging revenue.* Communications Sustainability (2026).

The study links a 2025 survey of 1,586 self-reported DC fast charging sessions in the United States and Canada with the commercial amenities within 250 m of each charging location, and estimates a two-part hurdle model of ancillary spending (whether anything was spent, and how much).

* Code archive (Zenodo): https://doi.org/10.5281/zenodo.XXXXXXX
* Data (Dryad): https://doi.org/10.5061/dryad.XXXXXXX

## Quick start

```bash
git clone https://github.com/jieweigrantli/EV-Charging-Local-Business.git
cd EV-Charging-Local-Business
# 1. download the dataset from Dryad and save it as data/filtered_dataset.csv
# 2. create the environment
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 3. run everything (about 3 minutes; --quick skips the Monte Carlo step)
python run_all.py
```

Optional: `Rscript code/R/crosscheck_two_part_model.R` re-estimates the two-part model in R (needs `sandwich` and `lmtest`) and reports the difference to the Python estimates.

Tested with Python 3.11.1; package versions are pinned in `requirements.txt`.

## What each script produces

| Script | Output |
|---|---|
| `01_descriptives.py` | Table 1, Supplementary Table 1, data for Figures 1, 2, 4 and Supplementary Fig. 6 |
| `02_two_part_model.py` | Table 2, Supplementary Tables 2 and 4, Supplementary Data 1 (amenity-specification robustness), data for Figure 3 |
| `03_heterogeneity.py` | Table 3, Supplementary Table 3 (logit by reason for charging, Wald tests) |
| `04_counterfactual.py` | Charge-duration counterfactuals (Supplementary Fig. 5) |
| `05_convergence.py` | Monte Carlo convergence analysis (Supplementary Figs. 1 and 2) |
| `06_figures.py` | Figures 1 to 4, Supplementary Figs. 1, 2, 5 and 6 |
| `07_map.py` | Supplementary Fig. 3 (map of charging locations) |
| `08_source_data.py` | `Source_Data.xlsx` (one sheet per figure panel) and Supplementary Data 2 (all model outputs) |

All outputs are written to `outputs/` (`tables/`, `figures/`, `source_data/`, `supplementary_data/`, `logs/`). Each script can also be run on its own from the `code/` folder; `common.py` holds the model specification and shared helpers. Supplementary Fig. 4 is a static illustration (`manuscript/image/`).

## Data

No survey data are distributed with the code. Download the analytic dataset (N = 1,586 sessions, with a variable dictionary) from Dryad and save it as `data/filtered_dataset.csv`; the scripts read it directly. See [`data/README.md`](data/README.md) for the variables.

The Dryad file omits the race and education variables to protect respondents. The scripts detect this, drop the five race/education predictors and print a warning; sample sizes are unchanged (N = 1,360 and 540) and coefficients differ slightly from the published values, which were estimated on the restricted file. The restricted file is available from the corresponding author (jwgli@ucdavis.edu) under a data use agreement; placed at the same path it reproduces the paper exactly. Another path can be given with `EVCLB_DATA=/path/to/file.csv`.

Included in the repository: `data/charger_coordinates.csv` (the public charger nearest to each marked charging location, unlinked to responses; Source Data of Supplementary Fig. 3) and `data/external/ev_owner_demographic_benchmarks.csv` (EV-owner percentages from the public NHTS 2022 and 2024 California Vehicle Survey microdata, for Supplementary Fig. 6). Natural Earth boundaries for the map are downloaded on first use.

## Model

* Extensive margin: binomial GLM (logit link) for `store_total_spent_calc > 0`, N = 1,360 sessions with complete covariates, 36 predictors.
* Intensive margin: OLS on ln(1 + spend) for the 540 sessions with positive spending, same predictors.
* HC3 robust standard errors; two-sided Wald tests; no multiple-comparison adjustment; Cox-Snell pseudo-R2.
* Stratified models: the extensive-margin logit within each reason-for-charging group, with joint Wald tests of coefficient equality from a fully interacted pooled logit.
* Counterfactuals: recycled predictions with Duan's smearing retransformation.

## Layout

```
code/                 numbered scripts, common.py, R/ cross-check
data/                 dataset (download), coordinates, benchmark percentages, variable dictionary
outputs/              tables, figures, Source Data, Supplementary Data, logs
manuscript/           LaTeX source of the main text and Supplementary Information
```

## License

Code: MIT (see `LICENSE`). Data: see the Dryad deposit.

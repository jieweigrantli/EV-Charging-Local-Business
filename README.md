# Electric vehicle charging supports local business spending beyond charging revenue

Replication code, Source Data and Supplementary Data for

> Li, J. G., Hardman, S. & Jenn, A. *Electric vehicle charging supports local business spending beyond charging revenue.*

The study links a 2025 survey of 1,586 self-reported DC fast charging sessions in the United States and Canada with the commercial amenities within 250 m of each charging location, and estimates a two-part hurdle model of ancillary spending (whether anything was spent, and how much).

* Code archive (Zenodo): [![DOI](https://zenodo.org/badge/1442414483.svg)](https://doi.org/10.5281/zenodo.23877269)
* Data (Dryad): https://doi.org/10.5061/dryad.v15dv42bb

## Quick start

```bash
git clone https://github.com/jieweigrantli/EV-Charging-Local-Business.git
cd EV-Charging-Local-Business
# 1. download filtered_dataset_public.csv from Dryad and save it in data/
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

All outputs are written to `outputs/` (`tables/`, `figures/`, `source_data/`, `supplementary_data/`, `logs/`), or to `outputs_public/` when the public Dryad file is used. Neither folder is part of the repository: the scripts create them. The published Source Data and Supplementary Data are available with the paper. Each script can also be run on its own from the `code/` folder; `common.py` holds the model specification and shared helpers.

## Data

No survey data are distributed with the code. Download the public dataset (`filtered_dataset_public.csv`, N = 1,586 sessions, with a variable dictionary) from Dryad and save it in `data/`; the scripts read it directly. See [`data/README.md`](data/README.md) for the variables.

The public file is de-identified (the Dryad README lists each step):

* **Kept:** spending, coded store categories, session characteristics and amenity counts within 250 m.
* **Banded:** distance from home and battery capacity.
* **Removed:** demographics, free text and response IDs.
* **Merged:** medical or dental store visits are combined with "Other".

The scripts detect the public file and:

* rebuild the model variables from the kept columns;
* omit the demographic and vehicle-use predictors;
* estimate the models on the same sessions as the paper (N = 1,360 and 540), using the `in_model_sample` flag.

The main results are reproduced, and coefficients differ slightly from the published values. The full analytic dataset is available from the corresponding author (jwgli@ucdavis.edu) on request; placed at `data/filtered_dataset.csv`, it reproduces the paper exactly. Another path can be given with `EVCLB_DATA=/path/to/file.csv`.

Included in the repository: `data/external/ev_owner_demographic_benchmarks.csv` (EV-owner percentages from the public NHTS 2022 and 2024 California Vehicle Survey microdata, for Supplementary Fig. 6). The charger coordinates behind the map (Supplementary Fig. 3) are not distributed, so `07_map.py` skips the map unless `data/charger_coordinates.csv` is present; Natural Earth boundaries for the map are downloaded on first use.

## Model

* Extensive margin: binomial GLM (logit link) for `store_total_spent_calc > 0`, N = 1,360 sessions with complete covariates, 36 predictors.
* Intensive margin: OLS on ln(1 + spend) for the 540 sessions with positive spending, same predictors.
* HC3 robust standard errors; two-sided Wald tests; no multiple-comparison adjustment; Cox-Snell pseudo-R2.
* Stratified models: the extensive-margin logit within each reason-for-charging group, with joint Wald tests of coefficient equality from a fully interacted pooled logit.
* Counterfactuals: recycled predictions with Duan's smearing retransformation.

## Layout

```
code/                 numbered scripts, common.py, R/ cross-check
data/                 dataset (download), benchmark percentages
outputs/              tables, figures, Source Data, Supplementary Data, logs (created by run_all.py; not tracked)
```

## License

Code: MIT (see `LICENSE`). Data: see the Dryad deposit.

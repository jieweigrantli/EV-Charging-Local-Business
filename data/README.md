# Data

Only `README.md`, `charger_coordinates.csv` and `external/ev_owner_demographic_benchmarks.csv` are tracked in git. The dataset files below are obtained from Dryad (public version) or from the authors (restricted version) and placed in this folder; they are listed in `.gitignore` so that they are never committed.

| File | Rows x columns | Description |
|---|---|---|
| `filtered_dataset.csv` | 1,586 x 181 | **Restricted** analytic dataset used for every published estimate: one row per self-reported DC fast charging session after the pre-specified filters (charge duration <= 5 h, wait time <= 2 h, session cost <= $100, respondent >= 18 years). Includes the race and education variables. Not distributed publicly; available from the corresponding author under a data use agreement. |
| `filtered_dataset.csv` (Dryad version) | 1,586 x 163 | **Public** version deposited in Dryad (https://doi.org/10.5061/dryad.v15dv42bb), to be saved under the same file name. Identical to the restricted file except that the 18 race and education columns (`race_*`, `edu_*`, `resp_education`) are removed to prevent re-identification. The pipeline runs on this file and omits the five race/education predictors, so the estimates differ slightly from the published ones. |
| `filtered_dataset_mapping.csv` | 163 rows | Variable dictionary distributed with the Dryad deposit: variable name, source (survey item code, Google Places API, derived) and type. |
| `charger_coordinates.csv` | 1,356 x 2 | Latitude/longitude of the public charger nearest to the location each respondent marked on the survey map (inferred during data processing, not reported by respondents), for the sessions in the model sample (n = 1,360; four locations geocoded outside North America are dropped). Rounded to four decimals, shuffled and without identifiers, so points cannot be linked to responses. Used for Supplementary Figure 3 and included in the Source Data. |
| `external/ev_owner_demographic_benchmarks.csv` | 16 rows | Percentages of California EV owners (2024 California Vehicle Survey, TSDC) and national EV owners (2022 NHTS, primary drivers of BEVs/PHEVs, person weights) by income, education and age bin. Derived from the public survey microdata (NHTS 2022 person file with person weights; CVS 2024 household and vehicle files); used for Supplementary Figure 6. |
| `external/ne_50m_admin_1_states_provinces/` | - | Natural Earth 1:50m state/province boundaries (public domain). Downloaded automatically by `code/07_map.py`; not stored in the repository. |

## Key variables

* `store_total_spent_calc` - ancillary spending during the session in US$ (sum of the amounts reported for up to two stores); `NaN` when no amount was reported. The extensive-margin outcome is `store_total_spent_calc > 0`.
* `store_total_spent_anything` - survey item "did you spend money?" (1/0); used only for the sensitivity version of the pooled Wald tests (Supplementary Data 2). All models use `store_total_spent_calc > 0`.
* `charge_duration`, `wait_time` - hours. `session_cost` - US$.
* `250m_*` - number of amenities of each type within 250 m of the reported charging location (Google Places API).
* `*_centered` - continuous covariates centred at the sample mean of respondents who reported them; the matching `*_exist` dummy equals 1 when the value was reported (missing values are set to the centred mean, i.e. zero).
* `charge_session_reason_{long_trip,routine,opportunity}` - reasons for charging (multiple selections allowed).
* `store_type_*`, `store2_type_*`, `store_spent`, `store2_spent` - category and amount for the first and second reported store.

The full list of survey items behind each variable is in `filtered_dataset_mapping.csv`; the survey instrument is reproduced in the Supplementary Information of the paper.

# Data

Only `README.md` and `external/ev_owner_demographic_benchmarks.csv` are tracked in git. The dataset files below come from Dryad (public version) or from the authors (full version). Save them in this folder; `.gitignore` keeps them out of the repository.

| File | Rows x columns | Description |
|---|---|---|
| `filtered_dataset_public.csv` | 1,586 x 43 | **Public** de-identified dataset, deposited in Dryad (https://doi.org/10.5061/dryad.v15dv42bb) together with its variable dictionary `filtered_dataset_public_mapping.csv`. It contains spending, coded store categories, session characteristics, amenity counts within 250 m, distance from home in bands, battery capacity in 10 kWh bands, and an `in_model_sample` flag for the 1,360 sessions in the published model. It has no demographic variables, no free text and no response IDs. The README of the Dryad deposit lists every de-identification step. |
| `filtered_dataset.csv` | 1,586 x 181 | **Full** analytic dataset used for every published estimate, including all demographic variables. It is not distributed publicly and is available from the corresponding author on request. With this file in place, the pipeline reproduces the paper exactly. |
| `charger_coordinates.csv` | 1,356 x 2 | Latitude and longitude of the public charger nearest to each reported charging location in the model sample, used only for the map in Supplementary Figure 3. **Not distributed**: combined with the amenity counts in the public file, the coordinates could link responses to specific charging sites. `code/07_map.py` skips the map when the file is absent. |
| `external/ev_owner_demographic_benchmarks.csv` | 16 rows | Percentages of California EV owners (2024 California Vehicle Survey, TSDC) and national EV owners (2022 NHTS, primary drivers of BEVs and PHEVs, person weights) by income, education and age group. Computed from the public survey microdata and used for Supplementary Figure 6. |
| `external/ne_50m_admin_1_states_provinces/` | - | Natural Earth 1:50m state and province boundaries (public domain). Downloaded automatically by `code/07_map.py`; not stored in the repository. |

## Running the code on the public file

`code/common.py` detects the public file and does three things:

* **Rebuilds the model variables.** `add_derived()` builds them from the columns in the file:
  * charge duration and wait time in hours;
  * `session_cost_log1p_centered`;
  * the distance and battery terms, using the band midpoints and the centring constants of the published model.
* **Uses the published sample.** It estimates the two-part model on the sessions with `in_model_sample = 1`, so N = 1,360 and 540 as in the paper.
* **Writes to a separate folder.** Outputs go to `outputs_public/`, so the outputs of the full file in `outputs/` are never overwritten.

With the public file, the models leave out the 16 demographic and vehicle-use predictors that it does not contain; `RESTRICTED_PREDICTORS` in `common.py` lists them. Analyses that need those variables are skipped:

* the demographic rows of Supplementary Table 1;
* Supplementary Figure 6;
* the household-children panel of Figure 3.

The main results are reproduced, with coefficients that differ slightly from the published values:

* charge duration;
* reason for charging;
* nearby medical facilities;
* standalone restrooms;
* battery capacity.

The grocery-store effect on the extensive margin (P = 0.10 in the paper) is not reproduced: P = 0.14 with the public file. Medical or dental store visits are merged into "Other" in the public file, so Table 1 shows seven categories instead of eight.

## Key variables

* `store_total_spent_calc` is ancillary spending during the session in US$: the amounts reported for up to two stores combined. It is empty when no amount was reported. The extensive-margin outcome is `store_total_spent_calc > 0`.
* `charge_session_spent_money` is the survey item "did you spend money?" (1/0). The code turns it into `store_total_spent_anything`. It is used only for the sensitivity version of the pooled Wald tests in Supplementary Data 2; all models use `store_total_spent_calc > 0`.
* `charge_duration` and `wait_time` are in hours. The public file stores them as the hours and minutes parts. `session_cost` is in US$.
* `250m_*` is the number of amenities of each type within 250 m of the reported charging location, from the Google Places API.
* `*_centered` columns are continuous covariates centred at the sample mean of respondents who reported them. The matching `*_exist` dummy equals 1 when the value was reported; missing values are set to the centred mean, which is zero. The full file stores these columns; with the public file the code rebuilds them.
* `charge_session_reason_{long_trip,routine,opportunity}` are the reasons for charging. Respondents could select more than one.
* `store_type_*`, `store2_type_*`, `store_spent` and `store2_spent` give the category and amount for the first and second reported store.

The variable dictionary on Dryad (`filtered_dataset_public_mapping.csv`) gives the survey item behind each variable. The survey instrument is reproduced in the Supplementary Information of the paper.

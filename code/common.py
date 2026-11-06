"""
Shared configuration and helper functions for the replication pipeline of

    Li, Hardman & Jenn, "Electric vehicle charging supports local business
    spending beyond charging revenue".

Every numbered script in this folder imports this module.  Nothing in here
performs data cleaning: the analysis starts from the analytic dataset
(N = 1,586 sessions) exactly as exported by the authors' data-processing
pipeline.

Two versions of the dataset exist:
  * data/filtered_dataset.csv         restricted file (all variables; available
                                      from the corresponding author on request)
  * data/filtered_dataset_public.csv  de-identified public file (Dryad); see
                                      data/README.md and the Dryad README
With the public file, the model variables are rebuilt from the kept columns by
`add_derived()`, the models omit the demographic and vehicle-use predictors
that are not in the public file, and outputs are written to outputs_public/ so
the published outputs in outputs/ are never overwritten.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# The restricted analytic dataset is used when it is present.  Otherwise the
# pipeline falls back to the public (Dryad) version; set EVCLB_DATA to force a
# particular file.
_default = DATA_DIR / "filtered_dataset.csv"
_public = DATA_DIR / "filtered_dataset_public.csv"
DATA_FILE = Path(os.environ.get("EVCLB_DATA", _default if _default.exists() else _public))
if not DATA_FILE.exists():
    sys.exit(
        f"Dataset not found: {DATA_FILE}\n"
        "Download filtered_dataset_public.csv from the Dryad deposit (https://doi.org/10.5061/dryad.v15dv42bb) "
        "and save it in data/, or set EVCLB_DATA to its path."
    )
# The public file has no demographic columns.
PUBLIC = "race_asian" not in pd.read_csv(DATA_FILE, nrows=0).columns

OUT_DIR = ROOT / ("outputs_public" if PUBLIC else "outputs")
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
SD_DIR = OUT_DIR / "source_data"          # per-panel CSVs -> Source Data.xlsx
SUPP_DIR = OUT_DIR / "supplementary_data"  # Supplementary Data xlsx files
LOG_DIR = OUT_DIR / "logs"
for _d in (FIG_DIR, TAB_DIR, SD_DIR, SUPP_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Model specification
# --------------------------------------------------------------------------- #
TARGET = "store_total_spent_calc"   # ancillary spending in US$ (NaN = none reported)

PREDICTORS = [
    "race_asian", "race_black", "race_hispanic",
    "gender_female", "edu_college", "edu_grad",
    "hh_adults", "hh_children",
    "charge_duration", "wait_time",
    "250m_department_store", "250m_entertainment", "250m_grocery", "250m_medical",
    "250m_park", "250m_religion", "250m_restaurant", "250m_standalone_bathroom",
    "250m_colocated_bathroom", "250m_retail", "250m_work",
    "veh_use_commuting", "veh_use_private", "veh_use_business", "veh_use_gigecon",
    "charge_session_reason_long_trip", "charge_session_reason_opportunity",
    "battery_capacity_exist", "distance_from_home_exist",
    "log_income_continuous_exist", "age_continuous_exist",
    "battery_capacity_centered", "distance_from_home_log1p_centered",
    "session_cost_log1p_centered", "log_income_centered", "age_centered",
]
# Predictors that are only in the restricted dataset (not on Dryad).
RESTRICTED_PREDICTORS = ["race_asian", "race_black", "race_hispanic", "gender_female", "edu_college", "edu_grad",
                         "hh_adults", "hh_children", "veh_use_commuting", "veh_use_private", "veh_use_business",
                         "veh_use_gigecon", "log_income_continuous_exist", "age_continuous_exist",
                         "log_income_centered", "age_centered"]

# Public file: 0/1 flag for the 1,360 sessions in the published model (complete
# cases on all 36 predictors), so the reduced public model uses the same sessions.
SAMPLE_FLAG = "in_model_sample"

# Centring constants of the published model (sample means of the reported values).
CENTER_DISTANCE_LOG1P = 3.4767063923
CENTER_BATTERY = 76.14056456
CENTER_SESSION_COST_LOG1P = 1.71971755

# Bands used in the public file: (lower, upper, label, value used in the model).
# A value x falls in a band when lower <= x < upper.  Labels are words ("2 to 5"),
# not "2-5", so spreadsheet software does not read them as dates.
DISTANCE_BANDS = [  # miles
    (0, 1e-9, "0", 0.0), (1e-9, 2, "0 to 2", 1.0), (2, 5, "2 to 5", 3.5), (5, 10, "5 to 10", 7.5),
    (10, 25, "10 to 25", 17.5), (25, 50, "25 to 50", 37.5), (50, 100, "50 to 100", 75.0),
    (100, 250, "100 to 250", 175.0), (250, 500, "250 to 500", 375.0), (500, np.inf, "500 or more", 750.0),
]
BATTERY_BANDS = [(0, 30, "under 30", 25.0)] + [
    (lo, lo + 10, f"{lo} to {lo + 10}", lo + 5.0) for lo in range(30, 140, 10)] + [(140, np.inf, "140 or more", 145.0)]

AMENITY_VARS = [c for c in PREDICTORS if c.startswith("250m_")]
REASON_VARS = {
    "Opportunity": "charge_session_reason_opportunity",
    "Long Trip": "charge_session_reason_long_trip",
    "Routine": "charge_session_reason_routine",
}

LABELS = {
    "race_asian": "Race: Asian",
    "race_black": "Race: Black",
    "race_hispanic": "Race: Hispanic",
    "gender_female": "Gender: Female (Dummy)",
    "edu_college": "Education: College Graduate",
    "edu_grad": "Education: Graduate/Professional",
    "hh_adults": "Household Adults (Count)",
    "hh_children": "Household Children (Count)",
    "charge_duration": "Charge Duration (Hours)",
    "wait_time": "Wait Time (Hours)",
    "250m_department_store": "Department Store (Count)",
    "250m_entertainment": "Entertainment (Count)",
    "250m_grocery": "Grocery Store (Count)",
    "250m_medical": "Medical Facility (Count)",
    "250m_park": "Park (Count)",
    "250m_religion": "Religious Facility (Count)",
    "250m_restaurant": "Restaurant (Count)",
    "250m_standalone_bathroom": "Standalone Restroom (Count)",
    "250m_colocated_bathroom": "Co-located Restroom (Count)",
    "250m_retail": "Retail Store (Count)",
    "250m_work": "Workplace (Count)",
    "veh_use_commuting": "Vehicle Use: Commuting",
    "veh_use_private": "Vehicle Use: Private",
    "veh_use_business": "Vehicle Use: Business",
    "veh_use_gigecon": "Vehicle Use: Gig Economy",
    "charge_session_reason_long_trip": "Charging Reason: Long Trip",
    "charge_session_reason_opportunity": "Charging Reason: Opportunity",
    "battery_capacity_exist": "Battery Reported (Dummy)",
    "distance_from_home_exist": "Distance Reported (Dummy)",
    "log_income_continuous_exist": "Income Reported (Dummy)",
    "age_continuous_exist": "Age Reported (Dummy)",
    "battery_capacity_centered": "Battery Capacity (kWh, centered)",
    "distance_from_home_log1p_centered": "Distance from Home (log, centered)",
    "session_cost_log1p_centered": "Session Cost (log, centered)",
    "log_income_centered": "Household Income (log, centered)",
    "age_centered": "Driver Age (Years, centered)",
    "const": "Constant",
}

# Axis labels for the marginal-probability panels (Figure 3)
X_AXIS_LABELS = {
    "battery_capacity_centered": "Battery capacity (kWh, centred at sample mean)",
    "charge_duration": "Charge duration (hours)",
    "hh_children": "Household children (count)",
    "250m_medical": "Medical facilities within 250 m (count)",
    "charge_session_reason_long_trip": "Charging reason: long trip",
    "charge_session_reason_opportunity": "Charging reason: opportunity",
    "250m_entertainment": "Entertainment venues within 250 m (count)",
    "250m_grocery": "Grocery stores within 250 m (count)",
    "250m_standalone_bathroom": "Standalone restrooms within 250 m (count)",
}

# Store categories (survey item G17Q52 / store_type_* columns)
STORE_TYPES = {
    "cafe": "Café/Restaurant/Bar",
    "work": "Place of Work/Study",
    "retail": "Retail/Clothing Store",
    "grocery": "Grocery Store",
    "department_store": "Department/Big Box Store",
    "medical": "Medical/Dental Facility",
    "entertainment": "Cinema/Theatre/Entertainment",
    "other": "Other",
}
STORE_PREFIXES = ["store_", "store2_"]

# Colours
PALETTE_PRIMARY = "#3C5488"
PALETTE_SECONDARY = "#E64B35"
PALETTE_TERTIARY = "#00A087"
PALETTE_QUATERNARY = "#4DBBD5"
REASON_COLORS = {"Opportunity": PALETTE_TERTIARY, "Long Trip": PALETTE_SECONDARY, "Routine": PALETTE_QUATERNARY}
REASON_ORDER = ["Opportunity", "Long Trip", "Routine"]


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
def band_label(x: pd.Series, bands) -> pd.Series:
    """Label of the band each value falls in (NaN stays NaN)."""
    out = pd.Series(pd.NA, index=x.index, dtype="object")
    for lo, hi, label, _ in bands:
        out[(x >= lo) & (x < hi)] = label
    return out


def band_value(labels: pd.Series, bands) -> pd.Series:
    """Representative value (band midpoint) for each band label."""
    return labels.map({label: v for _, _, label, v in bands}).astype(float)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the model variables from the columns kept in the public file.

    Each derived column is created only when it is absent, so the restricted
    file (which stores them) is left unchanged.  The formulas are the ones used
    to build the restricted file: missing values are set to 0 and flagged by an
    `_exist` indicator, and continuous variables are centred with the constants
    of the published model.
    """
    df = df.copy()
    num = lambda c: pd.to_numeric(df[c], errors="coerce")  # noqa: E731
    for v in ("charge_duration", "wait_time"):
        if v not in df.columns and f"{v}_hours" in df.columns:
            df[v] = num(f"{v}_hours").fillna(0) + num(f"{v}_minutes").fillna(0) / 60.0
    if "store_total_spent_anything" not in df.columns and "charge_session_spent_money" in df.columns:
        df["store_total_spent_anything"] = (num("charge_session_spent_money") == 1).astype(int)
    if "session_cost_log1p_centered" not in df.columns:
        df["session_cost_log1p_centered"] = np.log1p(num("session_cost")) - CENTER_SESSION_COST_LOG1P
    if "distance_from_home" not in df.columns and "distance_from_home_band" in df.columns:
        df["distance_from_home"] = band_value(df["distance_from_home_band"], DISTANCE_BANDS)
    if "distance_from_home_log1p_centered" not in df.columns:
        d = num("distance_from_home").fillna(0)
        df["distance_from_home_exist"] = (d > 0).astype(int)
        df["distance_from_home_log1p_centered"] = (np.log1p(d) - CENTER_DISTANCE_LOG1P) * df["distance_from_home_exist"]
    if "battery_capacity" not in df.columns and "battery_capacity_band" in df.columns:
        df["battery_capacity"] = band_value(df["battery_capacity_band"], BATTERY_BANDS)
    if "battery_capacity_centered" not in df.columns:
        b = num("battery_capacity").fillna(0)
        df["battery_capacity_exist"] = (b > 0).astype(int)
        df["battery_capacity_centered"] = (b - CENTER_BATTERY) * df["battery_capacity_exist"]
    if "response_id" not in df.columns and "record_id" in df.columns:
        df["response_id"] = df["record_id"]
    return df


def load_data(path: Path | None = None) -> pd.DataFrame:
    """Load the analytic dataset (one row per reported DC fast charging session)."""
    path = Path(path) if path else DATA_FILE
    df = add_derived(pd.read_csv(path, low_memory=False))
    print(f"[data] {path.name}: {df.shape[0]} sessions x {df.shape[1]} columns"
          + (" (public file; outputs -> outputs_public/)" if PUBLIC else ""))
    return df


def available_predictors(df: pd.DataFrame, predictors=PREDICTORS) -> list[str]:
    """Return the predictors present in `df`, warning about any that are missing
    (the public dataset has no demographic or vehicle-use predictors)."""
    missing = [c for c in predictors if c not in df.columns]
    if missing:
        print(
            "[warning] The following predictors are not in the dataset and are "
            f"omitted from all models: {missing}.\n"
            "          Estimates will therefore differ from the published values, "
            "which use the restricted dataset.",
            file=sys.stderr,
        )
    return [c for c in predictors if c in df.columns]


def design_matrix(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    X = df[predictors].apply(pd.to_numeric, errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- #
# Two-part hurdle model (logit for any spending; OLS on log(1 + spend) | spend > 0)
# --------------------------------------------------------------------------- #
class TwoPartResult:
    def __init__(self, glm, ols, X_glm, y_glm, X_ols, y_ols, df_glm, df_ols):
        self.glm, self.ols = glm, ols
        self.X_glm, self.y_glm, self.X_ols, self.y_ols = X_glm, y_glm, X_ols, y_ols
        self.df_glm, self.df_ols = df_glm, df_ols


def fit_two_part(df: pd.DataFrame, X: pd.DataFrame, target: str = TARGET, cov_type: str = "HC3",
                 complete_cases: bool = True) -> TwoPartResult:
    """Two-part hurdle model.

    Part 1: binomial GLM (logit link) on spend_anything = 1[target > 0]
            (sessions with no reported spending amount count as zero).
    Part 2: OLS on log(1 + target) for sessions with target > 0.
    Standard errors are heteroscedasticity-robust (HC3) in both parts.
    """
    X = X.copy()
    if complete_cases:
        keep = X.notna().all(axis=1)
        if SAMPLE_FLAG in df.columns:   # public file: same sessions as the published model
            keep &= df.loc[X.index, SAMPLE_FLAG] == 1
        X = X.loc[keep]
    y = pd.to_numeric(df.loc[X.index, target], errors="coerce")
    y_any = (y > 0).astype(int)

    X1 = sm.add_constant(X, has_constant="add")
    glm = sm.GLM(y_any, X1, family=sm.families.Binomial()).fit(cov_type=cov_type)

    pos = y > 0
    X2 = sm.add_constant(X.loc[pos], has_constant="add")
    y2 = np.log1p(y.loc[pos])
    ols = sm.OLS(y2, X2).fit(cov_type=cov_type)
    return TwoPartResult(glm, ols, X1, y_any, X2, y2, df.loc[X.index], df.loc[X2.index])


def mcfadden_r2(glm) -> float:
    return 1.0 - glm.deviance / glm.null_deviance


def cox_snell_r2(glm) -> float:
    n = glm.nobs
    return 1.0 - np.exp(2.0 * (glm.llnull - glm.llf) / n)


def coef_table(res, model_name: str = "") -> pd.DataFrame:
    """Tidy coefficient table with two-sided Wald p-values (HC3 SE)."""
    ci = res.conf_int()
    out = pd.DataFrame({
        "model": model_name,
        "term": res.params.index,
        "label": [LABELS.get(t, t) for t in res.params.index],
        "coef": res.params.values,
        "se": res.bse.values,
        "z_or_t": res.tvalues.values,
        "p_two_sided": res.pvalues.values,
        "ci_low": ci.iloc[:, 0].values,
        "ci_high": ci.iloc[:, 1].values,
    })
    return out


# --------------------------------------------------------------------------- #
# Formatting helpers for LaTeX tables
# --------------------------------------------------------------------------- #
def stars(p: float) -> str:
    if pd.isna(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def fmt_coef(coef: float, se: float, p: float, digits: int = 3) -> str:
    """'0.520*** (0.133)' with a LaTeX minus sign."""
    if pd.isna(coef):
        return "--"
    c = f"{coef:.{digits}f}".replace("-", "$-$")
    return f"{c}{stars(p)} ({se:.{digits}f})"


def fmt_num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace("-", "$-$")


def write_tex(name: str, text: str) -> Path:
    p = TAB_DIR / name
    p.write_text(text)
    print(f"[tex]  {p.relative_to(ROOT)}")
    return p


def write_csv(df: pd.DataFrame, path: Path, **kw) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=kw.pop("index", False), **kw)
    print(f"[csv]  {path.relative_to(ROOT)}")
    return path


# --------------------------------------------------------------------------- #
# Store visits / transactions (used for Table 1, Figure 1 and Figure 4)
# --------------------------------------------------------------------------- #
def store_visited_mask(df: pd.DataFrame, type_col: str, key: str) -> pd.Series:
    """A store of category `key` was visited (regression format uses 0/1; in the
    restricted file the 'other' category is recorded as free text)."""
    if key == "other" and df[type_col].dtype == object:
        return df[type_col].notna()
    col = df[type_col]
    return (col == 1) | (col == 1.0) | (col == "Yes")


def transactions_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (session, store slot) with a visited store category.

    `amount` is the self-reported amount spent at that store (NaN when the
    respondent did not report an amount).  Both the first and the second
    reported store are included.
    """
    rows = []
    for key, display in STORE_TYPES.items():
        for prefix in STORE_PREFIXES:
            type_col = f"{prefix}type_{key}"
            spent_col = f"{prefix}spent"
            if type_col not in df.columns:
                continue
            visited = store_visited_mask(df, type_col, key)
            sub = df.loc[visited]
            rows.append(pd.DataFrame({
                "response_id": sub["response_id"].values,
                "store_slot": prefix.rstrip("_"),
                "store_key": key,
                "store_type": display,
                "amount": pd.to_numeric(sub[spent_col], errors="coerce").values,
                "reason_opportunity": sub["charge_session_reason_opportunity"].values,
                "reason_long_trip": sub["charge_session_reason_long_trip"].values,
                "reason_routine": sub["charge_session_reason_routine"].values,
            }))
    return pd.concat(rows, ignore_index=True)


def reference_row(X: pd.DataFrame, exclude: str) -> dict:
    """Hold every other covariate at its mean (if > 2 unique values) or mode."""
    ref = {}
    for col in X.columns:
        if col == exclude:
            continue
        s = X[col]
        ref[col] = float(s.mean()) if s.nunique() > 2 else float(s.mode().iloc[0])
    return ref


def session_summary(df: pd.DataFrame) -> dict:
    """Headline counts used in the manuscript text."""
    return {
        "n_sessions": int(len(df)),
        "n_any_spend": int((pd.to_numeric(df[TARGET], errors="coerce") > 0).sum()),
    }

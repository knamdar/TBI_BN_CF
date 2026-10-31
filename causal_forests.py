"""Causal forest estimation of treatment-effect variation after traumatic brain injury.

Self-contained: data access, multiple imputation, the shared cohort preparation used by
the Bayesian network analysis, and the six treatment-outcome models with their
diagnostics and figures. Run all stages with `python causal_forests.py`, or a single
stage by name, e.g. `... causal_forests.py pooled_effects`. Outputs go to out/.

Data are not distributed with this code; see README.md.
"""
from __future__ import annotations
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import pandas as pd
import shap
import sys
import time
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pathlib import Path
from scipy import stats


# ============================================================================
# Data access
# ============================================================================
joblib.parallel_config(prefer="threads").__enter__()

# ---------------------------------------------------------------------------
# CONFIGURATION - set these to your own filenames. README.md describes what each
# file must contain and where to obtain it.
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")

FILES = {
    "analysis":        DATA_DIR / "analysis_cohort_coded.dta",
    "analysis_raw":       DATA_DIR / "analysis_cohort_raw.dta",
    "recruited_coded":    DATA_DIR / "recruited_cohort_coded.dta",
    "recruited_labelled": DATA_DIR / "recruited_cohort_labelled.dta",
}
CONTINUOUS_FILE = DATA_DIR / "continuous_covariates.csv"
N_IMPUTATIONS = 30


# Output directories. Every stage writes under out/; these are created once here so
# that any stage can be run on its own.
OUTDIR = Path("out")
for _d in [OUTDIR] + [OUTDIR / n for n in ['cf_cate', 'cf_partial', 'cf_shap', 'posteriors']]:
    _d.mkdir(parents=True, exist_ok=True)
NAME_MAP = {
    "Surgery": "CranialSurgery",
    "Stratum": "AdmissionUnit",
    "HospitalStay": "LengthHospitalStay",
    "InjCause": "InjuryCause",
    "PreInjASAPS": "PreInjuryASAPS",
}

TIER = {
    "Age": 1, "Sex": 1, "Race": 1, "EduLevel": 1, "MaritalStatus": 1, "PreInjASAPS": 1,
    "GCS": 2, "PupilReactivity": 2, "PupilSymmetry": 2, "Hypoxia": 2, "Hypotension": 2,
    "Glucose": 2, "Hemoglobin": 2, "InjuryISS": 2, "InjCause": 2, "MarshallCTScore": 2,
    "RotterdamCTScore": 2, "TSAH": 2, "SubduralHematoma": 2, "EpiduralHematoma": 2,
    "MassLesion": 2, "MidlineShift": 2, "SkullFracture": 2,
    "Surgery": 3, "ICPMonitor": 3, "Stratum": 3,
    "HospitalStay": 4,
    "QoLIBRI": 5, "FUWorkStatus": 5,
    "GOSE": 6,
}

TIER_NAME = {
    1: "pre-injury", 2: "presentation/baseline", 3: "acute hospital course",
    4: "discharge", 5: "follow-up (6 mo, concurrent with outcome)", 6: "outcome",
}

PRIMARY_OUTCOME = "GOSE"

SECONDARY_OUTCOME = "HospitalStay"

_ABSENT_PRESENT = {1: "Absent", 2: "Present"}

CODEBOOK = {
    "Age": {1: "< 15", 2: "16-30", 3: "31-45", 4: "46-60", 5: "61-75", 6: "> 76"},
    "Sex": {1: "Female", 2: "Male"},                      # inverse of the STATA label
    "Race": {1: "White European", 2: "White", 3: "White South American", 4: "Others"},
    "EduLevel": {0: "None, not currently in school", 1: "Currently in diploma/degree",
                 2: "Primary school", 3: "Secondary/High school",
                 4: "Post-High school training", 5: "College/University"},
    "MaritalStatus": {1: "Never married", 2: "Married", 3: "Living together",
                      4: "Divorced", 5: "Separated", 6: "Widowed", 9: "Others"},
    "PreInjASAPS": {1: "Normal/Healthy", 2: "Mild systemic disease",
                    3: "Severe systemic disease",
                    4: "Severe systemic disease, constant threat to life"},
    "GCS": {k: f"GCS total {k}" for k in range(3, 16)},
    "PupilReactivity": {0: "Both reactive", 1: "One reactive", 2: "None reactive"},  # CORRECTED
    "PupilSymmetry": {1: "Equal", 2: "Unequal R>L", 3: "Unequal L>R"},
    "Hypoxia": {0: "No", 1: "Single episode, short duration",
                2: "Multiple episodes, prolonged"},
    "Hypotension": {0: "No", 1: "Single episode, short duration",
                    2: "Multiple episodes, prolonged"},
    "Glucose": {1: "Normal", 2: "Pre-diabetes", 3: "Diabetes"},
    "Hemoglobin": {1: "Normal", 2: "Mild anaemia", 3: "Moderate anaemia",
                   4: "Severe anaemia"},
    "InjuryISS": {1: "Minor", 2: "Moderate", 3: "Severe", 4: "Critical"},
    "InjCause": {1: "Road traffic accident", 2: "Incidental fall",
                 3: "Non-intentional injury", 4: "Violence/Assault",
                 5: "Act of mass violence", 6: "Suicide attempt", 9: "Others"},
    "MarshallCTScore": {k: str(k) for k in range(1, 7)},
    "RotterdamCTScore": {k: str(k) for k in range(1, 7)},
    "TSAH": _ABSENT_PRESENT, "SubduralHematoma": _ABSENT_PRESENT,
    "EpiduralHematoma": _ABSENT_PRESENT, "MassLesion": _ABSENT_PRESENT,
    "MidlineShift": _ABSENT_PRESENT, "SkullFracture": _ABSENT_PRESENT,
    "Surgery": {0: "No", 1: "Yes"},
    "ICPMonitor": {0: "No", 1: "Yes"},
    "Stratum": {0: "Non-ICU (ER or ward)", 1: "ICU"},
    "HospitalStay": {1: "1 day", 2: "1 week", 3: "2 weeks", 4: "1 month",
                     5: "2 months", 6: "> 2 months"},
    "QoLIBRI": {1: "Low", 2: "Moderate", 3: "High"},
    "FUWorkStatus": {1: "Returned to previous work, same level/hours",
                     2: "Unable to work", 5: "Looking for work", 7: "Retired",
                     8: "Returned at increased level/hours",
                     9: "Returned at reduced level/hours",
                     10: "Change of job / different work",
                     11: "Special/sheltered employment"},
    "GOSE": {1: "1 (dead)", 2: "2 & 3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8"},
}

LEAKY = ["QoLIBRI", "FUWorkStatus", "HospitalStay"]

def load_analysis() -> pd.DataFrame:
    """One imputed copy of the analytic cohort, indexed by subject identifier."""
    df = pd.read_stata(FILES["analysis"], convert_categoricals=False)
    return df.set_index("subjectid")

def load_source(which: str = "analysis_raw") -> pd.DataFrame:
    """A pre-imputation file, with columns renamed to analysis-file names."""
    df = pd.read_stata(FILES[which], convert_categoricals=False)
    inv = {v: k for k, v in NAME_MAP.items()}
    return df.rename(columns=inv).set_index("subjectid")

def predictors(df: pd.DataFrame, leakage_free: bool = True) -> list[str]:
    """Predictor columns for a GOSE prognostic model.

    leakage_free=True  -> 26 predictors, tiers 1-3 only (the primary analysis)
    leakage_free=False -> 28 predictors, including the follow-up variables
    """
    cols = [c for c in df.columns if c not in (PRIMARY_OUTCOME,)]
    if leakage_free:
        cols = [c for c in cols if c not in LEAKY]
    else:
        cols = [c for c in cols if c != SECONDARY_OUTCOME]
    return cols

def decode(df: pd.DataFrame) -> pd.DataFrame:
    """Map numeric codes to corrected labels (for tables and figures, not modelling)."""
    out = df.copy()
    for col in out.columns:
        if col in CODEBOOK:
            out[col] = out[col].map(CODEBOOK[col]).astype("object")
    return out

CONTINUOUS = CONTINUOUS_FILE

TREATMENT_STATUS_UNKNOWN = {"Surgery": 667, "ICPMonitor": 0, "Stratum": 0}

def load_continuous() -> pd.DataFrame:
    """Continuous covariates, indexed by subject identifier.

    Columns: Age_years, ISS_total, LOS_days, Hb_baseline_gdl, Glucose_baseline_mmolL,
    GCS_total, GCS_motor, days_to_death, ICP_{start_day,end_day,n_readings,mean,max}.

    Two corrections are applied here rather than assumed of the input file:
    haemoglobin values recorded in g/L are rescaled to g/dL, and a negative length of
    stay is floored at zero. Both are no-ops on a file that already carries them.
    """
    df = pd.read_csv(CONTINUOUS, index_col=0)
    if "Hb_baseline_gdl" in df.columns:
        in_gl = df["Hb_baseline_gdl"] > 30          # no plausible g/dL value reaches 30
        df.loc[in_gl, "Hb_baseline_gdl"] /= 10.0
    if "LOS_days" in df.columns:
        df["LOS_days"] = df["LOS_days"].clip(lower=0)
    return df

def unknown_treatment_mask(treatment: str) -> pd.Series:
    """Boolean mask of patients whose status for `treatment` was imputed, not observed."""
    src = load_source("analysis_raw")   # already renamed to analysis-file names
    if treatment not in src.columns:
        raise KeyError(f"{treatment}: no pre-imputation source column available")
    return src[treatment].isna().reindex(load_analysis().index)

def tier_table() -> pd.DataFrame:
    rows = [{"Variable": v, "Tier": t, "Window": TIER_NAME[t],
             "Leakage_free_predictor": "n/a" if v == PRIMARY_OUTCOME
             else ("no" if v in LEAKY else "yes")}
            for v, t in TIER.items()]
    return pd.DataFrame(rows).sort_values(["Tier", "Variable"]).reset_index(drop=True)


# ============================================================================
# Multiple imputation
# ============================================================================
joblib.parallel_config(prefer="threads").__enter__()

OUT = Path("out")

IMPUTED_DIR = OUT / "imputed"

NOMINAL = [
    "Sex", "Race", "EduLevel", "MaritalStatus", "InjCause", "PupilSymmetry",
    "MarshallCTScore",                 # 5=evacuated vs 6=non-evacuated mass lesion: not ordered
    "Surgery", "Stratum", "ICPMonitor",
    "TSAH", "SubduralHematoma", "EpiduralHematoma", "MassLesion", "MidlineShift",
    "SkullFracture",
]

NUMERIC = [
    "Age_years", "ISS_total", "GCS", "PreInjASAPS", "Glucose", "Hemoglobin",
    "Hypoxia", "Hypotension", "PupilReactivity", "RotterdamCTScore",
    "GOSE",                            # outcome, included in the imputation model
]

DROP = ["QoLIBRI", "FUWorkStatus", "HospitalStay", "Age", "InjuryISS"]

RARE_COLLAPSE = {"InjCause": {5: 4}}

def build_missing_frame() -> pd.DataFrame:
    """Analysis-file values with NaN restored wherever the source value was missing.

    This recovers the true missing-data pattern on the analysis file's granular
    coding (the supplied 'not imputed' file is coarser for five variables, so it
    cannot be used directly).
    """
    a = load_analysis()
    s = load_source("analysis_raw").reindex(a.index)
    c = load_continuous().reindex(a.index)

    df = a.copy()
    for col in df.columns:
        if col in s.columns:
            df.loc[s[col].isna(), col] = np.nan
    # ICPMonitor has no source column; verified to have no missing values.
    assert df["ICPMonitor"].notna().all()

    df["Age_years"] = c["Age_years"]        # complete
    df["ISS_total"] = c["ISS_total"]
    df = df.drop(columns=[x for x in DROP if x in df.columns])

    for col, mapping in RARE_COLLAPSE.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)
    return df

def run_mice(m: int = 20, iterations: int = 5, seed: int = 20260219,
             include_outcome: bool = True, outdir: Path | None = None,
             verbose: bool = False):
    """Fit MICE and return m completed datasets, also written to `outdir`.

    include_outcome
        Whether GOSE enters the imputation model as a predictor of the other
        variables. The methodological default is True (von Hippel 2007; White,
        Royston & Wood 2011): omitting the outcome generates covariate values
        from a model in which the outcome played no part, attenuating their
        association with it. Set False only as a deliberate, declared choice.

        GOSE is never itself imputed -- it is complete by cohort construction,
        since the 698 patients without a six-month GOSE were excluded. When
        include_outcome is False it is simply held out of the predictor matrix
        and re-attached to each completed dataset unchanged.
    """
    import miceforest as mf

    outdir = Path(outdir) if outdir is not None else IMPUTED_DIR
    df = build_missing_frame()
    assert df["GOSE"].notna().all(), "GOSE must be complete before imputation"

    held = None
    if not include_outcome:
        held = df["GOSE"].copy()
        df = df.drop(columns=["GOSE"])

    for col in NOMINAL:
        if col in df.columns:
            df[col] = df[col].astype("category")

    ids = df.index                      # miceforest requires a RangeIndex
    kernel = mf.ImputationKernel(df.reset_index(drop=True), num_datasets=m,
                                 random_state=seed, save_all_iterations_data=True)
    # min_data_in_leaf guards against LightGBM emitting exactly 0.0 class
    # probabilities, which would make miceforest's mean-matching odds ratio infinite.
    kernel.mice(iterations, verbose=verbose, min_data_in_leaf=20)

    outdir.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(m):
        comp = kernel.complete_data(dataset=i)
        comp.index = ids                # restore subjectid
        if held is not None:
            comp["GOSE"] = held         # outcome re-attached, never imputed
        assert comp.notna().all().all(), f"dataset {i} still has missing values"
        comp.to_parquet(outdir / f"mice_{i:02d}.parquet")
        out.append(comp)
    return kernel, out

def run_mice_batched(m: int = 50, iterations: int = 5, seed: int = 20260219,
                     include_outcome: bool = True, outdir: Path | None = None,
                     batch: int = 10, num_threads: int = 4,
                     keep_first_kernel: Path | None = None):
    """MICE in sequential batches, to fit an 8 GB / 8-core machine.

    A single kernel with num_datasets=50 holds every dataset's LightGBM models and
    working copies resident at once. On this machine that drove the load average to
    ~23 on 8 cores while disk throughput collapsed -- the signature of swapping --
    and the run had to be abandoned after two hours with nothing written, because
    `mice()` writes only after all datasets finish.

    Running 5 kernels of 10 datasets instead cuts the peak footprint about fivefold,
    writes a tenth of the output every batch so progress is visible and partial
    results survive, and caps LightGBM's thread count so the datasets stop competing
    for the same cores. Each batch gets its own seed so the datasets remain distinct.
    """
    import gc
    import time as _time
    import miceforest as mf

    outdir = Path(outdir) if outdir is not None else IMPUTED_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    base = build_missing_frame()
    assert base["GOSE"].notna().all(), "GOSE must be complete before imputation"
    held = None
    if not include_outcome:
        held = base["GOSE"].copy()
        base = base.drop(columns=["GOSE"])
    for col in NOMINAL:
        if col in base.columns:
            base[col] = base[col].astype("category")
    ids = base.index

    n_batches = (m + batch - 1) // batch
    done, t0 = 0, _time.time()
    for b in range(n_batches):
        k = min(batch, m - done)
        # only the first kernel retains per-iteration history, for the convergence plot
        keep = keep_first_kernel is not None and b == 0
        kernel = mf.ImputationKernel(base.reset_index(drop=True), num_datasets=k,
                                     random_state=seed + b,
                                     save_all_iterations_data=keep)
        kernel.mice(iterations, num_threads=num_threads, min_data_in_leaf=20)
        for i in range(k):
            comp = kernel.complete_data(dataset=i)
            comp.index = ids
            if held is not None:
                comp["GOSE"] = held          # outcome re-attached, never imputed
            assert comp.notna().all().all(), f"dataset {done + i} still has gaps"
            comp.to_parquet(outdir / f"mice_{done + i:02d}.parquet")
        if keep:
            import pickle
            with open(keep_first_kernel, "wb") as f:
                pickle.dump(kernel, f)
        done += k
        del kernel
        gc.collect()
        print(f"batch {b + 1}/{n_batches}: {done}/{m} datasets written "
              f"({_time.time() - t0:.0f}s elapsed)", flush=True)
    return done

_IMPUTATION_CACHE: dict[str, list[pd.DataFrame]] = {}


def check_imputations(d: Path) -> tuple[str | None, list[pd.DataFrame] | None]:
    """Read a directory of imputed datasets, or say what is wrong with it.

    Every later stage treats the imputed datasets as interchangeable copies of one
    cohort, so they must agree on which patients are present, in what order, and with
    what columns -- not merely on their shape. Datasets are read once and kept, since
    each call to `prepare` would otherwise re-read all of them.

    The cache assumes the files do not change while a run is in progress: a directory
    that has already validated is not re-read or re-checked. `build_imputations`
    clears the cache after writing, so regenerating within one process is safe; if
    the files are replaced by any other means mid-run, start a fresh process.
    """
    key = str(d.resolve())
    if key in _IMPUTATION_CACHE:
        return None, _IMPUTATION_CACHE[key]
    if not d.exists():
        return f"no imputed datasets found in {d}", None
    found = sorted(d.glob("mice_*.parquet"))
    expected = [d / f"mice_{i:02d}.parquet" for i in range(N_IMPUTATIONS)]
    absent = [p.name for p in expected if not p.exists()]
    if absent or len(found) != N_IMPUTATIONS:
        return (f"{d} holds {len(found)} imputed datasets, expected {N_IMPUTATIONS}"
                + (f"; absent: {', '.join(absent)}" if absent else ""), None)
    try:
        frames = [pd.read_parquet(p) for p in expected]
    except Exception as exc:                      # truncated or corrupt parquet
        return f"could not read every imputed dataset in {d}: {exc}", None
    ref = frames[0]
    for i, df in enumerate(frames[1:], start=1):
        if not df.index.equals(ref.index):
            return (f"imputation {i} in {d} covers different patients, or the same "
                    f"patients in a different order, than imputation 0"), None
        if not df.columns.equals(ref.columns):
            return (f"imputation {i} in {d} has different columns than imputation 0: "
                    f"{sorted(set(df.columns) ^ set(ref.columns)) or 'same set, different order'}"), None
        if df.isna().any().any():
            return f"imputation {i} in {d} still contains missing values", None
    if ref.isna().any().any():
        return f"imputation 0 in {d} still contains missing values", None
    _IMPUTATION_CACHE[key] = frames
    return None, frames


def load_imputations(outdir: Path | None = None) -> list[pd.DataFrame]:
    d = Path(outdir) if outdir is not None else IMPUTED_DIR
    problem, frames = check_imputations(d)
    if problem:
        sys.exit(f"{problem}\n"
                 f"  the imputed datasets are produced by the 'build_imputations' stage."
                 f" Delete {d} and run:\n"
                 f"    PYTHONHASHSEED=0 python {Path(sys.argv[0]).name} build_imputations")
    return frames

def compare_marginals(imputations: list[pd.DataFrame]) -> pd.DataFrame:
    """Median-imputation vs MICE marginal distributions, for variables with missingness.

    Quantifies how far single median imputation displaces each variable's
    marginal distribution.
    """
    med = load_analysis()
    miss = build_missing_frame()
    rows = []
    for col in miss.columns:
        n_miss = int(miss[col].isna().sum())
        if n_miss == 0 or col not in med.columns:
            continue
        obs = miss[col].dropna()
        levels = sorted(set(np.unique(obs)) | set(np.unique(med[col].dropna())))
        if len(levels) > 12:
            continue
        for lev in levels:
            p_obs = float((obs == lev).mean())
            p_med = float((med[col] == lev).mean())
            p_mice = float(np.mean([(d[col].astype(float) == lev).mean() for d in imputations]))
            rows.append({
                "Variable": col, "n_missing": n_miss,
                "pct_missing": round(100 * n_miss / len(miss), 2),
                "Code": lev, "Meaning": CODEBOOK.get(col, {}).get(lev, ""),
                "pct_observed_only": round(100 * p_obs, 2),
                "pct_median_imputed": round(100 * p_med, 2),
                "pct_MICE": round(100 * p_mice, 2),
                "median_minus_observed": round(100 * (p_med - p_obs), 2),
                "MICE_minus_observed": round(100 * (p_mice - p_obs), 2),
            })
    return pd.DataFrame(rows)


# ============================================================================
# Bayesian network fitting and evaluation
# ============================================================================
joblib.parallel_config(prefer="threads").__enter__()

TARGET = "GOSE"

CLASSES = [1, 2, 4, 5, 6, 7, 8]

GCS_BAND_BINS = [2, 8, 12, 15]

GCS_BAND_LABELS = [3, 2, 1]

def _to_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Re-derive the discrete Age and InjuryISS bands from continuous values.

    Verified lossless: round-trip accuracy is 1.0000 for both against the original
    bands on observed rows.
    """
    out = df.copy()
    if "Age_years" in out.columns:
        out["Age"] = pd.cut(out["Age_years"], [-np.inf,15,30,45,60,75,np.inf],
                            labels=[1,2,3,4,5,6]).astype(int)
        out = out.drop(columns=["Age_years"])
    if "ISS_total" in out.columns:
        out["InjuryISS"] = pd.cut(out["ISS_total"], [-np.inf,8,16,25,np.inf],
                                  labels=[1,2,3,4]).astype(int)
        out = out.drop(columns=["ISS_total"])
    return out

FINAL_IMPUTATIONS = Path("out") / "imputed_m30_nogose"


def prepare(arm: str, gcs_spec: str = "band3", imputation: int = 0) -> pd.DataFrame:
    """Return a complete discrete frame for one comparison arm.

    arm  : 'arm1_median_full' | 'arm2_median_leakfree' | 'arm3_mice_leakfree'
    gcs_spec : 'band3' (primary) | 'total13' (as originally specified)
    imputation : which MICE dataset to use, for arm3 only
    """
    if arm == "arm3_mice_leakfree":
        df = _to_bands(load_imputations(FINAL_IMPUTATIONS)[imputation])
    else:
        df = load_analysis().copy()
        if arm == "arm2_median_leakfree":
            df = df.drop(columns=[c for c in LEAKY if c in df.columns])
        elif arm == "arm1_median_full":
            df = df.drop(columns=[SECONDARY_OUTCOME]) if False else df
        else:
            raise ValueError(arm)

    if gcs_spec == "band3":
        df["GCS"] = pd.cut(df["GCS"], GCS_BAND_BINS, labels=GCS_BAND_LABELS).astype(int)
    elif gcs_spec != "total13":
        raise ValueError(gcs_spec)

    return df.astype(int)


# ============================================================================
# Causal forest design and estimation
# ============================================================================
TREATMENTS = ["Surgery", "ICPMonitor", "Stratum"]

OUTCOMES = {"GOSE": "GOSE (ordinal, 1-8)", "HospitalStay": "Length of stay (days)"}

MODELS = {"CF-1": ("Surgery", "GOSE"), "CF-2": ("Stratum", "GOSE"),
          "CF-3": ("ICPMonitor", "GOSE"), "CF-4": ("Surgery", "HospitalStay"),
          "CF-5": ("Stratum", "HospitalStay"), "CF-6": ("ICPMonitor", "HospitalStay")}

MEDIATORS = {"Stratum": ["ICPMonitor", "Surgery"], "Surgery": [], "ICPMonitor": []}

FOREST = dict(n_estimators=1000, min_samples_leaf=100, max_depth=100,
              random_state=42, cv=5, criterion="het", discrete_treatment=True)

def cohort(treatment: str, imputation: int = 0) -> pd.Index:
    """Row index for one treatment's analysis cohort."""
    df = prepare("arm3_mice_leakfree", gcs_spec="band3", imputation=imputation)
    if treatment == "Surgery":
        return df.index[~unknown_treatment_mask("Surgery").reindex(df.index).fillna(False)]
    if treatment == "ICPMonitor":
        return df.index[df["Stratum"] == 1]
    return df.index

def covariates(treatment: str, columns, adjust: str = "main") -> list[str]:
    drop = {TARGET, treatment, *MEDIATORS[treatment]}
    if adjust == "strict":
        drop |= set(TREATMENTS)
    if treatment == "ICPMonitor":
        drop |= {"Stratum"}          # constant within the ICU-restricted cohort
    return [c for c in columns if c not in drop]


def require(path: str, producer: str) -> None:
    """Stop with an instruction if a prerequisite from an earlier stage is absent."""
    if not Path(path).exists():
        sys.exit(f"missing prerequisite: {path}\n"
                 f"  it is produced by the '{producer}' stage. Run:\n"
                 f"    PYTHONHASHSEED=0 python {Path(sys.argv[0]).name} {producer}")

def design(model: str, imputation: int = 0, adjust: str = "main"):
    """Return (X, T, Y) for one model and one imputed dataset."""
    treatment, outcome = MODELS[model]
    df = prepare("arm3_mice_leakfree", gcs_spec="band3", imputation=imputation)
    idx = cohort(treatment, imputation)
    if outcome == "HospitalStay":
        los = load_continuous()["LOS_days"]
        idx = idx.intersection(los.dropna().index)
        y = los.loc[idx].astype(float)
    else:
        y = df.loc[idx, TARGET].astype(float)
    X = df.loc[idx, covariates(treatment, df.columns, adjust)].astype(float)
    T = df.loc[idx, treatment].astype(int)
    return X, T, y

def fit(X, T, Y, n_jobs: int = 4, **kw):
    from econml.dml import CausalForestDML
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    cfg = {**FOREST, **kw}
    # econml requires a CLASSIFIER for model_t when discrete_treatment=True and a
    # default discrete_treatment=False, i.e. it regressed a 0/1 variable; keeping
    # both paths lets that specification be reproduced for the comparison table.
    seed = cfg["random_state"]
    mt = (RandomForestClassifier(random_state=seed) if cfg["discrete_treatment"]
          else RandomForestRegressor(random_state=seed))
    est = CausalForestDML(model_y=RandomForestRegressor(random_state=seed),
                          model_t=mt, n_jobs=n_jobs, **cfg)
    est.fit(Y=np.asarray(Y), T=np.asarray(T), X=np.asarray(X), cache_values=True)
    return est

def ate(est, X) -> dict:
    Xa = np.asarray(X)
    inf = est.ate_inference(Xa)
    cate = est.effect(Xa)
    return {"ATE": float(inf.mean_point), "SE": float(inf.stderr_mean),
            "CI_low": float(inf.conf_int_mean()[0]), "CI_high": float(inf.conf_int_mean()[1]),
            "CATE_sd": float(np.std(cate)), "CATE_min": float(cate.min()),
            "CATE_max": float(cate.max()), "n": int(len(Xa))}

def rubin(est, var, m: int) -> dict:
    """Rubin's rules for m estimates with their sampling variances."""
    est, var = np.asarray(est, float), np.asarray(var, float)
    if est.size != var.size or est.size != m:
        raise ValueError(f"rubin: {est.size} estimates, {var.size} variances, m={m}")
    if not (np.isfinite(est).all() and np.isfinite(var).all()):
        raise ValueError("rubin: estimates and variances must all be finite")
    if (var < 0).any():
        raise ValueError("rubin: sampling variances must be non-negative")
    Q, W = np.mean(est), np.mean(var)
    Bv = np.var(est, ddof=1)
    T_ = W + (1 + 1 / m) * Bv
    r = (1 + 1 / m) * Bv / W if W > 0 else np.inf
    df = (m - 1) * (1 + 1 / r) ** 2 if r > 0 else np.inf
    tc = stats.t.ppf(0.975, df)
    return dict(ATE=Q, SE=np.sqrt(T_), CI_low=Q - tc * np.sqrt(T_),
                CI_high=Q + tc * np.sqrt(T_), df=df,
                FMI=((1 + 1 / m) * Bv) / T_ if T_ > 0 else np.nan)


def propensity(X, T, seed: int = 42) -> np.ndarray:
    """Cross-fitted P(T=1|X) for the positivity and overlap diagnostic."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=20, random_state=seed, n_jobs=4)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_predict(clf, np.asarray(X), np.asarray(T), cv=cv, method="predict_proba")[:, 1]


# ============================================================================
# Figure style
# ============================================================================
FONT_DIR = Path("fonts")

FAMILY = "Cambria"

SIZE = 12

def boxed(ax, lw: float | None = None) -> None:
    """Draw all four spines, the convention used for the sub-figure panels.

    The house style hides the top and right spines; the causal-forest panels are
    deliberately exempt, because each panel is read as a self-contained sub-figure
    within a grid and the box is what separates it from its neighbours.
    """
    base = lw if lw is not None else mpl.rcParams.get("axes.linewidth", 0.8)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(base)

def nice_ticks(ax, axis: str = "x", nbins: int = 6, lo=None, hi=None):
    """Put ticks at round intervals and END THE AXIS ON A TICK.

    Replaces the earlier end_caps() approach. That one added an unlabelled minor
    tick at each axis limit, which lands at an arbitrary distance from the final
    major tick -- typically a fraction of a step -- so the axis appeared to finish
    with two ticks crowded together. Snapping the limit to the outermost round tick
    instead means the terminal tick is itself the cap, and every interval along the
    axis is equal.

    Returns the tick array, so a shared scale can be computed once and applied to
    several panels.
    """
    import matplotlib.ticker as mticker
    cur = ax.get_xlim() if axis == "x" else ax.get_ylim()
    lo = cur[0] if lo is None else lo
    hi = cur[1] if hi is None else hi
    ticks = mticker.MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(mticker.FixedLocator(ticks))
    if axis == "x":
        ax.set_xlim(ticks[0], ticks[-1])
    else:
        ax.set_ylim(ticks[0], ticks[-1])
    return ticks

def end_caps(ax, axes_to_cap: str = "xy") -> None:
    """Terminate each axis with a tick at its end points.

    Drawn as unlabelled minor ticks placed exactly at the axis limits, so they use
    matplotlib's own tick rendering and inherit tick length, width and colour.
    Call AFTER the final set_xlim/set_ylim, since the cap positions are read from
    the limits at call time.
    """
    import matplotlib.ticker as mticker
    length = mpl.rcParams.get("xtick.major.size", 3.5)
    if "x" in axes_to_cap:
        lo, hi = ax.get_xlim()
        ax.xaxis.set_minor_locator(mticker.FixedLocator([lo, hi]))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    if "y" in axes_to_cap:
        lo, hi = ax.get_ylim()
        ax.yaxis.set_minor_locator(mticker.FixedLocator([lo, hi]))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(which="minor", length=length, width=ax.spines["bottom"].get_linewidth())

def register_fonts(font_dir: Path | str = FONT_DIR) -> list[str]:
    """Register every .ttf in the project font directory with matplotlib."""
    font_dir = Path(font_dir)
    added = []
    for f in sorted(font_dir.glob("*.ttf")):
        font_manager.fontManager.addfont(str(f))
        added.append(font_manager.FontProperties(fname=str(f)).get_name())
    return sorted(set(added))

def use_house_style(size: int = SIZE, family: str = "IBM Plex Sans") -> None:
    """Uniform serif type at a single point size, with publication-grade defaults.

    House font is IBM Plex Sans 12 (SIL Open Font License), shipped in fonts/ so the
    figures are reproducible without relying on system fonts. Falls back to Helvetica
    then Arial, both of which Elsevier/Lancet artwork guidance lists as approved.
    """
    register_fonts()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [family, "Helvetica", "Arial", "DejaVu Sans"]
                           + mpl.rcParams["font.sans-serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": family, "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
        # uniform 12 pt: no role-mapped ladder, per the author's house convention
        "font.size": size,
        "axes.titlesize": size, "axes.labelsize": size,
        "xtick.labelsize": size, "ytick.labelsize": size,
        "legend.fontsize": size, "figure.titlesize": size,
        # axis defaults
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlelocation": "left", "axes.titleweight": "normal",
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,   # embed as TrueType for PLOS
    })


# ============================================================================
# Analysis stages
# ============================================================================

def stage_build_imputations():
    """Generate the imputed datasets every later stage depends on.

    Writes N_IMPUTATIONS completed datasets to FINAL_IMPUTATIONS as parquet files.
    The outcome is held out of the imputation model and re-attached unchanged, so
    imputation cannot induce predictor-outcome association. Run this once; it is by
    far the longest stage.
    """
    problem, _ = check_imputations(FINAL_IMPUTATIONS)
    if problem is None:
        print(f"{N_IMPUTATIONS} complete imputed datasets already present in "
              f"{FINAL_IMPUTATIONS}; delete them to regenerate.")
        return
    if FINAL_IMPUTATIONS.exists() and any(FINAL_IMPUTATIONS.glob("mice_*.parquet")):
        # A half-written or inconsistent set is worse than none: later stages would
        # refuse to run and the cause would not be obvious here.
        sys.exit(f"{problem}\n  delete {FINAL_IMPUTATIONS} and rerun this stage to "
                 "regenerate the full set.")
    run_mice(m=N_IMPUTATIONS, iterations=20, include_outcome=False,
             outdir=FINAL_IMPUTATIONS)
    _IMPUTATION_CACHE.pop(str(FINAL_IMPUTATIONS.resolve()), None)
    problem, frames = check_imputations(FINAL_IMPUTATIONS)
    if problem:
        sys.exit(f"the datasets just written are not usable: {problem}")
    print(f"wrote {len(frames)} imputed datasets to {FINAL_IMPUTATIONS}")


def stage_pooled_effects():
    """All six models across the imputations; pooled average treatment effects."""
    os.makedirs("out/cf_cate", exist_ok=True)
    M = N_IMPUTATIONS
    t0, rows = time.time(), []
    for model in MODELS:
        treat, out = MODELS[model]
        cates, index = [], None
        for i in range(M):
            X, T, Y = design(model, imputation=i)
            # Conditional effects are stacked column-wise below, one column per
            # imputed dataset, so row k must be the same patient in every column.
            # Cohort membership is invariant here -- Surgery models exclude patients
            # whose status was unobserved, read from the pre-imputation source, and
            # ICPMonitor models select on a variable with no missing values -- but the
            # stacking depends on it, so it is checked rather than assumed.
            if index is None:
                index = X.index
            elif not X.index.equals(index):
                sys.exit(f"{model}: cohort changed between imputations 0 and {i} "
                         f"({len(index)} vs {len(X.index)} patients, "
                         f"{len(index.difference(X.index))} not in both). Conditional "
                         "effects cannot be stacked across imputations.")
            est = fit(X, T, Y)
            a = ate(est, X)
            cates.append(est.effect(np.asarray(X)))
            rows.append({"model": model, "treatment": treat, "outcome": out,
                         "imputation": i, **a})
            print(f"[{model}] imp {i+1}/{M} ATE {a['ATE']:+.4f} "
                  f"({a['CI_low']:+.4f},{a['CI_high']:+.4f}) n={a['n']} "
                  f"t={time.time()-t0:.0f}s", flush=True)
        np.savez_compressed(f"out/cf_cate/{model}.npz", cate=np.column_stack(cates),
                            index=np.asarray(index))
    per = pd.DataFrame(rows)
    per.to_csv("out/cf_per_imputation.csv", index=False)
    pool = []
    for model, g in per.groupby("model"):
        p = rubin(g.ATE.values, g.SE.values ** 2, len(g))
        pool.append({"model": model, "treatment": g.treatment.iloc[0],
                     "outcome": g.outcome.iloc[0], "n": int(g.n.iloc[0]), **p,
                     "CATE_sd_mean": g.CATE_sd.mean()})
    pd.DataFrame(pool).to_csv("out/cf_ate_pooled.csv", index=False)
    print(f"\nCF MAIN DONE [{time.time()-t0:.0f}s]")
    print(pd.DataFrame(pool).round(4).to_string(index=False))

def stage_specification_and_propensity():
    """Specification comparison and cross-fitted propensity scores."""
    SPECS = {
        "original (mse, regressor T, continuous)": dict(criterion="mse", discrete_treatment=False),
        "+ discrete treatment (classifier T)":     dict(criterion="mse", discrete_treatment=True),
        "+ het criterion (final)":                 dict(criterion="het", discrete_treatment=True),
    }
    t0, rows, props = time.time(), [], []
    for model in MODELS:
        X, T, Y = design(model, imputation=0)
        for name, kw in SPECS.items():
            est = fit(X, T, Y, **kw)
            a = ate(est, X)
            rows.append({"model": model, "treatment": MODELS[model][0],
                         "outcome": MODELS[model][1], "specification": name, **a})
            print(f"[{model}] {name:42s} ATE {a['ATE']:+.4f} "
                  f"({a['CI_low']:+.4f},{a['CI_high']:+.4f}) t={time.time()-t0:.0f}s", flush=True)
        p = propensity(X, T)
        props.append(pd.DataFrame({"model": model, "treatment": MODELS[model][0],
                                   "T": np.asarray(T), "propensity": p}))
    pd.DataFrame(rows).to_csv("out/cf_specification_comparison.csv", index=False)
    pd.concat(props).to_csv("out/cf_propensity.csv", index=False)
    p = pd.concat(props)
    summ = []
    for m, g in p.groupby("model"):
        tr, ct = g[g["T"] == 1].propensity, g[g["T"] == 0].propensity
        lo, hi = max(tr.min(), ct.min()), min(tr.max(), ct.max())
        summ.append({"model": m, "n": len(g),
                     "propensity_min": g.propensity.min(), "propensity_max": g.propensity.max(),
                     "common_support_low": lo, "common_support_high": hi,
                     "outside_support_n": int(((g.propensity < lo) | (g.propensity > hi)).sum()),
                     "below_0.05_n": int((g.propensity < 0.05).sum()),
                     "above_0.95_n": int((g.propensity > 0.95).sum())})
    pd.DataFrame(summ).to_csv("out/cf_overlap_summary.csv", index=False)
    print("\n" + pd.DataFrame(summ).round(4).to_string(index=False))
    print(f"DIAG DONE [{time.time()-t0:.0f}s]")

def resumable(path: str, rows=None, imputations=None, features=None,
              column_values=None) -> bool:
    """True if a per-model checkpoint holds the complete result its stage would write.

    Stages resume from these files rather than refitting, so a file truncated by an
    interrupted run would otherwise be taken as finished and silently shorten the
    analysis. Presence is not enough, and neither is balanced coverage: a file
    carrying every imputation but only one feature of each is equally incomplete.
    Callers pass the exact row count, and where the rows are keyed by feature, the
    exact feature set that must appear within every imputation.
    """
    if not os.path.exists(path):
        return False
    try:
        d = pd.read_csv(path)
    except Exception:
        return False
    if d.empty or not np.isfinite(d.select_dtypes("number").to_numpy()).all():
        return False
    # Duplicate rows are not tested directly: the propensity checkpoints legitimately
    # repeat, since two patients can share a forest propensity and treatment status.
    # Where rows are keyed, the row count together with the imputation and feature
    # sets already excludes a duplicate -- it would displace a row that must be there.
    if rows is not None and len(d) != rows:
        return False
    if imputations is not None:
        if "imputation" not in d.columns or set(d.imputation) != set(imputations):
            return False
        if d.groupby("imputation").size().nunique() != 1:
            return False
    if features is not None:
        if "feature" not in d.columns:
            return False
        want = set(features)
        grouped = d.groupby("imputation").feature if "imputation" in d.columns else None
        if grouped is None:
            return set(d.feature) == want
        if any(set(g) != want for _, g in grouped):
            return False
    if column_values is not None:
        for col, want in column_values.items():
            if col not in d.columns or set(d[col]) != set(want):
                return False
    return True


def stage_overlap():
    """Propensity overlap pooled across imputations."""
    os.makedirs("out/cf_partial", exist_ok=True)
    t0 = time.time()
    for model in list(MODELS):
        ck = f"out/cf_partial/{model}_prop.csv"
        n_patients = len(design(model, imputation=0)[0])
        if resumable(ck, rows=N_IMPUTATIONS * n_patients,
                     imputations=range(N_IMPUTATIONS)):
            print(f"{model}: already done", flush=True); continue
        parts = []
        for i in range(N_IMPUTATIONS):
            X, T, Y = design(model, imputation=i)
            parts.append(pd.DataFrame({"model": model, "imputation": i,
                                       "T": np.asarray(T), "propensity": propensity(X, T)}))
        pd.concat(parts).to_csv(ck, index=False)
        print(f"{model}: {N_IMPUTATIONS} imputations, {len(parts[0])} patients each, "
              f"t={time.time()-t0:.0f}s", flush=True)

def stage_strict_adjustment():
    """Sensitivity analysis dropping all non-focal treatments from the covariates."""
    require("out/cf_ate_pooled.csv", "pooled_effects")
    M_STRICT = int(os.environ.get("M_STRICT", 30))
    os.makedirs("out/cf_partial", exist_ok=True)
    t0 = time.time()
    for model in list(MODELS):
        ck = f"out/cf_partial/{model}_strict.csv"
        if resumable(ck, rows=M_STRICT, imputations=range(M_STRICT)):
            print(f"{model}: already done", flush=True); continue
        rows = []
        for i in range(M_STRICT):
            X, T, Y = design(model, imputation=i, adjust="strict")
            est = fit(X, T, Y)
            rows.append({"model": model, "imputation": i, "n_covariates": X.shape[1],
                         **ate(est, X)})
        pd.DataFrame(rows).to_csv(ck, index=False)
        print(f"{model}: {M_STRICT} fits, p={rows[0]['n_covariates']}, "
              f"mean ATE {pd.DataFrame(rows).ATE.mean():+.4f}, t={time.time()-t0:.0f}s", flush=True)

    # Consolidate: pool the strict fits and set them beside the main estimates.
    per = pd.concat([pd.read_csv(f"out/cf_partial/{m}_strict.csv") for m in MODELS])
    per.to_csv("out/cf_strict_per_imputation.csv", index=False)
    main = pd.read_csv("out/cf_ate_pooled.csv").set_index("model")
    out = []
    for model, g in per.groupby("model", sort=False):
        p = rubin(g.ATE.values, g.SE.values ** 2, len(g))
        treat, outcome = MODELS[model]
        out.append({
            "model": model, "treatment": treat, "outcome": outcome,
            "imputations": g.imputation.nunique(),
            "p_main": design(model, imputation=0)[0].shape[1],
            "p_strict": int(g.n_covariates.iloc[0]),
            "ATE_main": round(main.loc[model, "ATE"], 4),
            "CI_main": f"({main.loc[model, 'CI_low']:.3f}, {main.loc[model, 'CI_high']:.3f})",
            "ATE_strict": round(p["ATE"], 4),
            "CI_strict": f"({p['CI_low']:.3f}, {p['CI_high']:.3f})",
            "diff": round(p["ATE"] - main.loc[model, "ATE"], 4),
            # The strict design removes the non-focal treatments from the covariates.
            # Where the focal treatment is the only one present, nothing is removed and
            # the two designs coincide.
            "identical_by_construction":
                "yes" if int(g.n_covariates.iloc[0]) == design(model, imputation=0)[0].shape[1]
                else "no"})
    s = pd.DataFrame(out).sort_values("model")
    s.to_csv("out/cf_strict_comparison.csv", index=False)
    print(s[["model", "p_main", "p_strict", "ATE_main", "ATE_strict", "diff"]].to_string(index=False))

def stage_leaf_size_sensitivity():
    """Robustness of the estimates to the minimum leaf size."""
    t0 = time.time()
    for model in list(MODELS):
        ck = f"out/cf_partial/{model}_hp.csv"
        if resumable(ck, rows=3, column_values={"min_samples_leaf": (50, 100, 200)}):
            print(f"{model}: already done", flush=True); continue
        X, T, Y = design(model, imputation=0)
        r = []
        for leaf in (50, 100, 200):
            est = fit(X, T, Y, min_samples_leaf=leaf)
            r.append({"model": model, "min_samples_leaf": leaf, **ate(est, X)})
        pd.DataFrame(r).to_csv(ck, index=False)
        print(f"{model}: " + " | ".join(f"leaf{d['min_samples_leaf']} {d['ATE']:+.4f}" for d in r)
              + f"  t={time.time()-t0:.0f}s", flush=True)

    hp = pd.concat([pd.read_csv(f"out/cf_partial/{m}_hp.csv") for m in MODELS])
    hp.to_csv("out/cf_hyperparameter_sensitivity.csv", index=False)
    print(hp.pivot_table(index="model", columns="min_samples_leaf",
                         values="ATE").round(4).to_string())

def stage_treatment_cooccurrence():
    """Co-occurrence of the three treatments."""
    frames = [prepare("arm3_mice_leakfree", gcs_spec="band3", imputation=i)
              for i in range(N_IMPUTATIONS)]
    unk = unknown_treatment_mask("Surgery").reindex(frames[0].index).fillna(False)
    rows = []
    for st in (1, 0):
        for ic in (1, 0):
            for sg in (1, 0):
                n = [int(((d.Surgery == sg) & (d.ICPMonitor == ic) & (d.Stratum == st)).sum())
                     for d in frames]
                obs = int(((frames[0].Surgery == sg) & (frames[0].ICPMonitor == ic)
                           & (frames[0].Stratum == st) & ~unk).sum())
                rows.append({"Stratum": "ICU" if st else "Non-ICU",
                             "ICP monitoring": "Yes" if ic else "No",
                             "Surgery": "Yes" if sg else "No",
                             "n_mean": round(float(np.mean(n)), 1),
                             "n_min": min(n), "n_max": max(n),
                             "pct_mean": round(100 * float(np.mean(n)) / len(frames[0]), 1),
                             "n_surgery_observed": obs})
    t = pd.DataFrame(rows)
    t.to_csv("out/TableS_treatment_cooccurrence.csv", index=False)
    print(t.to_string(index=False))
    pair = []
    for a, b in [("Surgery", "ICPMonitor"), ("Surgery", "Stratum"), ("ICPMonitor", "Stratum")]:
        both = [100 * float(((d[a] == 1) & (d[b] == 1)).mean()) for d in frames]
        phi = [float(np.corrcoef(d[a], d[b])[0, 1]) for d in frames]
        pair.append({"pair": f"{a} + {b}", "both_pct_mean": round(np.mean(both), 1),
                     "both_pct_range": f"{min(both):.1f}-{max(both):.1f}",
                     "phi_mean": round(np.mean(phi), 3)})
    pd.DataFrame(pair).to_csv("out/TableS_treatment_overlap_pairwise.csv", index=False)
    print("\n" + pd.DataFrame(pair).to_string(index=False))

def stage_importance_and_shap():
    """Forest covariate importance and mean |SHAP| per model."""
    os.makedirs("out/cf_shap", exist_ok=True)
    os.makedirs("out/cf_partial", exist_ok=True)
    NSHAP, M = 1000, N_IMPUTATIONS
    M_SHAP = 10
    t0, imp_rows, shap_rows = time.time(), [], []
    for model in MODELS:
        treat, out = MODELS[model]
        # Per-model checkpoint: a kernel reset mid-run must not cost the whole pass.
        ip, sp = f"out/cf_partial/{model}_imp.csv", f"out/cf_partial/{model}_shap.csv"
        if os.path.exists(ip) and os.path.exists(sp):
            imp_rows += pd.read_csv(ip).to_dict("records")
            shap_rows += pd.read_csv(sp).to_dict("records")
            print(f"[{model}] restored from checkpoint", flush=True)
            continue
        mi, ms = [], []
        for i in range(M):
            X, T, Y = design(model, imputation=i)
            est = fit(X, T, Y)
            imp = np.ravel(est.feature_importances_)
            mi += [{"model": model, "imputation": i, "feature": f, "importance": v}
                   for f, v in zip(X.columns, imp)]
            if i >= M_SHAP:
                print(f"[{model}] imp {i+1}/{M} (importance only) t={time.time()-t0:.0f}s", flush=True)
                continue
            Xs = X.sample(min(NSHAP, len(X)), random_state=42)
            sv = est.shap_values(Xs.to_numpy())
            k = list(sv.keys())[0]; k2 = list(sv[k].keys())[0]
            vals = sv[k][k2].values
            ms += [{"model": model, "imputation": i, "feature": f, "mean_abs_shap": v}
                   for f, v in zip(X.columns, np.abs(vals).mean(axis=0))]
            if i == 0:            # keep one full matrix per model for the beeswarm panels
                np.savez_compressed(f"out/cf_shap/{model}.npz", shap=vals,
                                    X=Xs.to_numpy(), features=np.array(list(X.columns)))
            print(f"[{model}] imp {i+1}/{M} t={time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(mi).to_csv(ip, index=False)
        pd.DataFrame(ms).to_csv(sp, index=False)
        imp_rows += mi; shap_rows += ms
    imp = pd.DataFrame(imp_rows)
    sh = pd.DataFrame(shap_rows)
    imp.to_csv("out/cf_feature_importance_raw.csv", index=False)
    sh.to_csv("out/cf_shap_raw.csv", index=False)
    for df, col, name in ((imp, "importance", "cf_feature_importance.csv"),
                          (sh, "mean_abs_shap", "cf_shap_importance.csv")):
        g = (df.groupby(["model", "feature"])[col].agg(["mean", "std"])
               .reset_index().rename(columns={"mean": f"{col}_mean", "std": f"{col}_sd"}))
        g["n_imputations"] = M if col == "importance" else M_SHAP
        g["rank"] = g.groupby("model")[f"{col}_mean"].rank(ascending=False).astype(int)
        g.sort_values(["model", "rank"]).to_csv(f"out/{name}", index=False)
    print(f"CF SHAP DONE [{time.time()-t0:.0f}s]")

def stage_shap_remaining_imputations():
    """Extends mean |SHAP| to the remaining imputed datasets."""
    require("out/cf_shap_raw.csv", "importance_and_shap")
    LO, HI, NSHAP = 10, N_IMPUTATIONS, 1000
    os.makedirs("out/cf_partial", exist_ok=True)
    t0 = time.time()
    for model in list(MODELS):
        ck = f"out/cf_partial/{model}_shap30.csv"
        feats = design(model, imputation=0)[0].columns
        if resumable(ck, rows=(HI - LO) * len(feats), imputations=range(LO, HI),
                     features=feats):
            print(f"{model}: already done", flush=True); continue
        rows = []
        for i in range(LO, HI):
            X, T, Y = design(model, imputation=i)
            est = fit(X, T, Y)
            Xs = X.sample(min(NSHAP, len(X)), random_state=42)
            sv = est.shap_values(Xs.to_numpy())
            k = list(sv.keys())[0]; k2 = list(sv[k].keys())[0]
            rows += [{"model": model, "imputation": i, "feature": f, "mean_abs_shap": v}
                     for f, v in zip(X.columns, np.abs(sv[k][k2].values).mean(axis=0))]
        pd.DataFrame(rows).to_csv(ck, index=False)
        print(f"{model}: imputations {LO}-{HI-1} done, t={time.time()-t0:.0f}s", flush=True)

    # Consolidate: the attributions computed here extend those written by
    # importance_and_shap, which covers the first LO imputations only. Recombine both
    # and rewrite the importance table over the full set.
    parts = [pd.read_csv("out/cf_shap_raw.csv")]
    parts += [pd.read_csv(f"out/cf_partial/{m}_shap30.csv") for m in MODELS]
    raw = pd.concat(parts).drop_duplicates(subset=["model", "imputation", "feature"])
    raw.to_csv("out/cf_shap_raw.csv", index=False)
    g = (raw.groupby(["model", "feature"]).mean_abs_shap.agg(["mean", "std"])
           .reset_index().rename(columns={"mean": "mean_abs_shap_mean",
                                          "std": "mean_abs_shap_sd"}))
    g["n_imputations"] = HI
    g["rank"] = g.groupby("model")["mean_abs_shap_mean"].rank(ascending=False).astype(int)
    g.sort_values(["model", "rank"]).to_csv("out/cf_shap_importance.csv", index=False)
    print(f"SHAP DONE over {raw.imputation.nunique()} imputations "
          f"[{time.time()-t0:.0f}s]", flush=True)
    print(g.sort_values(["model", "rank"]).groupby("model").head(1)
           .to_string(index=False))

def stage_effect_figures():
    """Distributions of conditional effects and covariate-importance panels."""
    require("out/cf_propensity.csv", "specification_and_propensity")
    require("out/cf_cate/CF-1.npz", "pooled_effects")
    use_house_style(12)
    NEG, POS = "#ffbd7d", "#7dcfff"
    NBIN = 80
    cate = {m: np.load(f"out/cf_cate/{m}.npz")["cate"].mean(axis=1) for m in MODELS}
    rows = [["CF-1", "CF-2", "CF-3"], ["CF-4", "CF-5", "CF-6"]]
    edges, lims, xticks = {}, {}, {}
    for row in rows:
        lo = min(0.0, min(cate[m].min() for m in row))
        hi = max(0.0, max(cate[m].max() for m in row))
        e = np.linspace(lo, hi, NBIN + 1)
        # The axis is snapped out to the nearest round tick on each side instead of
        # being padded by an arbitrary fraction, so it begins and ends ON a tick and
        # every interval is equal.
        t = mticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
        for m in row:
            edges[m], lims[m], xticks[m] = e, (t[0], t[-1]), t
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.6))
    for i, (ax, model) in enumerate(zip(axes.ravel(), MODELS)):
        n, bins, patches = ax.hist(cate[model], bins=edges[model], edgecolor="black",
                                   linewidth=0.4)
        for patch, left in zip(patches, bins[:-1]):
            patch.set_facecolor(NEG if left < 0 else POS)
        ax.axvline(0, color="#444", lw=1, ls="--")
        tr, out = MODELS[model]
        ax.set_title(f"{model}: {tr} \u2192 {out}", fontsize=12, loc="center")
        ax.set_xticks(xticks[model])
        ax.set_xlim(*lims[model])
        if i >= 3:                      # x-axis label on the bottom row only
            ax.set_xlabel("Conditional average treatment effect")
        ax.set_ylabel("Patients")
        nice_ticks(ax, "y", nbins=5, lo=0, hi=n.max())
        boxed(ax)
    fig.tight_layout(w_pad=3.2, h_pad=3.2)
    fig.savefig("out/fig_cf_cate.svg", bbox_inches="tight")
    p = pd.read_csv("out/cf_propensity.csv")
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.6))
    for i, (ax, model) in enumerate(zip(axes.ravel(), MODELS)):
        g = p[p.model == model]
        bins = np.linspace(0, 1, 41)
        h0, _, _ = ax.hist(g[g["T"] == 0].propensity, bins=bins, color=NEG, alpha=0.85,
                           label="Untreated")
        h1, _, _ = ax.hist(g[g["T"] == 1].propensity, bins=bins, color=POS, alpha=0.85,
                           label="Treated")
        tr, out = MODELS[model]
        ax.set_title(f"{model}: {tr} \u2192 {out}", fontsize=12, loc="center")
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_xlim(0, 1)
        if i >= 3:
            ax.set_xlabel("Estimated propensity score")
        ax.set_ylabel("Patient\u2013imputation observations")
        ax.legend(frameon=False)
        nice_ticks(ax, "y", nbins=5, lo=0, hi=max(h0.max(), h1.max()))
        boxed(ax)
    fig.tight_layout(w_pad=3.2, h_pad=3.2)
    fig.savefig("out/figS_cf_overlap.svg", bbox_inches="tight")
    print("figures written")

def stage_shap_figures():
    """SHAP beeswarm and aggregated attribution panels."""
    require("out/cf_shap_importance.csv", "importance_and_shap")
    require("out/cf_shap/CF-1.npz", "importance_and_shap")
    use_house_style(12)
    BAR, ERR = "#7dcfff", "#444444"
    TOP = 10
    def barh_panel(ax, g, value, sd, xlabel, scale=1.0, show_xlabel=True):
        y = np.arange(len(g))
        lo = np.minimum(scale * g[sd], scale * g[value])   # importance/|SHAP| is non-negative
        ax.barh(y, scale * g[value], xerr=np.vstack([lo, scale * g[sd]]), color=BAR,
                edgecolor="black", linewidth=0.4,
                error_kw=dict(ecolor=ERR, lw=0.8, capsize=2))
        ax.set_yticks(y)
        ax.set_yticklabels(g.feature, fontsize=12)
        ax.set_xlim(left=0)
        nice_ticks(ax, "x", nbins=5, lo=0, hi=ax.get_xlim()[1])
        if show_xlabel:
            ax.set_xlabel(xlabel)
    fi = pd.read_csv("out/cf_feature_importance.csv")
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.6))
    for i, (ax, model) in enumerate(zip(axes.ravel(), MODELS)):
        g = fi[fi.model == model].nsmallest(TOP, "rank").iloc[::-1]
        barh_panel(ax, g, "importance_mean", "importance_sd", "Relative importance (%)",
                   scale=100.0, show_xlabel=i >= 3)
        tr, out = MODELS[model]
        ax.set_title(f"{model}: {tr} \u2192 {out}", fontsize=12, loc="center")
        boxed(ax)
    fig.tight_layout(w_pad=3.2, h_pad=3.2)
    fig.savefig("out/fig_cf_importance.svg", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.4))
    for i, (ax, model) in enumerate(zip(axes.ravel(), MODELS)):
        d = np.load(f"out/cf_shap/{model}.npz", allow_pickle=True)
        feats = [str(f) for f in d["features"]]
        plt.sca(ax)
        shap.summary_plot(d["shap"], d["X"], feature_names=feats, max_display=TOP,
                          show=False, plot_size=None, color_bar=False, sort=True)
        tr, out = MODELS[model]
        ax.set_title(f"{model}: {tr} \u2192 {out}", fontsize=12, loc="center")
        # shap's own axis label, as in the originally submitted Figure 6
        ax.set_xlabel("SHAP value (impact on model output)" if i >= 3 else "")
        nice_ticks(ax, "x", nbins=5)
        boxed(ax)
    fig.tight_layout(w_pad=3.2, h_pad=3.2, rect=(0, 0, 0.93, 1))
    cax = fig.add_axes([0.945, 0.055, 0.013, 0.89])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=shap.plots.colors.red_blue),
                      cax=cax, ticks=[0, 1])
    cb.ax.set_yticklabels(["Low", "High"])
    cb.set_label("Covariate value", labelpad=-14)
    cb.outline.set_visible(False)
    fig.savefig("out/fig_cf_shap_beeswarm.svg", bbox_inches="tight")
    plt.close(fig)
    sh = pd.read_csv("out/cf_shap_importance.csv")
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.6))
    for i, (ax, model) in enumerate(zip(axes.ravel(), MODELS)):
        g = sh[sh.model == model].nsmallest(TOP, "rank").iloc[::-1]
        barh_panel(ax, g, "mean_abs_shap_mean", "mean_abs_shap_sd",
                   "Mean |SHAP| (impact on model output)", show_xlabel=i >= 3)
        tr, out = MODELS[model]
        ax.set_title(f"{model}: {tr} \u2192 {out}", fontsize=12, loc="center")
        boxed(ax)
    fig.tight_layout(w_pad=3.2, h_pad=3.2)
    fig.savefig("out/figS_cf_shap_top10.svg", bbox_inches="tight")
    plt.close(fig)
    for model in MODELS:
        d = np.load(f"out/cf_shap/{model}.npz", allow_pickle=True)
        feats = [str(f) for f in d["features"]]
        fig = plt.figure(figsize=(9.0, 7.5))
        shap.summary_plot(d["shap"], d["X"], feature_names=feats, max_display=TOP,
                          show=False, plot_size=None)
        fig.tight_layout()
        fig.savefig(f"out/figS_cf_beeswarm_{model}.svg", bbox_inches="tight")
        plt.close(fig)
    print("figures written")



# ============================================================================
# Command line
# ============================================================================
STAGES = {
    "build_imputations": stage_build_imputations,
    "pooled_effects": stage_pooled_effects,
    "specification_and_propensity": stage_specification_and_propensity,
    "overlap": stage_overlap,
    "strict_adjustment": stage_strict_adjustment,
    "leaf_size_sensitivity": stage_leaf_size_sensitivity,
    "treatment_cooccurrence": stage_treatment_cooccurrence,
    "importance_and_shap": stage_importance_and_shap,
    "shap_remaining_imputations": stage_shap_remaining_imputations,
    "effect_figures": stage_effect_figures,
    "shap_figures": stage_shap_figures,
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(STAGES)
    unknown = [w for w in wanted if w not in STAGES]
    if unknown:
        sys.exit(f"unknown stage(s): {unknown}\navailable: {list(STAGES)}")
    for w in wanted:
        print(f"\n=== {w} ===", flush=True)
        STAGES[w]()


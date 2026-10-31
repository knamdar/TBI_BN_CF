"""Bayesian network analysis of six-month functional outcome after traumatic brain injury.

Self-contained: data access, multiple imputation, model fitting, evaluation and figures.
Run all stages with `PYTHONHASHSEED=0 python bayesian_networks.py`, or a single stage by
name, e.g. `... bayesian_networks.py fit_and_pool`. Outputs are written to out/.

Data are not distributed with this code; see README.md.
"""
from __future__ import annotations
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
import pygraphviz as pgv
import statsmodels.api as sm
import sys
import time
from matplotlib import font_manager
from matplotlib.lines import Line2D
from pathlib import Path
from pgmpy.estimators import AIC, BIC
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("Set PYTHONHASHSEED=0 before starting Python. Hill-climbing structure "
             "search breaks score ties using string hashes, so without a fixed hash "
             "seed the learned structure differs between runs on identical data.\n"
             "    PYTHONHASHSEED=0 python bayesian_networks.py")



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

TREATMENTS = ["Surgery", "ICPMonitor", "Stratum"]

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


def require(path: str, producer: str) -> None:
    """Stop with an instruction if a prerequisite from an earlier stage is absent."""
    if not Path(path).exists():
        sys.exit(f"missing prerequisite: {path}\n"
                 f"  it is produced by the '{producer}' stage. Run:\n"
                 f"    PYTHONHASHSEED=0 python {Path(sys.argv[0]).name} {producer}")

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

S2_BLOCKED = [
    ("Stratum","EduLevel"), ("MassLesion","EduLevel"), ("Surgery","EduLevel"),
    ("InjCause","Age"), ("TSAH","Age"), ("SkullFracture","Age"),
    ("SubduralHematoma","Age"), ("MassLesion","Age"), ("Race","Age"),
    ("TSAH","InjCause"), ("Hemoglobin","Sex"), ("Age","MassLesion"),
    ("Age","SubduralHematoma"), ("Age","EpiduralHematoma"),
]

S2_FIXED = [
    ("Age","MaritalStatus"), ("Age","EduLevel"), ("Age","FUWorkStatus"),
    ("Age","PreInjASAPS"), ("TSAH","Surgery"), ("InjuryISS","Surgery"),
    ("GCS","Surgery"), ("InjCause","InjuryISS"), ("InjuryISS","GCS"),
    ("Surgery","HospitalStay"), ("Stratum","HospitalStay"), ("InjuryISS","HospitalStay"),
    ("ICPMonitor","HospitalStay"), ("GCS","HospitalStay"), ("InjuryISS","GOSE"),
    ("MarshallCTScore","GOSE"), ("ICPMonitor","GOSE"), ("GCS","GOSE"),
    ("GOSE","FUWorkStatus"), ("GOSE","QoLIBRI"),
    ("SubduralHematoma","MidlineShift"), ("SubduralHematoma","MassLesion"),
    ("MassLesion","MarshallCTScore"), ("MidlineShift","MarshallCTScore"),
    ("SkullFracture","MassLesion"), ("SkullFracture","EpiduralHematoma"),
    ("EpiduralHematoma","RotterdamCTScore"), ("MassLesion","RotterdamCTScore"),
    ("MidlineShift","RotterdamCTScore"), ("InjuryISS","Glucose"),
    ("InjCause","SkullFracture"),
    # ("Age","Sex") removed -- not causally justifiable
]

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

def fit_bn(train: pd.DataFrame, constrained: bool, scoretype: str = "aic",
           max_indegree=None, verbose: int = 0):
    """Learn structure (Hill-Climbing) and fit parameters.

    constrained=False -> BN-1, fully data-driven
    constrained=True  -> BN-2, Table S2 whitelist/blacklist restricted to present columns
    """
    import bnlearn as bn
    cols = set(train.columns)
    kw = {}
    if constrained:
        kw["black_list"] = [e for e in S2_BLOCKED if e[0] in cols and e[1] in cols]
        kw["fixed_edges"] = [e for e in S2_FIXED if e[0] in cols and e[1] in cols]
        kw["bw_list_method"] = "edges"
    if max_indegree is not None:
        kw["max_indegree"] = int(max_indegree)
    model = bn.structure_learning.fit(train, methodtype="hc", scoretype=scoretype,
                                      n_jobs=1, verbose=verbose, **kw)
    # n_jobs=1 is load-bearing: the default -1 forks one worker per core, each
    # carrying a copy of the model and data, which OOM-kills an 8 GB machine.
    fitted = bn.parameter_learning.fit(model, train, n_jobs=1, verbose=verbose)
    return model, fitted

def metrics(y_true: pd.Series, proba: pd.DataFrame) -> dict:
    """Full metric suite for an ordinal multi-class prediction.

    Nominal metrics are retained for comparability with the wider literature.
    Ordinal metrics, calibration and the dichotomised outcomes are the additions

    GOSE codes are 1,2,4,5,6,7,8 -- there is no 3, because code 2 is the released
    merged "2 & 3" class. Ordinal statistics are therefore computed on consecutive
    RANKS 1-7, and mean absolute error is reported in GOSE CATEGORIES rather than
    "units", since the raw codes are not equally spaced on the underlying scale.
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, mean_squared_error,
                                 cohen_kappa_score, brier_score_loss, log_loss)
    from sklearn.preprocessing import label_binarize
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import somersd

    classes = list(proba.columns)
    P = proba[classes].to_numpy()
    yhat = np.asarray(classes)[P.argmax(axis=1)]
    Y = label_binarize(y_true, classes=classes)
    rank = {c: i + 1 for i, c in enumerate(classes)}
    y_rank = y_true.map(rank).to_numpy()
    yhat_rank = pd.Series(yhat).map(rank).to_numpy()
    exp_rank = P @ np.arange(1, len(classes) + 1)

    m = {
        "accuracy": accuracy_score(y_true, yhat),
        "precision_macro": precision_score(y_true, yhat, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, yhat, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, yhat, average="macro", zero_division=0),
        "MSE_codes": mean_squared_error(y_true, yhat),
        # --- discrimination from the POSTERIOR (the correction) ---
        "AUC_micro": roc_auc_score(Y, P, average="micro"),
        "AUC_macro": roc_auc_score(Y, P, average="macro"),
        # --- the same quantity computed the ORIGINAL way, for comparison ---
        "AUC_micro_hardlabel": roc_auc_score(Y, label_binarize(yhat, classes=classes),
                                             average="micro"),
        "kappa_quadratic": cohen_kappa_score(y_rank, yhat_rank, weights="quadratic"),
        "MAE_categories": float(np.abs(y_rank - yhat_rank).mean()),
        "ordinal_c_index": float((somersd(y_rank, exp_rank).statistic + 1) / 2),
        "brier_multiclass": float(np.mean(np.sum((P - Y) ** 2, axis=1))),
        "log_loss": log_loss(y_true, P, labels=classes),
    }
    # GOSE 3 does not exist, so "unfavourable 1-4" is implemented as codes {1,2,4}
    for name, pos in [("unfavourable_1to4", [1, 2, 4]), ("mortality_GOSE1", [1])]:
        idx = [classes.index(c) for c in pos if c in classes]
        p = P[:, idx].sum(axis=1)
        y = y_true.isin(pos).astype(int).to_numpy()
        m[f"AUC_{name}"] = roc_auc_score(y, p)
        m[f"brier_{name}"] = brier_score_loss(y, p)
        # calibration slope / intercept on the logit scale
        eps = 1e-6
        lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)
        try:
            m[f"calib_slope_{name}"] = float(
                LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
                .fit(lp, y).coef_[0][0])
            m[f"calib_intercept_{name}"] = float(
                LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
                .fit(np.zeros_like(lp), y, sample_weight=None).intercept_[0]
                if False else
                LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000,
                                   fit_intercept=True).fit(lp, y).intercept_[0])
        except Exception:
            m[f"calib_slope_{name}"] = np.nan
            m[f"calib_intercept_{name}"] = np.nan
    return m

def posterior(fitted, test: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Full posterior P(target = k | predictors) for every test row.

    This is the replacement for the original hard-label AUC computation.
    """
    mdl = fitted["model"]
    in_model = [v for v in mdl.nodes() if v != target]
    X = test[in_model]
    proba = mdl.predict_probability(X)
    proba.index = test.index
    proba.columns = [int(str(c).split("_")[-1]) for c in proba.columns]
    return proba.reindex(sorted(proba.columns), axis=1)


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


def stage_attrition():
    """Included versus excluded patients across baseline characteristics."""
    src = load_source("recruited_coded")
    inc_idx = src.index[src["GOSE"].notna()]
    exc_idx = src.index[src["GOSE"].isna()]
    print(f"included {len(inc_idx)} | excluded {len(exc_idx)} | total {len(src)}")
    INC_COL = f"Included (n={len(inc_idx)})"
    EXC_COL = f"Excluded (n={len(exc_idx)})"
    cat = [c for c in predictors(src, leakage_free=True) if c != "GOSE"]
    rows = []
    SOURCE_RECODE = {
        "GCS": {1: "Mild (GCS 13-15)", 2: "Moderate (GCS 9-12)", 3: "Severe (GCS 3-8)"},
        "Stratum": {1: "Emergency room", 2: "Ward admission", 3: "Intensive care"},
    }
    dec = decode(src)
    for col, mapping in SOURCE_RECODE.items():
        dec[col] = src[col].map(mapping)
    for col in cat:
        if col not in dec.columns: continue
        a, b = dec.loc[inc_idx, col], dec.loc[exc_idx, col]
        for lev in sorted(set(a.dropna().unique()) | set(b.dropna().unique()), key=str):
            p1 = (a == lev).sum() / a.notna().sum()
            p2 = (b == lev).sum() / b.notna().sum()
            den = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
            smd = (p1 - p2) / den if den > 0 else 0.0
            rows.append({"Variable": col, "Level": str(lev),
                         INC_COL: f"{(a==lev).sum()} ({100*p1:.1f}%)",
                         EXC_COL: f"{(b==lev).sum()} ({100*p2:.1f}%)",
                         "SMD": round(smd, 3), "n_missing_inc": int(a.isna().sum()),
                         "n_missing_exc": int(b.isna().sum())})
    t = pd.DataFrame(rows)
    t["|SMD|"] = t.SMD.abs()
    t = t.sort_values("|SMD|", ascending=False)
    t.to_csv("out/TableS_included_vs_excluded.csv", index=False)
    n = len(t)
    print(f"\ncomparisons: {n} | max |SMD| {t['|SMD|'].max():.3f} | median {t['|SMD|'].median():.3f}")
    print(f"|SMD| >= 0.25 : {int((t['|SMD|']>=0.25).sum())}   >=0.20: {int((t['|SMD|']>=0.20).sum())}"
          f"   >=0.10: {int((t['|SMD|']>=0.10).sum())}")
    print("\nTOP 12:")
    print(t.head(12)[["Variable","Level",INC_COL,EXC_COL,"SMD"]].to_string(index=False))

def stage_association_heatmap():
    """Pairwise Cramer's V among the analysis variables, and the heatmap figure.

    V is the uncorrected statistic, sqrt(chi2 / (n * (min(shape) - 1))), averaged
    over the imputed datasets. Length of hospital stay is not a predictor and so is
    absent from the imputed data; its row and column use the observed banded value
    from the pre-imputation table, with unrecorded patients dropped pairwise.
    """

    use_house_style(12)
    GROUPS = [
        ("Demography",                ["Age", "Sex", "Race", "EduLevel", "MaritalStatus"]),
        ("Injury",                    ["PreInjASAPS", "InjCause", "InjuryISS"]),
        ("Presentation",              ["GCS", "PupilReactivity", "PupilSymmetry",
                                       "Hypoxia", "Hypotension"]),
        ("Laboratory",                ["Glucose", "Hemoglobin"]),
        ("Imaging (CT)",              ["MarshallCTScore", "RotterdamCTScore", "TSAH",
                                       "SubduralHematoma", "EpiduralHematoma",
                                       "MassLesion", "MidlineShift", "SkullFracture"]),
        ("Acute care",                ["Stratum", "ICPMonitor", "Surgery"]),
        ("Outcome",                   ["GOSE", "HospitalStay"]),
    ]
    ORDER = [v for _, vs in GROUPS for v in vs]
    def cramers_v(a, b):
        ct = pd.crosstab(a, b)
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return np.nan
        chi2 = chi2_contingency(ct, correction=False)[0]
        n = ct.to_numpy().sum()
        return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))

    frames = [prepare("arm3_mice_leakfree", gcs_spec="band3", imputation=i)
              for i in range(N_IMPUTATIONS)]
    cols = list(frames[0].columns)
    cube = np.full((len(frames), len(cols), len(cols)), np.nan)
    for k, df in enumerate(frames):
        for i in range(len(cols)):
            cube[k, i, i] = 1.0
            for j in range(i + 1, len(cols)):
                v = cramers_v(df[cols[i]], df[cols[j]])
                cube[k, i, j] = cube[k, j, i] = v
    V = pd.DataFrame(np.nanmean(cube, 0), index=cols, columns=cols)
    S = pd.DataFrame(np.nanstd(cube, 0, ddof=1), index=cols, columns=cols)

    # length of stay, observed rather than imputed, pairwise-complete
    obs = load_source("analysis_raw")["HospitalStay"].dropna()
    vals = np.full((len(frames), len(cols)), np.nan)
    for k, df in enumerate(frames):
        idx = df.index.intersection(obs.index)
        for j, c in enumerate(cols):
            vals[k, j] = cramers_v(obs.loc[idx], df.loc[idx, c])
    hs_mean, hs_sd = np.nanmean(vals, 0), np.nanstd(vals, 0, ddof=1)
    for M, vec, diag in ((V, hs_mean, 1.0), (S, hs_sd, 0.0)):
        M["HospitalStay"] = pd.Series(vec, index=cols)
        row = pd.Series(vec, index=cols).reindex(M.columns)
        row["HospitalStay"] = diag
        M.loc["HospitalStay"] = row
    V.round(4).to_csv("out/TABLES_cramersV_matrix.csv")
    S.round(4).to_csv("out/TABLES_cramersV_sd.csv")
    print(f"association matrix: {V.shape[0]} variables, "
          f"{int(obs.shape[0])} patients with a recorded length of stay")

    assert set(ORDER) == set(V.columns), set(ORDER) ^ set(V.columns)
    V = V.loc[ORDER, ORDER]
    V.round(4).to_csv('out/TABLES_cramersV_matrix.csv')
    M = V.to_numpy().copy()
    np.fill_diagonal(M, np.nan)
    vmax = float(np.ceil(np.nanmax(M) * 20) / 20)
    n = len(ORDER)
    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad('white')
    fig, ax = plt.subplots(figsize=(15.2, 12.8))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect='equal')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ORDER, rotation=90, fontsize=12)
    ax.set_yticklabels(ORDER, fontsize=12)
    ax.tick_params(length=0, pad=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5)
    ax.tick_params(which='minor', length=0)
    bounds, c = [], 0
    for _, vs in GROUPS:
        c += len(vs); bounds.append(c)
    for b in bounds[:-1]:
        ax.axhline(b - .5, color='#55617080', lw=1.6)
        ax.axvline(b - .5, color='#55617080', lw=1.6)
    start = 0
    for (label, vs), end in zip(GROUPS, bounds):
        mid = (start + end - 1) / 2
        ax.add_line(Line2D([-0.192, -0.192], [start - 0.38, end - 0.62],
                           transform=ax.get_yaxis_transform(), clip_on=False,
                           color='#546070', lw=1.4))
        ax.text(-0.318, mid, label, transform=ax.get_yaxis_transform(),
                ha='left', va='center', fontsize=12, color='#2B3440', linespacing=1.15)
        start = end
    cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02, shrink=0.82)
    cb.set_label("Cramér's V", fontsize=12, labelpad=18, rotation=270)
    cb.ax.tick_params(labelsize=12, length=0)
    cb.outline.set_visible(False)
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(f'out/fig1_cramersV.{ext}', dpi=600, bbox_inches='tight', facecolor='white')
    print(f"n={n}, colour scale 0-{vmax}, max off-diagonal {np.nanmax(M):.3f}")

def stage_imputation_convergence():
    """MICE chain convergence traces, extended to 30 iterations."""
    ITERATIONS, CHAINS, SEED = 30, 10, 20260219
    KERNEL_PKL = Path("out/mice_kernel_30iter.pkl")
    def fit():
        import miceforest as mf
        df = build_missing_frame()
        df = df.drop(columns=["GOSE"])          # no-GOSE, as for the analysis datasets
        for col in NOMINAL:
            if col in df.columns:
                df[col] = df[col].astype("category")
        miss = df.isna()                        # which cells were imputed
        kernel = mf.ImputationKernel(df.reset_index(drop=True), num_datasets=CHAINS,
                                     random_state=SEED, save_all_iterations_data=True)
        kernel.mice(ITERATIONS, num_threads=4, min_data_in_leaf=20)
        KERNEL_PKL.parent.mkdir(parents=True, exist_ok=True)
        with open(KERNEL_PKL, "wb") as f:
            pickle.dump({"kernel": kernel, "miss": miss}, f)
        return kernel, miss
    def traces(kernel, miss) -> pd.DataFrame:
        """Mean of the IMPUTED cells only, per variable per iteration per chain."""
        cols = [c for c in miss.columns if miss[c].any()]
        rows = []
        for d in range(CHAINS):
            for it in range(ITERATIONS + 1):
                comp = kernel.complete_data(dataset=d, iteration=it)
                for c in cols:
                    v = pd.to_numeric(comp[c][miss[c].to_numpy()], errors="coerce")
                    rows.append({"Variable": c, "iteration": it, "dataset": d,
                                 "mean_imputed": float(v.mean())})
        return pd.DataFrame(rows)
    def drift(tr: pd.DataFrame) -> pd.DataFrame:
        half = ITERATIONS // 2 + 1
        out = []
        for var, g in tr.groupby("Variable"):
            a = g[g.iteration == half].set_index("dataset").mean_imputed
            b = g[g.iteration == ITERATIONS].set_index("dataset").mean_imputed
            common = a.index.intersection(b.index)
            delta = (b.loc[common] - a.loc[common])
            agree = max((delta > 0).mean(), (delta < 0).mean())
            pct = 100.0 * (b.loc[common].mean() - a.loc[common].mean()) / abs(a.loc[common].mean())
            out.append({"Variable": var, f"iter{half}": a.mean(), f"iter{ITERATIONS}": b.mean(),
                        "pct_change_2nd_half": pct, "chains_agreeing": agree,
                        "converged": not (agree == 1.0 and abs(pct) > 3.0)})
        return pd.DataFrame(out).sort_values("pct_change_2nd_half",
                                             key=lambda s: s.abs(), ascending=False)
    kernel, miss = fit()
    tr = traces(kernel, miss)
    tr.to_csv("out/mice_convergence_30iter_traces.csv", index=False)
    dr = drift(tr)
    dr.to_csv("out/mice_convergence_30iter_drift.csv", index=False)
    print(dr.to_string(index=False))
    print(f"\nfailing at {ITERATIONS} iterations:",
              list(dr[~dr.converged].Variable) or "none")

def stage_imputation_convergence_figure():
    """Convergence diagnostic figure."""
    require("out/mice_convergence_30iter_traces.csv", "imputation_convergence")
    use_house_style(12)
    tr = pd.read_csv("out/mice_convergence_30iter_traces.csv")
    dr = pd.read_csv("out/mice_convergence_30iter_drift.csv").set_index("Variable")
    variables = sorted(tr.Variable.unique())
    ncol = 4
    nrow = int(np.ceil(len(variables) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 2.9 * nrow))
    for i, (ax, var) in enumerate(zip(axes.ravel(), variables)):
        g = tr[tr.Variable == var]
        for d, gd in g.groupby("dataset"):
            gd = gd.sort_values("iteration")
            ax.plot(gd.iteration, gd.mean_imputed, lw=0.8, alpha=0.75)
        ax.axvline(20, color="#444", lw=0.9, ls="--")
        failed = not bool(dr.loc[var, "converged"])
        ax.set_title(f"{var}{'  (not converged)' if failed else ''}", fontsize=12,
                     loc="center", color="#993300" if failed else "black")
        ax.set_xticks([0, 10, 20, 30])
        ax.set_xlim(0, 30)
        nice_ticks(ax, "y", nbins=4)
        if i >= (nrow - 1) * ncol:
            ax.set_xlabel("MICE iteration")
        ax.set_ylabel("Mean imputed value")
        boxed(ax)
    for ax in axes.ravel()[len(variables):]:
        ax.set_visible(False)
    fig.tight_layout(w_pad=3.2, h_pad=3.2)
    fig.savefig("out/figS_mice_convergence_30iter.svg", bbox_inches="tight")
    print(f"{len(variables)} variables, {tr.dataset.nunique()} chains, "
          f"iterations 0-{tr.iteration.max()}")
    print("not converged:", list(dr[~dr.converged].index) or "none")

def stage_fit_and_pool():
    """Per-imputation fits, posteriors, and the pooled performance tables."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("REFUSING TO RUN: PYTHONHASHSEED=0 must be set BEFORE the interpreter "
                 "starts. pgmpy's Hill-Climbing breaks score ties using string hashes, so "
                 "without a fixed hash seed the learned structure differs on every run "
                 "(BN-1 varied 56-61 edges on identical data). Relaunch as:\n"
                 "    PYTHONHASHSEED=0 python " + "<this script>")
    SEED, NBOOT, M = 42, 400, N_IMPUTATIONS
    CFG = dict(scoretype="aic", max_indegree=3)
    os.makedirs("out/posteriors", exist_ok=True)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    rows, edge_rows = [], []
    for name, con in [("BN-1", False), ("BN-2", True)]:
        for i in range(M):
            df = prepare("arm3_mice_leakfree", gcs_spec="band3", imputation=i)
            tr, te = train_test_split(df, test_size=0.3, stratify=df[TARGET],
                                      random_state=SEED)
            model, fitted = fit_bn(tr, constrained=con, **CFG)
            edges = model["model_edges"]
            edge_rows += [{"model": name, "imputation": i, "src": u, "dst": v}
                          for u, v in edges]

            pr = posterior(fitted, te).reindex(columns=CLASSES).fillna(0.0)
            pr = pr.div(pr.sum(axis=1), axis=0)
            pr.assign(GOSE_true=te[TARGET].values).to_csv(
                f"out/posteriors/{name}_imp{i:02d}.csv")

            dag = model["model"]
            rec = {"model": name, "imputation": i, "n_edges": len(edges),
                   "score_aic": AIC(tr).score(dag), "score_bic": BIC(tr).score(dag),
                   **metrics(te[TARGET], pr)}

            # per-class AUC and bootstrap variances (within-imputation variance for Rubin)
            Y = label_binarize(te[TARGET], classes=CLASSES)
            P = pr.to_numpy()
            for j, c in enumerate(CLASSES):
                rec[f"AUC_class{c}"] = roc_auc_score(Y[:, j], P[:, j])
            boot = {f"AUC_class{c}": [] for c in CLASSES}; boot["AUC_micro"] = []; boot["AUC_macro"] = []
            n = len(te)
            for _ in range(NBOOT):
                b = rng.integers(0, n, n)
                if Y[b].sum(axis=0).min() == 0:      # a class absent from the resample
                    continue
                boot["AUC_micro"].append(roc_auc_score(Y[b], P[b], average="micro"))
                boot["AUC_macro"].append(roc_auc_score(Y[b], P[b], average="macro"))
                for j, c in enumerate(CLASSES):
                    boot[f"AUC_class{c}"].append(roc_auc_score(Y[b][:, j], P[b][:, j]))
            for k, v in boot.items():
                rec[f"var_{k}"] = float(np.var(v, ddof=1)) if len(v) > 1 else np.nan
            rows.append(rec)
            print(f"[{name}] imp {i+1}/{M} edges={len(edges)} "
                  f"AUCmicro={rec['AUC_micro']:.4f} [{time.time()-t0:.0f}s]", flush=True)
    per = pd.DataFrame(rows)
    per.to_csv("out/bn_final_per_imputation.csv", index=False)
    pd.DataFrame(edge_rows).to_csv("out/bn_final_edges.csv", index=False)
    def rubin(est, var, m=M):
        """Pool m estimates with their within-imputation variances."""
        est, var = np.asarray(est, float), np.asarray(var, float)
        ok = ~(np.isnan(est) | np.isnan(var))
        est, var, m = est[ok], var[ok], ok.sum()
        Q, W = est.mean(), var.mean()
        Bv = est.var(ddof=1)
        T = W + (1 + 1 / m) * Bv
        r = (1 + 1 / m) * Bv / W if W > 0 else np.inf
        df = (m - 1) * (1 + 1 / r) ** 2 if r > 0 else np.inf
        tc = stats.t.ppf(0.975, df)
        fmi = ((1 + 1 / m) * Bv) / T if T > 0 else np.nan
        return dict(estimate=Q, se=np.sqrt(T), ci_low=Q - tc * np.sqrt(T),
                    ci_high=Q + tc * np.sqrt(T), df=df, FMI=fmi)
    t2 = []
    for name, g in per.groupby("model"):
        for c in CLASSES + ["micro", "macro"]:
            key = f"AUC_class{c}" if isinstance(c, int) else f"AUC_{c}"
            t2.append({"model": name, "GOSE_class": c, **rubin(g[key], g[f"var_{key}"])})
    pd.DataFrame(t2).to_csv("out/TABLE2_auc_by_gose_class.csv", index=False)
    METRICS = ["score_aic", "score_bic", "n_edges", "accuracy", "precision_macro",
               "recall_macro", "f1_macro", "MSE_codes", "AUC_micro",
               "AUC_micro_hardlabel", "kappa_quadratic", "MAE_categories",
               "ordinal_c_index", "AUC_unfavourable_1to4", "AUC_mortality_GOSE1",
               "brier_multiclass", "calib_slope_unfavourable_1to4",
               "calib_intercept_unfavourable_1to4"]
    t1 = []
    for met in METRICS:
        r = {"metric": met}
        for name, g in per.groupby("model"):
            r[f"{name}_mean"] = g[met].mean(); r[f"{name}_sd"] = g[met].std(ddof=1)
        t1.append(r)
    pd.DataFrame(t1).to_csv("out/TABLE1_bn_model_comparison.csv", index=False)
    print(f"TABLES DONE [{time.time()-t0:.0f}s]", flush=True)

def stage_stability_and_cv():
    """Edge stability, predictor-admissibility arms, repeated cross-validation."""
    require("out/bn_final_edges.csv", "fit_and_pool")
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("REFUSING TO RUN: PYTHONHASHSEED=0 must be set BEFORE the interpreter "
                 "starts. pgmpy's Hill-Climbing breaks score ties using string hashes, so "
                 "without a fixed hash seed the learned structure differs on every run "
                 "(BN-1 varied 56-61 edges on identical data). Relaunch as:\n"
                 "    PYTHONHASHSEED=0 python " + "<this script>")
    SEED, M = 42, N_IMPUTATIONS
    CFG = dict(scoretype="aic", max_indegree=3)
    t0 = time.time()
    e = pd.read_csv("out/bn_final_edges.csv")
    rows = []
    for model, g in e.groupby("model"):
        pair = g.assign(key=[tuple(sorted((a, b))) for a, b in zip(g.src, g.dst)])
        for k, gg in pair.groupby("key"):
            n1, n2 = k
            inc = gg.imputation.nunique() / M
            fwd = ((gg.src == n1) & (gg.dst == n2)).sum() / len(gg)
            rows.append({"model": model, "node_1": n1, "node_2": n2,
                         "inclusion_freq": round(inc, 3),
                         "P_dir_1to2": round(fwd, 3), "P_dir_2to1": round(1 - fwd, 3)})
    stab = pd.DataFrame(rows).sort_values(["model", "inclusion_freq"], ascending=[True, False])
    stab.to_csv("out/bn_structure_stability.csv", index=False)
    print(f"[stability] {len(stab)} distinct edges "
          f"({dict(stab.model.value_counts())}) [{time.time()-t0:.0f}s]", flush=True)
    arm_rows = []
    for arm in ["arm1_median_full", "arm2_median_leakfree", "arm3_mice_leakfree"]:
        for name, con in [("BN-1", False), ("BN-2", True)]:
            df = prepare(arm, gcs_spec="band3", imputation=0)
            tr, te = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df.GOSE)
            _, fitted = fit_bn(tr, constrained=con, **CFG)
            pr = posterior(fitted, te).reindex(columns=CLASSES).fillna(0.0)
            pr = pr.div(pr.sum(axis=1), axis=0)
            arm_rows.append({"arm": arm, "model": name, "n_predictors": df.shape[1] - 1,
                             **metrics(te.GOSE, pr)})
            print(f"[arm] {arm} {name} done [{time.time()-t0:.0f}s]", flush=True)
    pd.DataFrame(arm_rows).to_csv("out/bn_arm_comparison.csv", index=False)
    cv_rows, oof = [], []
    df = prepare("arm3_mice_leakfree", gcs_spec="band3", imputation=0)
    y = df[TARGET]
    for rep in range(5):
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED + rep)
        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            for name, con in [("BN-1", False), ("BN-2", True)]:
                _, fitted = fit_bn(df.iloc[tr], constrained=con, **CFG)
                pr = posterior(fitted, df.iloc[te]).reindex(columns=CLASSES).fillna(0.0)
                pr = pr.div(pr.sum(axis=1), axis=0)
                cv_rows.append({"model": name, "rep": rep, "fold": fold,
                                **metrics(y.iloc[te], pr)})
                oof.append(pr.assign(GOSE_true=y.iloc[te].values, model=name, rep=rep))
        print(f"[cv] rep {rep} done [{time.time()-t0:.0f}s]", flush=True)
    cvf = pd.DataFrame(cv_rows)
    cvf.to_csv("out/bn_cv_folds.csv", index=False)
    cvf.groupby("model").agg(["mean", "std"]).to_csv("out/bn_cv_summary.csv")
    pd.concat(oof).to_csv("out/bn_oof_predictions.csv")
    print(f"ALL REST DONE [{time.time()-t0:.0f}s]", flush=True)

def stage_unfavourable_auc():
    """Pooled AUC for the unfavourable-outcome dichotomy."""
    require("out/bn_final_per_imputation.csv", "fit_and_pool")
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("REFUSING TO RUN: PYTHONHASHSEED=0 must be set before interpreter start.")
    SEED, NBOOT, M = 42, 400, N_IMPUTATIONS
    CFG = dict(scoretype="aic", max_indegree=3)
    UNFAV = [1, 2, 4]
    def boot_var(y, p, seed):
        """Variance of the AUC over NBOOT resamples of the test set."""
        rng = np.random.default_rng(seed)
        n = len(y)
        vals = []
        for _ in range(NBOOT):
            idx = rng.integers(0, n, n)
            if len(np.unique(y[idx])) < 2:
                continue
            vals.append(roc_auc_score(y[idx], p[idx]))
        return float(np.var(vals, ddof=1)), len(vals)
    rows = []
    t0 = time.time()
    for name, con in [("BN-1", False), ("BN-2", True)]:
        for i in range(M):
            df = prepare("arm3_mice_leakfree", gcs_spec="band3", imputation=i)
            tr, te = train_test_split(df, test_size=0.3, stratify=df[TARGET],
                                      random_state=SEED)
            model, fitted = fit_bn(tr, constrained=con, **CFG)
            post = posterior(fitted, te)              # columns are the GOSE classes
            y_true = te[TARGET].to_numpy()
            p_unfav = post[[c for c in post.columns if int(c) in UNFAV]].sum(axis=1).to_numpy()  # cols are ints
            y_unfav = np.isin(y_true, UNFAV).astype(int)
            auc = roc_auc_score(y_unfav, p_unfav)
            var, nb = boot_var(y_unfav, p_unfav, seed=hash((name, i)) % (2**32))
            auc1 = roc_auc_score((y_true == 1).astype(int), post[1].to_numpy())
            rows.append({"model": name, "imputation": i, "AUC_unfavourable": auc,
                         "var_AUC_unfavourable": var, "n_boot_used": nb,
                         "AUC_class1_check": auc1, "n_test": len(te),
                         "n_unfav": int(y_unfav.sum())})
            print(f"[{name} imp{i:02d}] AUC_unfav={auc:.4f} var={var:.3e} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    per = pd.DataFrame(rows)
    per.to_csv("out/bn_unfav_auc_per_imputation.csv", index=False)
    ref = pd.read_csv("out/bn_final_per_imputation.csv")[["model", "imputation", "AUC_class1"]]
    j = per.merge(ref, on=["model", "imputation"])
    d = (j.AUC_class1_check - j.AUC_class1).abs()
    print(f"\nAUC_class1 reproduction: max |difference| = {d.max():.3e} over {len(j)} fits")
    assert d.max() < 1e-3, "refit does NOT match the pipeline that produced Table 2"
    def rubin(est, var, m=M):
        est, var = np.asarray(est, float), np.asarray(var, float)
        ok = ~(np.isnan(est) | np.isnan(var))
        est, var, m = est[ok], var[ok], ok.sum()
        Q, W = est.mean(), var.mean()
        Bv = est.var(ddof=1)
        T = W + (1 + 1 / m) * Bv
        r = (1 + 1 / m) * Bv / W if W > 0 else np.inf
        df = (m - 1) * (1 + 1 / r) ** 2 if r > 0 else np.inf
        tc = stats.t.ppf(0.975, df)
        fmi = ((1 + 1 / m) * Bv) / T if T > 0 else np.nan
        return dict(estimate=Q, se=np.sqrt(T), ci_low=Q - tc * np.sqrt(T),
                    ci_high=Q + tc * np.sqrt(T), df=df, FMI=fmi)
    out = [{"model": n, "GOSE_class": "unfavourable (1, 2 or 3, 4)",
            **rubin(g.AUC_unfavourable, g.var_AUC_unfavourable)}
           for n, g in per.groupby("model")]
    res = pd.DataFrame(out)
    res.to_csv("out/TABLE2_unfavourable_rubin.csv", index=False)
    print()
    print(res.to_string(index=False))

def stage_calibration_figure():
    """Calibration curves from pooled out-of-fold predictions."""
    require("out/bn_oof_predictions.csv", "stability_and_cv")
    use_house_style(size=12)
    oof = pd.read_csv("out/bn_oof_predictions.csv", index_col=0)
    defs = [("Unfavourable outcome (GOSE 1, 2 or 3, 4)", [1, 2, 4]), ("Mortality (GOSE 1)", [1])]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
    rows = []
    for ax, (title, codes) in zip(axes, defs):
        for mdl, col in [("BN-1", "#1F5FA8"), ("BN-2", "#B5392E")]:
            s = oof[oof.model == mdl]
            p = s[[str(c) for c in codes]].sum(axis=1).to_numpy()
            y = s.GOSE_true.isin(codes).astype(int).to_numpy()
            q = pd.qcut(p, 10, duplicates="drop")
            g = (pd.DataFrame({"p": p, "y": y, "q": q})
                 .groupby("q", observed=True).agg(["mean", "size"]))
            ax.plot(g[("p", "mean")], g[("y", "mean")], "o-", color=col, lw=1.6, ms=5, label=mdl)
            lo = np.clip(p, 1e-6, 1 - 1e-6)
            X = sm.add_constant(np.log(lo / (1 - lo)))
            r = sm.GLM(y, X, family=sm.families.Binomial()).fit()
            rows.append({"outcome": title, "model": mdl, "slope": r.params[1],
                         "intercept": r.params[0], "n": len(y)})
        ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(title, fontsize=12)       # panel identifier, not a figure title
        ax.legend(frameon=False)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig("out/fig_bn_calibration.svg", bbox_inches="tight")
    cal = pd.DataFrame(rows)
    cal.to_csv("out/bn_calibration.csv", index=False)
    print(cal.round(3).to_string(index=False))

def stage_network_figures():
    """Consensus network diagrams with edge-strength and centrality tables."""
    require("out/bn_structure_stability.csv", "stability_and_cv")
    GROUP_ORDER = ['Age', 'Sex', 'Race', 'EduLevel', 'MaritalStatus', 'PreInjASAPS', 'InjCause', 'InjuryISS', 'GCS', 'PupilReactivity', 'PupilSymmetry', 'Hypoxia', 'Hypotension', 'Glucose', 'Hemoglobin', 'MarshallCTScore', 'RotterdamCTScore', 'TSAH', 'SubduralHematoma', 'EpiduralHematoma', 'MassLesion', 'MidlineShift', 'SkullFracture', 'Stratum', 'ICPMonitor', 'Surgery', 'GOSE']
    FONTNAME, FONTSIZE = "IBM Plex Sans", 12
    SOLID, DOTTED = 0.91, 0.91
    LIGHT_EXCEPTIONS = {('Hypotension', 'Stratum')}
    MIN_IN, MAX_IN = 0.95, 2.00
    def wrap(name: str) -> str:
        """Split a CamelCase variable name onto separate lines.

        Graphviz honours width/height only when fixedsize='true', and a fixed-size
        circle does not grow to fit its label, so long names must be wrapped or the
        text spills outside the node."""
        out, cur = [], ""
        for ch in name:
            if ch.isupper() and cur and not cur.isupper():
                out.append(cur); cur = ch
            else:
                cur += ch
        out.append(cur)
        return "\\n".join(out)
    MODE = "perfigure"
    SUFFIX = ""
    stab = pd.read_csv('out/bn_structure_stability.csv')
    frames = [prepare('arm3_mice_leakfree', gcs_spec='band3', imputation=i)
              for i in range(N_IMPUTATIONS)]
    built = {}
    for model in ['BN-1', 'BN-2']:
        is_exc = pd.Series([(a, b) in LIGHT_EXCEPTIONS
                            for a, b in zip(stab.node_1, stab.node_2)], index=stab.index)
        sub = stab[(stab.model == model) & ((stab.inclusion_freq >= DOTTED) | is_exc)]

        # --- edge strengths using mutual information (averaged over the 30 fits) ---
        edge_strengths = []
        for _, r in sub.iterrows():
            a, b, p = r.node_1, r.node_2, float(r.P_dir_1to2)
            src, dst = (a, b) if p >= 0.5 else (b, a)
            strength = float(np.mean([mutual_info_score(d[a], d[b]) for d in frames]))
            # Every edge is drawn SOLID. Weak edges are conveyed by line weight, which
            # already encodes mutual information, rather than by a dashed style; the
            # sub-threshold exceptions are the thinnest lines in each figure as a result.
            tier = ('solid (below threshold, by exception)'
                    if float(r.inclusion_freq) < SOLID else 'solid')
            edge_strengths.append((src, dst, strength, tier))

        # Sorting is load-bearing: graphviz `dot` lays the graph out in the order
        # nodes and edges are declared, so an identical structure declared in a
        # different order draws differently. Sorting makes the figure reproducible.
        edge_strengths.sort(key=lambda t: (t[0], t[1]))
        G = nx.DiGraph()
        present = {n for e in edge_strengths for n in e[:2]}
        ordered = [n for n in GROUP_ORDER if n in present] + sorted(present - set(GROUP_ORDER))
        G.add_nodes_from(ordered)
        G.add_weighted_edges_from([(u, v, w) for u, v, w, _ in edge_strengths])
        built[model] = (G, nx.betweenness_centrality(G), edge_strengths, ordered)
    rows_e, rows_n = [], []
    gmax = max(max(c.values()) for _, c, _, _ in built.values())
    for model, stem in [('BN-1', 'out/fig2_bn1'), ('BN-2', 'out/fig3_bn2')]:
        G, centrality, edge_strengths, ordered = built[model]
        cref = gmax if MODE == "shared" else max(centrality.values())

        graph = pgv.AGraph(directed=True)
        observed_node_color = '#ffbd7d'
        for node in ordered:
            centrality_score = centrality.get(node, 0)
            if centrality_score > 0:
                # Non-zero betweenness: a true circle whose DIAMETER encodes centrality.
                size = MIN_IN + (centrality_score / cref) ** 0.5 * (MAX_IN - MIN_IN)
                graph.add_node(node, color=observed_node_color, style='filled',
                               shape='circle', fixedsize='true', width=size, height=size,
                               fontsize=FONTSIZE, fontname=FONTNAME, fontcolor='black',
                               label=wrap(node))
            else:
                # Betweenness exactly zero: label-fitted ellipse, deliberately a
                # different shape so it is not read as a small circle.
                graph.add_node(node, color=observed_node_color, style='filled',
                               shape='ellipse', fontsize=FONTSIZE, fontname=FONTNAME,
                               fontcolor='black', label=node)
        for u, v, strength, _tier in edge_strengths:
            width = strength * 10
            graph.add_edge(u, v, penwidth=width, style='solid')

        # BN-2 only: Hypotension descends from Stratum while Glucose descends from Age
        # and InjuryISS, so `dot` ranks them several levels apart and draws them at
        # opposite ends of the figure. Placing them on a shared rank groups the three
        # physiological variables together in the drawing. They share no edge, so the
        # constraint is admissible.
        if model == 'BN-2':
            graph.add_subgraph(['Glucose', 'Hypotension'], name='rank_physiology',
                               rank='same')

        graph.layout(prog='dot')
        graph.draw(f'{stem}{SUFFIX}.svg', format='svg')
        # ---------------------------------

        rows_e.extend({'model': model, 'source': u, 'target': v, 'tier': st,
                       'mutual_information': w, 'penwidth': w * 10}
                      for u, v, w, st in edge_strengths)
        rows_n.extend({'model': model, 'node': n, 'betweenness_centrality': centrality[n],
                       'shape': 'circle' if centrality[n] > 0 else 'ellipse',
                       'node_diameter_in': (MIN_IN + (centrality[n] / cref) ** 0.5 * (MAX_IN - MIN_IN))
                                           if centrality[n] > 0 else float('nan')}
                      for n in sorted(G.nodes()))

        n_dot = sum(1 for e in edge_strengths if e[3] == 'dotted')
        print(f"{model}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
              f"({G.number_of_edges()-n_dot} solid / {n_dot} dotted) | centrality max "
              f"{max(centrality.values()):.4f} -> node {MAX_IN:.2f} in "
              f"| bb={graph.graph_attr.get('bb','')}")
    pd.DataFrame(rows_e).to_csv('out/dag_edge_strength.csv', index=False)
    pd.DataFrame(rows_n).to_csv('out/dag_node_centrality.csv', index=False)



# ============================================================================
# Command line
# ============================================================================
STAGES = {
    "build_imputations": stage_build_imputations,
    "attrition": stage_attrition,
    "association_heatmap": stage_association_heatmap,
    "imputation_convergence": stage_imputation_convergence,
    "imputation_convergence_figure": stage_imputation_convergence_figure,
    "fit_and_pool": stage_fit_and_pool,
    "stability_and_cv": stage_stability_and_cv,
    "unfavourable_auc": stage_unfavourable_auc,
    "calibration_figure": stage_calibration_figure,
    "network_figures": stage_network_figures,
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(STAGES)
    unknown = [w for w in wanted if w not in STAGES]
    if unknown:
        sys.exit(f"unknown stage(s): {unknown}\navailable: {list(STAGES)}")
    for w in wanted:
        print(f"\n=== {w} ===", flush=True)
        STAGES[w]()


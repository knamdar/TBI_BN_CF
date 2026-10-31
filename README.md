# Prognostic modelling and treatment-effect estimation in traumatic brain injury

Code accompanying the manuscript submitted to *PLOS Digital Health*.

| file | analysis |
|---|---|
| `bayesian_networks.py` | Probabilistic prediction of the Glasgow Outcome Scale-Extended (GOSE) at six months, comparing a data-driven network structure with one constrained by clinical knowledge. |
| `causal_forests.py` | Exploratory estimates of how associations between cranial surgery, ICU admission or intracranial pressure monitoring and patient outcomes vary across patients. Outcomes are six-month GOSE and acute hospital length of stay. |

Each script is self-contained: data access, multiple imputation, model fitting,
evaluation and figures. There are no shared local modules.

---

## 1. Obtain the data

**No patient data are distributed with this code.** CENTER-TBI data are held under a
managed-access agreement; apply through the CENTER-TBI data-access procedure
(ClinicalTrials.gov NCT02210221).

Five files are needed. The first four are tabular exports of the cohort:

| configuration key | contents |
|---|---|
| `analysis` | the analytic cohort, variables coded numerically |
| `analysis_raw` | the same cohort before imputation, used to tell which treatment statuses were observed rather than filled in |
| `recruited_coded` | the full recruited cohort, coded, for the included-versus-excluded comparison |
| `recruited_labelled` | the same table with value labels rather than codes |

The fifth, `CONTINUOUS_FILE`, holds the continuous covariates. **It is not part of the
coded exports and must be assembled from the raw extracts.** Index it by subject
identifier, with these columns:

| column | used for |
|---|---|
| `Age_years`, `ISS_total` | re-deriving the age and injury-severity bands |
| `LOS_days` | the length-of-stay outcome of CF-4 to CF-6 |
| `Hb_baseline_gdl`, `Glucose_baseline_mmolL`, `GCS_total`, `GCS_motor` | auxiliary variables in the imputation model |
| `days_to_death`, `ICP_start_day`, `ICP_end_day`, `ICP_n_readings`, `ICP_mean`, `ICP_max` | descriptive only |

The loaders also apply two checks used during preparation of this table: haemoglobin
values recorded in g/L are converted to g/dL, and negative lengths of stay are set
to zero. These checks leave an already-cleaned file unchanged.

If your exports use different column names, extend `NAME_MAP` in either script rather
than renaming columns in the data itself.

## 2. Point the scripts at your files

Open the **CONFIGURATION** block at the top of either script — the first dozen lines
after the imports — and set the five paths to your own filenames. The other settings can remain as supplied.

## 3. Install

Use Python 3.11.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`pygraphviz` additionally needs a system Graphviz installation.

## 4. Run

```bash
export PYTHONHASHSEED=0

python bayesian_networks.py build_imputations    # once; the longest step
python bayesian_networks.py                      # then all remaining stages in order
python causal_forests.py                         # reuses the same imputed datasets
```

Either script runs every stage in order when called with no arguments, or a named
subset: `python causal_forests.py pooled_effects overlap`. Stages exchange results
through files in `out/`, so a stage needs the stages before it to have been run at
least once. A stage called before its prerequisite stops immediately and names the
stage to run first, rather than failing part-way through. Several causal-forest stages save per-model checkpoints in `out/cf_partial/`,
allowing interrupted runs to resume.

### `PYTHONHASHSEED=0` is required

Hill-climbing structure search iterates over dictionaries keyed by variable-name
strings, and Python randomises string hashing per process. With the seed unset,
repeated fits on identical data returned structures of 57, 58 and 59 edges, because
equally scoring candidate edges were visited in a different order. `bayesian_networks.py`
refuses to start unless it is set. It must be set in the environment **before** Python
starts; setting it inside the script has no effect.

## Stages

### `bayesian_networks.py`

| stage | what it does |
|---|---|
| `build_imputations` | Generate the imputed datasets every later stage depends on. |
| `attrition` | Included versus excluded patients across baseline characteristics. |
| `association_heatmap` | Pairwise Cramer's V among the analysis variables, and the heatmap figure. |
| `imputation_convergence` | MICE chain convergence traces, extended to 30 iterations. |
| `imputation_convergence_figure` | Convergence diagnostic figure. |
| `fit_and_pool` | Per-imputation fits, posteriors, and the pooled performance tables. |
| `stability_and_cv` | Edge stability, predictor-admissibility arms, repeated cross-validation. |
| `unfavourable_auc` | Pooled AUC for the unfavourable-outcome dichotomy. |
| `calibration_figure` | Calibration curves from pooled out-of-fold predictions. |
| `network_figures` | Consensus network diagrams with edge-strength and centrality tables. |

### `causal_forests.py`

| stage | what it does |
|---|---|
| `build_imputations` | Generate the imputed datasets every later stage depends on. |
| `pooled_effects` | All six models across the imputations; pooled average treatment effects. |
| `specification_and_propensity` | Specification comparison and cross-fitted propensity scores. |
| `overlap` | Propensity overlap pooled across imputations. |
| `strict_adjustment` | Sensitivity analysis dropping all non-focal treatments from the covariates. |
| `leaf_size_sensitivity` | Robustness of the estimates to the minimum leaf size. |
| `treatment_cooccurrence` | Co-occurrence of the three treatments. |
| `importance_and_shap` | Forest covariate importance and mean |SHAP| per model. |
| `shap_remaining_imputations` | Extends mean |SHAP| to the remaining imputed datasets. |
| `effect_figures` | Distributions of conditional effects and covariate-importance panels. |
| `shap_figures` | SHAP beeswarm and aggregated attribution panels. |

## Notes

* `build_imputations` writes the imputed datasets both scripts read. Run it once; if the
  datasets are already present the stage reports that and returns.
* Approximate runtimes on an 8-core machine: imputation a little over an hour, the 60
  network fits about 8 minutes, the association matrix about 90 seconds, SHAP about 33
  seconds per fit.
* The scripts are also run with `OMP_NUM_THREADS=4` and `ARROW_IO_THREADS=1`.
* `fonts/` holds the three IBM Plex Sans faces used by the figures, so they render
  identically without a system font installation. They are under the SIL Open Font
  License.
* The patient-selection flowchart in the manuscript is a drawn schematic containing no
  analysis and is not reproduced here.

## Licence

The analysis code is licensed under the BSD 3-Clause License; see `LICENSE`.
Copyright (c) 2026, Pascal N. Tyrrell.

The bundled IBM Plex Sans font files in `fonts/` are separately licensed under
the SIL Open Font License 1.1 and are not covered by the BSD 3-Clause License.

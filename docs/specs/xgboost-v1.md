# XGBoost v1 spec

Per-checkpoint XGBoost win-probability model. This is the v1 production model — its output is what feeds edge calculation against Polymarket odds.

Rationale for the strategic decisions lives in `discovery-checklist.md` sections 2–3. This doc is the *what*, not the *why*. Where a decision was refined beyond the checklist, the block below calls it out explicitly.

Scope:

- **4 models, one per checkpoint T ∈ {10, 15, 20, 25}.** Unified across tiers with `tier` as a categorical feature.
- **Must clear an acceptance threshold vs the logistic baseline** (see Acceptance) before we ship.
- **2026 is touched exactly once** at the end for honest evaluation.
- **Production refit happens after 2026 evaluation clears** — these refit models are the betting artifacts.

## Data + split

Inputs: `data/processed/checkpoint_{10,15,20,25}.csv` from the preprocessing pipeline.

**Refined temporal split** (refines the checklist's "train 2020–2024 / val 2025 / test 2026" — we subdivide both 2024 and 2025 to avoid any single slice being used for more than one decision):

| Split | Date range | Use |
|---|---|---|
| **Train** | 2020-01-01 – 2024-06-30 | Fit trees within each Optuna trial |
| **Early-stop val** | 2024-07-01 – 2024-12-31 | `early_stopping_rounds` inside each trial; picks `n_estimators` |
| **Optuna val** | 2025-01-01 – 2025-06-30 | Optuna trial objective; isotonic calibrator fit |
| **Acceptance val** | 2025-07-01 – 2025-12-31 | Raw acceptance gate; honest calibrated-metric eval |
| **Test** | 2026 full year | Final held-out evaluation — touched exactly once |
| **Production refit** | 2020-01-01 – 2025-12-31 | After 2026 clears, refit on all three vals + train with chosen hyperparams |

Why the two subdivisions:

- **2024 subdivision** (2024-07-01 cutoff): gives ~5K early-stop-val games per checkpoint and leaves ~33K training games per checkpoint. Enough signal to make in-trial early-stopping decisions stable without burning into 2025.
- **2025 subdivision** (2025-07-01 cutoff): separates the data Optuna optimizes against (H1, ~4K games) from the data the acceptance gate reads (H2, ~4K games). XGBoost never sees the acceptance-val set during tuning or calibration fit, so the raw-prob gate is honest. Isotonic is fit on Optuna val and evaluated on Acceptance val, so both raw and calibrated metrics reported for acceptance are out-of-sample with respect to every upstream fit.

Temporal cutoffs throughout respect no-future-leakage.

**2026 discipline:** no hyperparameter, feature, or architecture change in response to 2026 numbers. One evaluation, honest number. If we need to iterate after seeing 2026, we burn the test set and need a new held-out window.

## Features

Stage 3 produces a 63-column superset; stage 4 gates first-events per checkpoint, so the per-checkpoint CSVs on disk have 24 / 26 / 26 / 27 columns for T ∈ {10, 15, 20, 25}. The model consumes everything in the per-checkpoint CSV minus identifiers (see below). **Keep the `*diffat{T}` columns** (unlike the baseline, which dropped them for collinearity). XGBoost is tree-based: pre-computing the diff lets a single split express "golddiff > 3000" directly instead of a deep cascade of nested splits on raws. Raws + opp + diff is information-redundant but modeling-practical.

Per checkpoint T, the model consumes:

- **Numeric (12 cols):** the full `{raw, opp, diff}` triple for each of `gold`, `cs`, `kills`, `assists` at minute T.
- **Categorical (native):** `side` ∈ {Blue, Red}, `tier` ∈ {S, A, B}. Fed via xgboost native categorical support (`enable_categorical=True`), not one-hot.
- **Binary:** gated first-event flags — `firstblood` and `firstdragon` always; `firsttower` at T ≥ 20; `firstbaron` excluded entirely. Gates were realigned 2026-05-03 against direct livestats timing measurements (`docs/specs/livestats-firsts-timing.md`); see `preprocessing-pipeline.md` for the per-checkpoint table and rationale.

Identifiers (`gameid, date, year, league, patch, teamname, opp_teamname`) and `gamelength` are not fed to the model.

**No scaling.** Trees are scale-invariant.

**No feature engineering beyond what stage 3 already ships.** Any new feature idea goes through a discovery-checklist update, not this spec.

## Model

`xgboost.XGBClassifier` with:

- `objective='binary:logistic'`
- `eval_metric='logloss'`
- `tree_method='hist'` (CPU, fast, deterministic given a seed)
- `enable_categorical=True`
- `n_jobs=-1`
- `random_state=42`
- `early_stopping_rounds=50` (checked against early-stop val inside each trial)
- `n_estimators=2000` (upper bound; early stopping picks the actual count)

Dependency: `xgboost>=3.2,<4`.

## Hyperparameter tuning

**Optuna** (`optuna>=3.0,<5`), one independent study per checkpoint. 100 trials per study.

Search space:

| Param | Range | Dist |
|---|---|---|
| `max_depth` | 3–10 | int uniform |
| `learning_rate` | 0.01–0.3 | log-uniform |
| `min_child_weight` | 1–20 | int uniform |
| `subsample` | 0.6–1.0 | uniform |
| `colsample_bytree` | 0.6–1.0 | uniform |
| `reg_alpha` | 1e-8–10 | log-uniform |
| `reg_lambda` | 1e-8–10 | log-uniform |
| `gamma` | 1e-8–10 | log-uniform |

`n_estimators` is not tuned — set to 2000 with `early_stopping_rounds=50` on the early-stop val set; xgboost picks the actual count per trial.

**Sampler:** `optuna.samplers.TPESampler(seed=42)`.

**Pruner:** `optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50)` — kills trials whose val log-loss after 50 boosting rounds is worse than the median of completed trials at the same step. Expected to save ~30–50% compute.

**Objective:** Optuna-val log-loss (2025-H1). Each trial:

1. Fit `XGBClassifier` on train (2020 – 2024-06-30) with early stopping against early-stop val (2024-07-01 – 2024-12-31).
2. Predict on Optuna val (2025-H1), compute log-loss — this is Optuna's objective.
3. Report intermediate values after each boosting round to the pruner.

No nested CV. The 5-way temporal split is our guard against optimism.

## Calibration

After Optuna picks the best trial per checkpoint:

1. Refit the best-trial model on train + early-stop val combined (2020-01-01 – 2024-12-31) with the chosen hyperparams and the selected `n_estimators` (from the best trial's early-stopping count).
2. Predict on **Optuna val (2025-H1)** — these predictions feed isotonic fit.
3. Compute **10-bin weighted ECE** and a **reliability diagram** on the raw Optuna-val predictions.
4. **Calibration trigger: ECE > 0.01 on Optuna val.** If triggered, fit **isotonic regression** (`sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')`) on 2025-H1 predictions → 2025-H1 actuals. Production `predict_proba` chains model → isotonic.
5. **Platt fallback — quantitative rule.** Inspect the fitted isotonic calibrator. Fall back to **Platt scaling** (`LogisticRegression` on model logit vs actual) if **either**:
    - the calibrator has fewer than 15 distinct output values on the val prediction grid, **or**
    - the largest flat run on the isotonic step function — i.e., the widest span of input probabilities mapping to a single output value — exceeds 0.20 of the [0, 1] range.
    Both are cheap to compute from the fitted `IsotonicRegression.X_thresholds_` / `y_thresholds_` arrays.
6. **Predict on Acceptance val (2025-H2)** and record log-loss, Brier, AUC, accuracy, ECE on both **raw** and **calibrated** probabilities. These are the numbers that feed acceptance and runs.csv. They are out-of-sample with respect to Optuna (XGB never saw H2) and out-of-sample with respect to isotonic (fit on H1).

Acceptance gates on raw H2 metrics (baseline is uncalibrated, so comparing raw-to-raw is apples-to-apples). Calibrated H2 metrics are logged alongside for information — they are what the Kelly / bet pipeline actually sees, and because the cal fit was on a different slice, they're honest too.

## Acceptance criteria

**Committed up front, before seeing numbers.** If XGBoost fails at any checkpoint, we investigate — bug, leakage, or features genuinely lack nonlinear structure — before shipping.

Per checkpoint T, XGBoost's **raw (uncalibrated)** predictions on **Acceptance val (2025-H2)** must clear *both*:

- **H2 log-loss ≤ baseline H2 log-loss − 0.005**
- **H2 Brier ≤ baseline H2 Brier** (any improvement, no floor)

Raw-vs-raw comparison is intentional: the baseline is uncalibrated logistic; gating on calibrated XGB vs uncalibrated baseline would credit XGB for in-sample isotonic fit on the same set we gate on. H2-vs-H2 is also intentional: XGB never saw H2 during tuning or calibration fit, so the gate is honest.

Baseline-on-2025-H2 numbers (from `runs.csv` row `2026-04-21T11-45-07Z_baseline_logistic_h2`, produced by `scripts/models/baseline_logistic_h2_rerun.py` — predicts the already-fit 2020–2024 baseline model against the H2 slice, no retraining):

| T | H2 n | Baseline H2 log-loss | Baseline H2 Brier | XGBoost must hit |
|---|---|---|---|---|
| 10 | 3,274 | 0.5826 | 0.2001 | ≤ 0.5776 / ≤ 0.2001 |
| 15 | 3,274 | 0.5179 | 0.1744 | ≤ 0.5129 / ≤ 0.1744 |
| 20 | 3,271 | 0.4430 | 0.1459 | ≤ 0.4380 / ≤ 0.1459 |
| 25 | 3,176 | 0.2773 | 0.0848 | ≤ 0.2723 / ≤ 0.0848 |

Note: H2 baseline is a touch worse than the full-2025 baseline at T=15/20 and a touch better at T=10/25 — natural H1-vs-H2 distribution drift across the year (patches, rosters). Numbers above are what XGBoost is gated against; the full-2025 numbers are retained in `runs.csv` as historical context only.

AUC and accuracy are reported but not gating — log-loss + Brier are the calibration-aware metrics that matter for Kelly sizing.

## Test-set evaluation (2026)

Once all 4 checkpoints clear acceptance on 2025-H2:

1. **Atomic across all 4 checkpoints.** `--evaluate-test` loops T ∈ {10, 15, 20, 25} in one invocation and writes all four result sets at once. Per-checkpoint invocation is not supported — seeing T=10's 2026 number before running T=15 would bias any remaining judgement calls.
2. For each checkpoint: predict on 2026 test using the calibrated model, report **raw and calibrated** log-loss, Brier, AUC, accuracy, plus reliability diagram and 10-bin weighted ECE.
3. Expect some degradation vs 2025 val (Feats of Strength artifacts in 2025 train don't exist in 2026; Baron timing differs). Acceptable as long as calibration holds — reliability diagram + ECE are the primary checks, not raw log-loss.
4. **One evaluation. Touched once.** Any iteration on model, features, or hyperparameters after this burns the test set.

**CLI preconditions** — checked before the atomic loop starts, so we fail loudly instead of writing partial test results:

- **All 4 `metrics.json` files must exist.** Any missing → exit with an error naming the missing checkpoints and suggesting `python -m scripts.models.xgboost_v1` first.
- **All 4 must have `acceptance_passed == True`.** Any False → exit with an error listing the failing checkpoints and their `acceptance_diffs`. No partial eval.
- **None may already contain a `test` block or `test_n` field.** If any does → exit with an error "2026 already evaluated for T=…; the test set is touched once." No `--force` flag by design — to re-run, the operator must manually delete the `test` blocks and `reliability_test.png`s, which keeps re-runs a deliberate act.

## Production refit

Only runs after 2026 test evaluation clears. For each checkpoint:

1. Take the chosen hyperparams from the Optuna study's best trial.
2. **Second early-stopping pass for `n_estimators_prod`.** Hold out 2025-07-01 – 2025-12-31 (~4K games — this is physically the same slice as Acceptance val, which is fine because acceptance has already been scored by the time this step runs; we're just reusing the slice as an early-stopping signal, not a gate) as an early-stopping slice. Fit on 2020-01-01 – 2025-06-30 with the chosen hyperparams and `early_stopping_rounds=50` against that slice; record the selected `n_estimators_prod`. This replaces the earlier 1.1× heuristic with a calibrated-to-the-data count.
3. **Final refit on all data.** Refit on 2020-01-01 – 2025-12-31 using the chosen hyperparams and `n_estimators_prod` (no early stopping — count is locked from step 2). This is the shipped model.
4. **Carry the calibrator over unchanged** from the val-stage pipeline (isotonic, Platt, or `None`). No production-side calibrator refit. Rationale: the val-stage calibrator was fit on 2025-H1 predictions from a 2020–2024-trained model and evaluated honestly on 2025-H2; the production model adds ~10% more data, and isotonic's monotonicity bounds the resulting small distributional shift. Drift past that bound is caught by residual monitoring (next section), not pre-ship.
5. Serialize the (model, calibrator) pair as the production artifact. These are what the bet pipeline loads.

## Residual calibration monitoring

> **Deferred for v1 — not implemented.** No bet pipeline yet means no live P&L to protect, and the val-stage calibrator is fit honestly against a disjoint slice, so drift detection is theoretical in v1. The design below stands for when we revisit — likely once enough 2026 games have accumulated past the production fit timestamp to form a meaningful pool, and/or once the bet pipeline ships. Everything in this section is spec-only: no `--refit-calibrator` CLI, no `load_monitoring_pool` / `refit_calibrator_if_drifted` functions, no `calibrator_history/` or `monitoring_log.csv` artifacts, no associated tests in v1. The `calibrator_source` field in `metadata.json` still exists but only ever takes the value `"val_stage_carried_over"` until this section ships.

Post-deployment sanity check on calibrator drift. Runs outside the train → eval → ship flow — operator-invoked, not automatic. The flow is: (a1) intentionally ships the production model with a calibrator that was fit against a slightly-smaller-training-set sibling model, so we want a cheap signal if the resulting distributional mismatch (or meta shift in 2026) has broken calibration.

**Trigger condition (per checkpoint):** ≥ 1000 labeled 2026 games have passed through the bet pipeline since deployment (or since the last calibrator refit). LoL games resolve within hours, so this accumulates on the timescale of weeks, not quarters.

**Computation on the monitoring pool** (labeled 2026 games; raw model predictions passed through the current shipped calibrator):

- 10-bin weighted ECE (on calibrated probs)
- Brier score
- Log-loss

**Drift flag — either trigger:**

- ECE > 0.015 (slightly looser than the pre-ship 0.01; monitoring pools are smaller and noisier than 2025-H2), or
- Brier degraded by > 10% relative to the val-stage calibrated Brier recorded in `metrics.json`.

**Action on drift:** refit isotonic (or Platt, per the same pathology rule as the val stage) on `{monitoring_pool_raw_preds → monitoring_pool_actuals}`. **No model retrain, no hyperparameter change.** Archive the previous calibrator under `calibrator_history/` with a timestamp; swap the new one into `calibrator.pkl`.

This is a **signal**, not an automatic action. The operator can choose to keep the old calibrator if the drift looks like a one-off meta shift that is expected to reverse (e.g., a single-patch anomaly). The point of the check is visibility, not autonomy.

**Not part of test discipline.** Residual monitoring runs on 2026 data that has already been seen by `--evaluate-test` (the honest one-touch eval). It does not gate model architecture or hyperparameter decisions — only calibrator swap. That keeps the "2026 touched once" rule for model-level iteration intact.



`scripts/models/xgboost_v1.py` — pure functions mirroring the baseline/pipeline convention.

```python
NUMERIC_FEATURES(T)      -> list[str]     # 12 cols: raw+opp+diff × 4 stats
CATEGORICAL_FEATURES()   -> list[str]     # ["side", "tier"]
BINARY_FEATURES(T)       -> list[str]     # gated firsts

SEARCH_SPACE                              # dict consumed by Optuna objective
N_TRIALS = 100
SEED = 42

prepare_Xy(df, T) -> (X, y, feature_names)
    # cast categoricals; select features; no scaling

split_temporal(df) -> dict[str, DataFrame]
    # keys: "train", "early_stop_val", "optuna_val", "acceptance_val", "test"
    # cutoffs: 2020-01-01 / 2024-07-01 / 2025-01-01 / 2025-07-01 / 2026-01-01

make_objective(T, splits) -> Callable[[optuna.Trial], float]
    # returns Optuna objective closure: fits XGBClassifier with trial params,
    # early-stops on early_stop_val, scores optuna_val log-loss,
    # reports intermediate values to pruner

tune_checkpoint(T) -> optuna.Study
    # creates study with TPE sampler + MedianPruner, runs 100 trials

fit_calibrated(T, study, splits) -> (model, calibrator)
    # refits best-trial hyperparams on train + early_stop_val,
    # fits isotonic on optuna_val iff raw-prob ECE on optuna_val > 0.01,
    # falls back to Platt if isotonic is pathological
    # (<15 distinct outputs, or flat-run span > 0.20).
    # All ACCEPTANCE metrics are then scored on acceptance_val,
    # not optuna_val — keeps them out-of-sample w.r.t. both Optuna and isotonic.

ece_weighted(y_true, probs, n_bins=10) -> float
isotonic_is_pathological(calibrator) -> bool
    # distinct outputs <15 OR max flat-run span > 0.20 on [0,1]

evaluate(y_true, probs) -> dict          # log_loss, brier, auc, accuracy, ece
reliability_bins(y_true, probs, n_bins=10) -> DataFrame
check_acceptance(T, raw_metrics) -> (passed: bool, diffs: dict)
    # compares RAW metrics to baseline numbers, returns per-metric deltas
    # gate is intentionally on raw probs since baseline is uncalibrated

train_checkpoint(T) -> dict
    # tune -> fit_calibrated -> evaluate raw + calibrated on acceptance_val
    # -> check_acceptance(raw) -> writes artifacts, appends runs.csv row
    # metrics dict carries both raw_* and cal_* keys

evaluate_test_all() -> dict[int, dict]
    # ATOMIC across all 4 checkpoints. Loops T ∈ {10,15,20,25}:
    # predicts on 2026, writes test metrics (raw + cal) + reliability png.
    # Per-checkpoint evaluate_test(T) exists but is only invoked from here;
    # the CLI does not expose it separately.

fit_production(T) -> None
    # step 2: hold out 2025-07-01..2025-12-31, fit on 2020..2025-06-30
    #   with early_stopping_rounds=50 -> n_estimators_prod
    # step 3: refit on 2020..2025-12-31 with chosen hparams + n_estimators_prod
    # step 4: carry calibrator.pkl over from val stage unchanged
    # writes production artifact; only runs after --evaluate-test clears

# load_monitoring_pool, refit_calibrator_if_drifted — deferred, see Residual
# calibration monitoring section.

main()
    # loop T ∈ {10,15,20,25}: train_checkpoint
    # main() does NOT invoke evaluate_test_all or fit_production —
    # those are explicit CLI flags to enforce the "touched once"
    # and operator-in-the-loop disciplines
```

CLI:

```bash
python -m scripts.models.xgboost_v1                       # tune + fit + val acceptance check
python -m scripts.models.xgboost_v1 --evaluate-test       # 2026 honest eval (atomic across all 4 T, once)
python -m scripts.models.xgboost_v1 --fit-production      # final refit for deployment
# --refit-calibrator: deferred, see Residual calibration monitoring section
```

## Artifacts

Per checkpoint, under `data/models/xgboost_v1/checkpoint_{T}/`:

- `model.json` — xgboost-native save (JSON, version-stable across xgboost upgrades).
- `calibrator.pkl` — pickled isotonic (or Platt) regressor, or the sentinel `None` if calibration wasn't triggered.
- `study.pkl` — full Optuna study for post-hoc analysis (per-trial scores, param importances).
- `metrics.json` — `{best_params, n_estimators_selected, optuna_val_n, acceptance_val_n, test_n?, raw: {logloss, brier, auc, accuracy, ece}, cal: {logloss, brier, auc, accuracy, ece}, calibration_applied: bool, calibrator_kind: 'isotonic'|'platt'|null, acceptance_passed: bool, acceptance_diffs}`. `raw` and `cal` metric blocks are both evaluated on **acceptance_val (2025-H2)**; Optuna-val sizes are recorded for traceability but no metrics are stored there. Acceptance fields are computed against `raw`.
- `feature_importance.csv` — `feature, gain, weight, cover` from `booster.get_score()`; human-inspection aid.
- `reliability_val.png` — 10-bin reliability diagram on 2025-H2 acceptance val (both raw and calibrated overlays).
- `reliability_test.png` — only exists after `--evaluate-test` runs; 10-bin reliability on 2026.

Under `data/models/xgboost_v1/production/checkpoint_{T}/`:

- `model.json`, `calibrator.pkl`, `metadata.json` — the betting artifacts. Only exist after `--fit-production`. `metadata.json` records `n_estimators_prod` (from the second early-stopping pass), `fit_timestamp` (used later as the monitoring-pool cutoff), `calibrator_source` (always `"val_stage_carried_over"` in v1; reserved to take `"residual_refit:<timestamp>"` once monitoring ships), and the val-stage calibrated Brier (stored now as the future drift baseline).
- `calibrator_history/<timestamp>.pkl`, `monitoring_log.csv` — deferred, see Residual calibration monitoring section.

Appended to `data/experiments/runs.csv` — one row per checkpoint per `main()` invocation. Schema extends the baseline with two nullable columns: `val_logloss_cal`, `val_brier_cal`. Baseline rows leave them empty; XGBoost rows populate both. The existing `val_logloss` / `val_brier` columns continue to carry **raw** (uncalibrated) numbers so baseline-vs-XGB comparisons stay apples-to-apples. `val_n` for XGBoost rows is acceptance_val_n (2025-H2 size). Baseline rerun rows against 2025-H2 should set `notes="2025-H2 only"` to distinguish them from the original full-2025 baseline rows. `hparams_json` holds the Optuna best-trial params + `n_estimators_selected`.

## Testing

`tests/models/test_xgboost_v1.py` — mirrors baseline test conventions.

Small synthetic fixtures, no real-CSV slices.

- `prepare_Xy` — categorical columns typed correctly (pandas CategoricalDtype), diff columns present, gated firsts included at right T, no identifiers leak in, row count preserved.
- `split_temporal` — date boundaries correct; each split non-empty on realistic fixture; no row in multiple splits.
- `make_objective` — returns a float (log-loss), respects trial-suggested params, reports at least one intermediate value to the pruner.
- `tune_checkpoint` — runs a small (3-trial) study end-to-end on synthetic data, returns a study with `best_trial`.
- `fit_calibrated` — returns a model that predicts in [0,1]; calibrator is either `None` or an object with a `.predict(...)` method; fires on an intentionally miscalibrated fixture (ECE > 0.01) and stays at `None` on a well-calibrated fixture (ECE ≤ 0.01).
- `ece_weighted` — known-input cases match hand-computed values; equals 0 on perfectly calibrated probs.
- `isotonic_is_pathological` — flags a fitted isotonic with < 15 distinct outputs or flat-run span > 0.20; accepts a healthy one.
- `evaluate` — known-input cases (all-wrong / perfect / random) match expectations for log_loss, brier, auc, accuracy, ece.
- `check_acceptance` — reads from `raw_metrics` (not calibrated); passes when both metrics clear thresholds, fails when either doesn't, returns correct diff signs.
- `train_checkpoint` — runs end-to-end on synthetic data with a tiny trial budget, writes expected artifacts, appends exactly one `runs.csv` row with both raw and cal columns populated.
- `evaluate_test_all` — smoke test that it runs all 4 checkpoints in one invocation and writes all 4 result sets; asserts that a hypothetical per-T public entry point does not exist on the module.
- `fit_production` — smoke test that it runs end-to-end and writes the expected artifact; `metadata.json` records `n_estimators_prod` from a real (tiny-data) second early-stopping pass; `calibrator.pkl` is byte-identical to the val-stage calibrator.
- `refit_calibrator_if_drifted` — deferred, see Residual calibration monitoring section.

## How to run

```bash
# dependencies
pip install "xgboost>=3.2,<4" "optuna>=3.0,<5"

# tune + fit all 4 checkpoints, report val acceptance (gated on RAW probs)
python -m scripts.models.xgboost_v1

# honest 2026 evaluation — atomic across all 4 checkpoints, run ONCE, only after val acceptance
python -m scripts.models.xgboost_v1 --evaluate-test

# production refit (only after --evaluate-test clears)
python -m scripts.models.xgboost_v1 --fit-production

# --refit-calibrator (residual drift check): deferred, see Residual calibration monitoring section

# tests
python -m pytest tests/models/test_xgboost_v1.py
```

# Logistic baseline spec

Per-checkpoint logistic-regression baseline for v1 in-game win probability. Purely diagnostic: gives a lower bound that XGBoost must clearly beat. If XGBoost doesn't, there's a bug, leakage, or the features have no nonlinear structure worth exploiting.

Scope limits (locked):

- **No production deployment.** Baseline artifacts are reference only.
- **2026 is NOT touched here.** The 2026 test set is touched exactly once, later, when the final XGBoost model is evaluated.
- **No isotonic post-calibration.** We measure calibration (Brier + reliability diagram) but don't correct — logistic trained on log-loss is usually well-calibrated out of the box. Isotonic earns its keep on XGBoost, not here.
- **No walk-forward CV.** That's a separate exercise for Kelly sizing, run against a real candidate model.

## Data + split

Inputs: `data/processed/checkpoint_{10,15,20,25}.csv` from the preprocessing pipeline.

Temporal split:

| Split | Years | Use |
|---|---|---|
| Train | 2020–2024 | Fit the model |
| Val   | 2025      | `C` selection, metrics reported |
| Test (2026) | — | Not touched here |

## Features

Derived from the stage-4 checkpoint schema, with the collinear `*diffat{T}` columns dropped (see below). For checkpoint T:

- **Numeric (standardized):** the 8 raw/opp stats — `goldat{T}, opp_goldat{T}, csat{T}, opp_csat{T}, killsat{T}, opp_killsat{T}, assistsat{T}, opp_assistsat{T}`.
- **Binary (as-is):** `side_is_blue` (1 if `side == "Blue"`, else 0), plus the gated first-event flags (`firstblood` always; `firstdragon` and `firsttower` at T≥15; `firstbaron` at T=25).
- **Categorical (one-hot, S as baseline):** `tier` → `tier_A`, `tier_B`. Implemented via `pd.Categorical(tier, categories=["S","A","B"])` then `pd.get_dummies(..., drop_first=True)`; plain `drop_first=True` on the raw string column drops alphabetically-first (A), which is not what we want.

Identifiers (`gameid`, `date`, `year`, `league`, `patch`, `teamname`, `opp_teamname`) and `gamelength` are not fed to the model.

**Why drop `*diffat{T}`:** stage 3 asserts `diff == raw − opp` exactly, so the raw/opp/diff triple is perfectly collinear. XGBoost keeps all three because axis-aligned splits can't synthesize the diagonal cheaply; a linear model does that trivially with coefficients `(β, −β)` on `(raw, opp)`. Keeping diff would split L2 weight across collinear columns and make `coefficients.csv` meaningless (ridge spreads weight arbitrarily across exactly-collinear features). `{raw, opp}` spans the same 2D space as `{diff, sum}`, so it's strictly more expressive than diff-alone for a linear model — it can represent both a "lead" effect and a "pace" effect.

**Why `side_is_blue` (Blue=1) rather than Red=1:** Blue is the measured advantaged side (52–58% WR, `03_side_winrate.json`), so a positive coefficient means "the advantaged side" — matches the sign convention elsewhere in the spec (positive = favors this team's win). Keeps `coefficients.csv` readable without mental sign-flipping.

**Preprocessing order:** `StandardScaler` is fit on train numeric columns only (no data leakage from val). Binary and one-hot columns pass through untouched.

## Model

`sklearn.linear_model.LogisticRegressionCV` with:

- `penalty='l2'`
- `solver='lbfgs'`
- `max_iter=1000` (converges well under this on standardized features)
- `Cs=[0.01, 0.1, 1, 10, 100]`
- `cv=5`, `scoring='neg_log_loss'` — internal 5-fold CV on the training set picks `C`.
- `refit=True` — after CV, refit on the full training set using the chosen `C` (sklearn default).

No `random_state` — `lbfgs` is deterministic.

**Why `LogisticRegressionCV` on train, not a grid-search against val:** picking `C` on val and then reporting the same val metrics as the headline numbers is circular — the min of 5 noisy estimates is biased low. Selecting `C` via CV inside the training set leaves the 2025 val set untouched for honest scoring. Compute cost is negligible.

No class weighting — the label is ~50/50 by construction (one winner per game, filters are winner-agnostic; see discovery-checklist item "Class balance check").

## Metrics

Computed on the 2025 val set for each checkpoint:

- **Log-loss** (primary — matches tuning objective, rewards calibrated probabilities)
- **Brier score** (calibration + sharpness)
- **AUC** (ranking quality, scale-free)
- **Accuracy** at 0.5 threshold (human-readable sanity anchor)

Plus a **10-bin reliability diagram** saved as PNG per checkpoint — bins by predicted probability, plots mean predicted vs actual win rate. Diagonal = well-calibrated.

**Expected ranges** (rough, for sanity; from the literature on pro-only data):

| Checkpoint | Accuracy | AUC |
|---|---|---|
| T=10 | 0.60–0.68 | 0.65–0.75 |
| T=15 | 0.65–0.72 | 0.72–0.80 |
| T=20 | 0.70–0.78 | 0.78–0.86 |
| T=25 | 0.75–0.82 | 0.84–0.90 |

If T=10 lands above ~0.85 accuracy, investigate leakage before celebrating.

**2025-as-val caveat:** 2025 is a structurally anomalous year relative to both the 2020–2024 training pool and the 2026 deployment target — Baron was pushed to 25:00 (vs 20:00 before/after), Atakhan was live at 20:00, and Feats of Strength bled into `goldat20`/`goldat25` / `killsat20`/`killsat25` via compounding power advantage (~50% of 2025 games earned feats). Expect T=20 and T=25 val metrics to sit at or below the ranges above even if the model is sound. Don't read mediocre 20/25-min baseline numbers as "model is broken"; read them as "read XGBoost's 2026 test results, not the baseline's 2025 val results, for the honest performance estimate." See chat-summary "2026 meta context" and discovery section 4 for detail.

## Module shape

`scripts/models/baseline_logistic.py` exposes pure functions (mirrors the preprocessing pipeline convention):

```python
NUMERIC_FEATURES(T) -> list[str]          # the 8 raw/opp *at{T} columns (no diffs)
BINARY_FEATURES(T)  -> list[str]          # side_is_blue + gated firsts
C_GRID = (0.01, 0.1, 1, 10, 100)

prepare_Xy(df, T) -> (X: DataFrame, y: Series, feature_names: list[str])
    # tier → pd.Categorical(categories=["S","A","B"]) → get_dummies(drop_first=True)
    # side → side_is_blue (Blue=1, Red=0); select features; drop *diffat{T}

split_train_val(df) -> (train_df, val_df)
    # year ≤ 2024 → train; year == 2025 → val; year == 2026 → dropped

fit(X_train, y_train) -> (scaler: StandardScaler, model: LogisticRegressionCV)
    # scaler fit on numeric cols of train only; LogisticRegressionCV picks C via
    # internal 5-fold CV on train; model.C_ holds the chosen value

predict_proba(scaler, model, X, numeric_cols) -> np.ndarray
    # applies scaler to numeric cols, returns P(result=1)

evaluate(y_true, probs) -> dict          # log_loss, brier, auc, accuracy

reliability_bins(y_true, probs, n_bins=10) -> DataFrame
    # columns: bin_lo, bin_hi, n, mean_pred, mean_actual

train_checkpoint(T) -> artifacts
    # fit once (CV-internal C selection), evaluate on val, write artifacts,
    # return metrics + selected C

main()
    # loop T ∈ {10,15,20,25}, call train_checkpoint, append 4 rows to runs.csv
```

## Artifacts

Per checkpoint, under `data/models/baseline_logistic/checkpoint_{T}/`:

- `model.pkl` — pickled dict: `{"scaler", "model", "feature_names", "numeric_cols"}`.
- `metrics.json` — `{C_selected, train_n, val_n, log_loss, brier, auc, accuracy}`. (`C_selected` comes from `model.C_[0]`; the per-C CV scores are not reported — internal CV is an implementation detail, val is the honest score.)
- `coefficients.csv` — `feature, coefficient, abs_coefficient` sorted by `abs_coefficient` descending. Human-inspection aid — coefficients are on standardized features, so magnitude is directly comparable.
- `reliability.png` — 10-bin reliability diagram.

Appended to `data/experiments/runs.csv` (append-only, one row per checkpoint per run):

```
run_id, timestamp, model_type, checkpoint, hparams_json, train_n, val_n,
val_logloss, val_brier, val_auc, val_accuracy, notes
```

`run_id` = `{ISO-timestamp}_{model_type}` (one id per `main()` invocation; shared across the 4 checkpoint rows).

## Testing

`tests/models/test_baseline_logistic.py` — mirrors pipeline TDD conventions.

Test targets (small synthetic fixtures, no real-CSV slices):

- `prepare_Xy` — tier one-hot has **S as baseline** (columns `tier_A` and `tier_B` present, no `tier_S`) — guards against the `drop_first` alphabetical-sort pitfall; `side_is_blue` is 1 for Blue / 0 for Red; `*diffat{T}` columns are NOT in the output; gated firsts included at right T; no identifiers leak in; row count preserved.
- `split_train_val` — year boundaries correct (2024 in train, 2025 in val, 2026 in neither).
- `fit` — scaler fit on train only (feeding val stats doesn't leak), `LogisticRegressionCV` converges and exposes `model.C_`, `predict_proba` output in [0, 1] and sums to 1 across classes.
- `evaluate` — known-input cases: all-wrong → log_loss high, perfect → log_loss≈0, random → auc≈0.5.
- `reliability_bins` — correct bin counts, perfectly-calibrated synthetic input sits on diagonal.
- `train_checkpoint` — runs end-to-end on synthetic data, writes expected artifact files, returns metrics dict with expected keys including `C_selected`.

## How to run

```bash
# dependencies (one-time)
pip install scikit-learn matplotlib

# train all 4 baselines
python -m scripts.models.baseline_logistic

# tests
python -m pytest tests/models/
```

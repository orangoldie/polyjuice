"""
XGBoost v1 per-checkpoint win-probability model.

Per-checkpoint Optuna-tuned XGBClassifier with optional isotonic/Platt
calibration. Five-way temporal split:

    train           2020-01-01 .. 2024-06-30
    early_stop_val  2024-07-01 .. 2024-12-31    (early_stopping_rounds in-trial)
    optuna_val      2025-01-01 .. 2025-06-30    (Optuna objective; isotonic fit)
    acceptance_val  2025-07-01 .. 2025-12-31    (raw acceptance gate; honest cal eval)
    test            2026 full year              (touched once)

See docs/xgboost-v1.md for the full spec.

Inputs:  data/processed/checkpoint_{10,15,20,25}.csv
Outputs: data/models/xgboost_v1/checkpoint_{T}/{model.json, calibrator.pkl,
            study.pkl, metrics.json, feature_importance.csv, reliability_val.png,
            reliability_test.png after --evaluate-test}
         data/models/xgboost_v1/production/checkpoint_{T}/{model.json,
            calibrator.pkl, metadata.json} after --fit-production
         data/experiments/runs.csv (appended)
"""
import argparse
import json
import os
import pickle
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score  # noqa: E402


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_DIR = os.path.join(REPO_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "models", "xgboost_v1")
PROD_DIR = os.path.join(OUTPUT_DIR, "production")
RUNS_CSV = os.path.join(REPO_ROOT, "data", "experiments", "runs.csv")

CHECKPOINTS = (10, 15, 20, 25)
SEED = 42
N_TRIALS = 100

N_ESTIMATORS_MAX = 2000
EARLY_STOPPING_ROUNDS = 50
CALIBRATION_ECE_THRESHOLD = 0.01

SIDE_CATEGORIES = ["Blue", "Red"]
TIER_CATEGORIES = ["S", "A", "B"]

FIRST_GATES = {
    10: ["firstblood", "firstdragon"],
    15: ["firstblood", "firstdragon"],
    20: ["firstblood", "firstdragon", "firsttower"],
    25: ["firstblood", "firstdragon", "firsttower"],
}

SPLIT_CUTOFFS = {
    "train_start":          "2020-01-01",
    "early_stop_val_start": "2024-07-01",
    "optuna_val_start":     "2025-01-01",
    "acceptance_val_start": "2025-07-01",
    "test_start":           "2026-01-01",
    "test_end":             "2027-01-01",
}

# Raw baseline logloss/brier on 2025-H2 only (acceptance thresholds).
# Source: runs.csv row 2026-04-21T11-45-07Z_baseline_logistic_h2.
BASELINE_H2 = {
    10: {"logloss": 0.5826, "brier": 0.2001},
    15: {"logloss": 0.5179, "brier": 0.1744},
    20: {"logloss": 0.4430, "brier": 0.1459},
    25: {"logloss": 0.2773, "brier": 0.0848},
}
LOGLOSS_MARGIN = 0.005


_DIFF_NAMES = {"gold": "golddiff", "cs": "csdiff",
               "kills": "killdiff", "assists": "assistdiff"}


# ---------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------

def NUMERIC_FEATURES(T: int) -> list[str]:
    """12 cols: {raw, opp, diff} × {gold, cs, kills, assists} at minute T.

    Diff column names are irregular (killdiff not killsdiff, assistdiff not
    assistsdiff) — matches the preprocessed CSV schema.
    """
    feats = []
    for stat in ("gold", "cs", "kills", "assists"):
        feats.extend([
            f"{stat}at{T}",
            f"opp_{stat}at{T}",
            f"{_DIFF_NAMES[stat]}at{T}",
        ])
    return feats


def CATEGORICAL_FEATURES() -> list[str]:
    return ["side", "tier"]


def BINARY_FEATURES(T: int) -> list[str]:
    return list(FIRST_GATES[T])


def prepare_Xy(df: pd.DataFrame, T: int) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build (X, y, feature_names) for checkpoint T.

    Numeric → float. Categoricals → pandas CategoricalDtype with explicit
    category order so xgboost's native categorical support is stable across
    splits. Binary gated-firsts → int. No scaling; trees are scale-invariant.
    Identifiers and gamelength are dropped.
    """
    X = pd.DataFrame(index=df.index)
    for col in NUMERIC_FEATURES(T):
        X[col] = df[col].astype(float)
    X["side"] = pd.Categorical(df["side"], categories=SIDE_CATEGORIES)
    X["tier"] = pd.Categorical(df["tier"], categories=TIER_CATEGORIES)
    for flag in BINARY_FEATURES(T):
        X[flag] = df[flag].astype(int)

    y = df["result"].astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)
    return X, y, list(X.columns)


def split_temporal(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {train, early_stop_val, optuna_val, acceptance_val, test}.

    Half-open intervals [start, next_start). Expects df['date'] parseable
    to datetime.
    """
    dates = pd.to_datetime(df["date"])
    c = {k: pd.Timestamp(v) for k, v in SPLIT_CUTOFFS.items()}
    masks = {
        "train":          (dates >= c["train_start"])          & (dates < c["early_stop_val_start"]),
        "early_stop_val": (dates >= c["early_stop_val_start"]) & (dates < c["optuna_val_start"]),
        "optuna_val":     (dates >= c["optuna_val_start"])     & (dates < c["acceptance_val_start"]),
        "acceptance_val": (dates >= c["acceptance_val_start"]) & (dates < c["test_start"]),
        "test":           (dates >= c["test_start"])           & (dates < c["test_end"]),
    }
    return {name: df[mask].reset_index(drop=True) for name, mask in masks.items()}


# ---------------------------------------------------------------------
# Metrics / calibration helpers
# ---------------------------------------------------------------------

def reliability_bins(y_true, probs, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width bins on [0, 1]; per-bin n, mean_pred, mean_actual.

    The last bin is right-closed so probs exactly == 1.0 are counted.
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": 0,
                         "mean_pred": float("nan"), "mean_actual": float("nan")})
        else:
            rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": n,
                         "mean_pred": float(probs[mask].mean()),
                         "mean_actual": float(y_true[mask].mean())})
    return pd.DataFrame(rows)


def ece_weighted(y_true, probs, n_bins: int = 10) -> float:
    """Weighted Expected Calibration Error.

    ECE = Σ (n_bin / N) · |mean_pred_bin − mean_actual_bin|, bins with n=0 skipped.
    """
    bins = reliability_bins(y_true, probs, n_bins=n_bins)
    populated = bins[bins["n"] > 0]
    total = float(populated["n"].sum())
    if total == 0:
        return 0.0
    gap = (populated["mean_pred"] - populated["mean_actual"]).abs()
    return float((populated["n"] / total * gap).sum())


def evaluate(y_true, probs) -> dict:
    """log_loss, brier, auc, accuracy, ece."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = (probs >= 0.5).astype(int)
    auc = float(roc_auc_score(y_true, probs)) if len(set(y_true.tolist())) > 1 else float("nan")
    return {
        "log_loss": float(log_loss(y_true, probs, labels=[0, 1])),
        "brier":    float(brier_score_loss(y_true, probs)),
        "auc":      auc,
        "accuracy": float((preds == y_true).mean()),
        "ece":      ece_weighted(y_true, probs, n_bins=10),
    }


def isotonic_is_pathological(calibrator,
                             min_distinct: int = 15,
                             max_flat_span: float = 0.20) -> bool:
    """Return True if the fitted isotonic calibrator has collapsed structure.

    Triggers when EITHER:
      - fewer than `min_distinct` distinct output values, or
      - the largest span of input thresholds that map to a single output value
        exceeds `max_flat_span` of the [0, 1] range.
    Both indicate the step function is too degenerate for useful
    probability shaping → fall back to Platt scaling.

    Reads `X_thresholds_` / `y_thresholds_` from `sklearn.isotonic.IsotonicRegression`.
    """
    xs = np.asarray(calibrator.X_thresholds_, dtype=float)
    ys = np.asarray(calibrator.y_thresholds_, dtype=float)
    if len(np.unique(ys)) < min_distinct:
        return True

    max_span = 0.0
    i = 0
    while i < len(ys):
        j = i
        while j + 1 < len(ys) and ys[j + 1] == ys[i]:
            j += 1
        span = float(xs[j] - xs[i])
        if span > max_span:
            max_span = span
        i = j + 1
    return max_span > max_flat_span


def check_acceptance(T: int, raw_metrics: dict) -> tuple[bool, dict]:
    """Gate raw XGBoost metrics against baseline-H2 thresholds.

    Returns (passed, diffs). `passed` requires BOTH:
      - logloss beat baseline by ≥ LOGLOSS_MARGIN (0.005), AND
      - Brier ≤ baseline Brier (any improvement, no floor).
    """
    baseline = BASELINE_H2[T]
    logloss_diff = raw_metrics["log_loss"] - baseline["logloss"]
    brier_diff   = raw_metrics["brier"]    - baseline["brier"]
    logloss_ok = logloss_diff <= -LOGLOSS_MARGIN
    brier_ok   = brier_diff   <= 0.0
    passed = bool(logloss_ok and brier_ok)
    return passed, {
        "baseline_logloss": baseline["logloss"],
        "baseline_brier":   baseline["brier"],
        "xgb_logloss":      float(raw_metrics["log_loss"]),
        "xgb_brier":        float(raw_metrics["brier"]),
        "logloss_diff":     float(logloss_diff),
        "brier_diff":       float(brier_diff),
        "logloss_margin":   LOGLOSS_MARGIN,
        "logloss_ok":       bool(logloss_ok),
        "brier_ok":         bool(brier_ok),
    }


# ---------------------------------------------------------------------
# Optuna tuning
# ---------------------------------------------------------------------

def _xgb_base_params() -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "enable_categorical": True,
        "n_jobs": -1,
        "random_state": SEED,
    }


def _trial_params(trial: "optuna.Trial") -> dict:
    """Search space from the spec. int-uniform / uniform / log-uniform per docs."""
    return {
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma":            trial.suggest_float("gamma", 1e-8, 10.0, log=True),
    }


class _OptunaPruningCallback(xgb.callback.TrainingCallback):
    """Report the early-stop-val logloss after each boosting round to Optuna.

    Raises the prune signal via the `pruned` flag instead of an exception so
    xgboost's training loop unwinds cleanly; the caller re-raises
    optuna.TrialPruned() after fit returns.
    """
    def __init__(self, trial: "optuna.Trial", observation_key: str = "validation_0-logloss"):
        super().__init__()
        self.trial = trial
        self.observation_key = observation_key
        self.pruned = False

    def after_iteration(self, model, epoch, evals_log):  # noqa: D401
        ds, metric = self.observation_key.split("-", 1)
        if ds in evals_log and metric in evals_log[ds]:
            score = evals_log[ds][metric][-1]
            self.trial.report(float(score), epoch)
            if self.trial.should_prune():
                self.pruned = True
                return True
        return False


def make_objective(T: int, splits: dict[str, pd.DataFrame]):
    """Return an Optuna objective closure: fit → early-stop on early_stop_val →
    score optuna_val log-loss. Reports intermediates to the pruner.

    Stores the best-iteration count on the trial via `set_user_attr` so
    fit_calibrated can recover it later.
    """
    X_tr, y_tr, _ = prepare_Xy(splits["train"], T)
    X_es, y_es, _ = prepare_Xy(splits["early_stop_val"], T)
    X_opt, y_opt, _ = prepare_Xy(splits["optuna_val"], T)

    def objective(trial: "optuna.Trial") -> float:
        params = _trial_params(trial)
        callback = _OptunaPruningCallback(trial)
        model = xgb.XGBClassifier(
            **_xgb_base_params(),
            **params,
            n_estimators=N_ESTIMATORS_MAX,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            callbacks=[callback],
        )
        model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
        if callback.pruned:
            raise optuna.TrialPruned()
        n_estimators_selected = int(model.best_iteration) + 1
        trial.set_user_attr("n_estimators_selected", n_estimators_selected)
        probs = model.predict_proba(X_opt)[:, 1]
        return float(log_loss(y_opt.values, probs, labels=[0, 1]))

    return objective


def tune_checkpoint(T: int, splits: dict[str, pd.DataFrame],
                    n_trials: int = N_TRIALS, seed: int = SEED) -> optuna.Study:
    """Run an Optuna study with TPE sampler + MedianPruner, 100 trials."""
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(make_objective(T, splits), n_trials=n_trials, show_progress_bar=False)
    return study


# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------

class _PlattCalibrator:
    """Logistic-regression wrapper that exposes .predict(probs) → calibrated probs.

    Platt fallback for pathological isotonic fits. LR is fit on logit(raw) →
    actuals so the single feature is unbounded and well-conditioned.
    """
    def __init__(self, lr: LogisticRegression):
        self.lr = lr

    @staticmethod
    def _logit(probs: np.ndarray) -> np.ndarray:
        clipped = np.clip(probs, 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(self._logit(np.asarray(probs, dtype=float)))[:, 1]


def _apply_calibrator(calibrator, raw_probs: np.ndarray) -> np.ndarray:
    """Chain raw xgb probs through the calibrator. `None` → passthrough."""
    if calibrator is None:
        return np.asarray(raw_probs, dtype=float)
    return np.asarray(calibrator.predict(raw_probs), dtype=float)


def _fit_platt(raw_probs: np.ndarray, y_true: np.ndarray) -> _PlattCalibrator:
    lr = LogisticRegression()
    lr.fit(_PlattCalibrator._logit(raw_probs), np.asarray(y_true).astype(int))
    return _PlattCalibrator(lr)


def fit_calibrated(T: int, study: optuna.Study, splits: dict[str, pd.DataFrame],
                   ece_threshold: float = CALIBRATION_ECE_THRESHOLD):
    """Refit best-trial on train+early_stop_val, then fit an optional calibrator
    on optuna_val raw predictions.

    Returns (model, calibrator, calibrator_kind, n_estimators_selected).
    calibrator is None | IsotonicRegression | _PlattCalibrator.
    calibrator_kind is None | "isotonic" | "platt".
    """
    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    n_estimators_selected = int(best_trial.user_attrs["n_estimators_selected"])

    combined = pd.concat([splits["train"], splits["early_stop_val"]], ignore_index=True)
    X_tc, y_tc, _ = prepare_Xy(combined, T)

    model = xgb.XGBClassifier(
        **_xgb_base_params(),
        **best_params,
        n_estimators=n_estimators_selected,
    )
    model.fit(X_tc, y_tc, verbose=False)

    X_opt, y_opt, _ = prepare_Xy(splits["optuna_val"], T)
    raw_opt_probs = model.predict_proba(X_opt)[:, 1]

    if ece_weighted(y_opt.values, raw_opt_probs) <= ece_threshold:
        return model, None, None, n_estimators_selected

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_opt_probs, y_opt.values)
    if isotonic_is_pathological(iso):
        return model, _fit_platt(raw_opt_probs, y_opt.values), "platt", n_estimators_selected
    return model, iso, "isotonic", n_estimators_selected


# ---------------------------------------------------------------------
# Artifacts / plotting / runs.csv
# ---------------------------------------------------------------------

def _plot_reliability(y_true, raw_probs, cal_probs, T: int, path: str,
                      title: str | None = None, n_bins: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfect calibration")

    raw_bins = reliability_bins(y_true, raw_probs, n_bins=n_bins)
    raw_ne = raw_bins[raw_bins["n"] > 0]
    ax.plot(raw_ne["mean_pred"], raw_ne["mean_actual"], "o-", label="raw")

    if cal_probs is not None:
        cal_bins = reliability_bins(y_true, cal_probs, n_bins=n_bins)
        cal_ne = cal_bins[cal_bins["n"] > 0]
        ax.plot(cal_ne["mean_pred"], cal_ne["mean_actual"], "s-", label="calibrated")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed win rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title or f"Reliability — xgboost_v1 T={T}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _feature_importance_df(model: xgb.XGBClassifier) -> pd.DataFrame:
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    cover = booster.get_score(importance_type="cover")
    feats = sorted(set(gain) | set(weight) | set(cover))
    df = pd.DataFrame({
        "feature": feats,
        "gain":    [float(gain.get(f, 0.0)) for f in feats],
        "weight":  [float(weight.get(f, 0.0)) for f in feats],
        "cover":   [float(cover.get(f, 0.0)) for f in feats],
    })
    return df.sort_values("gain", ascending=False).reset_index(drop=True)


def _append_runs_row(row: dict, runs_csv: str) -> None:
    """Append a runs.csv row, widening the schema if new columns are introduced.

    Read-concat-write: robust when baseline rows have fewer columns than xgb
    rows. runs.csv is small (~10s of rows) so the overhead is negligible.
    """
    os.makedirs(os.path.dirname(runs_csv), exist_ok=True)
    new_row = pd.DataFrame([row])
    if os.path.exists(runs_csv):
        existing = pd.read_csv(runs_csv)
        combined = pd.concat([existing, new_row], ignore_index=True, sort=False)
    else:
        combined = new_row
    combined.to_csv(runs_csv, index=False)


# ---------------------------------------------------------------------
# train_checkpoint — tune + calibrate + evaluate + write artifacts
# ---------------------------------------------------------------------

def train_checkpoint(T: int,
                     input_dir: str = INPUT_DIR,
                     output_dir: str = OUTPUT_DIR,
                     runs_csv: str = RUNS_CSV,
                     run_id: str | None = None,
                     n_trials: int = N_TRIALS) -> dict:
    """Tune → fit_calibrated → evaluate raw + cal on acceptance_val → gate → write.

    Writes: model.json, calibrator.pkl, study.pkl, metrics.json,
    feature_importance.csv, reliability_val.png. Appends one row to runs.csv.
    Returns the metrics dict that was written to disk.
    """
    df = pd.read_csv(os.path.join(input_dir, f"checkpoint_{T}.csv"), low_memory=False)
    splits = split_temporal(df)

    study = tune_checkpoint(T, splits, n_trials=n_trials)
    model, calibrator, calibrator_kind, n_estimators_selected = fit_calibrated(
        T, study, splits
    )

    X_acc, y_acc, _ = prepare_Xy(splits["acceptance_val"], T)
    raw_probs = model.predict_proba(X_acc)[:, 1]
    cal_probs = _apply_calibrator(calibrator, raw_probs)

    raw_metrics = evaluate(y_acc.values, raw_probs)
    cal_metrics = evaluate(y_acc.values, cal_probs)
    passed, diffs = check_acceptance(T, raw_metrics)

    metrics = {
        "best_params":            dict(study.best_trial.params),
        "n_estimators_selected":  int(n_estimators_selected),
        "optuna_val_n":           int(len(splits["optuna_val"])),
        "acceptance_val_n":       int(len(X_acc)),
        "raw":                    raw_metrics,
        "cal":                    cal_metrics,
        "calibration_applied":    calibrator is not None,
        "calibrator_kind":        calibrator_kind,
        "acceptance_passed":      passed,
        "acceptance_diffs":       diffs,
    }

    ckpt_dir = os.path.join(output_dir, f"checkpoint_{T}")
    os.makedirs(ckpt_dir, exist_ok=True)

    model.save_model(os.path.join(ckpt_dir, "model.json"))
    with open(os.path.join(ckpt_dir, "calibrator.pkl"), "wb") as f:
        pickle.dump(calibrator, f)
    with open(os.path.join(ckpt_dir, "study.pkl"), "wb") as f:
        pickle.dump(study, f)
    with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    _feature_importance_df(model).to_csv(
        os.path.join(ckpt_dir, "feature_importance.csv"), index=False
    )
    _plot_reliability(
        y_acc.values, raw_probs, cal_probs if calibrator is not None else None, T,
        os.path.join(ckpt_dir, "reliability_val.png"),
        title=f"Reliability — xgboost_v1 T={T} on acceptance val (2025-H2)",
    )

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ") + "_xgboost_v1"
    _append_runs_row({
        "run_id":          run_id,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "model_type":      "xgboost_v1",
        "checkpoint":      T,
        "hparams_json":    json.dumps({
            **metrics["best_params"],
            "n_estimators_selected": metrics["n_estimators_selected"],
        }),
        "train_n":         int(len(splits["train"]) + len(splits["early_stop_val"])),
        "val_n":           metrics["acceptance_val_n"],
        "val_logloss":     raw_metrics["log_loss"],
        "val_brier":       raw_metrics["brier"],
        "val_auc":         raw_metrics["auc"],
        "val_accuracy":    raw_metrics["accuracy"],
        "val_logloss_cal": cal_metrics["log_loss"],
        "val_brier_cal":   cal_metrics["brier"],
        "notes":           f"acceptance_passed={passed}",
    }, runs_csv)

    return metrics


# ---------------------------------------------------------------------
# --evaluate-test — atomic across all 4 checkpoints
# ---------------------------------------------------------------------

def _load_val_stage(ckpt_dir: str):
    model = xgb.XGBClassifier(enable_categorical=True)
    model.load_model(os.path.join(ckpt_dir, "model.json"))
    with open(os.path.join(ckpt_dir, "calibrator.pkl"), "rb") as f:
        calibrator = pickle.load(f)
    with open(os.path.join(ckpt_dir, "metrics.json")) as f:
        metrics = json.load(f)
    return model, calibrator, metrics


def _evaluate_test_preconditions(output_dir: str) -> list[str]:
    errors: list[str] = []
    for T in CHECKPOINTS:
        metrics_path = os.path.join(output_dir, f"checkpoint_{T}", "metrics.json")
        if not os.path.exists(metrics_path):
            errors.append(
                f"T={T}: metrics.json missing — run `python -m scripts.models.xgboost_v1` first."
            )
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        if not m.get("acceptance_passed", False):
            errors.append(
                f"T={T}: acceptance_passed=False — investigate before touching 2026. "
                f"diffs={m.get('acceptance_diffs')}"
            )
        if "test" in m or "test_n" in m:
            errors.append(
                f"T={T}: 2026 already evaluated (metrics.json has a `test` block). "
                "The test set is touched once. To re-run deliberately, delete the "
                "`test` block from metrics.json and reliability_test.png."
            )
    return errors


def evaluate_test_all(input_dir: str = INPUT_DIR,
                      output_dir: str = OUTPUT_DIR) -> dict[int, dict]:
    """Predict on 2026 for all 4 checkpoints atomically. Touched once.

    Preconditions (checked before any writes): all 4 metrics.json exist, all
    acceptance_passed, none already has a test block. Any failure → SystemExit.
    """
    errors = _evaluate_test_preconditions(output_dir)
    if errors:
        raise SystemExit(
            "--evaluate-test preconditions failed:\n  " + "\n  ".join(errors)
        )

    results: dict[int, dict] = {}
    for T in CHECKPOINTS:
        ckpt_dir = os.path.join(output_dir, f"checkpoint_{T}")
        model, calibrator, metrics = _load_val_stage(ckpt_dir)

        df = pd.read_csv(os.path.join(input_dir, f"checkpoint_{T}.csv"), low_memory=False)
        splits = split_temporal(df)
        X_test, y_test, _ = prepare_Xy(splits["test"], T)
        raw_probs = model.predict_proba(X_test)[:, 1]
        cal_probs = _apply_calibrator(calibrator, raw_probs)

        raw_test = evaluate(y_test.values, raw_probs)
        cal_test = evaluate(y_test.values, cal_probs)

        metrics["test_n"] = int(len(X_test))
        metrics["test"] = {"raw": raw_test, "cal": cal_test}
        with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        _plot_reliability(
            y_test.values, raw_probs, cal_probs if calibrator is not None else None, T,
            os.path.join(ckpt_dir, "reliability_test.png"),
            title=f"Reliability — xgboost_v1 T={T} on 2026 test",
        )
        results[T] = {"raw": raw_test, "cal": cal_test, "n": int(len(X_test))}
    return results


# ---------------------------------------------------------------------
# --fit-production — final refit for deployment
# ---------------------------------------------------------------------

def fit_production(T: int,
                   input_dir: str = INPUT_DIR,
                   output_dir: str = OUTPUT_DIR) -> dict:
    """Second early-stopping pass → n_estimators_prod, then final refit on all
    data through 2025-12-31, then carry calibrator over from val stage.

    Writes model.json, calibrator.pkl, metadata.json under production/.
    """
    val_dir = os.path.join(output_dir, f"checkpoint_{T}")
    with open(os.path.join(val_dir, "metrics.json")) as f:
        val_metrics = json.load(f)
    with open(os.path.join(val_dir, "calibrator.pkl"), "rb") as f:
        calibrator = pickle.load(f)
    best_params = dict(val_metrics["best_params"])

    df = pd.read_csv(os.path.join(input_dir, f"checkpoint_{T}.csv"), low_memory=False)
    splits = split_temporal(df)

    # Step 2: second early-stopping pass — train 2020..2025-06-30, es on acceptance_val.
    es_train_df = pd.concat(
        [splits["train"], splits["early_stop_val"], splits["optuna_val"]],
        ignore_index=True,
    )
    X_es_tr, y_es_tr, _ = prepare_Xy(es_train_df, T)
    X_es_val, y_es_val, _ = prepare_Xy(splits["acceptance_val"], T)
    es_model = xgb.XGBClassifier(
        **_xgb_base_params(),
        **best_params,
        n_estimators=N_ESTIMATORS_MAX,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    es_model.fit(X_es_tr, y_es_tr, eval_set=[(X_es_val, y_es_val)], verbose=False)
    n_estimators_prod = int(es_model.best_iteration) + 1

    # Step 3: final refit on all data 2020..2025-12-31 with n_estimators_prod locked.
    all_train_df = pd.concat(
        [splits["train"], splits["early_stop_val"], splits["optuna_val"], splits["acceptance_val"]],
        ignore_index=True,
    )
    X_all, y_all, _ = prepare_Xy(all_train_df, T)
    prod_model = xgb.XGBClassifier(
        **_xgb_base_params(),
        **best_params,
        n_estimators=n_estimators_prod,
    )
    prod_model.fit(X_all, y_all, verbose=False)

    prod_ckpt = os.path.join(output_dir, "production", f"checkpoint_{T}")
    os.makedirs(prod_ckpt, exist_ok=True)
    prod_model.save_model(os.path.join(prod_ckpt, "model.json"))
    with open(os.path.join(prod_ckpt, "calibrator.pkl"), "wb") as f:
        pickle.dump(calibrator, f)

    metadata = {
        "checkpoint":           T,
        "best_params":          best_params,
        "n_estimators_prod":    n_estimators_prod,
        "train_n":              int(len(X_all)),
        "fit_timestamp":        datetime.now(timezone.utc).isoformat(),
        "calibrator_source":    "val_stage_carried_over",
        "calibrator_kind":      val_metrics.get("calibrator_kind"),
        "val_stage_cal_brier":  val_metrics.get("cal", {}).get("brier"),
    }
    with open(os.path.join(prod_ckpt, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--runs-csv", default=RUNS_CSV)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS,
                        help="Optuna trials per checkpoint (default 100).")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Atomic 2026 eval across all 4 T (touched once).")
    parser.add_argument("--fit-production", action="store_true",
                        help="Final production refit for deployment.")
    args = parser.parse_args()

    if args.evaluate_test and args.fit_production:
        raise SystemExit("--evaluate-test and --fit-production are mutually exclusive.")

    if args.evaluate_test:
        results = evaluate_test_all(args.input_dir, args.output_dir)
        for T, r in results.items():
            print(f"T={T}  n={r['n']:,}  "
                  f"raw_logloss={r['raw']['log_loss']:.4f}  raw_brier={r['raw']['brier']:.4f}  "
                  f"cal_logloss={r['cal']['log_loss']:.4f}  cal_brier={r['cal']['brier']:.4f}")
        return

    if args.fit_production:
        for T in CHECKPOINTS:
            md = fit_production(T, args.input_dir, args.output_dir)
            print(f"T={T}  n_estimators_prod={md['n_estimators_prod']}  "
                  f"train_n={md['train_n']:,}  calibrator_kind={md['calibrator_kind']}")
        return

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ") + "_xgboost_v1"
    print(f"run_id = {run_id}")
    for T in CHECKPOINTS:
        print(f"\n=== checkpoint T={T} ===")
        m = train_checkpoint(T, args.input_dir, args.output_dir, args.runs_csv,
                             run_id=run_id, n_trials=args.n_trials)
        print(f"  best_params={m['best_params']}")
        print(f"  n_estimators_selected={m['n_estimators_selected']}  "
              f"acceptance_val_n={m['acceptance_val_n']:,}")
        print(f"  raw:  logloss={m['raw']['log_loss']:.4f}  brier={m['raw']['brier']:.4f}  "
              f"auc={m['raw']['auc']:.4f}  acc={m['raw']['accuracy']:.4f}  "
              f"ece={m['raw']['ece']:.4f}")
        print(f"  cal:  logloss={m['cal']['log_loss']:.4f}  brier={m['cal']['brier']:.4f}  "
              f"auc={m['cal']['auc']:.4f}  acc={m['cal']['accuracy']:.4f}  "
              f"ece={m['cal']['ece']:.4f}  ({m['calibrator_kind']})")
        print(f"  acceptance_passed={m['acceptance_passed']}  diffs={m['acceptance_diffs']}")


if __name__ == "__main__":
    main()

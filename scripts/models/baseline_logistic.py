"""
Baseline logistic regression per checkpoint.

Diagnostic lower bound — gives a floor that XGBoost must clearly beat. If it
doesn't, investigate bug / leakage / feature set before further model work.

Default run trains the sibling model (2020-2024) scored on 2025 val. 2026 is
held out for --evaluate-test. --fit-production refits on 2020-2025 with an
optional isotonic calibrator. See docs/baseline-logistic.md.

Inputs:  data/processed/checkpoint_{10,15,20,25}.csv
Outputs (sibling): data/models/baseline_logistic/checkpoint_{T}/{model.pkl,
         metrics.json, coefficients.csv, reliability.png, metrics_2026.json?,
         reliability_2026.png?}
Outputs (production): data/models/baseline_logistic/production/checkpoint_{T}/{
         model.pkl, calibrator.pkl, metadata.json, metrics_2026.json?,
         reliability_2026.png?}
         data/experiments/runs.csv (appended)

CLI:
  python -m scripts.models.baseline_logistic                       # train sibling
  python -m scripts.models.baseline_logistic --fit-production      # production refit
  python -m scripts.models.baseline_logistic --evaluate-test       # atomic sibling 2026 eval, once
  python -m scripts.models.baseline_logistic --evaluate-test-production  # atomic production 2026 eval, once
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
import pandas as pd  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegressionCV  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_DIR = os.path.join(REPO_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "models", "baseline_logistic")
PROD_DIR = os.path.join(OUTPUT_DIR, "production")
RUNS_CSV = os.path.join(REPO_ROOT, "data", "experiments", "runs.csv")

CHECKPOINTS = (10, 15, 20, 25)
C_GRID = (0.01, 0.1, 1, 10, 100)
CV_SEED = 42

TIER_CATEGORIES = ["S", "A", "B"]  # S = baseline; dummies for A and B

FIRST_GATES = {
    10: ["firstblood"],
    15: ["firstblood", "firstdragon", "firsttower"],
    20: ["firstblood", "firstdragon", "firsttower"],
    25: ["firstblood", "firstdragon", "firsttower", "firstbaron"],
}

# Anything in X that is NOT in this set is numeric and gets standardized.
_NON_NUMERIC_FEATURE_NAMES = frozenset({
    "side_is_blue",
    "firstblood", "firstdragon", "firsttower", "firstbaron",
    "tier_A", "tier_B",
})

# Production refit cutoffs.
H1_START = "2025-01-01"
H1_END   = "2025-07-01"
H2_END   = "2026-01-01"
TRAIN_START = "2020-01-01"
TEST_START = "2026-01-01"
TEST_END   = "2027-01-01"

# If isotonic regresses H2 log-loss by more than this during production fit,
# fall back to no calibrator. Matches xgboost_v1's observed pattern.
FALLBACK_LOGLOSS_MARGIN = 0.005


def NUMERIC_FEATURES(T: int) -> list[str]:
    return [
        f"goldat{T}", f"opp_goldat{T}",
        f"csat{T}", f"opp_csat{T}",
        f"killsat{T}", f"opp_killsat{T}",
        f"assistsat{T}", f"opp_assistsat{T}",
    ]


def BINARY_FEATURES(T: int) -> list[str]:
    return ["side_is_blue"] + FIRST_GATES[T]


def prepare_Xy(df: pd.DataFrame, T: int) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build (X, y, feature_names) for checkpoint T.

    Drops the collinear *diffat{T} columns. tier is one-hot with S as baseline
    via explicit Categorical ordering (plain drop_first drops alphabetically
    first — A — which is not what we want).
    """
    X = pd.DataFrame(index=df.index)
    for col in NUMERIC_FEATURES(T):
        X[col] = df[col].astype(float)

    X["side_is_blue"] = (df["side"] == "Blue").astype(int)
    for flag in FIRST_GATES[T]:
        X[flag] = df[flag].astype(int)

    tier = pd.Categorical(df["tier"], categories=TIER_CATEGORIES)
    tier_dummies = pd.get_dummies(tier, prefix="tier", drop_first=True).astype(int)
    tier_dummies.index = df.index
    for col in tier_dummies.columns:
        X[col] = tier_dummies[col]

    y = df["result"].astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)
    return X, y, list(X.columns)


def split_train_val(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """year ≤ 2024 → train; year == 2025 → val; year == 2026 → dropped."""
    train = df[df["year"] <= 2024].reset_index(drop=True)
    val = df[df["year"] == 2025].reset_index(drop=True)
    return train, val


def numeric_cols_in(X: pd.DataFrame) -> list[str]:
    """Single source of truth: numeric = any X column not in the non-numeric set.

    Used both at fit time (scaler + model training) and at predict time
    (re-applying the scaler), so the two pipelines can't drift.
    """
    return [c for c in X.columns if c not in _NON_NUMERIC_FEATURE_NAMES]


def fit(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[StandardScaler, LogisticRegressionCV]:
    """Fit StandardScaler on train numeric cols, then LogisticRegressionCV on the
    combined (scaled-numeric + passthrough-binary/tier) feature matrix.

    Internal 5-fold CV over C_GRID picks C; val is never touched.
    """
    numeric_cols = numeric_cols_in(X_train)

    scaler = StandardScaler()
    scaler.fit(X_train[numeric_cols].values)

    X_fit = X_train.copy()
    X_fit[numeric_cols] = scaler.transform(X_train[numeric_cols].values)

    model = LogisticRegressionCV(
        Cs=list(C_GRID),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=CV_SEED),
        penalty="l2",
        solver="lbfgs",
        scoring="neg_log_loss",
        max_iter=1000,
        refit=True,
    )
    model.fit(X_fit.values, y_train.values)
    return scaler, model


def predict_proba(scaler: StandardScaler, model: LogisticRegressionCV,
                  X: pd.DataFrame, numeric_cols: list[str]) -> np.ndarray:
    """Apply train-fit scaler to numeric cols, return P(result=1)."""
    X = X.copy()
    X[numeric_cols] = scaler.transform(X[numeric_cols].values)
    return model.predict_proba(X.values)[:, 1]


def evaluate(y_true, probs) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = (probs >= 0.5).astype(int)
    auc = float(roc_auc_score(y_true, probs)) if len(set(y_true.tolist())) > 1 else float("nan")
    return {
        "log_loss": float(log_loss(y_true, probs, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probs)),
        "auc": auc,
        "accuracy": float((preds == y_true).mean()),
    }


def reliability_bins(y_true, probs, n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
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
                         "mean_pred": np.nan, "mean_actual": np.nan})
        else:
            rows.append({
                "bin_lo": float(lo), "bin_hi": float(hi), "n": n,
                "mean_pred": float(probs[mask].mean()),
                "mean_actual": float(y_true[mask].mean()),
            })
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


def _plot_reliability(bins: pd.DataFrame, path: str, label: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfect calibration")
    nonempty = bins[bins["n"] > 0]
    ax.plot(nonempty["mean_pred"], nonempty["mean_actual"], "o-", label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed win rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _slice_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = pd.to_datetime(df["date"])
    return df[(dates >= start) & (dates < end)].reset_index(drop=True)


def _append_runs_row(row: dict, runs_csv: str) -> None:
    os.makedirs(os.path.dirname(runs_csv), exist_ok=True)
    new_row = pd.DataFrame([row])
    if os.path.exists(runs_csv):
        existing = pd.read_csv(runs_csv)
        combined = pd.concat([existing, new_row], ignore_index=True, sort=False)
    else:
        combined = new_row
    combined.to_csv(runs_csv, index=False)


# ---------------------------------------------------------------------
# train_checkpoint — sibling fit (2020-2024 train, 2025 val)
# ---------------------------------------------------------------------

def train_checkpoint(T: int,
                     input_dir: str = INPUT_DIR,
                     output_dir: str = OUTPUT_DIR,
                     runs_csv: str = RUNS_CSV,
                     run_id: str | None = None) -> dict:
    """Train baseline at checkpoint T, write all 4 artifacts, append runs.csv."""
    input_path = os.path.join(input_dir, f"checkpoint_{T}.csv")
    df = pd.read_csv(input_path, low_memory=False)

    train_df, val_df = split_train_val(df)
    X_train, y_train, feature_names = prepare_Xy(train_df, T)
    X_val, y_val, _ = prepare_Xy(val_df, T)

    scaler, model = fit(X_train, y_train)
    C_selected = float(model.C_[0])

    numeric_cols = numeric_cols_in(X_train)
    probs = predict_proba(scaler, model, X_val, numeric_cols)
    metrics = evaluate(y_val, probs)
    metrics["C_selected"] = C_selected
    metrics["train_n"] = int(len(X_train))
    metrics["val_n"] = int(len(X_val))

    ckpt_dir = os.path.join(output_dir, f"checkpoint_{T}")
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(ckpt_dir, "model.pkl"), "wb") as f:
        pickle.dump({"scaler": scaler, "model": model,
                     "feature_names": feature_names,
                     "numeric_cols": numeric_cols}, f)

    with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    coefs = pd.DataFrame({"feature": feature_names, "coefficient": model.coef_[0]})
    coefs["abs_coefficient"] = coefs["coefficient"].abs()
    coefs = coefs.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    coefs.to_csv(os.path.join(ckpt_dir, "coefficients.csv"), index=False)

    bins = reliability_bins(y_val, probs, n_bins=10)
    _plot_reliability(bins, os.path.join(ckpt_dir, "reliability.png"),
                      label=f"baseline T={T}",
                      title=f"Reliability — baseline_logistic T={T}")

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ") + "_baseline_logistic"
    _append_runs_row({
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_type": "baseline_logistic",
        "checkpoint": T,
        "hparams_json": json.dumps({
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
            "Cs": list(C_GRID),
            "cv": 5,
            "scoring": "neg_log_loss",
            "C_selected": C_selected,
        }),
        "train_n": metrics["train_n"],
        "val_n": metrics["val_n"],
        "val_logloss": metrics["log_loss"],
        "val_brier": metrics["brier"],
        "val_auc": metrics["auc"],
        "val_accuracy": metrics["accuracy"],
        "notes": "",
    }, runs_csv)

    return metrics


# ---------------------------------------------------------------------
# --fit-production — refit on 2020-2025 with optional isotonic calibrator
# ---------------------------------------------------------------------

def fit_production(T: int,
                   input_dir: str = INPUT_DIR,
                   output_dir: str = OUTPUT_DIR) -> dict:
    """Production refit. Isotonic on sibling's 2025-H1 preds, evaluate on H2,
    drop calibrator if it regresses H2 logloss. Refit on 2020-2025; carry
    calibrator over unchanged.
    """
    sibling_dir = os.path.join(output_dir, f"checkpoint_{T}")
    with open(os.path.join(sibling_dir, "model.pkl"), "rb") as f:
        sibling = pickle.load(f)
    sib_scaler, sib_model = sibling["scaler"], sibling["model"]

    df = pd.read_csv(os.path.join(input_dir, f"checkpoint_{T}.csv"), low_memory=False)

    h1 = _slice_by_date(df, H1_START, H1_END)
    X_h1, y_h1, _ = prepare_Xy(h1, T)
    num_cols_h1 = numeric_cols_in(X_h1)
    probs_h1 = predict_proba(sib_scaler, sib_model, X_h1, num_cols_h1)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs_h1, y_h1.values)

    h2 = _slice_by_date(df, H1_END, H2_END)
    X_h2, y_h2, _ = prepare_Xy(h2, T)
    probs_h2_raw = predict_proba(sib_scaler, sib_model, X_h2, num_cols_h1)
    probs_h2_cal = np.asarray(iso.predict(probs_h2_raw))

    h2_raw_ll  = float(log_loss(y_h2, probs_h2_raw, labels=[0, 1]))
    h2_cal_ll  = float(log_loss(y_h2, probs_h2_cal, labels=[0, 1]))
    h2_raw_br  = float(brier_score_loss(y_h2, probs_h2_raw))
    h2_cal_br  = float(brier_score_loss(y_h2, probs_h2_cal))
    h2_raw_ece = float(ece_weighted(y_h2.values, probs_h2_raw))
    h2_cal_ece = float(ece_weighted(y_h2.values, probs_h2_cal))

    calibrator = iso
    calibrator_kind: str | None = "isotonic"
    fallback_reason: str | None = None
    if h2_cal_ll - h2_raw_ll > FALLBACK_LOGLOSS_MARGIN:
        calibrator = None
        calibrator_kind = None
        fallback_reason = (
            f"isotonic regresses H2 log-loss by {h2_cal_ll - h2_raw_ll:.4f} "
            f"(> {FALLBACK_LOGLOSS_MARGIN}); carrying None instead."
        )

    train_df = _slice_by_date(df, TRAIN_START, H2_END)
    X_tr, y_tr, feature_names = prepare_Xy(train_df, T)
    prod_scaler, prod_model = fit(X_tr, y_tr)
    numeric_cols = numeric_cols_in(X_tr)

    prod_ckpt = os.path.join(output_dir, "production", f"checkpoint_{T}")
    os.makedirs(prod_ckpt, exist_ok=True)

    with open(os.path.join(prod_ckpt, "model.pkl"), "wb") as f:
        pickle.dump({
            "scaler": prod_scaler,
            "model": prod_model,
            "feature_names": feature_names,
            "numeric_cols": numeric_cols,
        }, f)
    with open(os.path.join(prod_ckpt, "calibrator.pkl"), "wb") as f:
        pickle.dump(calibrator, f)

    metadata = {
        "checkpoint":                 T,
        "train_n":                    int(len(X_tr)),
        "train_range":                f"{TRAIN_START}..{H2_END}",
        "C_selected":                 float(prod_model.C_[0]),
        "fit_timestamp":              datetime.now(timezone.utc).isoformat(),
        "calibrator_kind":            calibrator_kind,
        "calibrator_fallback_reason": fallback_reason,
        "calibration_h1_n":           int(len(X_h1)),
        "calibration_h2_n":           int(len(X_h2)),
        "h2_raw_logloss":             h2_raw_ll,
        "h2_cal_logloss":             h2_cal_ll,
        "h2_raw_brier":               h2_raw_br,
        "h2_cal_brier":               h2_cal_br,
        "h2_raw_ece":                 h2_raw_ece,
        "h2_cal_ece":                 h2_cal_ece,
    }
    with open(os.path.join(prod_ckpt, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


# ---------------------------------------------------------------------
# --evaluate-test — sibling (2020-2024 model) scored on 2026
# --evaluate-test-production — production (2020-2025 model) scored on 2026
#
# Both are atomic across all 4 checkpoints and one-shot (runs.csv lock).
# ---------------------------------------------------------------------

def _check_2026_lock(runs_csv: str, notes_value: str) -> None:
    """Refuse to re-run if runs.csv already has rows with notes == `notes_value`.

    Exact match (not substring) so 'sibling 2026 test' and '2026 test production'
    don't cross-trigger each other.
    """
    if not os.path.exists(runs_csv):
        return
    runs = pd.read_csv(runs_csv)
    existing = runs[(runs["model_type"] == "baseline_logistic")
                    & (runs["notes"].fillna("") == notes_value)]
    if len(existing):
        raise SystemExit(
            f"runs.csv already has {len(existing)} baseline_logistic row(s) with "
            f"notes='{notes_value}' — this eval is one-shot. To re-run "
            "deliberately, delete those rows and the matching metrics_2026.json / "
            "reliability_2026.png files, then invoke again."
        )


def _eval_sibling_2026(T: int, input_dir: str, output_dir: str) -> tuple[dict, dict]:
    ckpt_dir = os.path.join(output_dir, f"checkpoint_{T}")
    with open(os.path.join(ckpt_dir, "model.pkl"), "rb") as f:
        payload = pickle.load(f)
    scaler, model = payload["scaler"], payload["model"]

    df = pd.read_csv(os.path.join(input_dir, f"checkpoint_{T}.csv"), low_memory=False)
    test = _slice_by_date(df, TEST_START, TEST_END)
    X, y, _ = prepare_Xy(test, T)
    numeric_cols = numeric_cols_in(X)
    probs = predict_proba(scaler, model, X, numeric_cols)

    metrics = evaluate(y, probs)
    metrics["ece"] = ece_weighted(y.values, probs, n_bins=10)
    metrics["test_n"] = int(len(X))
    metrics["C_selected"] = float(model.C_[0])

    with open(os.path.join(ckpt_dir, "metrics_2026.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    bins = reliability_bins(y, probs, n_bins=10)
    _plot_reliability(bins, os.path.join(ckpt_dir, "reliability_2026.png"),
                      label=f"baseline_logistic T={T} on 2026",
                      title=f"Reliability — baseline_logistic T={T} on 2026 test")

    with open(os.path.join(ckpt_dir, "metrics.json")) as f:
        orig = json.load(f)
    return metrics, {"train_n": int(orig["train_n"]),
                     "C_selected": float(model.C_[0])}


def evaluate_test_sibling_all(input_dir: str = INPUT_DIR,
                              output_dir: str = OUTPUT_DIR,
                              runs_csv: str = RUNS_CSV) -> dict[int, dict]:
    """Atomic 2026 eval of the sibling model across all 4 checkpoints. Once."""
    _check_2026_lock(runs_csv, "2026 test")
    run_id = (datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
              + "_baseline_logistic_2026")
    results: dict[int, dict] = {}
    for T in CHECKPOINTS:
        m, ctx = _eval_sibling_2026(T, input_dir, output_dir)
        _append_runs_row({
            "run_id":       run_id,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "model_type":   "baseline_logistic",
            "checkpoint":   T,
            "hparams_json": json.dumps({
                "penalty": "l2", "solver": "lbfgs", "max_iter": 1000,
                "Cs": list(C_GRID), "cv": 5, "scoring": "neg_log_loss",
                "C_selected": ctx["C_selected"],
            }),
            "train_n":      ctx["train_n"],
            "val_n":        m["test_n"],
            "val_logloss":  m["log_loss"],
            "val_brier":    m["brier"],
            "val_auc":      m["auc"],
            "val_accuracy": m["accuracy"],
            "notes":        "2026 test",
        }, runs_csv)
        results[T] = m
    return results


def _eval_production_2026(T: int, input_dir: str, prod_dir: str) -> tuple[dict, dict]:
    ckpt_dir = os.path.join(prod_dir, f"checkpoint_{T}")
    with open(os.path.join(ckpt_dir, "model.pkl"), "rb") as f:
        payload = pickle.load(f)
    with open(os.path.join(ckpt_dir, "calibrator.pkl"), "rb") as f:
        calibrator = pickle.load(f)
    with open(os.path.join(ckpt_dir, "metadata.json")) as f:
        meta = json.load(f)
    scaler, model = payload["scaler"], payload["model"]
    calibrator_kind = meta.get("calibrator_kind")

    df = pd.read_csv(os.path.join(input_dir, f"checkpoint_{T}.csv"), low_memory=False)
    test = _slice_by_date(df, TEST_START, TEST_END)
    X, y, _ = prepare_Xy(test, T)
    numeric_cols = numeric_cols_in(X)
    probs_raw = predict_proba(scaler, model, X, numeric_cols)
    probs_cal = (np.asarray(calibrator.predict(probs_raw))
                 if calibrator is not None else probs_raw)

    metrics_raw = evaluate(y, probs_raw)
    metrics_raw["ece"] = ece_weighted(y.values, probs_raw, n_bins=10)
    metrics_cal = evaluate(y, probs_cal)
    metrics_cal["ece"] = ece_weighted(y.values, probs_cal, n_bins=10)

    out = {
        "checkpoint":      T,
        "test_n":          int(len(X)),
        "calibrator_kind": calibrator_kind,
        "C_selected":      float(model.C_[0]),
        "raw":             metrics_raw,
        "calibrated":      metrics_cal,
    }
    with open(os.path.join(ckpt_dir, "metrics_2026.json"), "w") as f:
        json.dump(out, f, indent=2)

    probs_final = probs_cal
    bins = reliability_bins(y, probs_final, n_bins=10)
    _plot_reliability(
        bins,
        os.path.join(ckpt_dir, "reliability_2026.png"),
        label=f"production T={T} on 2026 (cal={calibrator_kind or 'none'})",
        title=f"Reliability — production baseline_logistic T={T} on 2026",
    )
    return out, meta


def evaluate_test_production_all(input_dir: str = INPUT_DIR,
                                 output_dir: str = OUTPUT_DIR,
                                 runs_csv: str = RUNS_CSV) -> dict[int, dict]:
    """Atomic 2026 eval of the production model across all 4 checkpoints. Once."""
    _check_2026_lock(runs_csv, "2026 test production")
    prod_dir = os.path.join(output_dir, "production")
    run_id = (datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
              + "_baseline_logistic_production_2026")
    results: dict[int, dict] = {}
    for T in CHECKPOINTS:
        out, meta = _eval_production_2026(T, input_dir, prod_dir)
        served = out["calibrated"] if out["calibrator_kind"] is not None else out["raw"]
        _append_runs_row({
            "run_id":       run_id,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "model_type":   "baseline_logistic",
            "checkpoint":   T,
            "hparams_json": json.dumps({
                "penalty": "l2", "solver": "lbfgs", "max_iter": 1000,
                "C_selected":      out["C_selected"],
                "train_range":     meta.get("train_range"),
                "calibrator_kind": out["calibrator_kind"],
            }),
            "train_n":      int(meta["train_n"]),
            "val_n":        out["test_n"],
            "val_logloss":  served["log_loss"],
            "val_brier":    served["brier"],
            "val_auc":      served["auc"],
            "val_accuracy": served["accuracy"],
            "notes":        "2026 test production",
        }, runs_csv)
        results[T] = out
    return results


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--runs-csv", default=RUNS_CSV)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fit-production", action="store_true",
                      help="Refit on 2020-2025 + optional isotonic; writes production/ artifacts.")
    mode.add_argument("--evaluate-test", action="store_true",
                      help="Atomic sibling 2026 eval across all 4 T (touched once).")
    mode.add_argument("--evaluate-test-production", action="store_true",
                      help="Atomic production 2026 eval across all 4 T (touched once).")
    args = parser.parse_args()

    if args.fit_production:
        print(f"{'T':>3} {'train_n':>8} {'C':>8} {'h2_raw_ll':>9} {'h2_cal_ll':>9} "
              f"{'h2_raw_ece':>10} {'h2_cal_ece':>10} calibrator")
        for T in CHECKPOINTS:
            m = fit_production(T, args.input_dir, args.output_dir)
            print(f"{T:>3} {m['train_n']:>8,} {m['C_selected']:>8.4f} "
                  f"{m['h2_raw_logloss']:>9.4f} {m['h2_cal_logloss']:>9.4f} "
                  f"{m['h2_raw_ece']:>10.4f} {m['h2_cal_ece']:>10.4f} "
                  f"{m['calibrator_kind'] or 'none'}"
                  + (f"  ({m['calibrator_fallback_reason']})"
                     if m['calibrator_fallback_reason'] else ""))
        return

    if args.evaluate_test:
        results = evaluate_test_sibling_all(args.input_dir, args.output_dir, args.runs_csv)
        print(f"{'T':>3} {'n':>6} {'logloss':>9} {'brier':>7} {'auc':>7} {'acc':>7} {'ece':>7}")
        for T, m in results.items():
            print(f"{T:>3} {m['test_n']:>6,} {m['log_loss']:>9.4f} {m['brier']:>7.4f} "
                  f"{m['auc']:>7.4f} {m['accuracy']:>7.4f} {m['ece']:>7.4f}")
        return

    if args.evaluate_test_production:
        results = evaluate_test_production_all(args.input_dir, args.output_dir, args.runs_csv)
        print(f"{'T':>3} {'n':>6} {'raw_ll':>7} {'cal_ll':>7} {'raw_br':>7} {'cal_br':>7} "
              f"{'raw_ece':>8} {'cal_ece':>8} calibrator")
        for T, m in results.items():
            print(f"{T:>3} {m['test_n']:>6,} "
                  f"{m['raw']['log_loss']:>7.4f} {m['calibrated']['log_loss']:>7.4f} "
                  f"{m['raw']['brier']:>7.4f} {m['calibrated']['brier']:>7.4f} "
                  f"{m['raw']['ece']:>8.4f} {m['calibrated']['ece']:>8.4f} "
                  f"{m['calibrator_kind'] or 'none'}")
        return

    # Default: train sibling per checkpoint.
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ") + "_baseline_logistic"
    print(f"run_id = {run_id}")
    for T in CHECKPOINTS:
        print(f"\n=== checkpoint T={T} ===")
        m = train_checkpoint(T, args.input_dir, args.output_dir, args.runs_csv, run_id=run_id)
        print(f"  C_selected={m['C_selected']:g}  train_n={m['train_n']:,}  val_n={m['val_n']:,}")
        print(f"  log_loss={m['log_loss']:.4f}  brier={m['brier']:.4f}  "
              f"auc={m['auc']:.4f}  acc={m['accuracy']:.4f}")


if __name__ == "__main__":
    main()

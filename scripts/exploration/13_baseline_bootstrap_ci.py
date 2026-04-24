"""Bootstrap 95% CIs on baseline_logistic evaluation metrics.

Quantifies evaluation-set sampling noise. Answers: when the sibling-vs-production
logloss differ by 0.04 or XGB-vs-baseline by 0.001, is that inside or outside
the CI band?

Resamples the eval set with replacement (not retraining — uncertainty we care
about is eval-set sampling variability, and this is fast). B=2000, seed=42.

Evaluations per checkpoint T:
    - sibling baseline (2020-2024) on 2025-H2
    - sibling baseline (2020-2024) on 2026
    - production baseline (2020-2025, w/ calibrator if shipped) on 2026

Output: data/experiments/13_baseline_bootstrap_ci.csv with (T, eval, metric,
point, mean, ci_lo, ci_hi, n_samples_used) rows.
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from scripts.models.baseline_logistic import (
    CHECKPOINTS,
    INPUT_DIR,
    OUTPUT_DIR,
    numeric_cols_in,
    predict_proba,
    prepare_Xy,
)
from scripts.models.xgboost_v1 import ece_weighted


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_CSV = os.path.join(REPO_ROOT, "data", "exploration",
                       "13_baseline_bootstrap_ci.csv")
PROD_DIR = os.path.join(OUTPUT_DIR, "production")

B = 2000
SEED = 42


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Five metrics; AUC is NaN when y is single-class in a bootstrap draw."""
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    preds = (p >= 0.5).astype(int)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier":    float(brier_score_loss(y, p)),
        "auc":      auc,
        "accuracy": float((preds == y).mean()),
        "ece":      float(ece_weighted(y, p, n_bins=10)),
    }


def _bootstrap(y: np.ndarray, p: np.ndarray, b: int = B, seed: int = SEED) -> dict:
    """Return {metric: (mean, ci_lo, ci_hi, n_used)} over b resamples."""
    rng = np.random.default_rng(seed)
    n = len(y)
    samples: dict[str, list[float]] = {
        k: [] for k in ("log_loss", "brier", "auc", "accuracy", "ece")
    }
    for _ in range(b):
        idx = rng.integers(0, n, size=n)
        m = _metrics(y[idx], p[idx])
        for k, v in m.items():
            if not np.isnan(v):
                samples[k].append(v)
    out = {}
    for k, vals in samples.items():
        arr = np.asarray(vals)
        out[k] = (
            float(arr.mean()),
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)),
            int(len(arr)),
        )
    return out


def _predict_sibling(T: int, start: str, end: str) -> tuple[np.ndarray, np.ndarray]:
    ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint_{T}")
    with open(os.path.join(ckpt_dir, "model.pkl"), "rb") as f:
        payload = pickle.load(f)
    df = pd.read_csv(os.path.join(INPUT_DIR, f"checkpoint_{T}.csv"), low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    sl = df[(df["date"] >= start) & (df["date"] < end)].reset_index(drop=True)
    X, y, _ = prepare_Xy(sl, T)
    p = predict_proba(payload["scaler"], payload["model"], X, numeric_cols_in(X))
    return y.values.astype(int), np.asarray(p, dtype=float)


def _predict_production(T: int, start: str, end: str) -> tuple[np.ndarray, np.ndarray]:
    ckpt_dir = os.path.join(PROD_DIR, f"checkpoint_{T}")
    with open(os.path.join(ckpt_dir, "model.pkl"), "rb") as f:
        payload = pickle.load(f)
    with open(os.path.join(ckpt_dir, "calibrator.pkl"), "rb") as f:
        calibrator = pickle.load(f)
    df = pd.read_csv(os.path.join(INPUT_DIR, f"checkpoint_{T}.csv"), low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    sl = df[(df["date"] >= start) & (df["date"] < end)].reset_index(drop=True)
    X, y, _ = prepare_Xy(sl, T)
    p_raw = predict_proba(payload["scaler"], payload["model"], X, numeric_cols_in(X))
    p = np.asarray(calibrator.predict(p_raw) if calibrator is not None else p_raw,
                   dtype=float)
    return y.values.astype(int), p


def _collect(T: int, eval_name: str, y: np.ndarray, p: np.ndarray) -> list[dict]:
    point = _metrics(y, p)
    boot = _bootstrap(y, p)
    rows = []
    for metric, (mean, lo, hi, used) in boot.items():
        rows.append({
            "T":              T,
            "eval":           eval_name,
            "n_eval":         int(len(y)),
            "metric":         metric,
            "point":          point[metric],
            "boot_mean":      mean,
            "ci_lo":          lo,
            "ci_hi":          hi,
            "ci_width":       hi - lo,
            "boot_samples":   used,
        })
    return rows


def main() -> None:
    rows: list[dict] = []
    for T in CHECKPOINTS:
        y_h2,   p_h2   = _predict_sibling(T,    "2025-07-01", "2026-01-01")
        y_2026, p_2026 = _predict_sibling(T,    "2026-01-01", "2027-01-01")
        y_prod, p_prod = _predict_production(T, "2026-01-01", "2027-01-01")
        rows += _collect(T, "sibling_h2",        y_h2,   p_h2)
        rows += _collect(T, "sibling_2026",      y_2026, p_2026)
        rows += _collect(T, "production_2026",   y_prod, p_prod)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"{'T':>3} {'eval':>18} {'metric':>9} {'point':>8} "
          f"{'mean':>8} {'ci_lo':>8} {'ci_hi':>8} {'width':>7}")
    for r in rows:
        print(f"{r['T']:>3} {r['eval']:>18} {r['metric']:>9} "
              f"{r['point']:>8.4f} {r['boot_mean']:>8.4f} "
              f"{r['ci_lo']:>8.4f} {r['ci_hi']:>8.4f} {r['ci_width']:>7.4f}")
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()

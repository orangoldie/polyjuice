"""Smooth calibration assessment for the GPR production model on 2026 S-tier.

Companion to scripts 20 and 25, which use 10 fixed-width and 10 quantile
bins respectively. Here we replace the histogram entirely with two smooth
views:

  1. Kernel-smoothed reliability curve (LOWESS): predicted prob on x,
     empirical winrate on y, with a bootstrap 95% band. No bin
     boundaries — the curve sees every prediction.

  2. K-fold cross-validated isotonic refit on 2026 itself: how much could
     log-loss / Brier improve if we recalibrated *on the test data* via
     a held-out CV procedure? This is the honest upper bound on what
     better calibration alone (no new features) can buy us, with no
     leakage from same-row training-and-application.

Outputs:
  data/exploration/26_gpr_smooth_calibration.csv      paired metrics
  data/exploration/26_gpr_smooth_calibration_T{T}.png reliability plots

The CV-refit metrics tell you whether the production calibrator is
already close to optimal (small delta) or leaving real money on the
table (large delta).
"""
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, brier_score_loss
from statsmodels.nonparametric.smoothers_lowess import lowess

from scripts.models.baseline_logistic import (
    OUTPUT_DIR as BASELINE_OUTPUT_DIR,
    numeric_cols_in as baseline_numeric_cols_in,
    predict_proba as baseline_predict_proba,
    prepare_Xy as baseline_prepare_Xy,
)
from scripts.models.baseline_logistic_gpr import (
    CHECKPOINTS,
    INPUT_DIR,
    OUTPUT_DIR as GPR_OUTPUT_DIR,
    numeric_cols_in as gpr_numeric_cols_in,
    predict_proba as gpr_predict_proba,
    prepare_Xy as gpr_prepare_Xy,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPLORATION_DIR = os.path.join(REPO_ROOT, "data", "exploration")
OUT_CSV = os.path.join(EXPLORATION_DIR, "26_gpr_smooth_calibration.csv")

N_BOOT = 2000
N_FOLDS = 5
LOWESS_FRAC = 0.4
SEED = 42
EPS = 1e-15


def _load_baseline(T):
    ckpt = os.path.join(BASELINE_OUTPUT_DIR, "production", f"checkpoint_{T}")
    with open(os.path.join(ckpt, "model.pkl"), "rb") as f:
        p = pickle.load(f)
    with open(os.path.join(ckpt, "calibrator.pkl"), "rb") as f:
        cal = pickle.load(f)
    return p["scaler"], p["model"], cal


def _load_gpr(T):
    ckpt = os.path.join(GPR_OUTPUT_DIR, "production", f"checkpoint_{T}")
    with open(os.path.join(ckpt, "model.pkl"), "rb") as f:
        p = pickle.load(f)
    with open(os.path.join(ckpt, "calibrator.pkl"), "rb") as f:
        cal = pickle.load(f)
    return p["scaler"], p["model"], p["mu_gpr"], cal


def _cv_isotonic(probs: np.ndarray, y: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Out-of-fold isotonic-refit predictions.

    For each test row, the isotonic correction is fit ONLY on the other
    folds — so the corrected prob is honest, not overfit to that row's
    own outcome.
    """
    out = np.zeros_like(probs, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(probs):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(probs[tr], y[tr])
        out[te] = np.asarray(iso.predict(probs[te]))
    return np.clip(out, EPS, 1 - EPS)


def _paired_bootstrap_logloss(y, p_a, p_b, rng, n_boot=N_BOOT):
    """Per-row LL difference (A − B), bootstrap CI on the mean delta.

    Positive delta means A has worse log-loss (B is better calibrated).
    """
    a = -(y * np.log(np.clip(p_a, EPS, 1-EPS))
          + (1-y) * np.log(np.clip(1-p_a, EPS, 1-EPS)))
    b = -(y * np.log(np.clip(p_b, EPS, 1-EPS))
          + (1-y) * np.log(np.clip(1-p_b, EPS, 1-EPS)))
    delta = a - b
    n = len(delta)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = delta[idx].mean(axis=1)
    return float(delta.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _bootstrap_lowess(probs, y, rng, n_boot=400):
    """Bootstrap LOWESS curves on (probs, y). Returns (xs, lo, hi, mean)
    over a fixed grid of x in [0, 1]."""
    grid = np.linspace(0.01, 0.99, 50)
    n = len(probs)
    curves = np.zeros((n_boot, len(grid)))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xs = probs[idx]
        ys = y[idx].astype(float)
        # LOWESS over (xs, ys), evaluated by interpolation onto the grid.
        sm = lowess(ys, xs, frac=LOWESS_FRAC, return_sorted=True, it=0)
        curves[b] = np.interp(grid, sm[:, 0], sm[:, 1], left=np.nan, right=np.nan)
    lo = np.nanpercentile(curves, 2.5, axis=0)
    hi = np.nanpercentile(curves, 97.5, axis=0)
    mean = np.nanmean(curves, axis=0)
    return grid, lo, hi, mean


def _plot_one(T, y, served_b, served_g, refit_b, refit_g, rng, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, name, served, refit in [
        (axes[0], "baseline", served_b, refit_b),
        (axes[1], "gpr", served_g, refit_g),
    ]:
        grid, lo, hi, mean = _bootstrap_lowess(served, y, rng)
        ax.fill_between(grid, lo, hi, alpha=0.25, label="LOWESS 95% band")
        ax.plot(grid, mean, lw=1.5, label="LOWESS mean")
        # Refit isotonic curve as a smooth correction reference.
        order = np.argsort(served)
        ax.plot(served[order], refit[order], lw=1.0, ls="--",
                label="CV isotonic refit", color="C2")
        ax.plot([0, 1], [0, 1], "k-", lw=0.5, alpha=0.5)
        ax.set_title(f"{name}  T={T}  n={len(y)}")
        ax.set_xlabel("predicted prob (served)")
        ax.set_ylabel("empirical winrate")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    os.makedirs(EXPLORATION_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []

    for T in CHECKPOINTS:
        df = pd.read_csv(os.path.join(INPUT_DIR, f"checkpoint_{T}_gpr.csv"),
                         low_memory=False)
        df["date"] = pd.to_datetime(df["date"])
        test = df[(df["date"] >= "2026-01-01") & (df["date"] < "2027-01-01")
                  & (df["tier"] == "S")].reset_index(drop=True)

        Xb, yb, _ = baseline_prepare_Xy(test, T)
        nb = baseline_numeric_cols_in(Xb)
        sb, mb, cb = _load_baseline(T)
        raw_b = baseline_predict_proba(sb, mb, Xb, nb)
        served_b = np.asarray(cb.predict(raw_b)) if cb is not None else raw_b
        refit_b = _cv_isotonic(served_b, yb.values, N_FOLDS, SEED)

        Xg, yg, _ = gpr_prepare_Xy(test, T)
        ng = gpr_numeric_cols_in(Xg)
        sg, mg, mu_g, cg = _load_gpr(T)
        raw_g = gpr_predict_proba(sg, mg, mu_g, Xg, ng)
        served_g = np.asarray(cg.predict(raw_g)) if cg is not None else raw_g
        refit_g = _cv_isotonic(served_g, yg.values, N_FOLDS, SEED)

        y_arr = yb.values

        for which, served, refit in [
            ("baseline", served_b, refit_b),
            ("gpr",      served_g, refit_g),
        ]:
            ll_served = float(log_loss(y_arr, served, labels=[0, 1]))
            ll_refit  = float(log_loss(y_arr, refit,  labels=[0, 1]))
            br_served = float(brier_score_loss(y_arr, served))
            br_refit  = float(brier_score_loss(y_arr, refit))
            d_mean, d_lo, d_hi = _paired_bootstrap_logloss(y_arr, served, refit, rng)
            rows.append({
                "T": T,
                "model": which,
                "n": len(y_arr),
                "served_logloss": ll_served,
                "refit_logloss":  ll_refit,
                "logloss_delta":  ll_served - ll_refit,
                "served_brier":   br_served,
                "refit_brier":    br_refit,
                "brier_delta":    br_served - br_refit,
                "paired_ll_delta_mean": d_mean,
                "paired_ll_delta_lo":   d_lo,
                "paired_ll_delta_hi":   d_hi,
                "paired_excludes_0":    (d_lo > 0) or (d_hi < 0),
            })

        _plot_one(T, y_arr, served_b, served_g, refit_b, refit_g, rng,
                  os.path.join(EXPLORATION_DIR, f"26_gpr_smooth_calibration_T{T}.png"))
        print(f"wrote 26_gpr_smooth_calibration_T{T}.png")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}\n")

    print("Honest CV-refit gain over served probs (positive delta = refit helps):\n")
    print(f"{'T':>3} {'model':<8} {'n':>4} {'served_LL':>9} {'refit_LL':>9} "
          f"{'LL_delta':>9} {'CI_lo':>9} {'CI_hi':>9} {'Br_delta':>9}  excl 0?")
    for _, r in out.iterrows():
        print(f"{int(r['T']):>3} {r['model']:<8} {int(r['n']):>4} "
              f"{r['served_logloss']:>9.4f} {r['refit_logloss']:>9.4f} "
              f"{r['logloss_delta']:>+9.4f} {r['paired_ll_delta_lo']:>+9.4f} "
              f"{r['paired_ll_delta_hi']:>+9.4f} {r['brier_delta']:>+9.4f}  "
              f"{'YES' if r['paired_excludes_0'] else 'no'}")


if __name__ == "__main__":
    main()

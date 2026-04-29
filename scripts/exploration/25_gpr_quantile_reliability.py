"""Quantile-binned reliability for the GPR production model on 2026 S-tier.

Sibling of script 20, which uses fixed 10pp-wide bins. Equal-width bins
have an n-imbalance problem: the model concentrates predictions near the
extremes, so mid-prob bins (the betting-relevant region) end up with
small n and wide CIs. This script bins by *quantile* of predicted
probability instead — every bin has roughly the same n, so CI widths are
comparable across bins and small-n collapse on the mid-prob region goes
away.

Bin edges are computed *per (T, model)* on that model's predicted-prob
distribution, so quantile cuts are honest to each model's own shape.
That means the bin boundaries differ between baseline and GPR within the
same T — read each model's table independently rather than aligning bins
across models.

Writes data/exploration/25_gpr_quantile_reliability.csv
"""
import os
import pickle

import numpy as np
import pandas as pd

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
OUT_CSV = os.path.join(EXPLORATION_DIR, "25_gpr_quantile_reliability.csv")

N_BINS = 10
N_BOOT = 5000
SEED = 42


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


def _quantile_bin_with_ci(probs, y, rng):
    """Bin by quantile of predicted prob, with bootstrap CI on per-bin gap.

    pd.qcut with duplicates="drop" handles the case where many predictions
    pile up at identical values (rare for logistic, common for tree
    models). Each bin's [lo, hi) is its empirical quantile boundary, so
    bin widths are unequal but bin counts are roughly equal.
    """
    n_total = len(probs)
    if n_total < N_BINS:
        return []
    # qcut returns the bin assignment per row; .cat.categories gives the
    # interval edges as pd.Interval objects.
    cats = pd.qcut(pd.Series(probs), q=N_BINS, duplicates="drop")
    rows = []
    for interval in cats.cat.categories:
        # qcut's first interval is left-open; treat all bins as [lo, hi]
        # for reporting (boundaries are quantile points, not nice numbers).
        mask = (cats == interval).to_numpy()
        n = int(mask.sum())
        if n < 5:
            continue
        cell_p = probs[mask]
        cell_y = y[mask].astype(float)
        gap = float(cell_y.mean() - cell_p.mean())
        idx = rng.integers(0, n, size=(N_BOOT, n))
        boot = cell_y[idx].mean(axis=1) - cell_p[idx].mean(axis=1)
        ci_lo = float(np.percentile(boot, 2.5))
        ci_hi = float(np.percentile(boot, 97.5))
        rows.append({
            "bin_lo":     float(interval.left),
            "bin_hi":     float(interval.right),
            "n":          n,
            "mean_pred":  float(cell_p.mean()),
            "actual":     float(cell_y.mean()),
            "gap":        gap,
            "ci_lo_95":   ci_lo,
            "ci_hi_95":   ci_hi,
            "excludes_0": bool((ci_lo > 0) or (ci_hi < 0)),
        })
    return rows


def main() -> None:
    os.makedirs(EXPLORATION_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    out_rows = []

    for T in CHECKPOINTS:
        df = pd.read_csv(os.path.join(INPUT_DIR, f"checkpoint_{T}_gpr.csv"),
                         low_memory=False)
        df["date"] = pd.to_datetime(df["date"])
        test = df[(df["date"] >= "2026-01-01") & (df["date"] < "2027-01-01")
                  & (df["tier"] == "S")].reset_index(drop=True)

        Xb, yb, _ = baseline_prepare_Xy(test, T)
        nb = baseline_numeric_cols_in(Xb)
        scaler_b, model_b, cal_b = _load_baseline(T)
        raw_b = baseline_predict_proba(scaler_b, model_b, Xb, nb)
        probs_b = np.asarray(cal_b.predict(raw_b)) if cal_b is not None else raw_b

        Xg, yg, _ = gpr_prepare_Xy(test, T)
        ng = gpr_numeric_cols_in(Xg)
        scaler_g, model_g, mu_g, cal_g = _load_gpr(T)
        raw_g = gpr_predict_proba(scaler_g, model_g, mu_g, Xg, ng)
        probs_g = np.asarray(cal_g.predict(raw_g)) if cal_g is not None else raw_g

        y_arr = yb.values

        for which, probs in (("baseline", probs_b), ("gpr", probs_g),
                             ("baseline_raw", raw_b), ("gpr_raw", raw_g)):
            for r in _quantile_bin_with_ci(probs, y_arr, rng):
                r.update({"T": T, "model": which})
                out_rows.append(r)

    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}\n")

    served = df_out[df_out["model"].isin(["baseline", "gpr"])]
    print("Quantile-binned (equal-n) reliability — served probs.\n"
          "Bin edges differ per (T, model) — each row's [lo, hi) is its own quantile range.\n")
    for T in CHECKPOINTS:
        for which in ("baseline", "gpr"):
            sub = served[(served["T"] == T) & (served["model"] == which)]
            if not len(sub):
                continue
            print(f"--- T={T}  model={which}  (S-tier 2026) ---")
            print(f"{'bin':<18} {'n':>4} {'pred':>6} {'actual':>7} "
                  f"{'gap':>8} {'CI_lo':>7} {'CI_hi':>7}  excl 0?")
            for _, r in sub.sort_values("bin_lo").iterrows():
                print(f"[{r['bin_lo']:.3f},{r['bin_hi']:.3f})  "
                      f"{int(r['n']):>4} {r['mean_pred']:>6.3f} {r['actual']:>7.3f} "
                      f"{r['gap']:>+8.3f} "
                      f"{r['ci_lo_95']:>+7.3f} {r['ci_hi_95']:>+7.3f}  "
                      f"{'YES' if r['excludes_0'] else 'no'}")
            print()


if __name__ == "__main__":
    main()

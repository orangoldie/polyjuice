"""Full per-bin reliability table for the GPR production model on 2026 S-tier.

Diagnostic-only follow-up to script 19, which only reports the
[0.65, 0.80) band. This script reports every populated decile bin with a
bootstrap 95% CI on (actual − mean_pred), for each T, S-tier-only.

For comparison, also runs the same bins on the baseline production model
on the same rows. Side-by-side lets us see whether GPR is widening gaps
across the whole probability range or just in specific bands.

Writes data/exploration/20_gpr_full_reliability.csv
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
OUT_CSV = os.path.join(EXPLORATION_DIR, "20_gpr_full_reliability.csv")

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


def _bin_with_ci(probs, y, rng):
    edges = np.linspace(0, 1, N_BINS + 1)
    rows = []
    for i in range(N_BINS):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < N_BINS - 1 else probs <= hi)
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
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "n":      n,
            "mean_pred":   float(cell_p.mean()),
            "actual":      float(cell_y.mean()),
            "gap":         gap,
            "ci_lo_95":    ci_lo,
            "ci_hi_95":    ci_hi,
            "excludes_0":  bool((ci_lo > 0) or (ci_hi < 0)),
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

        # Baseline served probs.
        Xb, yb, _ = baseline_prepare_Xy(test, T)
        nb = baseline_numeric_cols_in(Xb)
        scaler_b, model_b, cal_b = _load_baseline(T)
        raw_b = baseline_predict_proba(scaler_b, model_b, Xb, nb)
        probs_b = np.asarray(cal_b.predict(raw_b)) if cal_b is not None else raw_b

        # GPR served probs.
        Xg, yg, _ = gpr_prepare_Xy(test, T)
        ng = gpr_numeric_cols_in(Xg)
        scaler_g, model_g, mu_g, cal_g = _load_gpr(T)
        raw_g = gpr_predict_proba(scaler_g, model_g, mu_g, Xg, ng)
        probs_g = np.asarray(cal_g.predict(raw_g)) if cal_g is not None else raw_g

        y_arr = yb.values

        for which, probs in (("baseline", probs_b), ("gpr", probs_g),
                             ("baseline_raw", raw_b), ("gpr_raw", raw_g)):
            for r in _bin_with_ci(probs, y_arr, rng):
                r.update({"T": T, "model": which})
                out_rows.append(r)

    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}\n")

    # Pretty print: T × bin × model side-by-side, served only.
    print("Served probabilities (calibrated where calibrator kept; raw at T=25):\n")
    served = df_out[df_out["model"].isin(["baseline", "gpr"])]
    for T in CHECKPOINTS:
        sub = served[served["T"] == T]
        if not len(sub):
            continue
        print(f"--- T={T}  (S-tier 2026) ---")
        print(f"{'bin':<14} {'model':<8} {'n':>4} {'mean_pred':>9} "
              f"{'actual':>7} {'gap':>8} {'CI_lo':>7} {'CI_hi':>7}  excl 0?")
        bin_keys = sorted(sub.groupby(["bin_lo", "bin_hi"]).groups.keys())
        for (lo, hi) in bin_keys:
            for which in ("baseline", "gpr"):
                cell = sub[(sub["bin_lo"] == lo) & (sub["bin_hi"] == hi)
                           & (sub["model"] == which)]
                if not len(cell):
                    continue
                r = cell.iloc[0]
                print(f"[{r['bin_lo']:.2f},{r['bin_hi']:.2f})  {which:<8} "
                      f"{int(r['n']):>4} {r['mean_pred']:>9.3f} "
                      f"{r['actual']:>7.3f} {r['gap']:>+8.3f} "
                      f"{r['ci_lo_95']:>+7.3f} {r['ci_hi_95']:>+7.3f}  "
                      f"{'YES' if r['excludes_0'] else 'no'}")
        print()


if __name__ == "__main__":
    main()

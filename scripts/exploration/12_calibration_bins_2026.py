"""Per-bin calibration on 2026 raw production predictions.

For each checkpoint T, show decile bins of predicted probability with:
- n (sample size in bin)
- mean_pred (average probability we emitted)
- actual (observed win rate)
- gap (actual - mean_pred; positive = underconfident)
- 95% Wilson CI on actual

ECE is a single number. What matters for betting is *which bins are calibrated*.
A 70% prediction should give ~70% actual, with a CI that contains 0.70.
"""
import os
import pickle

import numpy as np
import pandas as pd

from scripts.models.baseline_logistic import (
    CHECKPOINTS,
    INPUT_DIR,
    OUTPUT_DIR,
    numeric_cols_in,
    predict_proba,
    prepare_Xy,
)


PROD = os.path.join(OUTPUT_DIR, "production")
EDGES = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])


def main() -> None:
    print(f"{'T':>3} {'bin':>12} {'n':>5} {'mean_pred':>10} "
          f"{'actual':>7} {'gap':>7} {'95% CI':>15}")
    for T in CHECKPOINTS:
        with open(os.path.join(PROD, f"checkpoint_{T}", "model.pkl"), "rb") as f:
            p = pickle.load(f)
        df = pd.read_csv(os.path.join(INPUT_DIR, f"checkpoint_{T}.csv"),
                         low_memory=False)
        df["date"] = pd.to_datetime(df["date"])
        test = df[(df["date"] >= "2026-01-01")
                  & (df["date"] < "2027-01-01")].reset_index(drop=True)
        X, y, _ = prepare_Xy(test, T)
        probs = predict_proba(p["scaler"], p["model"], X, numeric_cols_in(X))
        y_arr = y.values

        rows = []
        for i in range(len(EDGES) - 1):
            lo, hi = EDGES[i], EDGES[i + 1]
            mask = (probs >= lo) & (probs < hi)
            n = int(mask.sum())
            if n < 5:
                continue
            mp = float(probs[mask].mean())
            ac = float(y_arr[mask].mean())
            se = (ac * (1 - ac) / n) ** 0.5
            lo_ci, hi_ci = max(0.0, ac - 1.96 * se), min(1.0, ac + 1.96 * se)
            print(f"{T:>3}  [{lo:.1f},{hi:.1f}) {n:>5} "
                  f"{mp:>10.3f} {ac:>7.3f} {ac - mp:>+7.3f}   "
                  f"[{lo_ci:.2f},{hi_ci:.2f}]")
            rows.append({
                "bin_lo": round(lo, 2),
                "bin_hi": round(min(hi, 1.0), 2),
                "n": n,
                "mean_pred": mp,
                "actual": ac,
                "gap": ac - mp,
                "ci_lo": lo_ci,
                "ci_hi": hi_ci,
            })
        out_path = os.path.join(PROD, f"checkpoint_{T}",
                                "calibration_bins_2026.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print()


if __name__ == "__main__":
    main()

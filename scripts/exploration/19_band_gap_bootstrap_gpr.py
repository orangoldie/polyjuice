"""Bootstrap 95% CIs on the per-tier × per-T calibration gap in the
[0.65, 0.80) band, GPR production model on 2026.

Forks scripts/exploration/15_band_gap_bootstrap.py for the GPR variant —
implements success criterion #3 of docs/specs/gpr-feature.md:

  T=10 S-tier in-band gap narrows. Re-run the band-gap bootstrap on the new
  model and check that the [0.65, 0.80) S-tier T=10 gap shrinks vs the
  current baseline's −9.4pp. Required to be directionally correct (gap
  moves toward zero); CI on the new gap need not exclude zero given n=102.

Same logic as script 15, but loads `baseline_logistic_gpr` production
artifacts and reads `checkpoint_{T}_gpr.csv`.

Writes data/exploration/19_band_gap_bootstrap_gpr.csv.
"""
import os
import pickle

import numpy as np
import pandas as pd

from scripts.models.baseline_logistic_gpr import (
    CHECKPOINTS,
    INPUT_DIR,
    OUTPUT_DIR,
    numeric_cols_in,
    predict_proba,
    prepare_Xy,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPLORATION_DIR = os.path.join(REPO_ROOT, "data", "exploration")
OUT_CSV = os.path.join(EXPLORATION_DIR, "19_band_gap_bootstrap_gpr.csv")
PROD_DIR = os.path.join(OUTPUT_DIR, "production")

BAND_LO, BAND_HI = 0.65, 0.80
N_BOOT = 5000
SEED = 42
TIERS = ("S", "A", "B", "ALL")


def _load_production(T: int):
    ckpt = os.path.join(PROD_DIR, f"checkpoint_{T}")
    with open(os.path.join(ckpt, "model.pkl"), "rb") as f:
        payload = pickle.load(f)
    with open(os.path.join(ckpt, "calibrator.pkl"), "rb") as f:
        calibrator = pickle.load(f)
    return payload["scaler"], payload["model"], payload["mu_gpr"], calibrator


def main() -> None:
    os.makedirs(EXPLORATION_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []

    print(f"{'T':>3} {'tier':>4} {'n':>4} {'gap':>8} "
          f"{'CI lo':>7} {'CI hi':>7}  excludes 0?")
    for T in CHECKPOINTS:
        df = pd.read_csv(os.path.join(INPUT_DIR, f"checkpoint_{T}_gpr.csv"),
                         low_memory=False)
        df["date"] = pd.to_datetime(df["date"])
        test = df[(df["date"] >= "2026-01-01")
                  & (df["date"] < "2027-01-01")].reset_index(drop=True)
        X, y, _ = prepare_Xy(test, T)
        num_cols = numeric_cols_in(X)
        scaler, model, mu, calibrator = _load_production(T)
        raw = predict_proba(scaler, model, mu, X, num_cols)
        probs = (np.asarray(calibrator.predict(raw))
                 if calibrator is not None else raw)
        y_arr = y.values
        tier_arr = test["tier"].values

        for tier in TIERS:
            tier_mask = (np.ones_like(tier_arr, dtype=bool) if tier == "ALL"
                         else (tier_arr == tier))
            band_mask = tier_mask & (probs >= BAND_LO) & (probs < BAND_HI)
            n = int(band_mask.sum())
            if n < 5:
                continue
            cell_p = probs[band_mask]
            cell_y = y_arr[band_mask].astype(float)
            point_gap = float(cell_y.mean() - cell_p.mean())

            idx = rng.integers(0, n, size=(N_BOOT, n))
            boot_gaps = cell_y[idx].mean(axis=1) - cell_p[idx].mean(axis=1)
            ci_lo = float(np.percentile(boot_gaps, 2.5))
            ci_hi = float(np.percentile(boot_gaps, 97.5))
            excludes_zero = (ci_lo > 0) or (ci_hi < 0)

            print(f"{T:>3} {tier:>4} {n:>4} {point_gap:>+8.3f} "
                  f"{ci_lo:>+7.3f} {ci_hi:>+7.3f}  "
                  f"{'YES' if excludes_zero else 'no'}")
            rows.append({
                "T": T, "tier": tier, "n": n,
                "mean_pred": float(cell_p.mean()),
                "actual":    float(cell_y.mean()),
                "gap":       point_gap,
                "ci_lo_95":  ci_lo,
                "ci_hi_95":  ci_hi,
                "excludes_zero": bool(excludes_zero),
            })

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()

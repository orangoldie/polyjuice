"""Per-tier × per-bin calibration on 2026, with tight bins around 65-80%.

Phase 1 diagnostic for the "calibration in the 70-75% betting band" problem.
Two passes per checkpoint T:
  - production : 2020-2025 model + its production calibrator
  - sibling    : 2020-2024 model, raw (the "drop 2025" ablation)

For each (variant, T, tier, bin) reports n, mean_pred, actual, gap, 95% Wilson CI.

Bin edges deliberately oversample 65-80% — that's the band the user bets in,
so we want bin n large enough to actually distinguish "miscalibrated" from
"noisy."

Writes:
  data/exploration/14_calibration_per_tier_bucketed.csv  (long format)
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


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPLORATION_DIR = os.path.join(REPO_ROOT, "data", "exploration")
OUT_CSV = os.path.join(EXPLORATION_DIR, "14_calibration_per_tier_bucketed.csv")

PROD_DIR = os.path.join(OUTPUT_DIR, "production")

# Tight bins through the betting band, coarse outside it. Right edge open;
# last edge >1 so probs of exactly 1.0 land in the top bin.
EDGES = np.array([0.0, 0.5, 0.6, 0.65, 0.70, 0.725, 0.75, 0.775, 0.80, 0.85, 1.0001])
TIERS = ("S", "A", "B", "ALL")


def _load_production(T: int):
    ckpt = os.path.join(PROD_DIR, f"checkpoint_{T}")
    with open(os.path.join(ckpt, "model.pkl"), "rb") as f:
        payload = pickle.load(f)
    with open(os.path.join(ckpt, "calibrator.pkl"), "rb") as f:
        calibrator = pickle.load(f)
    return payload["scaler"], payload["model"], calibrator


def _load_sibling(T: int):
    ckpt = os.path.join(OUTPUT_DIR, f"checkpoint_{T}")
    with open(os.path.join(ckpt, "model.pkl"), "rb") as f:
        payload = pickle.load(f)
    return payload["scaler"], payload["model"]


def _bin_rows(variant: str, T: int, tier: str,
              probs: np.ndarray, y: np.ndarray) -> list[dict]:
    rows = []
    for i in range(len(EDGES) - 1):
        lo, hi = EDGES[i], EDGES[i + 1]
        mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        mp = float(probs[mask].mean())
        ac = float(y[mask].mean())
        # Wilson 95% CI for a binomial proportion (better behaved than normal
        # at small n / extreme p than the SE approximation in script 12).
        z = 1.96
        denom = 1 + z * z / n
        center = (ac + z * z / (2 * n)) / denom
        half = (z * np.sqrt(ac * (1 - ac) / n + z * z / (4 * n * n))) / denom
        ci_lo = max(0.0, center - half)
        ci_hi = min(1.0, center + half)
        rows.append({
            "variant":   variant,
            "T":         T,
            "tier":      tier,
            "bin_lo":    round(float(lo), 4),
            "bin_hi":    round(float(min(hi, 1.0)), 4),
            "n":         n,
            "mean_pred": mp,
            "actual":    ac,
            "gap":       ac - mp,
            "ci_lo":     ci_lo,
            "ci_hi":     ci_hi,
        })
    return rows


def _print_block(variant: str, T: int, rows: list[dict]) -> None:
    print(f"\n--- {variant}  T={T} ---")
    print(f"{'tier':>4} {'bin':>14} {'n':>5} {'mean_pred':>10} "
          f"{'actual':>7} {'gap':>7} {'95% CI':>15}")
    for r in rows:
        bin_str = f"[{r['bin_lo']:.3f},{r['bin_hi']:.3f})"
        print(f"{r['tier']:>4} {bin_str:>14} {r['n']:>5} "
              f"{r['mean_pred']:>10.3f} {r['actual']:>7.3f} "
              f"{r['gap']:>+7.3f}   [{r['ci_lo']:.2f},{r['ci_hi']:.2f}]")


def main() -> None:
    os.makedirs(EXPLORATION_DIR, exist_ok=True)
    all_rows: list[dict] = []

    for T in CHECKPOINTS:
        df = pd.read_csv(os.path.join(INPUT_DIR, f"checkpoint_{T}.csv"),
                         low_memory=False)
        df["date"] = pd.to_datetime(df["date"])
        test = df[(df["date"] >= "2026-01-01")
                  & (df["date"] < "2027-01-01")].reset_index(drop=True)
        X, y, _ = prepare_Xy(test, T)
        num_cols = numeric_cols_in(X)
        y_arr = y.values
        tier_arr = test["tier"].values

        prod_scaler, prod_model, prod_cal = _load_production(T)
        prod_raw = predict_proba(prod_scaler, prod_model, X, num_cols)
        prod_probs = (np.asarray(prod_cal.predict(prod_raw))
                      if prod_cal is not None else prod_raw)

        sib_scaler, sib_model = _load_sibling(T)
        sib_probs = predict_proba(sib_scaler, sib_model, X, num_cols)

        for variant, probs in (("production", prod_probs),
                               ("sibling_no_2025", sib_probs)):
            block: list[dict] = []
            for tier in TIERS:
                if tier == "ALL":
                    mask = np.ones(len(y_arr), dtype=bool)
                else:
                    mask = (tier_arr == tier)
                if mask.sum() == 0:
                    continue
                block.extend(_bin_rows(variant, T, tier, probs[mask], y_arr[mask]))
            _print_block(variant, T, block)
            all_rows.extend(block)

    pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()

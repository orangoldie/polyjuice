"""How confident are we in baseline_logistic predictions on 2026?

For each predicted-probability bin, report empirical win rate with a 95%
bootstrap CI. Interpretation:
  - actual ≈ mean_pred:  model is calibrated. When it says 65%, reality is 65%.
  - actual > mean_pred:  model is UNDERCONFIDENT. Bet the favored side at
                         market-matching odds; you'll win more than market expects.
  - actual < mean_pred:  model is OVERCONFIDENT. If you trust the model and
                         bet the favored side, you'll lose to expectation.
  - CI is wide:          n in this bin is small; treat with caution regardless.

Two views:
  1. 2026 all-tier (n≈2,556 per T) — the broad calibration picture.
  2. 2026 S-tier only (n=687) — the slice that matches our betting universe
     if we focus on top-tier matches (where Polymarket liquidity is highest).

Also reports a 'pooled' view that stacks all checkpoints — useful for
gauging how often the model speaks at each confidence level overall.
"""
import json
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
OUT_CSV = os.path.join(EXPLORATION_DIR, "24_baseline_betting_calibration.csv")
PROD_DIR = os.path.join(OUTPUT_DIR, "production")

N_BINS = 10
N_BOOT = 5000
SEED = 42


def _load(T):
    ckpt = os.path.join(PROD_DIR, f"checkpoint_{T}")
    with open(os.path.join(ckpt, "model.pkl"), "rb") as f:
        p = pickle.load(f)
    with open(os.path.join(ckpt, "calibrator.pkl"), "rb") as f:
        cal = pickle.load(f)
    with open(os.path.join(ckpt, "metadata.json")) as f:
        meta = json.load(f)
    return p["scaler"], p["model"], cal, meta


def _bin_with_ci(probs, y, rng):
    edges = np.linspace(0, 1, N_BINS + 1)
    rows = []
    for i in range(N_BINS):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & ((probs < hi) if i < N_BINS - 1 else (probs <= hi))
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": 0,
                         "mean_pred": np.nan, "actual": np.nan, "gap": np.nan,
                         "ci_lo": np.nan, "ci_hi": np.nan})
            continue
        cell_p = probs[mask]
        cell_y = y[mask].astype(float)
        gap = float(cell_y.mean() - cell_p.mean())
        # Bootstrap on the actual win rate (CI on the empirical proportion).
        if n >= 5:
            idx = rng.integers(0, n, size=(N_BOOT, n))
            boot_actual = cell_y[idx].mean(axis=1)
            ci_lo = float(np.percentile(boot_actual, 2.5))
            ci_hi = float(np.percentile(boot_actual, 97.5))
        else:
            ci_lo = ci_hi = float("nan")
        rows.append({
            "bin_lo":    float(lo),
            "bin_hi":    float(hi),
            "n":         n,
            "mean_pred": float(cell_p.mean()),
            "actual":    float(cell_y.mean()),
            "gap":       gap,
            "ci_lo":     ci_lo,
            "ci_hi":     ci_hi,
        })
    return rows


def _verdict(row):
    if row["n"] < 20:
        return "TOO FEW (n<20)"
    # CI on actual; check whether mean_pred lies inside it.
    inside = (row["mean_pred"] >= row["ci_lo"]) and (row["mean_pred"] <= row["ci_hi"])
    if inside:
        return "calibrated"
    # mean_pred outside CI on actual — direction matters.
    if row["actual"] > row["mean_pred"]:
        return "UNDERCONF (bet model side)"
    return "OVERCONF (avoid)"


def main():
    os.makedirs(EXPLORATION_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    out_rows = []

    # Score 2026 with baseline production model, all tier and S-tier.
    pooled = {"all": {"probs": [], "y": []}, "stier": {"probs": [], "y": []}}

    for T in CHECKPOINTS:
        df = pd.read_csv(os.path.join(INPUT_DIR, f"checkpoint_{T}.csv"),
                         low_memory=False)
        df["date"] = pd.to_datetime(df["date"])
        test = df[(df["date"] >= "2026-01-01")
                  & (df["date"] < "2027-01-01")].reset_index(drop=True)

        scaler, model, cal, _ = _load(T)
        X, y, _ = prepare_Xy(test, T)
        nc = numeric_cols_in(X)
        raw = predict_proba(scaler, model, X, nc)
        probs = np.asarray(cal.predict(raw)) if cal is not None else raw
        y_arr = y.values

        stier_mask = test["tier"].values == "S"

        for slice_name, mask in (("all", np.ones(len(X), dtype=bool)),
                                 ("stier", stier_mask)):
            for r in _bin_with_ci(probs[mask], y_arr[mask], rng):
                r.update({"T": T, "slice": slice_name})
                r["verdict"] = _verdict(r) if r["n"] > 0 else "empty"
                out_rows.append(r)
            pooled[slice_name]["probs"].append(probs[mask])
            pooled[slice_name]["y"].append(y_arr[mask])

    # Pooled across all T.
    for slice_name in ("all", "stier"):
        all_p = np.concatenate(pooled[slice_name]["probs"])
        all_y = np.concatenate(pooled[slice_name]["y"])
        for r in _bin_with_ci(all_p, all_y, rng):
            r.update({"T": "pooled", "slice": slice_name})
            r["verdict"] = _verdict(r) if r["n"] > 0 else "empty"
            out_rows.append(r)

    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(OUT_CSV, index=False)

    # Pretty-print: pooled views first (the headline), then per-T S-tier.
    def _print_table(rows, title):
        print(f"\n{title}")
        print("-" * len(title))
        print(f"{'bin':<14} {'n':>5} {'pred':>6} {'actual':>7} {'CI on actual':>14} "
              f"{'gap':>8} verdict")
        for r in rows:
            if r["n"] == 0:
                continue
            ci = f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
            print(f"[{r['bin_lo']:.2f},{r['bin_hi']:.2f})  "
                  f"{int(r['n']):>5,} {r['mean_pred']:>6.3f} {r['actual']:>7.3f} "
                  f"{ci:>14} {r['gap']:>+8.3f} {r['verdict']}")

    print("=" * 80)
    print("BASELINE_LOGISTIC on 2026 — calibration with 95% CI on empirical win rate")
    print("=" * 80)

    _print_table(
        df_out[(df_out["T"] == "pooled") & (df_out["slice"] == "all")].to_dict("records"),
        "POOLED across all 4 checkpoints - ALL TIERS (n~10K predictions)"
    )
    _print_table(
        df_out[(df_out["T"] == "pooled") & (df_out["slice"] == "stier")].to_dict("records"),
        "POOLED across all 4 checkpoints - S-TIER ONLY (n~2.7K predictions)"
    )
    for T in CHECKPOINTS:
        _print_table(
            df_out[(df_out["T"] == T) & (df_out["slice"] == "stier")].to_dict("records"),
            f"T={T} S-tier 2026"
        )

    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()

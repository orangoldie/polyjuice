"""
Riot prior sanity check (pre-fit diagnostic for the GPR feature spec).

Question: does Riot's published expected-score formula
  W_e = 1 / (1 + 10^(-gpr_diff / 600))
actually calibrate on the slice of games we'll train on, or is the 65%
S-tier accuracy figure they report a different distribution than ours?

If W_e is roughly calibrated (empirical WR tracks p_riot, slope ~ 1, no
monotonicity violations), then the learned-coefficient design in the spec
is well-motivated and the fixed-coefficient-offset Future Option remains
a viable fallback. If it isn't, both designs are suspect and the prior
gets revisited before we burn cycles on the full refit.

Rows used: T=10 has the broadest coverage and `gpr_diff` is per-game (not
per-checkpoint), so the calibration is the same at every T. Filter to
has_gpr=1 AND year <= 2025 to avoid touching the 2026 holdout.

Outputs:
  data/exploration/16_gpr_prior_calibration.csv  -- per-decile bin table
  data/exploration/16_gpr_prior_calibration.png  -- empirical vs predicted plot
  stdout: one-line MAD summary
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "data" / "processed" / "checkpoint_10_gpr.csv"
OUT_DIR = REPO_ROOT / "data" / "exploration"
CSV_OUT = OUT_DIR / "16_gpr_prior_calibration.csv"
PNG_OUT = OUT_DIR / "16_gpr_prior_calibration.png"

TRAINING_YEAR_MAX = 2025
N_BINS = 10


def riot_we(gpr_diff: np.ndarray) -> np.ndarray:
    """Riot's published expected-score formula."""
    return 1.0 / (1.0 + np.power(10.0, -gpr_diff / 600.0))


def main() -> int:
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    rated = df[(df["has_gpr"] == 1) & (df["year"] <= TRAINING_YEAR_MAX)].copy()
    n = len(rated)
    if n == 0:
        print("no rated training rows — aborting")
        return 1
    print(f"Rated training rows (has_gpr=1, year<={TRAINING_YEAR_MAX}): {n:,}")

    rated["p_riot"] = riot_we(rated["gpr_diff"].to_numpy())

    # Bin by deciles of gpr_diff. Use qcut with duplicates='drop' so heavy
    # mass at gpr_diff~0 doesn't crash the binning.
    rated["bin"] = pd.qcut(rated["gpr_diff"], q=N_BINS, duplicates="drop")
    grouped = rated.groupby("bin", observed=True).agg(
        n=("result", "size"),
        emp_wr=("result", "mean"),
        mean_p_riot=("p_riot", "mean"),
        mean_gpr_diff=("gpr_diff", "mean"),
    ).reset_index()
    grouped["gap"] = grouped["emp_wr"] - grouped["mean_p_riot"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(CSV_OUT, index=False)

    # Headline metrics. n-weighted MAD is the spec's "one-line summary".
    weights = grouped["n"].to_numpy()
    abs_gaps = grouped["gap"].abs().to_numpy()
    mad = float(np.average(abs_gaps, weights=weights))

    # Slope of empirical WR vs p_riot. Calibrated prior => slope ~ 1.
    x = grouped["mean_p_riot"].to_numpy()
    y = grouped["emp_wr"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1, w=weights)

    # Monotonicity: is empirical WR weakly non-decreasing in p_riot order?
    order = np.argsort(x)
    y_ord = y[order]
    n_inversions = int(np.sum(np.diff(y_ord) < 0))

    print("\n=== Bin table ===")
    for _, r in grouped.iterrows():
        print(
            f"  bin gpr_diff~{r['mean_gpr_diff']:>+8.1f}  "
            f"n={int(r['n']):>4}  p_riot={r['mean_p_riot']:.3f}  "
            f"emp_wr={r['emp_wr']:.3f}  gap={r['gap']:+.3f}"
        )
    print(
        f"\nMAD (n-weighted) = {mad:.4f}  "
        f"slope = {slope:.3f}  intercept = {intercept:+.3f}  "
        f"monotonicity violations = {n_inversions}/{len(grouped) - 1}"
    )

    # Plot.
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", label="perfect calibration")
    ax.scatter(x, y, s=np.sqrt(weights) * 6, alpha=0.8, label="bin (size ∝ √n)")
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, slope * xs + intercept, color="C1", label=f"weighted fit slope={slope:.3f}")
    ax.set_xlabel("Riot $W_e$ (mean per bin)")
    ax.set_ylabel("Empirical win rate")
    ax.set_title(
        f"GPR prior calibration on training slice "
        f"(rated games, year≤{TRAINING_YEAR_MAX}, n={n:,})\n"
        f"MAD={mad:.3f}, slope={slope:.3f}"
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=130)

    print(f"\nWrote {CSV_OUT}")
    print(f"Wrote {PNG_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

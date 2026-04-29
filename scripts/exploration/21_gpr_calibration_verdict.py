"""GPR vs baseline: which model is better calibrated for betting?

Reads data/exploration/20_gpr_full_reliability.csv and produces:
  1. GPR-only bin table per T, with overconfident/underconfident labels.
  2. Side-by-side weighted ECE per T (lower is better calibrated).
  3. Count of bins where the gap CI excludes zero (real miscalibrations).

Betting interpretation:
  - actual > mean_pred  =>  model is UNDERCONFIDENT  =>  bet model side
  - actual < mean_pred  =>  model is OVERCONFIDENT   =>  bet against model
  CI excluding zero in either direction = the miscalibration is statistically
  real on this slice (n=687 S-tier 2026 games).
"""
import os
import sys

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO_ROOT, "data", "exploration", "20_gpr_full_reliability.csv")
OUT_TXT = os.path.join(REPO_ROOT, "data", "exploration",
                       "21_gpr_calibration_verdict.txt")


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)

    def flush(self):
        for st in self._streams:
            st.flush()


def main():
    df = pd.read_csv(SRC)
    df = df[df["model"].isin(["baseline", "gpr"])].copy()

    fh = open(OUT_TXT, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, fh)

    print("=" * 72)
    print("GPR served probabilities — every populated bin (S-tier 2026)")
    print("=" * 72)
    g = df[df["model"] == "gpr"]
    for T in (10, 15, 20, 25):
        sub = g[g["T"] == T].sort_values("bin_lo")
        print(f"\n--- T={T} ---")
        print(f"{'bin':<14} {'n':>4} {'pred':>6} {'actual':>7} {'gap':>8} "
              f"{'CI_lo':>7} {'CI_hi':>7}  verdict")
        for _, r in sub.iterrows():
            if r["excludes_0"]:
                if r["gap"] > 0:
                    verdict = "UNDERCONF (real)"
                else:
                    verdict = "OVERCONF (real)"
            else:
                verdict = "calibrated within noise"
            print(f"[{r['bin_lo']:.2f},{r['bin_hi']:.2f})  "
                  f"{int(r['n']):>4} {r['mean_pred']:>6.3f} {r['actual']:>7.3f} "
                  f"{r['gap']:>+8.3f} {r['ci_lo_95']:>+7.3f} {r['ci_hi_95']:>+7.3f}  "
                  f"{verdict}")

    print("\n" + "=" * 72)
    print("Side-by-side calibration quality — baseline vs GPR")
    print("=" * 72)
    print(f"{'T':>3} {'base_ECE':>9} {'gpr_ECE':>9} "
          f"{'base_real_miscal':>17} {'gpr_real_miscal':>16} winner")
    totals = {"baseline": [0, 0, 0], "gpr": [0, 0, 0]}  # weighted_ece_num, n, real_miscal
    for T in (10, 15, 20, 25):
        sub = df[df["T"] == T]
        out = {}
        for which in ("baseline", "gpr"):
            s = sub[sub["model"] == which]
            total = s["n"].sum()
            ece = float((s["n"] / total * s["gap"].abs()).sum())
            real = int(s["excludes_0"].sum())
            out[which] = (ece, real, len(s))
            totals[which][0] += float((s["n"] * s["gap"].abs()).sum())
            totals[which][1] += int(total)
            totals[which][2] += real
        b, g = out["baseline"], out["gpr"]
        winner = ("baseline" if b[0] < g[0] else
                  "gpr"      if g[0] < b[0] else "tie")
        print(f"{T:>3} {b[0]:>9.4f} {g[0]:>9.4f} "
              f"{b[1]:>10d}/{b[2]:<6d} {g[1]:>9d}/{g[2]:<6d} {winner}")

    print()
    print("Pooled across all T (one big reliability assessment):")
    for which in ("baseline", "gpr"):
        num, n, real = totals[which]
        ece = num / n if n else 0.0
        print(f"  {which:<10} pooled ECE={ece:.4f}   "
              f"real miscalibrated bins={real}")

    sys.stdout = sys.__stdout__
    fh.close()
    print(f"\nWrote {OUT_TXT}")


if __name__ == "__main__":
    main()

"""Diagnose whether GPR's 2026-holdout failure is a distribution-shift story.

Two checks:

  1. Sibling model (trained 2020-2024) scored on 2025 S-tier. The original
     val numbers were on ALL tiers in 2025; this re-runs them on the S-tier
     slice that's directly comparable to the 2026 S-tier holdout. If GPR
     improves log-loss on rated 2025 S-tier, the feature works on data
     adjacent to training.

  2. Anchor-staleness distribution on 2026 S-tier rated games. For each
     rated game, compute (game_date - snapshot_date) in days. Histogram
     the staleness; bucket into [0-15], [15-45], [45-180] days; report
     paired log-loss delta per bucket. If the failure concentrates in the
     stale bucket, that's the mechanism — late-2025 ratings predicting
     early-2026 games.
"""
import json
import os
import pickle
import sys

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
SNAPSHOTS_PATH = os.path.join(REPO_ROOT, "data", "gpr", "snapshots.csv")
ALIASES_PATH = os.path.join(REPO_ROOT, "data", "gpr", "team_aliases.csv")
OUT_TXT = os.path.join(REPO_ROOT, "data", "exploration",
                       "23_gpr_distribution_shift.txt")
EPS = 1e-15


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)

    def flush(self):
        for st in self._streams:
            st.flush()


_fh = open(OUT_TXT, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _fh)


def _logloss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _load_baseline_sibling(T):
    ckpt = os.path.join(BASELINE_OUTPUT_DIR, f"checkpoint_{T}")
    with open(os.path.join(ckpt, "model.pkl"), "rb") as f:
        p = pickle.load(f)
    return p["scaler"], p["model"], None


def _load_gpr_sibling(T):
    ckpt = os.path.join(GPR_OUTPUT_DIR, f"checkpoint_{T}")
    with open(os.path.join(ckpt, "model.pkl"), "rb") as f:
        p = pickle.load(f)
    return p["scaler"], p["model"], p["mu_gpr"], None


# ---------------- 1. Sibling on 2025 S-tier rated subset ----------------
print("=" * 72)
print("1. Sibling models on 2025 S-tier (the 'adjacent to training' test)")
print("=" * 72)
print(f"{'T':>3} {'slice':<10} {'n':>5} {'baseline_LL':>12} {'gpr_LL':>10} {'delta':>8}")
for T in CHECKPOINTS:
    df = pd.read_csv(os.path.join(INPUT_DIR, f"checkpoint_{T}_gpr.csv"),
                     low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    val = df[(df["date"] >= "2025-01-01") & (df["date"] < "2026-01-01")
             & (df["tier"] == "S")].reset_index(drop=True)

    Xb, yb, _ = baseline_prepare_Xy(val, T)
    nb = baseline_numeric_cols_in(Xb)
    sb_scaler, sb_model, _ = _load_baseline_sibling(T)
    pb = baseline_predict_proba(sb_scaler, sb_model, Xb, nb)

    Xg, yg, _ = gpr_prepare_Xy(val, T)
    ng = gpr_numeric_cols_in(Xg)
    sg_scaler, sg_model, sg_mu, _ = _load_gpr_sibling(T)
    pg = gpr_predict_proba(sg_scaler, sg_model, sg_mu, Xg, ng)

    y_arr = yb.to_numpy()
    rated = Xg["has_gpr"].values == 1
    for slice_name, mask in (("overall", np.ones(len(Xg), dtype=bool)),
                             ("rated", rated),
                             ("unrated", ~rated)):
        if mask.sum() == 0:
            continue
        ll_b = float(_logloss(y_arr[mask], pb[mask]).mean())
        ll_g = float(_logloss(y_arr[mask], pg[mask]).mean())
        print(f"{T:>3} {slice_name:<10} {int(mask.sum()):>5,} "
              f"{ll_b:>12.4f} {ll_g:>10.4f} {ll_b-ll_g:>+8.4f}")


# ---------------- 2. Anchor staleness on 2026 S-tier ----------------
print()
print("=" * 72)
print("2. Anchor staleness on 2026 S-tier rated games")
print("=" * 72)

aliases = (pd.read_csv(ALIASES_PATH, dtype=str).fillna(""))
oe_to_gpr = {r["oe_teamname"]: r["gpr_team_name"]
             for _, r in aliases.iterrows()
             if r["oe_teamname"] and r["gpr_team_name"]}
snapshots = pd.read_csv(SNAPSHOTS_PATH)
snapshots["snapshot_date"] = pd.to_datetime(snapshots["snapshot_date"], utc=True)
snapshots = snapshots.sort_values("snapshot_date").reset_index(drop=True)

# Use T=10 holdout (same gameid set across all T).
df = pd.read_csv(os.path.join(INPUT_DIR, "checkpoint_10_gpr.csv"),
                 low_memory=False)
df["date_utc"] = pd.to_datetime(df["date"], utc=True)
test = df[(df["date_utc"] >= "2026-01-01") & (df["date_utc"] < "2027-01-01")
          & (df["tier"] == "S") & (df["has_gpr"] == 1)].reset_index(drop=True)


def staleness_for_team(team_name_oe, game_date):
    g = oe_to_gpr.get(team_name_oe)
    if g is None:
        return np.nan
    sub = snapshots[(snapshots["team_name"] == g)
                    & (snapshots["snapshot_date"] < game_date)]
    if not len(sub):
        return np.nan
    latest = sub["snapshot_date"].max()
    return (game_date - latest).days


stale_team = test.apply(
    lambda r: staleness_for_team(r["teamname"], r["date_utc"]), axis=1)
stale_opp = test.apply(
    lambda r: staleness_for_team(r["opp_teamname"], r["date_utc"]), axis=1)
stale_max = pd.concat([stale_team, stale_opp], axis=1).max(axis=1)

print("Distribution of max(team_staleness, opp_staleness) in days:")
print(f"  n={len(stale_max)}  median={stale_max.median():.0f}d  "
      f"p25={stale_max.quantile(0.25):.0f}d  p75={stale_max.quantile(0.75):.0f}d  "
      f"max={stale_max.max():.0f}d")
print(f"  buckets:")
for lo, hi, name in [(0, 15, "fresh (<15d)"),
                     (15, 45, "stale (15-45d)"),
                     (45, 180, "very stale (45-180d)"),
                     (180, 999, "ancient (180+d)")]:
    bucket_mask = (stale_max >= lo) & (stale_max < hi)
    print(f"    {name:<24} n={int(bucket_mask.sum()):>4}")

# Now compute paired log-loss delta per staleness bucket using PRODUCTION
# probs (the served probs from the actual evaluation).
def _load_baseline_prod(T):
    ckpt = os.path.join(BASELINE_OUTPUT_DIR, "production", f"checkpoint_{T}")
    with open(os.path.join(ckpt, "model.pkl"), "rb") as f:
        p = pickle.load(f)
    with open(os.path.join(ckpt, "calibrator.pkl"), "rb") as f:
        cal = pickle.load(f)
    with open(os.path.join(ckpt, "metadata.json")) as f:
        meta = json.load(f)
    return p["scaler"], p["model"], cal, meta


def _load_gpr_prod(T):
    ckpt = os.path.join(GPR_OUTPUT_DIR, "production", f"checkpoint_{T}")
    with open(os.path.join(ckpt, "model.pkl"), "rb") as f:
        p = pickle.load(f)
    with open(os.path.join(ckpt, "calibrator.pkl"), "rb") as f:
        cal = pickle.load(f)
    with open(os.path.join(ckpt, "metadata.json")) as f:
        meta = json.load(f)
    return p["scaler"], p["model"], p["mu_gpr"], cal, meta


print()
print("Per-checkpoint paired LL by staleness bucket (rated S-tier 2026, raw probs):")
print(f"{'T':>3} {'bucket':<22} {'n':>4} {'baseline_LL':>12} {'gpr_LL':>10} {'delta':>8}")
for T in CHECKPOINTS:
    dfT = pd.read_csv(os.path.join(INPUT_DIR, f"checkpoint_{T}_gpr.csv"),
                      low_memory=False)
    dfT["date_utc"] = pd.to_datetime(dfT["date"], utc=True)
    sub = dfT[(dfT["date_utc"] >= "2026-01-01") & (dfT["date_utc"] < "2027-01-01")
              & (dfT["tier"] == "S") & (dfT["has_gpr"] == 1)].reset_index(drop=True)

    s_team = sub.apply(
        lambda r: staleness_for_team(r["teamname"], r["date_utc"]), axis=1)
    s_opp = sub.apply(
        lambda r: staleness_for_team(r["opp_teamname"], r["date_utc"]), axis=1)
    s = pd.concat([s_team, s_opp], axis=1).max(axis=1)

    Xb, yb, _ = baseline_prepare_Xy(sub, T)
    nb = baseline_numeric_cols_in(Xb)
    bs, bm, _, _ = _load_baseline_prod(T)
    pb = baseline_predict_proba(bs, bm, Xb, nb)

    Xg, yg, _ = gpr_prepare_Xy(sub, T)
    ng = gpr_numeric_cols_in(Xg)
    gs, gm, gmu, _, _ = _load_gpr_prod(T)
    pg = gpr_predict_proba(gs, gm, gmu, Xg, ng)

    y_arr = yb.to_numpy()
    for lo, hi, name in [(0, 15, "fresh (<15d)"),
                         (15, 45, "stale (15-45d)"),
                         (45, 180, "very stale (45-180d)"),
                         (180, 999, "ancient (180+d)")]:
        m = ((s >= lo) & (s < hi)).to_numpy()
        if m.sum() < 10:
            continue
        ll_b = float(_logloss(y_arr[m], pb[m]).mean())
        ll_g = float(_logloss(y_arr[m], pg[m]).mean())
        print(f"{T:>3} {name:<22} {int(m.sum()):>4,} "
              f"{ll_b:>12.4f} {ll_g:>10.4f} {ll_b-ll_g:>+8.4f}")

sys.stdout = sys.__stdout__
_fh.close()
print(f"Wrote {OUT_TXT}")

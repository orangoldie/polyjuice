"""Paired bootstrap CI on 2026 S-tier log-loss: baseline vs baseline+GPR.

Implements success criterion #1 of docs/specs/gpr-feature.md:

  Aggregate log-loss improves across all 4 checkpoints, paired-bootstrap CI
  for (baseline+GPR − baseline) excludes zero (improvement direction).

  - Resample unit:    gameid
  - Per-game stat:    mean of per-(side, T) log-loss over checkpoints reached
  - Pairing:          (baseline_score[gameid], gpr_score[gameid])

Both models served per their own production calibrator decision (calibrated
if metadata.calibrator_kind != null, raw otherwise) — i.e., whichever
probabilities each model would actually serve in production.

Reports the bootstrap CI on the per-game (baseline − gpr) delta. A negative
delta means GPR is better. Spec wants the CI to exclude 0 in the improvement
direction (delta strictly positive).

Also prints the same comparison on the rated/unrated slice — the breakout
the user asked for: improvement should concentrate on rated rows.

Writes data/exploration/18_gpr_paired_bootstrap.csv with the per-checkpoint
and aggregate numbers.
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd

from scripts.models.baseline_logistic import (
    CHECKPOINTS,
    INPUT_DIR,
    OUTPUT_DIR as BASELINE_OUTPUT_DIR,
    TEST_END,
    TEST_START,
    _slice_by_date,
    numeric_cols_in as baseline_numeric_cols_in,
    predict_proba as baseline_predict_proba,
    prepare_Xy as baseline_prepare_Xy,
)
from scripts.models.baseline_logistic_gpr import (
    OUTPUT_DIR as GPR_OUTPUT_DIR,
    numeric_cols_in as gpr_numeric_cols_in,
    predict_proba as gpr_predict_proba,
    prepare_Xy as gpr_prepare_Xy,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPLORATION_DIR = os.path.join(REPO_ROOT, "data", "exploration")
OUT_CSV = os.path.join(EXPLORATION_DIR, "18_gpr_paired_bootstrap.csv")

N_BOOT = 5000
SEED = 42

EPS = 1e-15  # log-loss clip


def _logloss_per_row(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _load_baseline_prod(T: int):
    ckpt = os.path.join(BASELINE_OUTPUT_DIR, "production", f"checkpoint_{T}")
    with open(os.path.join(ckpt, "model.pkl"), "rb") as f:
        payload = pickle.load(f)
    with open(os.path.join(ckpt, "calibrator.pkl"), "rb") as f:
        calibrator = pickle.load(f)
    with open(os.path.join(ckpt, "metadata.json")) as f:
        meta = json.load(f)
    return payload["scaler"], payload["model"], calibrator, meta


def _load_gpr_prod(T: int):
    ckpt = os.path.join(GPR_OUTPUT_DIR, "production", f"checkpoint_{T}")
    with open(os.path.join(ckpt, "model.pkl"), "rb") as f:
        payload = pickle.load(f)
    with open(os.path.join(ckpt, "calibrator.pkl"), "rb") as f:
        calibrator = pickle.load(f)
    with open(os.path.join(ckpt, "metadata.json")) as f:
        meta = json.load(f)
    return payload["scaler"], payload["model"], payload["mu_gpr"], calibrator, meta


def _score_baseline(T: int, df_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw_probs, served_probs). served = calibrated if metadata says
    calibrator_kind is not None, else raw."""
    scaler, model, calibrator, meta = _load_baseline_prod(T)
    X, _y, _ = baseline_prepare_Xy(df_test, T)
    num_cols = baseline_numeric_cols_in(X)
    raw = baseline_predict_proba(scaler, model, X, num_cols)
    if meta.get("calibrator_kind") is not None and calibrator is not None:
        served = np.asarray(calibrator.predict(raw))
    else:
        served = raw
    return raw, served


def _score_gpr(T: int, df_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    scaler, model, mu, calibrator, meta = _load_gpr_prod(T)
    X, _y, _ = gpr_prepare_Xy(df_test, T)
    num_cols = gpr_numeric_cols_in(X)
    raw = gpr_predict_proba(scaler, model, mu, X, num_cols)
    if meta.get("calibrator_kind") is not None and calibrator is not None:
        served = np.asarray(calibrator.predict(raw))
    else:
        served = raw
    return raw, served


def _build_long_table() -> pd.DataFrame:
    """Build a long-format table of per-(gameid, side, T) log-losses for both
    models, restricted to 2026 S-tier. One row per (gameid, side, T)."""
    parts = []
    for T in CHECKPOINTS:
        # Baseline reads checkpoint_{T}.csv; GPR reads checkpoint_{T}_gpr.csv.
        # The latter is a strict superset of the former in columns and has the
        # same row identity (gameid, side). Use the GPR file so we have
        # `has_gpr` for the rated/unrated breakdown without an extra join.
        df = pd.read_csv(os.path.join(INPUT_DIR, f"checkpoint_{T}_gpr.csv"),
                         low_memory=False)
        test = _slice_by_date(df, TEST_START, TEST_END)
        test = test[test["tier"] == "S"].reset_index(drop=True)

        y = test["result"].astype(int).to_numpy()
        p_base_raw, p_base_srv = _score_baseline(T, test)
        p_gpr_raw, p_gpr_srv = _score_gpr(T, test)

        parts.append(pd.DataFrame({
            "gameid":         test["gameid"].values,
            "side":           test["side"].values,
            "T":              T,
            "has_gpr":        test["has_gpr"].astype(int).values,
            "y":              y,
            "ll_base_raw":    _logloss_per_row(y, p_base_raw),
            "ll_base_served": _logloss_per_row(y, p_base_srv),
            "ll_gpr_raw":     _logloss_per_row(y, p_gpr_raw),
            "ll_gpr_served":  _logloss_per_row(y, p_gpr_srv),
        }))
    long = pd.concat(parts, ignore_index=True)
    return long


def _per_game_stat(long: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Collapse the long table to one row per gameid: mean LL over reached
    checkpoints × sides. `mode` is 'raw' or 'served'."""
    base_col = f"ll_base_{mode}"
    gpr_col = f"ll_gpr_{mode}"
    grouped = long.groupby("gameid").agg(
        ll_base=(base_col, "mean"),
        ll_gpr=(gpr_col, "mean"),
        has_gpr=("has_gpr", "max"),
        n_rows=(base_col, "size"),
    ).reset_index()
    grouped["delta"] = grouped["ll_base"] - grouped["ll_gpr"]
    return grouped


def _bootstrap_ci(deltas: np.ndarray, rng: np.random.Generator,
                  n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Return (mean, ci_lo_95, ci_hi_95) bootstrapped over the given deltas."""
    n = len(deltas)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = deltas[idx].mean(axis=1)
    return float(deltas.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _per_T_stat(long: pd.DataFrame, T: int, mode: str) -> pd.DataFrame:
    """Per-checkpoint variant. `mode` is 'raw' or 'served'."""
    base_col = f"ll_base_{mode}"
    gpr_col = f"ll_gpr_{mode}"
    sub = long[long["T"] == T]
    grouped = sub.groupby("gameid").agg(
        ll_base=(base_col, "mean"),
        ll_gpr=(gpr_col, "mean"),
        has_gpr=("has_gpr", "max"),
    ).reset_index()
    grouped["delta"] = grouped["ll_base"] - grouped["ll_gpr"]
    return grouped


def main() -> None:
    os.makedirs(EXPLORATION_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Building long table over 2026 S-tier holdout...")
    long = _build_long_table()
    print(f"  {len(long):,} (gameid, side, T) rows  "
          f"({long['gameid'].nunique():,} unique gameids)")

    rows = []
    for mode in ("served", "raw"):
        agg = _per_game_stat(long, mode)
        print(f"\n=== Aggregate ({mode}) — per-game mean LL across reached T × sides ===")
        print(f"{'slice':<10} {'n_games':>8} {'baseline_LL':>12} {'gpr_LL':>10} "
              f"{'delta':>9} {'CI_lo':>9} {'CI_hi':>9}  excludes 0?")

        for slice_name, mask in [
            ("overall", np.ones(len(agg), dtype=bool)),
            ("rated",   agg["has_gpr"].values == 1),
            ("unrated", agg["has_gpr"].values == 0),
        ]:
            sub = agg[mask]
            deltas = sub["delta"].to_numpy()
            mean_d, ci_lo, ci_hi = _bootstrap_ci(deltas, rng)
            excludes = (ci_lo > 0) or (ci_hi < 0)
            ll_b = float(sub["ll_base"].mean()) if len(sub) else float("nan")
            ll_g = float(sub["ll_gpr"].mean()) if len(sub) else float("nan")
            print(f"{slice_name:<10} {len(sub):>8,} {ll_b:>12.4f} {ll_g:>10.4f} "
                  f"{mean_d:>+9.4f} {ci_lo:>+9.4f} {ci_hi:>+9.4f}  "
                  f"{'YES' if excludes else 'no'}")
            rows.append({
                "scope":         "aggregate",
                "mode":          mode,
                "T":             "all",
                "slice":         slice_name,
                "n_games":       int(len(sub)),
                "baseline_ll":   ll_b,
                "gpr_ll":        ll_g,
                "delta_mean":    mean_d,
                "delta_ci_lo":   ci_lo,
                "delta_ci_hi":   ci_hi,
                "excludes_zero": bool(excludes),
            })

        print(f"\n=== Per-checkpoint ({mode}) ===")
        print(f"{'T':>3} {'slice':<10} {'n_games':>8} {'baseline_LL':>12} {'gpr_LL':>10} "
              f"{'delta':>9} {'CI_lo':>9} {'CI_hi':>9}  excludes 0?")
        for T in CHECKPOINTS:
            per_T = _per_T_stat(long, T, mode)
            for slice_name, mask in [
                ("overall", np.ones(len(per_T), dtype=bool)),
                ("rated",   per_T["has_gpr"].values == 1),
                ("unrated", per_T["has_gpr"].values == 0),
            ]:
                sub = per_T[mask]
                deltas = sub["delta"].to_numpy()
                mean_d, ci_lo, ci_hi = _bootstrap_ci(deltas, rng)
                excludes = (ci_lo > 0) or (ci_hi < 0)
                ll_b = float(sub["ll_base"].mean()) if len(sub) else float("nan")
                ll_g = float(sub["ll_gpr"].mean()) if len(sub) else float("nan")
                print(f"{T:>3} {slice_name:<10} {len(sub):>8,} {ll_b:>12.4f} {ll_g:>10.4f} "
                      f"{mean_d:>+9.4f} {ci_lo:>+9.4f} {ci_hi:>+9.4f}  "
                      f"{'YES' if excludes else 'no'}")
                rows.append({
                    "scope":         "per_T",
                    "mode":          mode,
                    "T":             T,
                    "slice":         slice_name,
                    "n_games":       int(len(sub)),
                    "baseline_ll":   ll_b,
                    "gpr_ll":        ll_g,
                    "delta_mean":    mean_d,
                    "delta_ci_lo":   ci_lo,
                    "delta_ci_hi":   ci_hi,
                    "excludes_zero": bool(excludes),
                })

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()

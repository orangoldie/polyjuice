"""
Baseline logistic regression per checkpoint, augmented with GPR features.

Refit of `baseline_logistic` with three new columns from stage 5:
  - gpr_team       (numeric, scaled)
  - gpr_opp_team   (numeric, scaled)
  - has_gpr        (binary passthrough)

`gpr_diff` is persisted in checkpoint_{T}_gpr.csv but dropped at
prepare_Xy time, mirroring how baseline drops `*diffat{T}`. This keeps the
feature space at {gpr_team, gpr_opp_team} so the model can express both a
rating-gap effect (β_team − β_opp) and a rating-level effect (β_team + β_opp).

Two design points the spec calls out, both implemented here:

  1. Standardization of GPR columns is fit on rated rows only. Unrated
     rows have gpr_team/gpr_opp_team imputed to the rated mean BEFORE
     scaling, so post-standardization they sit at zero. Without this,
     the rated-vs-unrated mean shift would leak into β_team/β_opp under
     l2, biasing the headline `β_team − β_opp` contrast.

  2. Standard errors and the coefficient covariance — needed for the
     contrast z-stat in success-criterion #2 — come from an unregularized
     statsmodels Logit refit on identical scaled features, reported as a
     diagnostic alongside the production sklearn fit. (sklearn's
     LogisticRegressionCV does not expose SE / Cov.)

Inputs:  data/processed/checkpoint_{10,15,20,25}_gpr.csv
Outputs: data/models/baseline_logistic_gpr/checkpoint_{T}/{model.pkl,
         metrics.json, coefficients.csv, gpr_diagnostics.json,
         reliability.png}

CLI:
  python -m scripts.models.baseline_logistic_gpr                       # train sibling per checkpoint
  python -m scripts.models.baseline_logistic_gpr --fit-production      # production refit on 2020-2025
  python -m scripts.models.baseline_logistic_gpr --evaluate-test-production  # atomic 2026 S-tier eval, once
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from scripts.models.baseline_logistic import (
    CHECKPOINTS,
    C_GRID,
    CV_SEED,
    FALLBACK_LOGLOSS_MARGIN,
    FIRST_GATES,
    H1_END,
    H1_START,
    H2_END,
    NUMERIC_FEATURES,
    TEST_END,
    TEST_START,
    TIER_CATEGORIES,
    TRAIN_START,
    _append_runs_row,
    _plot_reliability,
    _slice_by_date,
    ece_weighted,
    evaluate,
    reliability_bins,
    split_train_val,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_DIR = os.path.join(REPO_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "models", "baseline_logistic_gpr")
PROD_DIR = os.path.join(OUTPUT_DIR, "production")
RUNS_CSV = os.path.join(REPO_ROOT, "data", "experiments", "runs.csv")

GPR_NUMERIC = ("gpr_team", "gpr_opp_team")
GPR_BINARY = ("has_gpr",)

_NON_NUMERIC_FEATURE_NAMES = frozenset({
    "side_is_blue",
    "firstblood", "firstdragon", "firsttower", "firstbaron",
    "tier_A", "tier_B",
    "has_gpr",
})


def prepare_Xy(df: pd.DataFrame, T: int) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build (X, y, feature_names) for checkpoint T including GPR columns.

    Mirrors baseline.prepare_Xy exactly, then appends gpr_team, gpr_opp_team
    (numeric) and has_gpr (binary). gpr_diff is intentionally not surfaced —
    it's exactly collinear with the {team, opp} pair under l2.
    """
    X = pd.DataFrame(index=df.index)
    for col in NUMERIC_FEATURES(T):
        X[col] = df[col].astype(float)

    X["side_is_blue"] = (df["side"] == "Blue").astype(int)
    for flag in FIRST_GATES[T]:
        X[flag] = df[flag].astype(int)

    tier = pd.Categorical(df["tier"], categories=TIER_CATEGORIES)
    tier_dummies = pd.get_dummies(tier, prefix="tier", drop_first=True).astype(int)
    tier_dummies.index = df.index
    for col in tier_dummies.columns:
        X[col] = tier_dummies[col]

    for col in GPR_NUMERIC:
        X[col] = df[col].astype(float)
    for col in GPR_BINARY:
        X[col] = df[col].astype(int)

    y = df["result"].astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)
    return X, y, list(X.columns)


def numeric_cols_in(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in _NON_NUMERIC_FEATURE_NAMES]


def _impute_unrated_to_rated_mean(X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Set gpr_team/gpr_opp_team on has_gpr=0 rows to the rated-mean.

    Returns the modified frame (copy) and the mu dict used, so val-time
    predictions can apply the same imputation.

    Why: the population mean shift between rated and unrated rows would
    otherwise leak into β_team/β_opp under l2 (since unrated rows are
    zero-coded). Imputing to the rated mean confines the population shift
    to has_gpr's coefficient.
    """
    X = X.copy()
    rated_mask = X["has_gpr"] == 1
    if rated_mask.sum() == 0:
        # Degenerate — no rated rows. Leave columns at their literal zeros.
        # Down-stream scaler will still standardize; the contrast just
        # won't be identifiable. Caller will surface this in diagnostics.
        return X, {c: 0.0 for c in GPR_NUMERIC}

    mu = {c: float(X.loc[rated_mask, c].mean()) for c in GPR_NUMERIC}
    unrated_mask = ~rated_mask
    if unrated_mask.any():
        for c in GPR_NUMERIC:
            X.loc[unrated_mask, c] = mu[c]
    return X, mu


def fit(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[StandardScaler, LogisticRegressionCV, dict]:
    """Fit scaler + LogisticRegressionCV with rated-only imputation upstream.

    Returns (scaler, model, mu) — mu is the rated-row mean of the GPR
    numeric columns from the training set, reused at predict time.
    """
    X_imp, mu = _impute_unrated_to_rated_mean(X_train)
    numeric_cols = numeric_cols_in(X_imp)

    scaler = StandardScaler()
    scaler.fit(X_imp[numeric_cols].values)

    X_fit = X_imp.copy()
    X_fit[numeric_cols] = scaler.transform(X_imp[numeric_cols].values)

    model = LogisticRegressionCV(
        Cs=list(C_GRID),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=CV_SEED),
        penalty="l2",
        solver="lbfgs",
        scoring="neg_log_loss",
        max_iter=1000,
        refit=True,
    )
    model.fit(X_fit.values, y_train.values)
    return scaler, model, mu


def predict_proba(scaler: StandardScaler, model: LogisticRegressionCV,
                  mu: dict, X: pd.DataFrame, numeric_cols: list[str]) -> np.ndarray:
    """Re-apply train-time GPR imputation, then scaler, then model."""
    X = X.copy()
    unrated_mask = X["has_gpr"] == 0
    if unrated_mask.any():
        for c in GPR_NUMERIC:
            X.loc[unrated_mask, c] = mu[c]
    X[numeric_cols] = scaler.transform(X[numeric_cols].values)
    return model.predict_proba(X.values)[:, 1]


def gpr_contrast_diagnostics(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    feature_names: list[str],
) -> dict:
    """Refit unregularized statsmodels Logit on the same scaled features
    to get standard errors + coefficient covariance for the GPR triple.

    Reported as a diagnostic — the production model stays the sklearn l2
    fit. The unregularized SE here is an honest approximation; with our
    regularization strength (C in 0.01–100) the effect on the contrast SE
    is small relative to the contrast scale we're testing against.

    Reports for each of {gpr_team, gpr_opp_team, has_gpr}:
      coefficient, standard error, z-stat
    plus the linear contrast `β_team − β_opp` with SE from the full
    coefficient covariance:
      Var(β_team − β_opp) = Var(β_team) + Var(β_opp) − 2·Cov(β_team, β_opp)
    """
    cols_to_check = ["gpr_team", "gpr_opp_team", "has_gpr"]
    missing = [c for c in cols_to_check if c not in feature_names]
    if missing:
        return {"error": f"feature columns missing: {missing}"}

    X_with_const = sm.add_constant(X_train_scaled, has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sm.Logit(y_train.values, X_with_const.values).fit(disp=False, maxiter=200)
    except Exception as e:  # convergence / separation
        return {"error": f"statsmodels Logit fit failed: {e!r}"}

    names_with_const = ["const"] + list(feature_names)
    params = pd.Series(result.params, index=names_with_const)
    bse = pd.Series(result.bse, index=names_with_const)
    cov = pd.DataFrame(result.cov_params(), index=names_with_const, columns=names_with_const)

    out: dict = {}
    for c in cols_to_check:
        beta = float(params[c])
        se = float(bse[c])
        out[c] = {
            "coef": beta,
            "se": se,
            "z": beta / se if se > 0 else float("nan"),
        }

    # β_team − β_opp contrast.
    bt, bo = float(params["gpr_team"]), float(params["gpr_opp_team"])
    var_t = float(cov.loc["gpr_team", "gpr_team"])
    var_o = float(cov.loc["gpr_opp_team", "gpr_opp_team"])
    cov_to = float(cov.loc["gpr_team", "gpr_opp_team"])
    var_contrast = var_t + var_o - 2.0 * cov_to
    se_contrast = float(np.sqrt(var_contrast)) if var_contrast > 0 else float("nan")
    contrast = bt - bo
    out["contrast_team_minus_opp"] = {
        "coef": contrast,
        "se": se_contrast,
        "z": contrast / se_contrast if se_contrast and se_contrast > 0 else float("nan"),
    }
    out["statsmodels_converged"] = bool(result.mle_retvals.get("converged", False))
    out["statsmodels_n"] = int(result.nobs)
    return out


def train_checkpoint(T: int,
                     input_dir: str = INPUT_DIR,
                     output_dir: str = OUTPUT_DIR,
                     run_id: str | None = None) -> dict:
    """Sibling fit at checkpoint T (2020-2024 train, 2025 val).

    Writes model.pkl, metrics.json, coefficients.csv, gpr_diagnostics.json,
    reliability.png.
    """
    df = pd.read_csv(os.path.join(input_dir, f"checkpoint_{T}_gpr.csv"), low_memory=False)

    train_df, val_df = split_train_val(df)
    X_train, y_train, feature_names = prepare_Xy(train_df, T)
    X_val, y_val, _ = prepare_Xy(val_df, T)

    scaler, model, mu = fit(X_train, y_train)
    C_selected = float(model.C_[0])
    numeric_cols = numeric_cols_in(X_train)

    probs = predict_proba(scaler, model, mu, X_val, numeric_cols)
    metrics = evaluate(y_val, probs)
    metrics["C_selected"] = C_selected
    metrics["train_n"] = int(len(X_train))
    metrics["val_n"] = int(len(X_val))
    metrics["train_n_rated"] = int(X_train["has_gpr"].sum())
    metrics["val_n_rated"] = int(X_val["has_gpr"].sum())

    # Re-build the train scaled frame for the statsmodels diagnostic so it
    # operates on identical features as the sklearn fit.
    X_imp, _ = _impute_unrated_to_rated_mean(X_train)
    X_scaled = X_imp.copy()
    X_scaled[numeric_cols] = scaler.transform(X_imp[numeric_cols].values)
    diagnostics = gpr_contrast_diagnostics(X_scaled, y_train, feature_names)

    ckpt_dir = os.path.join(output_dir, f"checkpoint_{T}")
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(ckpt_dir, "model.pkl"), "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "model": model,
            "mu_gpr": mu,
            "feature_names": feature_names,
            "numeric_cols": numeric_cols,
        }, f)

    with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    coefs = pd.DataFrame({"feature": feature_names, "coefficient": model.coef_[0]})
    coefs["abs_coefficient"] = coefs["coefficient"].abs()
    coefs = coefs.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    coefs.to_csv(os.path.join(ckpt_dir, "coefficients.csv"), index=False)

    with open(os.path.join(ckpt_dir, "gpr_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)

    bins = reliability_bins(y_val, probs, n_bins=10)
    _plot_reliability(bins, os.path.join(ckpt_dir, "reliability.png"),
                      label=f"baseline_gpr T={T}",
                      title=f"Reliability — baseline_logistic_gpr T={T}")

    return {**metrics, "diagnostics": diagnostics}


# ---------------------------------------------------------------------
# --fit-production — refit on 2020-2025 with optional isotonic calibrator
# ---------------------------------------------------------------------

def fit_production(T: int,
                   input_dir: str = INPUT_DIR,
                   output_dir: str = OUTPUT_DIR) -> dict:
    """GPR analogue of baseline_logistic.fit_production.

    Same calibration flow: isotonic on sibling preds over 2025-H1, evaluate
    on 2025-H2, drop calibrator if it regresses H2 logloss past the margin.
    Then refit on 2020-2025 (production) and carry the calibrator over.

    Differences from baseline:
      - reads checkpoint_{T}_gpr.csv
      - sibling/production payloads carry mu_gpr (the rated-row mean used
        for the impute-to-rated-mean trick at scaler-fit time)
    """
    sibling_dir = os.path.join(output_dir, f"checkpoint_{T}")
    with open(os.path.join(sibling_dir, "model.pkl"), "rb") as f:
        sibling = pickle.load(f)
    sib_scaler, sib_model, sib_mu = sibling["scaler"], sibling["model"], sibling["mu_gpr"]

    df = pd.read_csv(os.path.join(input_dir, f"checkpoint_{T}_gpr.csv"), low_memory=False)

    h1 = _slice_by_date(df, H1_START, H1_END)
    X_h1, y_h1, _ = prepare_Xy(h1, T)
    num_cols_h1 = numeric_cols_in(X_h1)
    probs_h1 = predict_proba(sib_scaler, sib_model, sib_mu, X_h1, num_cols_h1)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs_h1, y_h1.values)

    h2 = _slice_by_date(df, H1_END, H2_END)
    X_h2, y_h2, _ = prepare_Xy(h2, T)
    probs_h2_raw = predict_proba(sib_scaler, sib_model, sib_mu, X_h2, num_cols_h1)
    probs_h2_cal = np.asarray(iso.predict(probs_h2_raw))

    h2_raw_ll  = float(log_loss(y_h2, probs_h2_raw, labels=[0, 1]))
    h2_cal_ll  = float(log_loss(y_h2, probs_h2_cal, labels=[0, 1]))
    h2_raw_br  = float(brier_score_loss(y_h2, probs_h2_raw))
    h2_cal_br  = float(brier_score_loss(y_h2, probs_h2_cal))
    h2_raw_ece = float(ece_weighted(y_h2.values, probs_h2_raw))
    h2_cal_ece = float(ece_weighted(y_h2.values, probs_h2_cal))

    calibrator = iso
    calibrator_kind: str | None = "isotonic"
    fallback_reason: str | None = None
    if h2_cal_ll - h2_raw_ll > FALLBACK_LOGLOSS_MARGIN:
        calibrator = None
        calibrator_kind = None
        fallback_reason = (
            f"isotonic regresses H2 log-loss by {h2_cal_ll - h2_raw_ll:.4f} "
            f"(> {FALLBACK_LOGLOSS_MARGIN}); carrying None instead."
        )

    train_df = _slice_by_date(df, TRAIN_START, H2_END)
    X_tr, y_tr, feature_names = prepare_Xy(train_df, T)
    prod_scaler, prod_model, prod_mu = fit(X_tr, y_tr)
    numeric_cols = numeric_cols_in(X_tr)

    prod_ckpt = os.path.join(output_dir, "production", f"checkpoint_{T}")
    os.makedirs(prod_ckpt, exist_ok=True)

    with open(os.path.join(prod_ckpt, "model.pkl"), "wb") as f:
        pickle.dump({
            "scaler": prod_scaler,
            "model": prod_model,
            "mu_gpr": prod_mu,
            "feature_names": feature_names,
            "numeric_cols": numeric_cols,
        }, f)
    with open(os.path.join(prod_ckpt, "calibrator.pkl"), "wb") as f:
        pickle.dump(calibrator, f)

    metadata = {
        "checkpoint":                 T,
        "train_n":                    int(len(X_tr)),
        "train_n_rated":              int(X_tr["has_gpr"].sum()),
        "train_range":                f"{TRAIN_START}..{H2_END}",
        "C_selected":                 float(prod_model.C_[0]),
        "fit_timestamp":              datetime.now(timezone.utc).isoformat(),
        "calibrator_kind":            calibrator_kind,
        "calibrator_fallback_reason": fallback_reason,
        "calibration_h1_n":           int(len(X_h1)),
        "calibration_h2_n":           int(len(X_h2)),
        "h2_raw_logloss":             h2_raw_ll,
        "h2_cal_logloss":             h2_cal_ll,
        "h2_raw_brier":               h2_raw_br,
        "h2_cal_brier":               h2_cal_br,
        "h2_raw_ece":                 h2_raw_ece,
        "h2_cal_ece":                 h2_cal_ece,
        "mu_gpr":                     prod_mu,
    }
    with open(os.path.join(prod_ckpt, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


# ---------------------------------------------------------------------
# --evaluate-test-production — production model on 2026 S-tier holdout
# ---------------------------------------------------------------------

def _check_2026_lock(runs_csv: str, notes_value: str) -> None:
    """Refuse to re-run if runs.csv already has rows with notes == `notes_value`
    for model_type='baseline_logistic_gpr'. One-shot guard.
    """
    if not os.path.exists(runs_csv):
        return
    runs = pd.read_csv(runs_csv)
    existing = runs[(runs["model_type"] == "baseline_logistic_gpr")
                    & (runs["notes"].fillna("") == notes_value)]
    if len(existing):
        raise SystemExit(
            f"runs.csv already has {len(existing)} baseline_logistic_gpr row(s) with "
            f"notes='{notes_value}' — this eval is one-shot. To re-run "
            "deliberately, delete those rows and the matching metrics_2026.json / "
            "reliability_2026.png files, then invoke again."
        )


def _eval_subset(y, probs, mask) -> dict:
    """Compute metrics on a boolean-masked subset, or return zero-filled dict."""
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    sub_y = y[mask]
    sub_p = probs[mask]
    out = evaluate(sub_y, sub_p)
    out["ece"] = ece_weighted(np.asarray(sub_y), sub_p, n_bins=10)
    out["n"] = n
    return out


def _eval_production_2026(T: int, input_dir: str, prod_dir: str) -> tuple[dict, dict]:
    """Score 2026 S-tier holdout. Emit overall metrics + rated/unrated split.

    Saves metrics_2026.json with the structure:
      {
        checkpoint, calibrator_kind, C_selected, test_n,
        overall: {raw, calibrated},
        rated:   {raw, calibrated},
        unrated: {raw, calibrated},
      }
    Each leaf is the standard log_loss/brier/auc/accuracy/ece dict (+ n).
    """
    ckpt_dir = os.path.join(prod_dir, f"checkpoint_{T}")
    with open(os.path.join(ckpt_dir, "model.pkl"), "rb") as f:
        payload = pickle.load(f)
    with open(os.path.join(ckpt_dir, "calibrator.pkl"), "rb") as f:
        calibrator = pickle.load(f)
    with open(os.path.join(ckpt_dir, "metadata.json")) as f:
        meta = json.load(f)
    scaler, model, mu = payload["scaler"], payload["model"], payload["mu_gpr"]
    calibrator_kind = meta.get("calibrator_kind")

    df = pd.read_csv(os.path.join(input_dir, f"checkpoint_{T}_gpr.csv"), low_memory=False)
    test = _slice_by_date(df, TEST_START, TEST_END)
    test = test[test["tier"] == "S"].reset_index(drop=True)  # S-tier holdout per spec

    X, y, _ = prepare_Xy(test, T)
    numeric_cols = numeric_cols_in(X)
    probs_raw = predict_proba(scaler, model, mu, X, numeric_cols)
    probs_cal = (np.asarray(calibrator.predict(probs_raw))
                 if calibrator is not None else probs_raw)

    y_arr = y.values
    rated_mask = X["has_gpr"].values == 1
    unrated_mask = ~rated_mask
    overall_mask = np.ones(len(X), dtype=bool)

    out = {
        "checkpoint":      T,
        "test_n":          int(len(X)),
        "test_n_rated":    int(rated_mask.sum()),
        "test_n_unrated":  int(unrated_mask.sum()),
        "calibrator_kind": calibrator_kind,
        "C_selected":      float(model.C_[0]),
        "overall": {
            "raw":        _eval_subset(y_arr, probs_raw, overall_mask),
            "calibrated": _eval_subset(y_arr, probs_cal, overall_mask),
        },
        "rated": {
            "raw":        _eval_subset(y_arr, probs_raw, rated_mask),
            "calibrated": _eval_subset(y_arr, probs_cal, rated_mask),
        },
        "unrated": {
            "raw":        _eval_subset(y_arr, probs_raw, unrated_mask),
            "calibrated": _eval_subset(y_arr, probs_cal, unrated_mask),
        },
    }
    with open(os.path.join(ckpt_dir, "metrics_2026.json"), "w") as f:
        json.dump(out, f, indent=2)

    probs_final = probs_cal
    bins = reliability_bins(y, probs_final, n_bins=10)
    _plot_reliability(
        bins,
        os.path.join(ckpt_dir, "reliability_2026.png"),
        label=f"production GPR T={T} on 2026 S-tier (cal={calibrator_kind or 'none'})",
        title=f"Reliability — baseline_logistic_gpr production T={T} on 2026 S-tier",
    )
    return out, meta


def evaluate_test_production_all(input_dir: str = INPUT_DIR,
                                 output_dir: str = OUTPUT_DIR,
                                 runs_csv: str = RUNS_CSV) -> dict[int, dict]:
    """Atomic 2026 S-tier eval of the GPR production model across all 4 T."""
    _check_2026_lock(runs_csv, "2026 test production S-tier")
    prod_dir = os.path.join(output_dir, "production")
    run_id = (datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
              + "_baseline_logistic_gpr_production_2026")
    results: dict[int, dict] = {}
    for T in CHECKPOINTS:
        out, meta = _eval_production_2026(T, input_dir, prod_dir)
        served = (out["overall"]["calibrated"]
                  if out["calibrator_kind"] is not None else out["overall"]["raw"])
        _append_runs_row({
            "run_id":       run_id,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "model_type":   "baseline_logistic_gpr",
            "checkpoint":   T,
            "hparams_json": json.dumps({
                "penalty": "l2", "solver": "lbfgs", "max_iter": 1000,
                "C_selected":      out["C_selected"],
                "train_range":     meta.get("train_range"),
                "calibrator_kind": out["calibrator_kind"],
            }),
            "train_n":      int(meta["train_n"]),
            "val_n":        out["test_n"],
            "val_logloss":  served["log_loss"],
            "val_brier":    served["brier"],
            "val_auc":      served["auc"],
            "val_accuracy": served["accuracy"],
            "notes":        "2026 test production S-tier",
        }, runs_csv)
        results[T] = out
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--runs-csv", default=RUNS_CSV)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fit-production", action="store_true",
                      help="Refit on 2020-2025 + optional isotonic; writes production/ artifacts.")
    mode.add_argument("--evaluate-test-production", action="store_true",
                      help="Atomic production 2026 S-tier eval across all 4 T (touched once).")
    args = parser.parse_args()

    if args.fit_production:
        print(f"{'T':>3} {'train_n':>8} {'rated':>6} {'C':>8} "
              f"{'h2_raw_ll':>9} {'h2_cal_ll':>9} "
              f"{'h2_raw_ece':>10} {'h2_cal_ece':>10} calibrator")
        for T in CHECKPOINTS:
            m = fit_production(T, args.input_dir, args.output_dir)
            print(f"{T:>3} {m['train_n']:>8,} {m['train_n_rated']:>6,} "
                  f"{m['C_selected']:>8.4f} "
                  f"{m['h2_raw_logloss']:>9.4f} {m['h2_cal_logloss']:>9.4f} "
                  f"{m['h2_raw_ece']:>10.4f} {m['h2_cal_ece']:>10.4f} "
                  f"{m['calibrator_kind'] or 'none'}"
                  + (f"  ({m['calibrator_fallback_reason']})"
                     if m['calibrator_fallback_reason'] else ""))
        return

    if args.evaluate_test_production:
        results = evaluate_test_production_all(args.input_dir, args.output_dir, args.runs_csv)
        print(f"\n=== Overall (S-tier) ===")
        print(f"{'T':>3} {'n':>5} {'rated':>5} "
              f"{'raw_ll':>7} {'cal_ll':>7} {'raw_ece':>8} {'cal_ece':>8} cal")
        for T, m in results.items():
            o = m["overall"]
            print(f"{T:>3} {m['test_n']:>5,} {m['test_n_rated']:>5,} "
                  f"{o['raw']['log_loss']:>7.4f} {o['calibrated']['log_loss']:>7.4f} "
                  f"{o['raw']['ece']:>8.4f} {o['calibrated']['ece']:>8.4f} "
                  f"{m['calibrator_kind'] or 'none'}")
        print(f"\n=== Rated (has_gpr=1) ===")
        print(f"{'T':>3} {'n':>5} {'raw_ll':>7} {'cal_ll':>7} {'raw_auc':>8} {'cal_auc':>8}")
        for T, m in results.items():
            r = m["rated"]
            if r["raw"].get("n", 0) == 0:
                print(f"{T:>3} {0:>5}  (empty)")
                continue
            print(f"{T:>3} {r['raw']['n']:>5,} "
                  f"{r['raw']['log_loss']:>7.4f} {r['calibrated']['log_loss']:>7.4f} "
                  f"{r['raw']['auc']:>8.4f} {r['calibrated']['auc']:>8.4f}")
        print(f"\n=== Unrated (has_gpr=0) ===")
        print(f"{'T':>3} {'n':>5} {'raw_ll':>7} {'cal_ll':>7} {'raw_auc':>8} {'cal_auc':>8}")
        for T, m in results.items():
            u = m["unrated"]
            if u["raw"].get("n", 0) == 0:
                print(f"{T:>3} {0:>5}  (empty)")
                continue
            print(f"{T:>3} {u['raw']['n']:>5,} "
                  f"{u['raw']['log_loss']:>7.4f} {u['calibrated']['log_loss']:>7.4f} "
                  f"{u['raw']['auc']:>8.4f} {u['calibrated']['auc']:>8.4f}")
        return

    # Default: train sibling per checkpoint.
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ") + "_baseline_logistic_gpr"
    print(f"run_id = {run_id}")
    for T in CHECKPOINTS:
        print(f"\n=== checkpoint T={T} ===")
        m = train_checkpoint(T, args.input_dir, args.output_dir, run_id=run_id)
        print(f"  C_selected={m['C_selected']:g}  "
              f"train_n={m['train_n']:,} ({m['train_n_rated']:,} rated)  "
              f"val_n={m['val_n']:,} ({m['val_n_rated']:,} rated)")
        print(f"  log_loss={m['log_loss']:.4f}  brier={m['brier']:.4f}  "
              f"auc={m['auc']:.4f}  acc={m['accuracy']:.4f}")
        d = m["diagnostics"]
        if "error" in d:
            print(f"  GPR diagnostic ERROR: {d['error']}")
            continue
        print(f"  GPR coefs (statsmodels SE):")
        for c in ("gpr_team", "gpr_opp_team", "has_gpr"):
            x = d[c]
            print(f"    {c:<22} coef={x['coef']:+.4f}  SE={x['se']:.4f}  z={x['z']:+.2f}")
        c = d["contrast_team_minus_opp"]
        print(f"    {'b_team - b_opp':<22} coef={c['coef']:+.4f}  SE={c['se']:.4f}  z={c['z']:+.2f}")


if __name__ == "__main__":
    main()

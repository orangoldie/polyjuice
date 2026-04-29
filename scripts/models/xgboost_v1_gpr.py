"""
XGBoost v1 + GPR per-checkpoint win-probability model.

Diagnostic clone of xgboost_v1.py with the four GPR columns added to the
feature matrix. Per docs/specs/gpr-feature.md, tree models keep all four
(`gpr_team`, `gpr_opp_team`, `gpr_diff`, `has_gpr`) — no collinearity issue
under axis-aligned splits, and the diff column makes a useful split feature.

No scaling, no rated-only-scaler trick (those are linear-model concerns).
Unrated rows have `gpr_team`/`gpr_opp_team`/`gpr_diff` = 0 from stage 5;
trees can carve `has_gpr` directly to gate the GPR signal.

Reads checkpoint_{T}_gpr.csv. Writes to data/models/xgboost_v1_gpr/.

Acceptance gate uses baseline_logistic_gpr's H2 raw-logloss numbers (from
data/models/baseline_logistic_gpr/production/checkpoint_{T}/metadata.json)
so we're comparing XGB-with-GPR to the linear-with-GPR baseline.

CLI mirrors xgboost_v1.py.
"""
import argparse
import json
import os
import pickle
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402

# Reuse all the helpers from xgboost_v1; we only override prepare_Xy / paths /
# acceptance gate / a couple of file names. Keeping the helpers in one place
# avoids drift between the two model scripts.
from scripts.models.xgboost_v1 import (
    CALIBRATION_ECE_THRESHOLD,
    CHECKPOINTS,
    EARLY_STOPPING_ROUNDS,
    FIRST_GATES,
    LOGLOSS_MARGIN,
    N_ESTIMATORS_MAX,
    N_TRIALS,
    SEED,
    SIDE_CATEGORIES,
    SPLIT_CUTOFFS,
    TIER_CATEGORIES,
    _DIFF_NAMES,
    _PlattCalibrator,
    _apply_calibrator,
    _append_runs_row,
    _feature_importance_df,
    _fit_platt,
    _plot_reliability,
    _trial_params,
    _xgb_base_params,
    ece_weighted,
    evaluate,
    isotonic_is_pathological,
    reliability_bins,
    split_temporal,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_DIR = os.path.join(REPO_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "models", "xgboost_v1_gpr")
PROD_DIR = os.path.join(OUTPUT_DIR, "production")
RUNS_CSV = os.path.join(REPO_ROOT, "data", "experiments", "runs.csv")
BASELINE_LOGISTIC_GPR_DIR = os.path.join(REPO_ROOT, "data", "models",
                                          "baseline_logistic_gpr", "production")

GPR_NUMERIC = ("gpr_team", "gpr_opp_team", "gpr_diff")
GPR_BINARY = ("has_gpr",)


def NUMERIC_FEATURES(T: int) -> list[str]:
    """xgboost_v1's 12 in-game cols + 3 GPR numerics."""
    feats = []
    for stat in ("gold", "cs", "kills", "assists"):
        feats.extend([
            f"{stat}at{T}",
            f"opp_{stat}at{T}",
            f"{_DIFF_NAMES[stat]}at{T}",
        ])
    feats.extend(GPR_NUMERIC)
    return feats


def CATEGORICAL_FEATURES() -> list[str]:
    return ["side", "tier"]


def BINARY_FEATURES(T: int) -> list[str]:
    return list(FIRST_GATES[T]) + list(GPR_BINARY)


def prepare_Xy(df: pd.DataFrame, T: int) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    X = pd.DataFrame(index=df.index)
    for col in NUMERIC_FEATURES(T):
        X[col] = df[col].astype(float)
    X["side"] = pd.Categorical(df["side"], categories=SIDE_CATEGORIES)
    X["tier"] = pd.Categorical(df["tier"], categories=TIER_CATEGORIES)
    for flag in BINARY_FEATURES(T):
        X[flag] = df[flag].astype(int)

    y = df["result"].astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)
    return X, y, list(X.columns)


def _load_baseline_h2() -> dict[int, dict]:
    """Read baseline_logistic_gpr's per-T H2 raw logloss / brier from metadata.

    Used as the acceptance gate so we compare XGB-with-GPR to logistic-with-GPR.
    """
    out = {}
    for T in CHECKPOINTS:
        with open(os.path.join(BASELINE_LOGISTIC_GPR_DIR, f"checkpoint_{T}",
                               "metadata.json")) as f:
            meta = json.load(f)
        out[T] = {"logloss": float(meta["h2_raw_logloss"]),
                  "brier":   float(meta["h2_raw_brier"])}
    return out


def check_acceptance(T: int, raw_metrics: dict, baseline_h2: dict) -> tuple[bool, dict]:
    baseline = baseline_h2[T]
    logloss_diff = raw_metrics["log_loss"] - baseline["logloss"]
    brier_diff   = raw_metrics["brier"]    - baseline["brier"]
    logloss_ok = logloss_diff <= -LOGLOSS_MARGIN
    brier_ok   = brier_diff   <= 0.0
    return bool(logloss_ok and brier_ok), {
        "baseline_logloss": baseline["logloss"],
        "baseline_brier":   baseline["brier"],
        "xgb_logloss":      float(raw_metrics["log_loss"]),
        "xgb_brier":        float(raw_metrics["brier"]),
        "logloss_diff":     float(logloss_diff),
        "brier_diff":       float(brier_diff),
        "logloss_margin":   LOGLOSS_MARGIN,
        "logloss_ok":       bool(logloss_ok),
        "brier_ok":         bool(brier_ok),
    }


class _OptunaPruningCallback(xgb.callback.TrainingCallback):
    def __init__(self, trial: "optuna.Trial",
                 observation_key: str = "validation_0-logloss"):
        super().__init__()
        self.trial = trial
        self.observation_key = observation_key
        self.pruned = False

    def after_iteration(self, model, epoch, evals_log):
        ds, metric = self.observation_key.split("-", 1)
        if ds in evals_log and metric in evals_log[ds]:
            score = evals_log[ds][metric][-1]
            self.trial.report(float(score), epoch)
            if self.trial.should_prune():
                self.pruned = True
                return True
        return False


def make_objective(T: int, splits: dict[str, pd.DataFrame]):
    X_tr, y_tr, _ = prepare_Xy(splits["train"], T)
    X_es, y_es, _ = prepare_Xy(splits["early_stop_val"], T)
    X_opt, y_opt, _ = prepare_Xy(splits["optuna_val"], T)

    def objective(trial: "optuna.Trial") -> float:
        params = _trial_params(trial)
        callback = _OptunaPruningCallback(trial)
        model = xgb.XGBClassifier(
            **_xgb_base_params(),
            **params,
            n_estimators=N_ESTIMATORS_MAX,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            callbacks=[callback],
        )
        model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
        if callback.pruned:
            raise optuna.TrialPruned()
        n_estimators_selected = int(model.best_iteration) + 1
        trial.set_user_attr("n_estimators_selected", n_estimators_selected)
        probs = model.predict_proba(X_opt)[:, 1]
        return float(log_loss(y_opt.values, probs, labels=[0, 1]))

    return objective


def tune_checkpoint(T: int, splits: dict[str, pd.DataFrame],
                    n_trials: int = N_TRIALS, seed: int = SEED) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(make_objective(T, splits), n_trials=n_trials, show_progress_bar=False)
    return study


def fit_calibrated(T: int, study: optuna.Study, splits: dict[str, pd.DataFrame],
                   ece_threshold: float = CALIBRATION_ECE_THRESHOLD):
    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    n_estimators_selected = int(best_trial.user_attrs["n_estimators_selected"])

    combined = pd.concat([splits["train"], splits["early_stop_val"]], ignore_index=True)
    X_tc, y_tc, _ = prepare_Xy(combined, T)

    model = xgb.XGBClassifier(
        **_xgb_base_params(),
        **best_params,
        n_estimators=n_estimators_selected,
    )
    model.fit(X_tc, y_tc, verbose=False)

    X_opt, y_opt, _ = prepare_Xy(splits["optuna_val"], T)
    raw_opt_probs = model.predict_proba(X_opt)[:, 1]

    if ece_weighted(y_opt.values, raw_opt_probs) <= ece_threshold:
        return model, None, None, n_estimators_selected

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_opt_probs, y_opt.values)
    if isotonic_is_pathological(iso):
        return model, _fit_platt(raw_opt_probs, y_opt.values), "platt", n_estimators_selected
    return model, iso, "isotonic", n_estimators_selected


def train_checkpoint(T: int,
                     input_dir: str = INPUT_DIR,
                     output_dir: str = OUTPUT_DIR,
                     runs_csv: str = RUNS_CSV,
                     run_id: str | None = None,
                     n_trials: int = N_TRIALS,
                     baseline_h2: dict | None = None) -> dict:
    df = pd.read_csv(os.path.join(input_dir, f"checkpoint_{T}_gpr.csv"),
                     low_memory=False)
    splits = split_temporal(df)

    study = tune_checkpoint(T, splits, n_trials=n_trials)
    model, calibrator, calibrator_kind, n_estimators_selected = fit_calibrated(
        T, study, splits
    )

    X_acc, y_acc, _ = prepare_Xy(splits["acceptance_val"], T)
    raw_probs = model.predict_proba(X_acc)[:, 1]
    cal_probs = _apply_calibrator(calibrator, raw_probs)

    raw_metrics = evaluate(y_acc.values, raw_probs)
    cal_metrics = evaluate(y_acc.values, cal_probs)
    if baseline_h2 is None:
        baseline_h2 = _load_baseline_h2()
    passed, diffs = check_acceptance(T, raw_metrics, baseline_h2)

    metrics = {
        "best_params":            dict(study.best_trial.params),
        "n_estimators_selected":  int(n_estimators_selected),
        "optuna_val_n":           int(len(splits["optuna_val"])),
        "acceptance_val_n":       int(len(X_acc)),
        "raw":                    raw_metrics,
        "cal":                    cal_metrics,
        "calibration_applied":    calibrator is not None,
        "calibrator_kind":        calibrator_kind,
        "acceptance_passed":      passed,
        "acceptance_diffs":       diffs,
    }

    ckpt_dir = os.path.join(output_dir, f"checkpoint_{T}")
    os.makedirs(ckpt_dir, exist_ok=True)

    model.save_model(os.path.join(ckpt_dir, "model.json"))
    with open(os.path.join(ckpt_dir, "calibrator.pkl"), "wb") as f:
        pickle.dump(calibrator, f)
    with open(os.path.join(ckpt_dir, "study.pkl"), "wb") as f:
        pickle.dump(study, f)
    with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    _feature_importance_df(model).to_csv(
        os.path.join(ckpt_dir, "feature_importance.csv"), index=False
    )
    _plot_reliability(
        y_acc.values, raw_probs, cal_probs if calibrator is not None else None, T,
        os.path.join(ckpt_dir, "reliability_val.png"),
        title=f"Reliability — xgboost_v1_gpr T={T} on acceptance val (2025-H2)",
    )

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ") + "_xgboost_v1_gpr"
    _append_runs_row({
        "run_id":          run_id,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "model_type":      "xgboost_v1_gpr",
        "checkpoint":      T,
        "hparams_json":    json.dumps({
            **metrics["best_params"],
            "n_estimators_selected": metrics["n_estimators_selected"],
        }),
        "train_n":         int(len(splits["train"]) + len(splits["early_stop_val"])),
        "val_n":           metrics["acceptance_val_n"],
        "val_logloss":     raw_metrics["log_loss"],
        "val_brier":       raw_metrics["brier"],
        "val_auc":         raw_metrics["auc"],
        "val_accuracy":    raw_metrics["accuracy"],
        "val_logloss_cal": cal_metrics["log_loss"],
        "val_brier_cal":   cal_metrics["brier"],
        "notes":           f"acceptance_passed={passed}",
    }, runs_csv)

    return metrics


def _load_val_stage(ckpt_dir: str):
    model = xgb.XGBClassifier(enable_categorical=True)
    model.load_model(os.path.join(ckpt_dir, "model.json"))
    with open(os.path.join(ckpt_dir, "calibrator.pkl"), "rb") as f:
        calibrator = pickle.load(f)
    with open(os.path.join(ckpt_dir, "metrics.json")) as f:
        metrics = json.load(f)
    return model, calibrator, metrics


def evaluate_test_all(input_dir: str = INPUT_DIR,
                      output_dir: str = OUTPUT_DIR,
                      stier_only: bool = False) -> dict[int, dict]:
    """Score 2026 test set across all 4 T using the sibling val-stage model.

    Unlike xgboost_v1's gated version, this runs regardless of acceptance —
    we want the 2026 numbers as a diagnostic even if the model didn't beat
    the gate. Pass `stier_only=True` to filter the test slice to tier=='S',
    which is what we want for apples-to-apples comparison with
    baseline_logistic_gpr's S-tier-only holdout.
    """
    results: dict[int, dict] = {}
    for T in CHECKPOINTS:
        ckpt_dir = os.path.join(output_dir, f"checkpoint_{T}")
        model, calibrator, metrics = _load_val_stage(ckpt_dir)

        df = pd.read_csv(os.path.join(input_dir, f"checkpoint_{T}_gpr.csv"),
                         low_memory=False)
        splits = split_temporal(df)
        test_df = splits["test"]
        if stier_only:
            test_df = test_df[test_df["tier"] == "S"].reset_index(drop=True)
        X_test, y_test, _ = prepare_Xy(test_df, T)
        raw_probs = model.predict_proba(X_test)[:, 1]
        cal_probs = _apply_calibrator(calibrator, raw_probs)

        # Rated/unrated split alongside overall.
        has_gpr = X_test["has_gpr"].values == 1
        results[T] = {
            "n":           int(len(X_test)),
            "n_rated":     int(has_gpr.sum()),
            "raw":         evaluate(y_test.values, raw_probs),
            "cal":         evaluate(y_test.values, cal_probs),
            "raw_rated":   evaluate(y_test.values[has_gpr], raw_probs[has_gpr]) if has_gpr.any() else None,
            "raw_unrated": evaluate(y_test.values[~has_gpr], raw_probs[~has_gpr]) if (~has_gpr).any() else None,
            "calibrator_kind": metrics.get("calibrator_kind"),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--runs-csv", default=RUNS_CSV)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--evaluate-test-stier", action="store_true",
                        help="Score 2026 S-tier holdout with the sibling model.")
    args = parser.parse_args()

    if args.evaluate_test_stier:
        results = evaluate_test_all(args.input_dir, args.output_dir, stier_only=True)
        print(f"\n=== xgboost_v1_gpr on 2026 S-tier holdout ===")
        print(f"{'T':>3} {'n':>5} {'rated':>5} "
              f"{'raw_ll':>8} {'cal_ll':>8} {'raw_auc':>8} "
              f"{'rated_ll':>9} {'unrated_ll':>11} cal")
        for T, r in results.items():
            rr = r["raw_rated"] or {"log_loss": float("nan")}
            ru = r["raw_unrated"] or {"log_loss": float("nan")}
            print(f"{T:>3} {r['n']:>5,} {r['n_rated']:>5,} "
                  f"{r['raw']['log_loss']:>8.4f} {r['cal']['log_loss']:>8.4f} "
                  f"{r['raw']['auc']:>8.4f} "
                  f"{rr['log_loss']:>9.4f} {ru['log_loss']:>11.4f} "
                  f"{r['calibrator_kind'] or 'none'}")
        return

    baseline_h2 = _load_baseline_h2()
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ") + "_xgboost_v1_gpr"
    print(f"run_id = {run_id}  baseline_h2 from baseline_logistic_gpr/production")
    for T in CHECKPOINTS:
        print(f"\n=== checkpoint T={T} (n_trials={args.n_trials}) ===")
        m = train_checkpoint(T, args.input_dir, args.output_dir, args.runs_csv,
                             run_id=run_id, n_trials=args.n_trials,
                             baseline_h2=baseline_h2)
        print(f"  best_params={m['best_params']}")
        print(f"  n_estimators_selected={m['n_estimators_selected']}  "
              f"acceptance_val_n={m['acceptance_val_n']:,}")
        print(f"  raw:  logloss={m['raw']['log_loss']:.4f}  brier={m['raw']['brier']:.4f}  "
              f"auc={m['raw']['auc']:.4f}  ece={m['raw']['ece']:.4f}")
        print(f"  cal:  logloss={m['cal']['log_loss']:.4f}  brier={m['cal']['brier']:.4f}  "
              f"ece={m['cal']['ece']:.4f}  ({m['calibrator_kind']})")
        print(f"  acceptance_passed={m['acceptance_passed']}  "
              f"diffs={m['acceptance_diffs']}")


if __name__ == "__main__":
    main()

"""Tests for scripts/models/xgboost_v1.py.

Mirrors baseline test conventions: small synthetic fixtures, no real-CSV
slices. See docs/xgboost-v1.md for the spec.

Optuna / XGBoost fits are kept tiny (few-trial studies on few-hundred-row
fixtures) to keep the suite fast.
"""
import json
import pickle

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from scripts.models.xgboost_v1 import (
    BASELINE_H2,
    BINARY_FEATURES,
    CALIBRATION_ECE_THRESHOLD,
    CATEGORICAL_FEATURES,
    CHECKPOINTS,
    FIRST_GATES,
    LOGLOSS_MARGIN,
    NUMERIC_FEATURES,
    SIDE_CATEGORIES,
    TIER_CATEGORIES,
    check_acceptance,
    ece_weighted,
    evaluate,
    evaluate_test_all,
    fit_calibrated,
    fit_production,
    isotonic_is_pathological,
    make_objective,
    prepare_Xy,
    reliability_bins,
    split_temporal,
    train_checkpoint,
    tune_checkpoint,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

def make_row(T, gameid="g1", date="2022-06-01", tier="S", side="Blue",
             result=1, gold_diff=3000.0, **overrides):
    row = {
        "gameid":       gameid,
        "date":         date,
        "year":         int(date[:4]),
        "league":       "LCK",
        "patch":        "13.1",
        "teamname":     "T1",
        "opp_teamname": "GenG",
        "side":         side,
        "tier":         tier,
        f"goldat{T}":      20000.0 + gold_diff / 2,
        f"opp_goldat{T}":  20000.0 - gold_diff / 2,
        f"golddiffat{T}":  float(gold_diff),
        f"csat{T}":        8.0 * T,
        f"opp_csat{T}":    7.0 * T,
        f"csdiffat{T}":    float(T),
        f"killsat{T}":     float(T // 5),
        f"opp_killsat{T}": float(max(0, (T // 5) - 1)),
        f"killdiffat{T}":  1.0,
        f"assistsat{T}":     float((T // 5) + 2),
        f"opp_assistsat{T}": float(T // 5),
        f"assistdiffat{T}":  2.0,
        "gamelength":   T * 60 + 100,
        "result":       result,
    }
    for flag in FIRST_GATES[T]:
        row[flag] = 1 if flag == "firstblood" else 0
    row.update(overrides)
    return row


# Date within each split, keyed by split name.
_SPLIT_SAMPLE_DATES = {
    "train":          ["2020-02-01", "2021-06-01", "2022-08-15", "2023-03-10", "2024-05-20"],
    "early_stop_val": ["2024-08-10", "2024-10-01", "2024-12-20"],
    "optuna_val":     ["2025-02-14", "2025-04-01", "2025-06-20"],
    "acceptance_val": ["2025-08-01", "2025-10-15", "2025-12-20"],
    "test":           ["2026-02-14", "2026-06-01", "2026-11-10"],
}


def synthetic_splits_frame(T, per_date=20, seed=0):
    """Multi-split frame: rows spread across all 5 temporal buckets.

    gold_diff > 0 → result=1 (learnable signal).
    """
    rng = np.random.default_rng(seed)
    rows = []
    gid = 0
    for _split, dates in _SPLIT_SAMPLE_DATES.items():
        for d in dates:
            for _ in range(per_date):
                diff = float(rng.normal(0, 2500))
                winner = 1 if diff > 0 else 0
                rows.append(make_row(
                    T,
                    gameid=f"g_{gid}",
                    date=d,
                    tier=str(rng.choice(TIER_CATEGORIES)),
                    side=str(rng.choice(SIDE_CATEGORIES)),
                    result=winner,
                    gold_diff=diff,
                ))
                gid += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Feature-selection functions
# ---------------------------------------------------------------------

@pytest.mark.parametrize("T", CHECKPOINTS)
def test_numeric_features_twelve_raw_opp_diff(T):
    feats = NUMERIC_FEATURES(T)
    assert len(feats) == 12
    for stat in ("gold", "cs", "kills", "assists"):
        assert f"{stat}at{T}" in feats
        assert f"opp_{stat}at{T}" in feats
    for stat in ("golddiff", "csdiff", "killdiff", "assistdiff"):
        assert f"{stat}at{T}" in feats, f"{stat}at{T} (tree-friendly) must be kept"


def test_categorical_features_side_tier():
    assert CATEGORICAL_FEATURES() == ["side", "tier"]


@pytest.mark.parametrize("T,expected", [
    (10, ["firstblood"]),
    (15, ["firstblood", "firstdragon", "firsttower"]),
    (20, ["firstblood", "firstdragon", "firsttower"]),
    (25, ["firstblood", "firstdragon", "firsttower", "firstbaron"]),
])
def test_binary_features_gated(T, expected):
    assert BINARY_FEATURES(T) == expected


# ---------------------------------------------------------------------
# prepare_Xy
# ---------------------------------------------------------------------

def test_prepare_Xy_categoricals_typed_correctly():
    df = pd.DataFrame([make_row(15, tier="A", side="Red")])
    X, _, _ = prepare_Xy(df, 15)
    assert isinstance(X["side"].dtype, pd.CategoricalDtype)
    assert isinstance(X["tier"].dtype, pd.CategoricalDtype)
    assert list(X["side"].cat.categories) == SIDE_CATEGORIES
    assert list(X["tier"].cat.categories) == TIER_CATEGORIES


@pytest.mark.parametrize("T", CHECKPOINTS)
def test_prepare_Xy_diff_columns_present(T):
    df = pd.DataFrame([make_row(T)])
    X, _, _ = prepare_Xy(df, T)
    for stat in ("golddiff", "csdiff", "killdiff", "assistdiff"):
        assert f"{stat}at{T}" in X.columns, f"{stat}at{T} kept for trees"


def test_prepare_Xy_gated_firsts_at_10():
    df = pd.DataFrame([make_row(10)])
    X, _, _ = prepare_Xy(df, 10)
    assert "firstblood" in X.columns
    for flag in ("firstdragon", "firsttower", "firstbaron"):
        assert flag not in X.columns


def test_prepare_Xy_gated_firsts_at_25():
    df = pd.DataFrame([make_row(25)])
    X, _, _ = prepare_Xy(df, 25)
    for flag in ("firstblood", "firstdragon", "firsttower", "firstbaron"):
        assert flag in X.columns


def test_prepare_Xy_identifiers_not_in_features():
    df = pd.DataFrame([make_row(15)])
    X, _, _ = prepare_Xy(df, 15)
    for col in ("gameid", "date", "year", "league", "patch",
                "teamname", "opp_teamname", "gamelength", "result"):
        assert col not in X.columns


def test_prepare_Xy_row_count_preserved():
    rows = [make_row(15, gameid=f"g{i}") for i in range(7)]
    X, y, _ = prepare_Xy(pd.DataFrame(rows), 15)
    assert len(X) == 7 and len(y) == 7


def test_prepare_Xy_y_is_result():
    df = pd.DataFrame([make_row(15, result=1, gameid="a"),
                       make_row(15, result=0, gameid="b")])
    _, y, _ = prepare_Xy(df, 15)
    assert list(y) == [1, 0]


# ---------------------------------------------------------------------
# split_temporal
# ---------------------------------------------------------------------

def test_split_temporal_all_five_populated_on_fixture():
    df = synthetic_splits_frame(15, per_date=5)
    splits = split_temporal(df)
    for name in ("train", "early_stop_val", "optuna_val", "acceptance_val", "test"):
        assert len(splits[name]) > 0, f"{name} unexpectedly empty"


def test_split_temporal_no_overlap():
    df = synthetic_splits_frame(15, per_date=5)
    splits = split_temporal(df)
    seen = set()
    for name, part in splits.items():
        ids = set(part["gameid"])
        assert seen.isdisjoint(ids), f"{name} overlaps earlier splits"
        seen |= ids


def test_split_temporal_boundary_dates():
    rows = [
        make_row(15, gameid="t1", date="2024-06-30"),   # train (last day)
        make_row(15, gameid="es1", date="2024-07-01"),  # early_stop_val (first day)
        make_row(15, gameid="es2", date="2024-12-31"),  # early_stop_val (last day)
        make_row(15, gameid="o1", date="2025-01-01"),   # optuna_val (first day)
        make_row(15, gameid="o2", date="2025-06-30"),   # optuna_val (last day)
        make_row(15, gameid="a1", date="2025-07-01"),   # acceptance_val (first day)
        make_row(15, gameid="a2", date="2025-12-31"),   # acceptance_val (last day)
        make_row(15, gameid="te1", date="2026-01-01"),  # test (first day)
    ]
    splits = split_temporal(pd.DataFrame(rows))
    assert list(splits["train"]["gameid"]) == ["t1"]
    assert list(splits["early_stop_val"]["gameid"]) == ["es1", "es2"]
    assert list(splits["optuna_val"]["gameid"]) == ["o1", "o2"]
    assert list(splits["acceptance_val"]["gameid"]) == ["a1", "a2"]
    assert list(splits["test"]["gameid"]) == ["te1"]


# ---------------------------------------------------------------------
# ece_weighted
# ---------------------------------------------------------------------

def test_ece_weighted_perfect_is_zero():
    # 1000 points where empirical rate within each bin == mean prob.
    rng = np.random.default_rng(1)
    probs = rng.random(10_000)
    y = (rng.random(10_000) < probs).astype(int)
    ece = ece_weighted(y, probs, n_bins=10)
    assert ece < 0.02  # asymptotically 0; finite sample noise


def test_ece_weighted_known_input():
    # Two bins: [0, 0.5) all pred=0.25 actual=0, [0.5, 1.0] all pred=0.75 actual=1.
    # With 2 bins: gap=(|0.25-0|, |0.75-1|)=(0.25, 0.25), weights 0.5 each → ECE=0.25.
    probs = np.array([0.25]*4 + [0.75]*4)
    y     = np.array([0]*4    + [1]*4)
    assert ece_weighted(y, probs, n_bins=2) == pytest.approx(0.25)


def test_ece_weighted_all_miscalibrated():
    # Predict 0.9, always lose → gap = 0.9, one bin.
    probs = np.full(100, 0.9)
    y     = np.zeros(100, dtype=int)
    assert ece_weighted(y, probs, n_bins=10) == pytest.approx(0.9)


# ---------------------------------------------------------------------
# reliability_bins
# ---------------------------------------------------------------------

def test_reliability_bins_counts_and_columns():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    probs = rng.random(500)
    out = reliability_bins(y, probs, n_bins=10)
    assert {"bin_lo", "bin_hi", "n", "mean_pred", "mean_actual"}.issubset(set(out.columns))
    assert len(out) == 10
    assert int(out["n"].sum()) == 500


# ---------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------

def test_evaluate_perfect():
    y = np.array([0, 1, 0, 1])
    probs = np.array([0.01, 0.99, 0.01, 0.99])
    m = evaluate(y, probs)
    assert m["log_loss"] < 0.05
    assert m["brier"] < 0.01
    assert m["auc"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["ece"] < 0.05


def test_evaluate_all_wrong():
    y = np.array([0, 1, 0, 1])
    probs = np.array([0.99, 0.01, 0.99, 0.01])
    m = evaluate(y, probs)
    assert m["log_loss"] > 3.0
    assert m["accuracy"] == 0.0


def test_evaluate_random_auc_near_half():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 2000)
    probs = rng.random(2000)
    m = evaluate(y, probs)
    assert 0.45 < m["auc"] < 0.55


# ---------------------------------------------------------------------
# isotonic_is_pathological
# ---------------------------------------------------------------------

def test_isotonic_pathological_too_few_distinct_outputs():
    # Collapsed isotonic — only 2 unique outputs.
    iso = IsotonicRegression(out_of_bounds="clip")
    xs = np.linspace(0.05, 0.95, 50)
    ys = (xs > 0.5).astype(int)
    iso.fit(xs, ys)
    assert isotonic_is_pathological(iso)


def test_isotonic_not_pathological_on_healthy_fit():
    rng = np.random.default_rng(3)
    raw = rng.random(2000)
    actual = (rng.random(2000) < raw).astype(int)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw, actual)
    assert not isotonic_is_pathological(iso)


# ---------------------------------------------------------------------
# check_acceptance
# ---------------------------------------------------------------------

def test_check_acceptance_passes_when_both_clear():
    T = 15
    raw = {"log_loss": BASELINE_H2[T]["logloss"] - LOGLOSS_MARGIN - 0.001,
           "brier":    BASELINE_H2[T]["brier"] - 0.001}
    passed, diffs = check_acceptance(T, raw)
    assert passed is True
    assert diffs["logloss_ok"] is True
    assert diffs["brier_ok"] is True
    assert diffs["logloss_diff"] < 0
    assert diffs["brier_diff"] < 0


def test_check_acceptance_fails_logloss_too_close():
    T = 15
    raw = {"log_loss": BASELINE_H2[T]["logloss"] - LOGLOSS_MARGIN / 2,
           "brier":    BASELINE_H2[T]["brier"] - 0.001}
    passed, diffs = check_acceptance(T, raw)
    assert passed is False
    assert diffs["logloss_ok"] is False
    assert diffs["brier_ok"] is True


def test_check_acceptance_fails_brier_regresses():
    T = 15
    raw = {"log_loss": BASELINE_H2[T]["logloss"] - LOGLOSS_MARGIN - 0.01,
           "brier":    BASELINE_H2[T]["brier"] + 0.001}
    passed, diffs = check_acceptance(T, raw)
    assert passed is False
    assert diffs["logloss_ok"] is True
    assert diffs["brier_ok"] is False


def test_check_acceptance_reads_raw_not_cal():
    """Fail if someone accidentally threaded calibrated metrics in."""
    T = 15
    raw = {"log_loss": BASELINE_H2[T]["logloss"] + 0.05,
           "brier":    BASELINE_H2[T]["brier"] + 0.01}
    passed, _ = check_acceptance(T, raw)
    assert passed is False


# ---------------------------------------------------------------------
# make_objective + tune_checkpoint
# ---------------------------------------------------------------------

def test_make_objective_returns_float_and_respects_params():
    df = synthetic_splits_frame(15, per_date=15, seed=0)
    splits = split_temporal(df)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    objective = make_objective(15, splits)
    study.optimize(objective, n_trials=1)
    trial = study.best_trial
    # Returns a finite float.
    assert np.isfinite(trial.value)
    # Records the early-stop-selected tree count.
    assert trial.user_attrs.get("n_estimators_selected", 0) >= 1
    # Records at least one intermediate value (pruner feed).
    assert len(trial.intermediate_values) >= 1


def test_tune_checkpoint_three_trial_study():
    df = synthetic_splits_frame(15, per_date=15, seed=0)
    splits = split_temporal(df)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = tune_checkpoint(15, splits, n_trials=3, seed=0)
    assert study.best_trial is not None
    assert np.isfinite(study.best_value)
    assert 0 < len(study.trials) <= 3


# ---------------------------------------------------------------------
# fit_calibrated
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def well_calibrated_study_splits():
    """Reusable (study, splits) from a tiny learnable fixture — expensive to recompute."""
    df = synthetic_splits_frame(15, per_date=20, seed=0)
    splits = split_temporal(df)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = tune_checkpoint(15, splits, n_trials=3, seed=0)
    return study, splits


def test_fit_calibrated_predicts_in_unit_interval(well_calibrated_study_splits):
    study, splits = well_calibrated_study_splits
    model, _, _, _ = fit_calibrated(15, study, splits)
    X_opt, _, _ = prepare_Xy(splits["optuna_val"], 15)
    probs = model.predict_proba(X_opt)[:, 1]
    assert probs.min() >= 0 and probs.max() <= 1


def test_fit_calibrated_calibrator_has_predict_when_fired(well_calibrated_study_splits):
    """Force-fire the calibration path by setting the threshold to -1 so any
    non-zero ECE triggers."""
    study, splits = well_calibrated_study_splits
    _, calibrator, kind, _ = fit_calibrated(15, study, splits, ece_threshold=-1.0)
    assert calibrator is not None
    assert hasattr(calibrator, "predict")
    assert kind in ("isotonic", "platt")


def test_fit_calibrated_none_when_well_calibrated(well_calibrated_study_splits):
    """Threshold at 1.1 → no ECE can trigger; must return None calibrator."""
    study, splits = well_calibrated_study_splits
    _, calibrator, kind, _ = fit_calibrated(15, study, splits, ece_threshold=1.1)
    assert calibrator is None
    assert kind is None


# ---------------------------------------------------------------------
# train_checkpoint end-to-end
# ---------------------------------------------------------------------

def _write_fixture_csv(tmp_path, T, per_date=20, seed=0):
    df = synthetic_splits_frame(T, per_date=per_date, seed=seed)
    input_dir = tmp_path / "processed"
    input_dir.mkdir(exist_ok=True)
    df.to_csv(input_dir / f"checkpoint_{T}.csv", index=False)
    return input_dir


def test_train_checkpoint_writes_artifacts_and_runs_row(tmp_path):
    input_dir = _write_fixture_csv(tmp_path, T=15, per_date=20, seed=0)
    output_dir = tmp_path / "models"
    runs_csv = tmp_path / "runs.csv"
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    metrics = train_checkpoint(
        T=15,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        runs_csv=str(runs_csv),
        run_id="test_run",
        n_trials=3,
    )

    ckpt_dir = output_dir / "checkpoint_15"
    for name in ("model.json", "calibrator.pkl", "study.pkl", "metrics.json",
                 "feature_importance.csv", "reliability_val.png"):
        assert (ckpt_dir / name).exists(), f"{name} not written"

    persisted = json.loads((ckpt_dir / "metrics.json").read_text())
    for key in ("best_params", "n_estimators_selected", "optuna_val_n",
                "acceptance_val_n", "raw", "cal", "calibration_applied",
                "calibrator_kind", "acceptance_passed", "acceptance_diffs"):
        assert key in persisted
        assert key in metrics
    for block in ("raw", "cal"):
        for subkey in ("log_loss", "brier", "auc", "accuracy", "ece"):
            assert subkey in persisted[block]

    runs = pd.read_csv(runs_csv)
    assert len(runs) == 1
    assert runs.iloc[0]["run_id"] == "test_run"
    assert int(runs.iloc[0]["checkpoint"]) == 15
    # Both raw and cal columns populated (not NaN).
    for col in ("val_logloss", "val_brier", "val_logloss_cal", "val_brier_cal"):
        assert not pd.isna(runs.iloc[0][col]), f"{col} must be populated"


# ---------------------------------------------------------------------
# evaluate_test_all — atomicity + preconditions + no per-T entry point
# ---------------------------------------------------------------------

def test_no_public_per_T_evaluate_test_entry_point():
    """Per-checkpoint 2026 eval is explicitly unsupported (see spec)."""
    import scripts.models.xgboost_v1 as mod
    for name in ("evaluate_test", "evaluate_test_checkpoint"):
        assert not hasattr(mod, name), (
            f"{name} must not exist — --evaluate-test is atomic across all 4 T."
        )


def _train_all_four(tmp_path, per_date=15):
    """Run train_checkpoint for all 4 T on the same synthetic fixture."""
    output_dir = tmp_path / "models"
    runs_csv = tmp_path / "runs.csv"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for T in CHECKPOINTS:
        input_dir = _write_fixture_csv(tmp_path, T=T, per_date=per_date, seed=T)
        train_checkpoint(
            T=T,
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            runs_csv=str(runs_csv),
            run_id="test_run",
            n_trials=2,
        )
    return output_dir


def _force_acceptance(output_dir, value: bool):
    for T in CHECKPOINTS:
        mp = output_dir / f"checkpoint_{T}" / "metrics.json"
        m = json.loads(mp.read_text())
        m["acceptance_passed"] = value
        mp.write_text(json.dumps(m))


def test_evaluate_test_all_errors_when_no_metrics(tmp_path):
    with pytest.raises(SystemExit, match="metrics.json missing"):
        evaluate_test_all(
            input_dir=str(tmp_path / "processed"),
            output_dir=str(tmp_path / "models"),
        )


def test_evaluate_test_all_errors_when_acceptance_failed(tmp_path):
    output_dir = _train_all_four(tmp_path)
    _force_acceptance(output_dir, False)
    input_dir = tmp_path / "processed"
    with pytest.raises(SystemExit, match="acceptance_passed=False"):
        evaluate_test_all(input_dir=str(input_dir), output_dir=str(output_dir))


def test_evaluate_test_all_errors_when_already_evaluated(tmp_path):
    output_dir = _train_all_four(tmp_path)
    _force_acceptance(output_dir, True)
    # Seed an existing test block in one metrics.json to trip the precondition.
    mp = output_dir / "checkpoint_15" / "metrics.json"
    m = json.loads(mp.read_text())
    m["test_n"] = 100
    m["test"] = {"raw": {}, "cal": {}}
    mp.write_text(json.dumps(m))
    input_dir = tmp_path / "processed"
    with pytest.raises(SystemExit, match="already evaluated"):
        evaluate_test_all(input_dir=str(input_dir), output_dir=str(output_dir))


def test_evaluate_test_all_writes_four_result_sets(tmp_path):
    output_dir = _train_all_four(tmp_path)
    _force_acceptance(output_dir, True)
    input_dir = tmp_path / "processed"
    results = evaluate_test_all(input_dir=str(input_dir), output_dir=str(output_dir))
    assert set(results.keys()) == set(CHECKPOINTS)
    for T in CHECKPOINTS:
        ckpt_dir = output_dir / f"checkpoint_{T}"
        assert (ckpt_dir / "reliability_test.png").exists()
        m = json.loads((ckpt_dir / "metrics.json").read_text())
        assert "test" in m
        assert "test_n" in m
        for block in ("raw", "cal"):
            for key in ("log_loss", "brier", "auc", "accuracy", "ece"):
                assert key in m["test"][block]


# ---------------------------------------------------------------------
# fit_production smoke test
# ---------------------------------------------------------------------

def test_fit_production_writes_artifacts_and_carries_calibrator(tmp_path):
    output_dir = _train_all_four(tmp_path, per_date=15)
    _force_acceptance(output_dir, True)
    input_dir = tmp_path / "processed"

    md = fit_production(T=15, input_dir=str(input_dir), output_dir=str(output_dir))

    prod = output_dir / "production" / "checkpoint_15"
    assert (prod / "model.json").exists()
    assert (prod / "calibrator.pkl").exists()
    assert (prod / "metadata.json").exists()

    metadata = json.loads((prod / "metadata.json").read_text())
    assert metadata["n_estimators_prod"] >= 1
    assert metadata["calibrator_source"] == "val_stage_carried_over"
    assert "fit_timestamp" in metadata
    assert md["n_estimators_prod"] == metadata["n_estimators_prod"]

    # calibrator.pkl byte-identical to val-stage.
    val_bytes = (output_dir / "checkpoint_15" / "calibrator.pkl").read_bytes()
    prod_bytes = (prod / "calibrator.pkl").read_bytes()
    assert val_bytes == prod_bytes

    # Calibrator unpickles fine either way.
    pickle.loads(prod_bytes)

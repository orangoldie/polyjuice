"""Tests for scripts/models/baseline_logistic.py.

Mirrors the preprocessing-pipeline TDD conventions: small synthetic fixtures,
no real-CSV slices. See docs/baseline-logistic.md for the spec.
"""
import json

import numpy as np
import pandas as pd
import pytest

from scripts.models.baseline_logistic import (
    BINARY_FEATURES,
    CHECKPOINTS,
    C_GRID,
    NUMERIC_FEATURES,
    evaluate,
    fit,
    predict_proba,
    prepare_Xy,
    reliability_bins,
    split_train_val,
    train_checkpoint,
)


FIRST_GATES_PER_T = {
    10: ["firstblood", "firstdragon"],
    15: ["firstblood", "firstdragon"],
    20: ["firstblood", "firstdragon", "firsttower"],
    25: ["firstblood", "firstdragon", "firsttower"],
}


# --- fixtures ---------------------------------------------------------

def make_checkpoint_row(T, gameid="g1", year=2023, tier="S", side="Blue",
                        result=1, **overrides):
    row = {
        "gameid": gameid,
        "date": f"{year}-06-01",
        "year": year,
        "league": "LCK",
        "patch": "13.1",
        "teamname": "T1",
        "opp_teamname": "GenG",
        "side": side,
        "tier": tier,
        f"goldat{T}": 100.0 * T,
        f"opp_goldat{T}": 95.0 * T,
        f"golddiffat{T}": 5.0 * T,
        f"csat{T}": 8.0 * T,
        f"opp_csat{T}": 7.0 * T,
        f"csdiffat{T}": float(T),
        f"killsat{T}": float(T // 5),
        f"opp_killsat{T}": float((T // 5) - 1),
        f"killdiffat{T}": 1.0,
        f"assistsat{T}": float((T // 5) + 2),
        f"opp_assistsat{T}": float(T // 5),
        f"assistdiffat{T}": 2.0,
        "gamelength": T * 60 + 100,
        "result": result,
    }
    for flag in FIRST_GATES_PER_T[T]:
        row[flag] = 1 if flag == "firstblood" else 0
    row.update(overrides)
    return row


def _synthetic_frame(T, n=400, seed=0, year=2022):
    """Training frame with a learnable signal so LR converges quickly."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        diff = rng.normal(0, 3000)
        winner = 1 if diff > 0 else 0
        row = make_checkpoint_row(
            T, gameid=f"syn_{i}", year=year,
            tier=str(rng.choice(["S", "A", "B"])),
            side=str(rng.choice(["Blue", "Red"])),
            result=winner,
        )
        row[f"goldat{T}"] = 20000.0 + diff / 2
        row[f"opp_goldat{T}"] = 20000.0 - diff / 2
        row[f"golddiffat{T}"] = float(diff)
        rows.append(row)
    return pd.DataFrame(rows)


# --- NUMERIC_FEATURES / BINARY_FEATURES -------------------------------

@pytest.mark.parametrize("T", CHECKPOINTS)
def test_numeric_features_eight_raw_opp_stats_no_diffs(T):
    feats = NUMERIC_FEATURES(T)
    assert len(feats) == 8
    for stat in ("gold", "opp_gold", "cs", "opp_cs",
                 "kills", "opp_kills", "assists", "opp_assists"):
        assert f"{stat}at{T}" in feats, f"{stat}at{T} missing"
    for stat in ("golddiff", "csdiff", "killdiff", "assistdiff"):
        assert f"{stat}at{T}" not in feats, f"{stat}at{T} must be dropped (collinear)"


@pytest.mark.parametrize("T,expected_firsts", [
    (10, ["firstblood", "firstdragon"]),
    (15, ["firstblood", "firstdragon"]),
    (20, ["firstblood", "firstdragon", "firsttower"]),
    (25, ["firstblood", "firstdragon", "firsttower"]),
])
def test_binary_features_is_side_is_blue_plus_gated_firsts(T, expected_firsts):
    feats = BINARY_FEATURES(T)
    assert "side_is_blue" in feats
    for flag in expected_firsts:
        assert flag in feats
    for flag in ("firstblood", "firstdragon", "firsttower", "firstbaron"):
        if flag not in expected_firsts:
            assert flag not in feats


# --- prepare_Xy() -----------------------------------------------------

def test_prepare_Xy_tier_S_is_baseline():
    df = pd.DataFrame([
        make_checkpoint_row(15, tier="S"),
        make_checkpoint_row(15, gameid="g2", tier="A"),
        make_checkpoint_row(15, gameid="g3", tier="B"),
    ])
    X, _, _ = prepare_Xy(df, 15)
    assert "tier_A" in X.columns
    assert "tier_B" in X.columns
    assert "tier_S" not in X.columns


def test_prepare_Xy_tier_onehot_values_correct():
    df = pd.DataFrame([
        make_checkpoint_row(15, tier="S", gameid="g_s"),
        make_checkpoint_row(15, tier="A", gameid="g_a"),
        make_checkpoint_row(15, tier="B", gameid="g_b"),
    ])
    X, _, _ = prepare_Xy(df, 15)
    assert X.iloc[0]["tier_A"] == 0 and X.iloc[0]["tier_B"] == 0  # S
    assert X.iloc[1]["tier_A"] == 1 and X.iloc[1]["tier_B"] == 0  # A
    assert X.iloc[2]["tier_A"] == 0 and X.iloc[2]["tier_B"] == 1  # B


def test_prepare_Xy_side_is_blue_binarized():
    df = pd.DataFrame([
        make_checkpoint_row(10, side="Blue", gameid="g_b"),
        make_checkpoint_row(10, side="Red", gameid="g_r"),
    ])
    X, _, _ = prepare_Xy(df, 10)
    assert "side_is_blue" in X.columns
    assert "side" not in X.columns
    assert X.iloc[0]["side_is_blue"] == 1
    assert X.iloc[1]["side_is_blue"] == 0


@pytest.mark.parametrize("T", CHECKPOINTS)
def test_prepare_Xy_no_diff_columns_in_output(T):
    df = pd.DataFrame([make_checkpoint_row(T)])
    X, _, _ = prepare_Xy(df, T)
    for stat in ("golddiff", "csdiff", "killdiff", "assistdiff"):
        assert f"{stat}at{T}" not in X.columns, \
            f"{stat}at{T} (collinear with raw/opp) must be dropped"


def test_prepare_Xy_identifiers_not_in_features():
    df = pd.DataFrame([make_checkpoint_row(15)])
    X, _, _ = prepare_Xy(df, 15)
    for col in ("gameid", "date", "year", "league", "patch",
                "teamname", "opp_teamname", "gamelength", "result"):
        assert col not in X.columns


def test_prepare_Xy_gated_firsts_at_10():
    df = pd.DataFrame([make_checkpoint_row(10)])
    X, _, _ = prepare_Xy(df, 10)
    assert "firstblood" in X.columns
    assert "firstdragon" in X.columns
    assert "firsttower" not in X.columns
    assert "firstbaron" not in X.columns


def test_prepare_Xy_gated_firsts_at_25():
    df = pd.DataFrame([make_checkpoint_row(25)])
    X, _, _ = prepare_Xy(df, 25)
    for flag in ("firstblood", "firstdragon", "firsttower"):
        assert flag in X.columns
    assert "firstbaron" not in X.columns


def test_prepare_Xy_row_count_preserved():
    rows = [make_checkpoint_row(15, gameid=f"g{i}") for i in range(7)]
    df = pd.DataFrame(rows)
    X, y, _ = prepare_Xy(df, 15)
    assert len(X) == 7
    assert len(y) == 7


def test_prepare_Xy_y_is_result():
    df = pd.DataFrame([
        make_checkpoint_row(15, result=1, gameid="a"),
        make_checkpoint_row(15, result=0, gameid="b"),
    ])
    _, y, _ = prepare_Xy(df, 15)
    assert list(y) == [1, 0]


def test_prepare_Xy_feature_names_match_X_columns():
    df = pd.DataFrame([make_checkpoint_row(15)])
    X, _, feature_names = prepare_Xy(df, 15)
    assert list(X.columns) == feature_names


# --- split_train_val() -----------------------------------------------

def test_split_train_val_year_boundaries():
    rows = [make_checkpoint_row(15, gameid=f"g{y}", year=y)
            for y in range(2020, 2027)]
    df = pd.DataFrame(rows)
    train, val = split_train_val(df)
    assert set(train["year"]) == {2020, 2021, 2022, 2023, 2024}
    assert set(val["year"]) == {2025}


def test_split_train_val_2026_dropped():
    rows = [
        make_checkpoint_row(15, gameid="t", year=2022),
        make_checkpoint_row(15, gameid="v", year=2025),
        make_checkpoint_row(15, gameid="x", year=2026),
    ]
    df = pd.DataFrame(rows)
    train, val = split_train_val(df)
    kept = set(train["gameid"]) | set(val["gameid"])
    assert "x" not in kept


# --- fit() and predict_proba() ---------------------------------------

def test_fit_produces_model_with_C_selected():
    df = _synthetic_frame(15, n=400)
    X, y, _ = prepare_Xy(df, 15)
    _, model = fit(X, y)
    assert hasattr(model, "C_")
    assert float(model.C_[0]) in C_GRID


def test_predict_proba_in_unit_interval():
    df = _synthetic_frame(15, n=400)
    X, y, _ = prepare_Xy(df, 15)
    scaler, model = fit(X, y)
    probs = predict_proba(scaler, model, X, NUMERIC_FEATURES(15))
    assert probs.min() >= 0
    assert probs.max() <= 1


def test_predict_proba_does_not_refit_scaler():
    """Feeding val data through predict_proba must not mutate the train-fit scaler."""
    df_train = _synthetic_frame(15, n=300, seed=0)
    X_tr, y_tr, _ = prepare_Xy(df_train, 15)
    scaler, model = fit(X_tr, y_tr)
    mean_before = scaler.mean_.copy()

    df_val = _synthetic_frame(15, n=200, seed=99)
    X_val, _, _ = prepare_Xy(df_val, 15)
    predict_proba(scaler, model, X_val, NUMERIC_FEATURES(15))
    np.testing.assert_array_equal(scaler.mean_, mean_before)


def test_fit_learns_signal():
    """Sanity: on the synthetic gold-diff-drives-win fixture, accuracy beats 0.8."""
    df = _synthetic_frame(15, n=800, seed=0)
    X, y, _ = prepare_Xy(df, 15)
    scaler, model = fit(X, y)
    probs = predict_proba(scaler, model, X, NUMERIC_FEATURES(15))
    preds = (probs >= 0.5).astype(int)
    assert (preds == y.values).mean() > 0.8


# --- evaluate() ------------------------------------------------------

def test_evaluate_perfect_predictions():
    y = np.array([0, 1, 0, 1])
    probs = np.array([0.01, 0.99, 0.01, 0.99])
    m = evaluate(y, probs)
    assert m["log_loss"] < 0.05
    assert m["brier"] < 0.01
    assert m["auc"] == 1.0
    assert m["accuracy"] == 1.0


def test_evaluate_all_wrong_predictions():
    y = np.array([0, 1, 0, 1])
    probs = np.array([0.99, 0.01, 0.99, 0.01])
    m = evaluate(y, probs)
    assert m["log_loss"] > 3.0
    assert m["accuracy"] == 0.0


def test_evaluate_random_predictions_auc_near_half():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 2000)
    probs = rng.random(2000)
    m = evaluate(y, probs)
    assert 0.45 < m["auc"] < 0.55


# --- reliability_bins() ----------------------------------------------

def test_reliability_bins_counts_and_columns():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    probs = rng.random(500)
    out = reliability_bins(y, probs, n_bins=10)
    assert {"bin_lo", "bin_hi", "n", "mean_pred", "mean_actual"}.issubset(set(out.columns))
    assert len(out) == 10
    assert int(out["n"].sum()) == 500


def test_reliability_bins_perfect_calibration_hugs_diagonal():
    rng = np.random.default_rng(1)
    probs = rng.random(10_000)
    y = (rng.random(10_000) < probs).astype(int)
    out = reliability_bins(y, probs, n_bins=10)
    nonempty = out[out["n"] > 50]
    diffs = (nonempty["mean_actual"] - nonempty["mean_pred"]).abs()
    assert (diffs < 0.05).all()


# --- train_checkpoint() end-to-end -----------------------------------

def test_train_checkpoint_writes_artifacts_and_returns_metrics(tmp_path):
    # Mix of 2022 train rows and 2025 val rows
    train_df = _synthetic_frame(15, n=300, seed=0, year=2022)
    val_df = _synthetic_frame(15, n=150, seed=1, year=2025)
    df = pd.concat([train_df, val_df], ignore_index=True)

    input_dir = tmp_path / "processed"
    input_dir.mkdir()
    df.to_csv(input_dir / "checkpoint_15.csv", index=False)

    output_dir = tmp_path / "models"
    runs_csv = tmp_path / "runs.csv"

    metrics = train_checkpoint(
        T=15,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        runs_csv=str(runs_csv),
        run_id="test_run",
    )

    ckpt_dir = output_dir / "checkpoint_15"
    assert (ckpt_dir / "model.pkl").exists()
    assert (ckpt_dir / "metrics.json").exists()
    assert (ckpt_dir / "coefficients.csv").exists()
    assert (ckpt_dir / "reliability.png").exists()

    persisted = json.loads((ckpt_dir / "metrics.json").read_text())
    for key in ("C_selected", "train_n", "val_n", "log_loss",
                "brier", "auc", "accuracy"):
        assert key in persisted
        assert key in metrics

    assert runs_csv.exists()
    runs = pd.read_csv(runs_csv)
    assert len(runs) == 1
    assert runs.iloc[0]["run_id"] == "test_run"
    assert int(runs.iloc[0]["checkpoint"]) == 15


def test_train_checkpoint_coefficients_sorted_by_abs(tmp_path):
    train_df = _synthetic_frame(10, n=300, seed=0, year=2022)
    val_df = _synthetic_frame(10, n=150, seed=1, year=2025)
    df = pd.concat([train_df, val_df], ignore_index=True)

    input_dir = tmp_path / "processed"
    input_dir.mkdir()
    df.to_csv(input_dir / "checkpoint_10.csv", index=False)

    train_checkpoint(
        T=10,
        input_dir=str(input_dir),
        output_dir=str(tmp_path / "models"),
        runs_csv=str(tmp_path / "runs.csv"),
        run_id="test",
    )
    coefs = pd.read_csv(tmp_path / "models" / "checkpoint_10" / "coefficients.csv")
    abs_col = coefs["abs_coefficient"].values
    assert (abs_col[:-1] >= abs_col[1:]).all(), "coefficients.csv should be sorted desc by abs_coefficient"

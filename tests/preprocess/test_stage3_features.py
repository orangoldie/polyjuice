"""Tests for stage 3: feature engineering + column pruning + tier join."""
import math

import numpy as np
import pandas as pd
import pytest

from scripts.preprocess.stage3_features import (
    CHECKPOINTS,
    OUTPUT_COLUMNS,
    run,
    validate,
)


TIER_MAP = {"LCK": "S", "LEC": "S", "TCL": "B"}


def make_input_row(gameid: str = "g1", league: str = "LCK", **overrides) -> dict:
    """Build one stage-2-output row: identifiers + raw stats + firsts + noise columns."""
    row = {
        "gameid": gameid,
        "date": "2023-01-15",
        "year": 2023,
        "league": league,
        "patch": "13.1",
        "teamname": "t_blue",
        "opp_teamname": "t_red",
        "gamelength": 1800,
        "result": 1,
        "side": "Blue",
        "position": "team",
        "firstblood": 1,
        "firstdragon": 0,
        "firsttower": 1,
        "firstbaron": 0,
        # Noise columns that stage 3 must drop:
        "damagetochampions": 99999,
        "visionscore": 123,
        "dragons": 3,
        "barons": 1,
        "atakhans": 0,
    }
    for T in CHECKPOINTS:
        row[f"goldat{T}"] = 100 * T
        row[f"opp_goldat{T}"] = 90 * T
        row[f"golddiffat{T}"] = 10 * T
        row[f"csat{T}"] = 8 * T
        row[f"opp_csat{T}"] = 7 * T
        row[f"csdiffat{T}"] = T
        row[f"killsat{T}"] = T // 5
        row[f"opp_killsat{T}"] = (T // 5) - 1
        row[f"assistsat{T}"] = (T // 5) + 2
        row[f"opp_assistsat{T}"] = T // 5
    row.update(overrides)
    return row


# --- run() -------------------------------------------------------------

def test_run_output_has_exactly_the_locked_columns():
    df = pd.DataFrame([make_input_row("g1")])
    out = run(df, TIER_MAP)
    assert set(out.columns) == set(OUTPUT_COLUMNS)


def test_run_preserves_row_count():
    df = pd.DataFrame([make_input_row(f"g{i}") for i in range(5)])
    out = run(df, TIER_MAP)
    assert len(out) == 5


def test_run_joins_tier_from_league():
    df = pd.DataFrame([
        make_input_row("g1", league="LCK"),
        make_input_row("g2", league="TCL"),
    ])
    out = run(df, TIER_MAP).set_index("gameid")
    assert out.loc["g1", "tier"] == "S"
    assert out.loc["g2", "tier"] == "B"


def test_run_computes_killdiff_for_all_checkpoints():
    df = pd.DataFrame([make_input_row("g1", killsat15=7, opp_killsat15=3)])
    out = run(df, TIER_MAP)
    assert out.iloc[0]["killdiffat15"] == 4


def test_run_computes_assistdiff_for_all_checkpoints():
    df = pd.DataFrame([make_input_row("g1", assistsat20=12, opp_assistsat20=8)])
    out = run(df, TIER_MAP)
    assert out.iloc[0]["assistdiffat20"] == 4


def test_run_drops_noise_columns():
    df = pd.DataFrame([make_input_row("g1")])
    out = run(df, TIER_MAP)
    for dropped in ("damagetochampions", "visionscore", "dragons", "barons", "atakhans", "position"):
        assert dropped not in out.columns


def test_run_preserves_nan_in_late_checkpoint_stats():
    """Short games have NaN at *at25 columns; stage 3 must pass them through untouched."""
    row = make_input_row("g1", gamelength=1400)
    for col in ("goldat25", "opp_goldat25", "golddiffat25",
                "csat25", "opp_csat25", "csdiffat25",
                "killsat25", "opp_killsat25",
                "assistsat25", "opp_assistsat25"):
        row[col] = math.nan
    out = run(pd.DataFrame([row]), TIER_MAP)
    assert out.iloc[0]["goldat25"] != out.iloc[0]["goldat25"]  # NaN != NaN
    assert out.iloc[0]["killdiffat25"] != out.iloc[0]["killdiffat25"]  # NaN - NaN = NaN


# --- validate() --------------------------------------------------------

def _valid_output_row(gameid: str = "g1", **overrides) -> dict:
    row = {
        "gameid": gameid,
        "date": "2023-01-15",
        "year": 2023,
        "league": "LCK",
        "patch": "13.1",
        "teamname": "t_blue",
        "opp_teamname": "t_red",
        "gamelength": 1800,
        "result": 1,
        "side": "Blue",
        "tier": "S",
        "firstblood": 1,
        "firstdragon": 0,
        "firsttower": 1,
        "firstbaron": 0,
    }
    for T in CHECKPOINTS:
        row[f"goldat{T}"] = 100 * T
        row[f"opp_goldat{T}"] = 90 * T
        row[f"golddiffat{T}"] = 10 * T
        row[f"csat{T}"] = 8 * T
        row[f"opp_csat{T}"] = 7 * T
        row[f"csdiffat{T}"] = T
        row[f"killsat{T}"] = T // 5
        row[f"opp_killsat{T}"] = (T // 5) - 1
        row[f"killdiffat{T}"] = 1
        row[f"assistsat{T}"] = (T // 5) + 2
        row[f"opp_assistsat{T}"] = T // 5
        row[f"assistdiffat{T}"] = 2
    row.update(overrides)
    return row


def test_validate_passes_on_valid_output():
    df = pd.DataFrame([_valid_output_row("g1")])
    validate(df)


def test_validate_raises_on_missing_column():
    df = pd.DataFrame([_valid_output_row("g1")]).drop(columns=["tier"])
    with pytest.raises(AssertionError, match="missing"):
        validate(df)


def test_validate_raises_on_extra_column():
    df = pd.DataFrame([_valid_output_row("g1")])
    df["some_extra"] = 42
    with pytest.raises(AssertionError, match="extra"):
        validate(df)


def test_validate_raises_on_null_side():
    df = pd.DataFrame([_valid_output_row("g1")])
    df.loc[0, "side"] = None
    with pytest.raises(AssertionError, match="side"):
        validate(df)


def test_validate_raises_on_null_tier():
    df = pd.DataFrame([_valid_output_row("g1")])
    df.loc[0, "tier"] = None
    with pytest.raises(AssertionError, match="tier"):
        validate(df)


def test_validate_raises_on_null_firstblood():
    df = pd.DataFrame([_valid_output_row("g1")])
    df.loc[0, "firstblood"] = None
    with pytest.raises(AssertionError, match="firstblood"):
        validate(df)


def test_validate_raises_on_null_gamelength():
    df = pd.DataFrame([_valid_output_row("g1")])
    df.loc[0, "gamelength"] = None
    with pytest.raises(AssertionError, match="gamelength"):
        validate(df)


def test_validate_allows_nan_in_late_checkpoint_stats():
    """Short games have NaN across the whole *at25 family — stage 3 must tolerate it."""
    df = pd.DataFrame([_valid_output_row("g1")])
    at25_cols = [c for c in df.columns if c.endswith("at25")]
    assert at25_cols, "fixture sanity: expected some at25 columns"
    for col in at25_cols:
        df.loc[0, col] = np.nan
    validate(df)  # does not raise


def test_validate_raises_on_wrong_killdiff():
    df = pd.DataFrame([_valid_output_row("g1", killsat10=5, opp_killsat10=2, killdiffat10=999)])
    with pytest.raises(AssertionError, match="killdiff"):
        validate(df)


def test_validate_raises_on_wrong_assistdiff():
    df = pd.DataFrame([_valid_output_row("g1", assistsat25=10, opp_assistsat25=4, assistdiffat25=999)])
    with pytest.raises(AssertionError, match="assistdiff"):
        validate(df)


def test_validate_raises_on_tier_outside_SAB():
    df = pd.DataFrame([_valid_output_row("g1", tier="C")])
    with pytest.raises(AssertionError, match="tier"):
        validate(df)


def test_validate_raises_on_side_outside_blue_red():
    df = pd.DataFrame([_valid_output_row("g1", side="Purple")])
    with pytest.raises(AssertionError, match="side"):
        validate(df)


def test_validate_raises_on_first_flag_not_binary():
    df = pd.DataFrame([_valid_output_row("g1", firstdragon=2)])
    with pytest.raises(AssertionError, match="firstdragon"):
        validate(df)


def test_validate_raises_on_result_not_binary():
    df = pd.DataFrame([_valid_output_row("g1", result=5)])
    with pytest.raises(AssertionError, match="result"):
        validate(df)

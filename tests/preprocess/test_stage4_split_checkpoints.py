"""Tests for stage 4: per-checkpoint row filter + column selection."""
import math

import numpy as np
import pandas as pd
import pytest

from scripts.preprocess.stage4_split_checkpoints import (
    CHECKPOINTS,
    FIRST_GATES,
    output_columns,
    run,
    validate,
)


# --- fixtures ---------------------------------------------------------

def make_stage3_row(gameid: str = "g1", gamelength: int = 1800, **overrides) -> dict:
    """Build one stage-3-output row: all 63 locked columns populated."""
    row = {
        "gameid": gameid,
        "date": "2023-01-15",
        "year": 2023,
        "league": "LCK",
        "patch": "13.1",
        "teamname": "t_blue",
        "opp_teamname": "t_red",
        "side": "Blue",
        "tier": "S",
        "firstblood": 1,
        "firstdragon": 0,
        "firsttower": 1,
        "firstbaron": 0,
        "gamelength": gamelength,
        "result": 1,
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


# --- output_columns() -------------------------------------------------

@pytest.mark.parametrize("T", CHECKPOINTS)
def test_output_columns_includes_only_this_checkpoints_stats(T):
    cols = set(output_columns(T))
    # Own stats in
    for stat in ("gold", "opp_gold", "golddiff", "cs", "opp_cs", "csdiff",
                 "kills", "opp_kills", "killdiff",
                 "assists", "opp_assists", "assistdiff"):
        assert f"{stat}at{T}" in cols, f"{stat}at{T} missing from T={T} schema"
    # Other checkpoints out
    for other in CHECKPOINTS:
        if other == T:
            continue
        for stat in ("gold", "cs", "kills", "assists"):
            assert f"{stat}at{other}" not in cols, \
                f"{stat}at{other} leaked into T={T} schema"


def test_output_columns_firstblood_at_every_checkpoint():
    for T in CHECKPOINTS:
        assert "firstblood" in output_columns(T)


def test_output_columns_firstdragon_gated_at_15_and_later():
    assert "firstdragon" not in output_columns(10)
    for T in (15, 20, 25):
        assert "firstdragon" in output_columns(T)


def test_output_columns_firsttower_gated_at_15_and_later():
    assert "firsttower" not in output_columns(10)
    for T in (15, 20, 25):
        assert "firsttower" in output_columns(T)


def test_output_columns_firstbaron_only_at_25():
    for T in (10, 15, 20):
        assert "firstbaron" not in output_columns(T)
    assert "firstbaron" in output_columns(25)


@pytest.mark.parametrize("T", CHECKPOINTS)
def test_output_columns_has_identifiers_static_gamelength_result(T):
    cols = set(output_columns(T))
    for required in ("gameid", "date", "year", "league", "patch",
                     "teamname", "opp_teamname",
                     "side", "tier", "gamelength", "result"):
        assert required in cols, f"{required} missing from T={T} schema"


# --- run() ------------------------------------------------------------

@pytest.mark.parametrize("T", CHECKPOINTS)
def test_run_drops_games_with_null_checkpoint_stat(T):
    """If any of T's 12 stat columns is null, the row is dropped."""
    populated = make_stage3_row("populated")
    unpopulated = make_stage3_row("unpopulated")
    unpopulated[f"goldat{T}"] = math.nan
    df = pd.DataFrame([populated, unpopulated])
    out = run(df, T)
    assert set(out["gameid"]) == {"populated"}


@pytest.mark.parametrize("T", CHECKPOINTS)
def test_run_keeps_short_but_populated_rows(T):
    """Fuzzy-boundary games (gamelength < T*60 but T-frame populated) are kept."""
    df = pd.DataFrame([make_stage3_row("fuzzy", gamelength=T * 60 - 30)])
    out = run(df, T)
    assert len(out) == 1
    assert out.iloc[0]["gameid"] == "fuzzy"


@pytest.mark.parametrize("T", CHECKPOINTS)
def test_run_drops_long_but_null_rows(T):
    """Games with gamelength >= T*60 but missing the T-frame are dropped."""
    row = make_stage3_row("long_null", gamelength=T * 60 + 100)
    row[f"opp_csat{T}"] = math.nan
    df = pd.DataFrame([row])
    out = run(df, T)
    assert len(out) == 0


@pytest.mark.parametrize("T", CHECKPOINTS)
def test_run_output_has_exactly_the_checkpoint_schema(T):
    df = pd.DataFrame([make_stage3_row("g1", gamelength=1800)])
    out = run(df, T)
    assert set(out.columns) == set(output_columns(T))


def test_run_preserves_identifier_values():
    df = pd.DataFrame([make_stage3_row("g1", gamelength=1800, teamname="T1", opp_teamname="GenG")])
    out = run(df, 15)
    assert out.iloc[0]["teamname"] == "T1"
    assert out.iloc[0]["opp_teamname"] == "GenG"


def test_run_does_not_mutate_input():
    df = pd.DataFrame([make_stage3_row("g1", gamelength=1800)])
    before_cols = list(df.columns)
    before_len = len(df)
    run(df, 10)
    assert list(df.columns) == before_cols
    assert len(df) == before_len


def test_run_drops_all_rows_when_all_unpopulated():
    rows = []
    for gid in ("a", "b"):
        r = make_stage3_row(gid)
        for col in ("goldat25", "opp_goldat25", "csat25", "opp_csat25",
                    "killsat25", "opp_killsat25", "assistsat25", "opp_assistsat25"):
            r[col] = math.nan
        rows.append(r)
    out = run(pd.DataFrame(rows), 25)
    assert len(out) == 0
    assert set(out.columns) == set(output_columns(25))


def test_run_excludes_late_checkpoint_stats_for_short_model():
    """At T=10, `goldat25` etc. should not appear — even if still present in input."""
    df = pd.DataFrame([make_stage3_row("g1", gamelength=1800)])
    out = run(df, 10)
    for col in ("goldat25", "killsat20", "csat15", "assistdiffat20"):
        assert col not in out.columns


# --- validate() -------------------------------------------------------

def _valid_checkpoint_row(T: int, gameid: str = "g1", **overrides) -> dict:
    """Build a row that validate(T) should accept."""
    row = {
        "gameid": gameid,
        "date": "2023-01-15",
        "year": 2023,
        "league": "LCK",
        "patch": "13.1",
        "teamname": "t_blue",
        "opp_teamname": "t_red",
        "side": "Blue",
        "tier": "S",
        "gamelength": T * 60 + 100,
        "result": 1,
    }
    for flag in FIRST_GATES[T]:
        row[flag] = 1 if flag == "firstblood" else 0
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


@pytest.mark.parametrize("T", CHECKPOINTS)
def test_validate_passes_on_valid_output(T):
    df = pd.DataFrame([_valid_checkpoint_row(T)])
    validate(df, T)


@pytest.mark.parametrize("T", CHECKPOINTS)
def test_validate_raises_on_missing_column(T):
    df = pd.DataFrame([_valid_checkpoint_row(T)]).drop(columns=["tier"])
    with pytest.raises(AssertionError, match="missing"):
        validate(df, T)


@pytest.mark.parametrize("T", CHECKPOINTS)
def test_validate_raises_on_extra_column(T):
    df = pd.DataFrame([_valid_checkpoint_row(T)])
    df["some_extra"] = 42
    with pytest.raises(AssertionError, match="extra"):
        validate(df, T)


def test_validate_raises_on_wrong_checkpoint_stats():
    """A T=15 dataframe fed to validate(T=10) has the wrong stat columns."""
    df = pd.DataFrame([_valid_checkpoint_row(15)])
    with pytest.raises(AssertionError):
        validate(df, 10)


@pytest.mark.parametrize("T", CHECKPOINTS)
def test_validate_tolerates_short_gamelength_when_stats_populated(T):
    """Fuzzy-boundary rows are valid if the T-frame is populated."""
    df = pd.DataFrame([_valid_checkpoint_row(T, gamelength=T * 60 - 30)])
    validate(df, T)


@pytest.mark.parametrize("T", CHECKPOINTS)
def test_validate_raises_on_nan_in_checkpoint_stat(T):
    df = pd.DataFrame([_valid_checkpoint_row(T)])
    df.loc[0, f"goldat{T}"] = np.nan
    with pytest.raises(AssertionError):
        validate(df, T)


@pytest.mark.parametrize("T", CHECKPOINTS)
def test_validate_raises_on_nan_in_firstblood(T):
    df = pd.DataFrame([_valid_checkpoint_row(T)])
    df.loc[0, "firstblood"] = np.nan
    with pytest.raises(AssertionError, match="firstblood"):
        validate(df, T)


def test_validate_raises_on_side_outside_blue_red():
    df = pd.DataFrame([_valid_checkpoint_row(10, side="Purple")])
    with pytest.raises(AssertionError, match="side"):
        validate(df, 10)


def test_validate_raises_on_tier_outside_SAB():
    df = pd.DataFrame([_valid_checkpoint_row(15, tier="C")])
    with pytest.raises(AssertionError, match="tier"):
        validate(df, 15)


def test_validate_raises_on_result_not_binary():
    df = pd.DataFrame([_valid_checkpoint_row(20, result=5)])
    with pytest.raises(AssertionError, match="result"):
        validate(df, 20)


def test_validate_raises_on_firstbaron_not_binary_at_25():
    df = pd.DataFrame([_valid_checkpoint_row(25, firstbaron=2)])
    with pytest.raises(AssertionError, match="firstbaron"):
        validate(df, 25)


def test_validate_passes_on_empty_frame_with_correct_schema():
    """A valid-but-empty frame (e.g. nothing survived the gamelength filter) should pass."""
    df = pd.DataFrame(columns=output_columns(25))
    validate(df, 25)

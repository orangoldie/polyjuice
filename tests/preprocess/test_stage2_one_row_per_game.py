"""Tests for stage 2: collapse to one row per game."""
import itertools

import pandas as pd
import pytest

from scripts.preprocess.stage2_one_row_per_game import run, validate


TIER_MAP = {"LCK": "S", "LEC": "S", "TCL": "B"}


def make_game(gameid: str = "g1", league: str = "LCK", winner_side: str = "Blue") -> list[dict]:
    """12 rows (10 player + 2 team) matching stage 1's output shape."""
    rows = []
    for side in ("Blue", "Red"):
        teamname = f"{gameid}_{side}"
        for pos in ("top", "jng", "mid", "bot", "sup"):
            rows.append({
                "gameid": gameid, "league": league, "position": pos,
                "side": side, "result": 1 if side == winner_side else 0,
                "teamname": teamname,
            })
    for side in ("Blue", "Red"):
        rows.append({
            "gameid": gameid, "league": league, "position": "team",
            "side": side, "result": 1 if side == winner_side else 0,
            "teamname": f"{gameid}_{side}",
        })
    return rows


def make_games(n: int, league: str = "LCK") -> list[dict]:
    return list(itertools.chain.from_iterable(make_game(f"g{i}", league=league) for i in range(n)))


# --- run() -------------------------------------------------------------

def test_run_drops_all_player_rows():
    df = pd.DataFrame(make_game("g1"))
    out = run(df, seed=42)
    assert (out["position"] == "team").all()


def test_run_keeps_exactly_one_row_per_gameid():
    df = pd.DataFrame(make_games(3))
    out = run(df, seed=42)
    assert len(out) == 3
    assert set(out["gameid"]) == {"g0", "g1", "g2"}


def test_run_is_deterministic_for_same_seed():
    df = pd.DataFrame(make_games(20))
    out1 = run(df, seed=42)
    out2 = run(df, seed=42)
    pd.testing.assert_frame_equal(out1, out2)


def test_run_different_seeds_produce_different_picks():
    df = pd.DataFrame(make_games(100))
    a = run(df, seed=42).set_index("gameid")["side"]
    b = run(df, seed=7).set_index("gameid")["side"]
    assert (a != b).any(), "two seeds produced identical picks across 100 games — RNG may be broken"


def test_run_produces_roughly_balanced_side_over_many_games():
    df = pd.DataFrame(make_games(1000))
    out = run(df, seed=42)
    blue_frac = (out["side"] == "Blue").mean()
    assert 0.4 <= blue_frac <= 0.6


def test_run_preserves_input_columns():
    df = pd.DataFrame(make_game("g1"))
    df["goldat10"] = 5000
    out = run(df, seed=42)
    assert "goldat10" in out.columns


def test_run_attaches_opp_teamname():
    df = pd.DataFrame(make_game("g1"))
    out = run(df, seed=42)
    assert "opp_teamname" in out.columns
    row = out.iloc[0]
    expected_opp = f"g1_{'Red' if row['side'] == 'Blue' else 'Blue'}"
    assert row["opp_teamname"] == expected_opp
    assert row["opp_teamname"] != row["teamname"]


def test_run_picks_one_of_the_two_existing_team_rows():
    """The chosen row's (gameid, side, result) must match one of the input team rows exactly."""
    df = pd.DataFrame(make_game("g1", winner_side="Red"))
    out = run(df, seed=42)
    assert len(out) == 1
    chosen_side = out.iloc[0]["side"]
    chosen_result = out.iloc[0]["result"]
    # Blue lost, Red won in this game
    expected = {("Blue", 0), ("Red", 1)}
    assert (chosen_side, chosen_result) in expected


# --- validate() --------------------------------------------------------

def _valid_row(gameid: str = "g1", side: str = "Blue", league: str = "LCK") -> dict:
    opp_side = "Red" if side == "Blue" else "Blue"
    return {
        "gameid": gameid, "league": league, "position": "team",
        "side": side, "result": 1,
        "teamname": f"{gameid}_{side}",
        "opp_teamname": f"{gameid}_{opp_side}",
    }


def test_validate_passes_on_valid_output():
    df = pd.DataFrame([_valid_row("g1", "Blue"), _valid_row("g2", "Red")])
    validate(df, TIER_MAP)


def test_validate_raises_on_duplicate_gameid():
    df = pd.DataFrame([_valid_row("g1", "Blue"), _valid_row("g1", "Red")])
    with pytest.raises(AssertionError, match="gameid"):
        validate(df, TIER_MAP)


def test_validate_raises_on_player_row_present():
    df = pd.DataFrame([_valid_row("g1"), _valid_row("g2", "Red")])
    df.loc[0, "position"] = "top"
    with pytest.raises(AssertionError, match="team"):
        validate(df, TIER_MAP)


def test_validate_raises_on_side_imbalance():
    # 100 all-Blue rows: diff=100, 5σ=5*sqrt(100)=50 → fails.
    df = pd.DataFrame([_valid_row(f"g{i}", "Blue") for i in range(100)])
    with pytest.raises(AssertionError, match="side imbalance"):
        validate(df, TIER_MAP)


def test_validate_raises_on_unmapped_league():
    df = pd.DataFrame([
        _valid_row("g1", "Blue", league="UnknownLeague"),
        _valid_row("g2", "Red"),
    ])
    with pytest.raises(AssertionError, match="tier"):
        validate(df, TIER_MAP)


def test_validate_raises_on_unexpected_side_value():
    df = pd.DataFrame([_valid_row("g1", "Blue"), _valid_row("g2", "Purple")])
    with pytest.raises(AssertionError, match="side"):
        validate(df, TIER_MAP)


def test_validate_raises_on_null_opp_teamname():
    df = pd.DataFrame([_valid_row("g1", "Blue"), _valid_row("g2", "Red")])
    df.loc[0, "opp_teamname"] = None
    with pytest.raises(AssertionError, match="opp_teamname"):
        validate(df, TIER_MAP)


def test_validate_raises_when_opp_teamname_equals_teamname():
    df = pd.DataFrame([_valid_row("g1", "Blue"), _valid_row("g2", "Red")])
    df.loc[0, "opp_teamname"] = df.loc[0, "teamname"]
    with pytest.raises(AssertionError, match="opp_teamname"):
        validate(df, TIER_MAP)

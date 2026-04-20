"""Tests for stage 1: load and filter raw Oracle CSVs."""
import pandas as pd
import pytest

from scripts.preprocess.stage1_load_and_filter import run, validate


TIER_MAP = {"LCK": "S", "LEC": "S", "TCL": "B"}


def make_game(
    gameid: str = "g1",
    league: str = "LCK",
    year: int = 2023,
    winner_side: str = "Blue",
    complete: bool = True,
    gamelength: int = 1800,
) -> list[dict]:
    """Build 12 rows for one game: 10 player rows + 2 team rows."""
    rows = []
    completeness = "complete" if complete else "partial"
    for side in ("Blue", "Red"):
        teamname = f"{gameid}_{side}_team"
        for pos in ("top", "jng", "mid", "bot", "sup"):
            rows.append({
                "gameid": gameid,
                "datacompleteness": completeness,
                "year": year,
                "league": league,
                "position": pos,
                "side": side,
                "result": 1 if side == winner_side else 0,
                "gamelength": gamelength,
                "teamname": teamname,
            })
    for side in ("Blue", "Red"):
        rows.append({
            "gameid": gameid,
            "datacompleteness": completeness,
            "year": year,
            "league": league,
            "position": "team",
            "side": side,
            "result": 1 if side == winner_side else 0,
            "gamelength": gamelength,
            "teamname": f"{gameid}_{side}_team",
            "firstblood": 1 if side == "Blue" else 0,
            "firstdragon": 1 if side == "Blue" else 0,
            "firsttower": 1 if side == "Blue" else 0,
            "firstbaron": 1 if side == "Blue" else 0,
        })
    return rows


# --- run() -------------------------------------------------------------

def test_run_keeps_a_normal_game():
    df = pd.DataFrame(make_game(gameid="g1"))
    out = run(df, TIER_MAP)
    assert set(out["gameid"]) == {"g1"}
    assert len(out) == 12


def test_run_drops_incomplete_games():
    df = pd.DataFrame(
        make_game(gameid="g1", complete=True)
        + make_game(gameid="g2", complete=False)
    )
    out = run(df, TIER_MAP)
    assert set(out["gameid"]) == {"g1"}


def test_run_drops_games_before_2020():
    df = pd.DataFrame(
        make_game(gameid="old", year=2019)
        + make_game(gameid="ok", year=2020)
    )
    out = run(df, TIER_MAP)
    assert set(out["gameid"]) == {"ok"}


def test_run_keeps_2026_games():
    df = pd.DataFrame(make_game(gameid="latest", year=2026))
    out = run(df, TIER_MAP)
    assert set(out["gameid"]) == {"latest"}


def test_run_drops_games_from_unmapped_leagues():
    df = pd.DataFrame(
        make_game(gameid="g1", league="LCK")
        + make_game(gameid="g2", league="LATAM_NATIONALS")
    )
    out = run(df, TIER_MAP)
    assert set(out["gameid"]) == {"g1"}


def test_run_drops_games_with_null_teamname():
    """If any team row has null teamname, drop the whole gameid."""
    ok = make_game(gameid="ok")
    bad = make_game(gameid="bad")
    for row in bad:
        if row["position"] == "team" and row["side"] == "Red":
            row["teamname"] = None
    df = pd.DataFrame(ok + bad)
    out = run(df, TIER_MAP)
    assert set(out["gameid"]) == {"ok"}


def test_run_drops_self_matches():
    """A game where both team rows have the same teamname is a self-match — drop it."""
    ok = make_game(gameid="ok")
    bad = make_game(gameid="bad")
    for row in bad:
        if row["position"] == "team":
            row["teamname"] = "Academy"
    df = pd.DataFrame(ok + bad)
    out = run(df, TIER_MAP)
    assert set(out["gameid"]) == {"ok"}


def test_run_drops_games_with_null_first_event_flag():
    """If any team row has a null first-event flag, drop the whole gameid."""
    ok = make_game(gameid="ok")
    bad = make_game(gameid="bad")
    for row in bad:
        if row["position"] == "team" and row["side"] == "Red":
            row["firstbaron"] = None
    df = pd.DataFrame(ok + bad)
    out = run(df, TIER_MAP)
    assert set(out["gameid"]) == {"ok"}


def test_run_drops_no_winner_games():
    """A gameid whose team rows both have result==0 is dropped entirely."""
    ok = make_game(gameid="ok")
    bad = make_game(gameid="bad")
    for row in bad:
        if row["position"] == "team":
            row["result"] = 0
    df = pd.DataFrame(ok + bad)
    out = run(df, TIER_MAP)
    assert set(out["gameid"]) == {"ok"}


def test_run_preserves_all_12_rows_per_surviving_game():
    df = pd.DataFrame(make_game(gameid="g1") + make_game(gameid="g2"))
    out = run(df, TIER_MAP)
    assert (out.groupby("gameid").size() == 12).all()


# --- validate() --------------------------------------------------------

def test_validate_passes_on_valid_input():
    df = pd.DataFrame(make_game(gameid="g1"))
    validate(df, TIER_MAP)  # does not raise


def test_validate_raises_on_incomplete_row():
    df = pd.DataFrame(make_game(gameid="g1"))
    df.loc[0, "datacompleteness"] = "partial"
    with pytest.raises(AssertionError, match="datacompleteness"):
        validate(df, TIER_MAP)


def test_validate_raises_on_year_before_2020():
    df = pd.DataFrame(make_game(gameid="g1", year=2019))
    with pytest.raises(AssertionError, match="year"):
        validate(df, TIER_MAP)


def test_validate_raises_on_year_after_2026():
    df = pd.DataFrame(make_game(gameid="g1", year=2027))
    with pytest.raises(AssertionError, match="year"):
        validate(df, TIER_MAP)


def test_validate_raises_on_unmapped_league():
    df = pd.DataFrame(make_game(gameid="g1", league="UnknownLeague"))
    with pytest.raises(AssertionError, match="tier"):
        validate(df, TIER_MAP)


def test_validate_raises_when_gameid_has_wrong_row_count():
    df = pd.DataFrame(make_game(gameid="g1")).iloc[:-1]  # 11 rows
    with pytest.raises(AssertionError, match="12 rows"):
        validate(df, TIER_MAP)


def test_validate_raises_on_null_team_result():
    df = pd.DataFrame(make_game(gameid="g1"))
    first_team_idx = df[df["position"] == "team"].index[0]
    df.loc[first_team_idx, "result"] = None
    with pytest.raises(AssertionError, match="result"):
        validate(df, TIER_MAP)


def test_validate_raises_on_null_team_side():
    df = pd.DataFrame(make_game(gameid="g1"))
    first_team_idx = df[df["position"] == "team"].index[0]
    df.loc[first_team_idx, "side"] = None
    with pytest.raises(AssertionError, match="side"):
        validate(df, TIER_MAP)


def test_validate_raises_on_null_team_gamelength():
    df = pd.DataFrame(make_game(gameid="g1"))
    first_team_idx = df[df["position"] == "team"].index[0]
    df.loc[first_team_idx, "gamelength"] = None
    with pytest.raises(AssertionError, match="gamelength"):
        validate(df, TIER_MAP)


def test_validate_raises_on_self_match():
    df = pd.DataFrame(make_game(gameid="g1"))
    df.loc[df["position"] == "team", "teamname"] = "OneTeam"
    with pytest.raises(AssertionError, match="teamname"):
        validate(df, TIER_MAP)


def test_validate_raises_on_null_team_teamname():
    df = pd.DataFrame(make_game(gameid="g1"))
    first_team_idx = df[df["position"] == "team"].index[0]
    df.loc[first_team_idx, "teamname"] = None
    with pytest.raises(AssertionError, match="teamname"):
        validate(df, TIER_MAP)


def test_validate_raises_on_null_first_event_flag():
    df = pd.DataFrame(make_game(gameid="g1"))
    first_team_idx = df[df["position"] == "team"].index[0]
    df.loc[first_team_idx, "firstdragon"] = None
    with pytest.raises(AssertionError, match="firstdragon"):
        validate(df, TIER_MAP)


def test_validate_raises_on_no_winner():
    df = pd.DataFrame(make_game(gameid="g1"))
    df.loc[df["position"] == "team", "result"] = 0
    with pytest.raises(AssertionError, match="winner"):
        validate(df, TIER_MAP)


def test_validate_raises_on_two_winners():
    df = pd.DataFrame(make_game(gameid="g1"))
    df.loc[df["position"] == "team", "result"] = 1
    with pytest.raises(AssertionError, match="winner"):
        validate(df, TIER_MAP)

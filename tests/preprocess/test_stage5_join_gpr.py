"""Tests for stage 5: per-game cutoff join of GPR snapshots."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.preprocess.stage5_join_gpr import (
    NEW_COLUMNS,
    run,
    validate,
)


# --- fixtures ---------------------------------------------------------

def make_checkpoint_row(
    gameid: str = "g1",
    date: str = "2024-06-15 12:00:00",
    teamname: str = "T_alpha",
    opp_teamname: str = "T_beta",
    **overrides,
) -> dict:
    """Stage-4 T=10 row shape, just enough for run() to operate on."""
    row = {
        "gameid": gameid,
        "date": date,
        "year": int(date[:4]),
        "league": "LCK",
        "patch": "14.12",
        "teamname": teamname,
        "opp_teamname": opp_teamname,
        "side": "Blue",
        "tier": "S",
        "goldat10": 15000, "opp_goldat10": 15000, "golddiffat10": 0,
        "csat10": 300, "opp_csat10": 300, "csdiffat10": 0,
        "killsat10": 1, "opp_killsat10": 1, "killdiffat10": 0,
        "assistsat10": 1, "opp_assistsat10": 1, "assistdiffat10": 0,
        "firstblood": 1,
        "gamelength": 1800,
        "result": 1,
    }
    row.update(overrides)
    return row


def make_snapshot(team_name: str, snapshot_date: str, power_score: float, team_id: str = "tid") -> dict:
    return {
        "team_id": team_id,
        "team_code": team_name[:3].upper(),
        "team_name": team_name,
        "home_league_slug": "lck",
        "snapshot_date": snapshot_date,
        "power_score": power_score,
        "team_elo": power_score - 20,
        "rank": 1,
    }


SNAPSHOT_COLS = [
    "team_id", "team_code", "team_name", "home_league_slug",
    "snapshot_date", "power_score", "team_elo", "rank",
]


def _make_snapshots(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=SNAPSHOT_COLS)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], utc=True)
    return df.sort_values("snapshot_date").reset_index(drop=True)


# --- happy path -------------------------------------------------------

def test_basic_two_team_rated_join():
    df = pd.DataFrame([make_checkpoint_row()])
    snapshots = _make_snapshots([
        make_snapshot("T_alpha", "2024-06-01T00:00:00Z", 1500),
        make_snapshot("T_beta",  "2024-06-01T00:00:00Z", 1400),
    ])
    aliases = {"T_alpha": "T_alpha", "T_beta": "T_beta"}

    out = run(df, snapshots, aliases)

    assert out.loc[0, "has_gpr"] == 1
    assert out.loc[0, "gpr_team"] == 1500
    assert out.loc[0, "gpr_opp_team"] == 1400
    assert out.loc[0, "gpr_diff"] == 100
    validate(out)


def test_takes_latest_qualifying_snapshot():
    df = pd.DataFrame([make_checkpoint_row(date="2024-06-15 12:00:00")])
    snapshots = _make_snapshots([
        make_snapshot("T_alpha", "2024-06-01T00:00:00Z", 1500),
        make_snapshot("T_alpha", "2024-06-10T00:00:00Z", 1600),  # latest <  game date
        make_snapshot("T_alpha", "2024-06-20T00:00:00Z", 1700),  # post-game, must be ignored
        make_snapshot("T_beta",  "2024-05-01T00:00:00Z", 1400),
    ])
    aliases = {"T_alpha": "T_alpha", "T_beta": "T_beta"}

    out = run(df, snapshots, aliases)
    assert out.loc[0, "gpr_team"] == 1600


# --- strict-< boundary ------------------------------------------------

def test_strict_less_than_boundary_excludes_same_instant_snapshot():
    """A snapshot dateCalculated == game_date to the second is treated as
    same-day-as-or-after-game and must NOT join."""
    df = pd.DataFrame([make_checkpoint_row(
        date="2024-06-15 12:00:00",
        teamname="T_alpha", opp_teamname="T_beta",
    )])
    snapshots = _make_snapshots([
        # T_alpha: only snapshot is exactly at game time → must be excluded
        make_snapshot("T_alpha", "2024-06-15T12:00:00Z", 1500),
        make_snapshot("T_beta",  "2024-06-01T00:00:00Z", 1400),
    ])
    aliases = {"T_alpha": "T_alpha", "T_beta": "T_beta"}

    out = run(df, snapshots, aliases)
    # T_alpha has no qualifying snapshot → has_gpr=0, all GPR cols zero.
    assert out.loc[0, "has_gpr"] == 0
    assert out.loc[0, "gpr_team"] == 0
    assert out.loc[0, "gpr_opp_team"] == 0


def test_one_second_before_qualifies():
    df = pd.DataFrame([make_checkpoint_row(
        date="2024-06-15 12:00:00",
        teamname="T_alpha", opp_teamname="T_beta",
    )])
    snapshots = _make_snapshots([
        make_snapshot("T_alpha", "2024-06-15T11:59:59Z", 1500),
        make_snapshot("T_beta",  "2024-06-01T00:00:00Z", 1400),
    ])
    out = run(df, snapshots, {"T_alpha": "T_alpha", "T_beta": "T_beta"})
    assert out.loc[0, "has_gpr"] == 1
    assert out.loc[0, "gpr_team"] == 1500


# --- both-teams-rated requirement -------------------------------------

def test_only_one_team_rated_zeros_everything():
    df = pd.DataFrame([make_checkpoint_row(teamname="T_alpha", opp_teamname="T_unrated")])
    snapshots = _make_snapshots([
        make_snapshot("T_alpha", "2024-06-01T00:00:00Z", 1500),
        # No snapshot for T_unrated.
    ])
    out = run(df, snapshots, {"T_alpha": "T_alpha", "T_unrated": "T_unrated"})
    # Even though T_alpha has a snapshot, has_gpr=0 because the opponent doesn't.
    # gpr_team must be zeroed too — the spec keeps unrated rows clean.
    assert out.loc[0, "has_gpr"] == 0
    assert out.loc[0, "gpr_team"] == 0
    assert out.loc[0, "gpr_opp_team"] == 0
    assert out.loc[0, "gpr_diff"] == 0


def test_neither_team_in_alias_table():
    df = pd.DataFrame([make_checkpoint_row(teamname="UnknownX", opp_teamname="UnknownY")])
    snapshots = _make_snapshots([
        make_snapshot("T_alpha", "2024-06-01T00:00:00Z", 1500),
    ])
    out = run(df, snapshots, {})  # empty alias table
    assert out.loc[0, "has_gpr"] == 0


def test_alias_resolves_oe_name_to_gpr_name():
    """OE 'Cloud9' maps to GPR 'Cloud9 Kia' via aliases — same team, sponsor change."""
    df = pd.DataFrame([make_checkpoint_row(teamname="Cloud9", opp_teamname="NRG")])
    snapshots = _make_snapshots([
        make_snapshot("Cloud9 Kia", "2024-06-01T00:00:00Z", 1450),
        make_snapshot("NRG Kia",    "2024-06-01T00:00:00Z", 1380),
    ])
    aliases = {"Cloud9": "Cloud9 Kia", "NRG": "NRG Kia"}
    out = run(df, snapshots, aliases)
    assert out.loc[0, "has_gpr"] == 1
    assert out.loc[0, "gpr_team"] == 1450
    assert out.loc[0, "gpr_opp_team"] == 1380


# --- cross-year-boundary anchoring ------------------------------------

def test_january_game_anchors_to_prior_year_end_snapshot():
    """A 2025-01 game whose only qualifying snapshot is the 2024 year-end
    snapshot must join to that snapshot (no in-year history exists yet)."""
    df = pd.DataFrame([make_checkpoint_row(date="2025-01-15 03:00:00")])
    snapshots = _make_snapshots([
        # 2024 year-end snapshot
        make_snapshot("T_alpha", "2024-11-02T00:00:00Z", 1550),
        make_snapshot("T_beta",  "2024-11-02T00:00:00Z", 1420),
        # First 2025 snapshot, AFTER the game date — must not be selected
        make_snapshot("T_alpha", "2025-01-22T00:00:00Z", 1600),
        make_snapshot("T_beta",  "2025-01-22T00:00:00Z", 1450),
    ])
    out = run(df, snapshots, {"T_alpha": "T_alpha", "T_beta": "T_beta"})
    assert out.loc[0, "has_gpr"] == 1
    assert out.loc[0, "gpr_team"] == 1550
    assert out.loc[0, "gpr_opp_team"] == 1420


# --- pre-coverage window ----------------------------------------------

def test_pregpr_era_unrated():
    """Games before any snapshot exists must be has_gpr=0."""
    df = pd.DataFrame([make_checkpoint_row(date="2022-06-15 12:00:00")])
    snapshots = _make_snapshots([
        make_snapshot("T_alpha", "2024-02-10T00:00:00Z", 1500),
        make_snapshot("T_beta",  "2024-02-10T00:00:00Z", 1400),
    ])
    out = run(df, snapshots, {"T_alpha": "T_alpha", "T_beta": "T_beta"})
    assert out.loc[0, "has_gpr"] == 0


# --- invariants & schema ----------------------------------------------

def test_gpr_diff_invariant_holds_after_run():
    """Stage-3 *diff invariant analogue: gpr_diff == gpr_team - gpr_opp_team exactly."""
    rows = [
        make_checkpoint_row(gameid=f"g{i}", date="2024-06-15 12:00:00",
                            teamname="T_alpha", opp_teamname="T_beta")
        for i in range(5)
    ]
    df = pd.DataFrame(rows)
    snapshots = _make_snapshots([
        make_snapshot("T_alpha", "2024-06-01T00:00:00Z", 1517.3),
        make_snapshot("T_beta",  "2024-06-01T00:00:00Z", 1402.9),
    ])
    out = run(df, snapshots, {"T_alpha": "T_alpha", "T_beta": "T_beta"})
    assert (out["gpr_diff"] == out["gpr_team"] - out["gpr_opp_team"]).all()
    validate(out)


def test_validate_catches_diff_violation():
    """Synthetic violation of the gpr_diff invariant must be caught by validate()."""
    df = pd.DataFrame([make_checkpoint_row()])
    snapshots = _make_snapshots([
        make_snapshot("T_alpha", "2024-06-01T00:00:00Z", 1500),
        make_snapshot("T_beta",  "2024-06-01T00:00:00Z", 1400),
    ])
    out = run(df, snapshots, {"T_alpha": "T_alpha", "T_beta": "T_beta"})
    out.loc[0, "gpr_diff"] = 999.0
    with pytest.raises(AssertionError, match="gpr_diff invariant"):
        validate(out)


def test_validate_catches_nonzero_unrated():
    df = pd.DataFrame([make_checkpoint_row(teamname="UnknownX", opp_teamname="UnknownY")])
    out = run(df, _make_snapshots([]), {})
    assert out.loc[0, "has_gpr"] == 0
    out.loc[0, "gpr_team"] = 1000.0  # corrupt
    out.loc[0, "gpr_diff"] = 1000.0
    with pytest.raises(AssertionError, match="nonzero"):
        validate(out)


def test_output_preserves_input_columns_plus_four():
    df = pd.DataFrame([make_checkpoint_row()])
    snapshots = _make_snapshots([
        make_snapshot("T_alpha", "2024-06-01T00:00:00Z", 1500),
        make_snapshot("T_beta",  "2024-06-01T00:00:00Z", 1400),
    ])
    out = run(df, snapshots, {"T_alpha": "T_alpha", "T_beta": "T_beta"})
    extra = set(out.columns) - set(df.columns)
    assert extra == set(NEW_COLUMNS)
    # Original columns retained in order.
    assert list(out.columns)[: len(df.columns)] == list(df.columns)


def test_row_order_preserved():
    """merge_asof requires sorted left input internally; we must restore order."""
    rows = [
        make_checkpoint_row(gameid="g_late",  date="2024-08-01 12:00:00",
                            teamname="T_alpha", opp_teamname="T_beta"),
        make_checkpoint_row(gameid="g_early", date="2024-03-01 12:00:00",
                            teamname="T_alpha", opp_teamname="T_beta"),
        make_checkpoint_row(gameid="g_mid",   date="2024-06-01 12:00:00",
                            teamname="T_alpha", opp_teamname="T_beta"),
    ]
    df = pd.DataFrame(rows)
    snapshots = _make_snapshots([
        make_snapshot("T_alpha", "2024-02-01T00:00:00Z", 1500),
        make_snapshot("T_beta",  "2024-02-01T00:00:00Z", 1400),
        make_snapshot("T_alpha", "2024-05-01T00:00:00Z", 1550),
        make_snapshot("T_beta",  "2024-05-01T00:00:00Z", 1450),
    ])
    out = run(df, snapshots, {"T_alpha": "T_alpha", "T_beta": "T_beta"})
    assert list(out["gameid"]) == ["g_late", "g_early", "g_mid"]
    # Each row attaches the correct latest-prior snapshot for that date:
    assert out.loc[out["gameid"] == "g_early", "gpr_team"].iloc[0] == 1500
    assert out.loc[out["gameid"] == "g_mid",   "gpr_team"].iloc[0] == 1550
    assert out.loc[out["gameid"] == "g_late",  "gpr_team"].iloc[0] == 1550

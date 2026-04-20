"""
End-to-end validation: pick N random games per year from the raw Oracle's
Elixir dataset and verify their rows in 03_features.csv and each
checkpoint_{T}.csv, column-by-column.

For every sampled game:
  1. Locate the two team rows in the raw CSV (chunked read — never loads
     the full file into memory).
  2. Look up the pipeline row in 03_features.csv. Stage 2 picks one of
     the two team rows randomly (seed=42); the output's `side` column
     tells us which row was picked. The test uses that to compute every
     expected column (raw stats from the picked row, `opp_*` stats from
     the sibling row, engineered diffs, identifiers, tier, label).
  3. For each T in {10, 15, 20, 25}, check checkpoint_{T}.csv: the game
     must appear iff all 12 T-stat columns are populated on the stage 3
     row, and every column must match.

Games that were correctly dropped by stage 1 (unmapped tier, partial
datacompleteness, etc.) pass by confirming they're absent from every
pipeline output.
"""
import glob
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest

from scripts.preprocess.stage1_load_and_filter import (
    DATA_DIR,
    TIER_MAP_PATH,
    load_tier_map,
)
from scripts.preprocess.stage3_features import OUTPUT_COLUMNS as STAGE3_COLUMNS
from scripts.preprocess.stage4_split_checkpoints import (
    CHECKPOINTS,
    checkpoint_stat_columns,
    output_columns as checkpoint_columns,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "03_features.csv"
CHECKPOINT_FMT = str(REPO_ROOT / "data" / "processed" / "checkpoint_{T}.csv")

YEARS = range(2020, 2027)
SAMPLES_PER_YEAR = int(os.environ.get("E2E_PER_YEAR", "2"))
SEED = int(os.environ.get("E2E_SEED", "7"))


# --- sampling (runs at collection time) -------------------------------

def _sample_gameids() -> list[tuple[int, str]]:
    """Read just the gameid column from each year's raw CSV; pick N per year."""
    if not Path(DATA_DIR).exists():
        return []
    rng = random.Random(SEED)
    out: list[tuple[int, str]] = []
    for year in YEARS:
        paths = sorted(glob.glob(os.path.join(DATA_DIR, f"{year}_*.csv")))
        if not paths:
            continue
        gids: set[str] = set()
        for p in paths:
            gids.update(pd.read_csv(p, usecols=["gameid"])["gameid"].dropna().unique())
        for gid in rng.sample(sorted(gids), k=SAMPLES_PER_YEAR):
            out.append((year, gid))
    return out


SAMPLES = _sample_gameids()


# --- fixtures ---------------------------------------------------------

@pytest.fixture(scope="session")
def pipeline_outputs():
    if not FEATURES_PATH.exists():
        pytest.skip("pipeline outputs missing — run `python scripts/build_dataset.py` first")
    features = pd.read_csv(FEATURES_PATH, low_memory=False).set_index("gameid")
    checkpoints = {
        T: pd.read_csv(CHECKPOINT_FMT.format(T=T), low_memory=False).set_index("gameid")
        for T in CHECKPOINTS
    }
    return features, checkpoints


@pytest.fixture(scope="session")
def tier_map():
    return load_tier_map(TIER_MAP_PATH)


@pytest.fixture(scope="session")
def raw_rows_by_gameid() -> dict[str, pd.DataFrame]:
    """Chunked-load only the sampled gameids from raw CSVs, keyed by gameid."""
    if not SAMPLES:
        pytest.skip("raw data/lol-historical-data/ not present")
    gids_by_year: dict[int, set[str]] = defaultdict(set)
    for year, gid in SAMPLES:
        gids_by_year[year].add(gid)

    collected: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for year, gids in gids_by_year.items():
        for p in sorted(glob.glob(os.path.join(DATA_DIR, f"{year}_*.csv"))):
            for chunk in pd.read_csv(p, chunksize=100_000, low_memory=False):
                match = chunk[chunk["gameid"].isin(gids)]
                if len(match):
                    for gid, group in match.groupby("gameid"):
                        collected[gid].append(group)
    return {gid: pd.concat(frames, ignore_index=True) for gid, frames in collected.items()}


# --- helpers ----------------------------------------------------------

def values_equal(a, b) -> bool:
    """NaN==NaN is True; normalize numpy/python numeric scalars."""
    a_nan = isinstance(a, float) and math.isnan(a)
    b_nan = isinstance(b, float) and math.isnan(b)
    if a_nan and b_nan:
        return True
    if a_nan or b_nan:
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def compute_expected(picked: pd.Series, other: pd.Series, tier_map: dict[str, str]) -> dict:
    exp = {
        "gameid": picked["gameid"],
        "date": picked["date"],
        "year": int(picked["year"]),
        "league": picked["league"],
        "patch": picked["patch"],
        "teamname": picked["teamname"],
        "opp_teamname": other["teamname"],
        "side": picked["side"],
        "tier": tier_map[picked["league"]],
        "firstblood": int(picked["firstblood"]),
        "firstdragon": int(picked["firstdragon"]),
        "firsttower": int(picked["firsttower"]),
        "firstbaron": int(picked["firstbaron"]),
        "gamelength": int(picked["gamelength"]),
        "result": int(picked["result"]),
    }
    for T in CHECKPOINTS:
        exp[f"goldat{T}"] = picked[f"goldat{T}"]
        exp[f"opp_goldat{T}"] = other[f"goldat{T}"]
        exp[f"golddiffat{T}"] = picked[f"golddiffat{T}"]
        exp[f"csat{T}"] = picked[f"csat{T}"]
        exp[f"opp_csat{T}"] = other[f"csat{T}"]
        exp[f"csdiffat{T}"] = picked[f"csdiffat{T}"]
        exp[f"killsat{T}"] = picked[f"killsat{T}"]
        exp[f"opp_killsat{T}"] = other[f"killsat{T}"]
        exp[f"killdiffat{T}"] = picked[f"killsat{T}"] - other[f"killsat{T}"]
        exp[f"assistsat{T}"] = picked[f"assistsat{T}"]
        exp[f"opp_assistsat{T}"] = other[f"assistsat{T}"]
        exp[f"assistdiffat{T}"] = picked[f"assistsat{T}"] - other[f"assistsat{T}"]
    return exp


def diagnose_stage1_drop(team_rows: pd.DataFrame, tier_map: dict[str, str]) -> str:
    if len(team_rows) == 0:
        return "no team rows in raw"
    if not (team_rows["datacompleteness"] == "complete").all():
        return f"datacompleteness={sorted(set(team_rows['datacompleteness']))}"
    if team_rows["league"].iloc[0] not in tier_map:
        return f"league '{team_rows['league'].iloc[0]}' has no tier mapping"
    if team_rows["teamname"].isna().any():
        return "null teamname on a team row"
    if team_rows["teamname"].nunique() == 1:
        return f"academy self-match ('{team_rows['teamname'].iloc[0]}')"
    for flag in ("firstblood", "firstdragon", "firsttower", "firstbaron"):
        if team_rows[flag].isna().any():
            return f"null {flag}"
    winners = (team_rows["result"] == 1).sum()
    if winners != 1:
        return f"winner count = {winners}"
    return "unknown — passes every obvious filter, investigate"


def compare_row(expected: dict, actual: dict, cols: list[str]) -> list[str]:
    return [
        f"{c}: expected={expected[c]!r}, actual={actual[c]!r}"
        for c in cols
        if not values_equal(expected[c], actual[c])
    ]


def output_row_dict(df: pd.DataFrame, gid: str) -> dict:
    row = df.loc[gid].to_dict()
    row["gameid"] = gid
    return row


# --- tests ------------------------------------------------------------

def _param(sample: tuple[int, str]) -> pytest.param:
    year, gid = sample
    return pytest.param(year, gid, id=f"{year}-{gid}")


@pytest.mark.parametrize(
    "year,gameid",
    [_param(s) for s in SAMPLES]
    or [pytest.param(None, None, marks=pytest.mark.skip(reason="raw data/ missing"))],
)
def test_game_passes_through_pipeline_correctly(
    year, gameid, pipeline_outputs, tier_map, raw_rows_by_gameid
):
    features, checkpoints = pipeline_outputs
    game_rows = raw_rows_by_gameid[gameid]
    team_rows = game_rows[game_rows["position"] == "team"]

    assert len(team_rows) == 2, \
        f"raw data for {gameid}: expected 2 team rows, got {len(team_rows)}"

    in_features = gameid in features.index
    in_checkpoints = {T: gameid in checkpoints[T].index for T in CHECKPOINTS}

    # Case 1: game was dropped at stage 1 → must be absent from every output.
    if not in_features:
        reason = diagnose_stage1_drop(team_rows, tier_map)
        for T, present in in_checkpoints.items():
            assert not present, \
                f"{gameid} dropped at stage 1 ({reason}) but still present in checkpoint_{T}"
        pytest.skip(f"correctly dropped at stage 1: {reason}")

    # Case 2: game is in 03_features.csv → row must appear exactly once.
    matches = features.loc[[gameid]]
    assert len(matches) == 1, \
        f"{gameid} appears {len(matches)} times in 03_features.csv (expected 1)"

    actual_s3 = output_row_dict(features, gameid)
    picked_side = actual_s3["side"]
    picked = team_rows[team_rows["side"] == picked_side].iloc[0]
    other = team_rows[team_rows["side"] != picked_side].iloc[0]

    expected = compute_expected(picked, other, tier_map)

    # Stage 3: every column matches.
    s3_diffs = compare_row(expected, actual_s3, STAGE3_COLUMNS)
    assert not s3_diffs, (
        f"{gameid} (picked side={picked_side}, {picked['teamname']} vs {other['teamname']}) — "
        f"stage 3 column mismatches:\n  " + "\n  ".join(s3_diffs)
    )

    # Stage 4: inclusion/exclusion policy + per-column match when included.
    for T in CHECKPOINTS:
        stat_cols = checkpoint_stat_columns(T)
        populated = all(not (isinstance(actual_s3[c], float) and math.isnan(actual_s3[c]))
                        for c in stat_cols)
        present = in_checkpoints[T]

        assert populated == present, (
            f"{gameid}: T={T} populated={populated}, but present_in_checkpoint={present}"
        )
        if not present:
            continue

        cp_matches = checkpoints[T].loc[[gameid]]
        assert len(cp_matches) == 1, \
            f"{gameid} appears {len(cp_matches)} times in checkpoint_{T} (expected 1)"

        cp_diffs = compare_row(expected, output_row_dict(checkpoints[T], gameid), checkpoint_columns(T))
        assert not cp_diffs, (
            f"{gameid} — checkpoint_{T} column mismatches:\n  " + "\n  ".join(cp_diffs)
        )

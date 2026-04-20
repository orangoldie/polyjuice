"""
Stage 1: load raw Oracle's Elixir CSVs and apply row-level filters.

Filters:
  - datacompleteness == "complete"
  - year in 2020..2026
  - league present in the tier map
  - gameid has at least one team row with result == 1
  - both team rows have a non-null teamname (needed to derive opp_teamname)
  - the two team rows have distinct teamnames (drop academy self-matches)
  - no team row has a null first-event flag (firstblood/dragon/tower/baron)

Output: data/processed/01_filtered.csv  (all 12 rows per surviving game,
all columns preserved).
"""
import argparse
import glob
import json
import os

import pandas as pd


YEAR_MIN = 2020
YEAR_MAX = 2026

FIRST_FLAGS = ("firstblood", "firstdragon", "firsttower", "firstbaron")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data", "lol-historical-data")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "01_filtered.csv")
TIER_MAP_PATH = os.path.join(REPO_ROOT, "data", "exploration", "07_league_tier_matching.json")


def load_tier_map(path: str = TIER_MAP_PATH) -> dict[str, str]:
    with open(path) as f:
        data = json.load(f)
    return {m["league"]: m["tier"] for m in data["matched"]}


def load_raw_csvs(data_dir: str = DATA_DIR, years: range = range(YEAR_MIN, YEAR_MAX + 1)) -> pd.DataFrame:
    frames = []
    for year in years:
        for path in sorted(glob.glob(os.path.join(data_dir, f"{year}_*.csv"))):
            frames.append(pd.read_csv(path, low_memory=False))
    return pd.concat(frames, ignore_index=True)


def run(df: pd.DataFrame, tier_map: dict[str, str]) -> pd.DataFrame:
    df = df[df["datacompleteness"] == "complete"]
    df = df[df["year"].between(YEAR_MIN, YEAR_MAX)]
    df = df[df["league"].isin(tier_map.keys())]

    team_rows = df[df["position"] == "team"]
    winning_gameids = team_rows.loc[team_rows["result"] == 1, "gameid"].unique()
    df = df[df["gameid"].isin(winning_gameids)]

    # Drop gameids where any team row is missing teamname — we need both sides'
    # names to derive opp_teamname in stage 2, and to identify the team.
    team_rows = df[df["position"] == "team"]
    null_name_gids = team_rows.loc[team_rows["teamname"].isna(), "gameid"].unique()
    df = df[~df["gameid"].isin(null_name_gids)]

    # Drop gameids where both team rows have the same teamname — academy/scrim
    # self-matches (e.g. LAS 2020-21); the outcome is structurally meaningless.
    team_rows = df[df["position"] == "team"]
    name_pairs = team_rows.groupby("gameid")["teamname"].nunique()
    self_match_gids = name_pairs[name_pairs < 2].index
    df = df[~df["gameid"].isin(self_match_gids)]

    # Drop gameids where any team row has a null first-event flag — these come
    # from data-quality-flagged leagues (LPL 2020-21, LDL) and the games are
    # long enough that the nulls mean "missing data" not "didn't happen".
    team_rows = df[df["position"] == "team"]
    null_firsts_mask = team_rows[list(FIRST_FLAGS)].isna().any(axis=1)
    null_firsts_gids = team_rows.loc[null_firsts_mask, "gameid"].unique()
    df = df[~df["gameid"].isin(null_firsts_gids)]

    return df.reset_index(drop=True)


def validate(df: pd.DataFrame, tier_map: dict[str, str]) -> None:
    assert (df["datacompleteness"] == "complete").all(), \
        "non-complete datacompleteness rows present"

    assert df["year"].between(YEAR_MIN, YEAR_MAX).all(), \
        f"year outside {YEAR_MIN}-{YEAR_MAX}"

    assert df["league"].isin(tier_map.keys()).all(), \
        "league without tier mapping present"

    row_counts = df.groupby("gameid").size()
    bad = row_counts[row_counts != 12]
    assert bad.empty, f"gameid(s) without 12 rows: {bad.head().to_dict()}"

    team_rows = df[df["position"] == "team"]
    for col in ("result", "side", "gamelength", "teamname", *FIRST_FLAGS):
        assert team_rows[col].notna().all(), f"null {col} on team row"

    winners_per_game = team_rows.groupby("gameid")["result"].sum()
    bad = winners_per_game[winners_per_game != 1]
    assert bad.empty, f"gameid(s) without exactly 1 winner: {bad.head().to_dict()}"

    distinct_names = team_rows.groupby("gameid")["teamname"].nunique()
    bad = distinct_names[distinct_names != 2]
    assert bad.empty, f"gameid(s) where the 2 team rows share a teamname: {bad.head().to_dict()}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--tier-map", default=TIER_MAP_PATH)
    args = parser.parse_args()

    print("Loading raw CSVs...")
    df = load_raw_csvs(args.data_dir)
    print(f"  loaded {len(df):,} raw rows from {YEAR_MIN}-{YEAR_MAX}")

    tier_map = load_tier_map(args.tier_map)
    print(f"  tier map: {len(tier_map)} leagues")

    print("Filtering...")
    out = run(df, tier_map)
    print(f"  kept {len(out):,} rows ({out['gameid'].nunique():,} games)")

    print("Validating...")
    validate(out, tier_map)
    print("  ok")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

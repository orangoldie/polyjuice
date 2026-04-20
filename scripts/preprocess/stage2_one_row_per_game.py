"""
Stage 2: collapse to one row per game.

Drops the 10 player rows; for each gameid, randomly picks one of the 2
team rows (Blue or Red) with a fixed RNG seed. Balancing `side` ~50/50
makes it a learnable categorical feature, and lets inference frame the
prediction as P(Team X wins) from Team X's own perspective.

Input:  data/processed/01_filtered.csv
Output: data/processed/02_one_row.csv
"""
import argparse
import math
import os

import numpy as np
import pandas as pd

from scripts.preprocess.stage1_load_and_filter import TIER_MAP_PATH, load_tier_map


def _attach_opp_teamname(team_rows: pd.DataFrame) -> pd.DataFrame:
    """Attach `opp_teamname` — the teamname of the other side in the same gameid."""
    by_gameid_side = team_rows.pivot(index="gameid", columns="side", values="teamname")
    blue = team_rows["gameid"].map(by_gameid_side["Blue"])
    red = team_rows["gameid"].map(by_gameid_side["Red"])
    out = team_rows.copy()
    out["opp_teamname"] = np.where(out["side"] == "Blue", red.to_numpy(), blue.to_numpy())
    return out


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "01_filtered.csv")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "02_one_row.csv")

SEED = 42


def run(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    team_rows = df[df["position"] == "team"]
    team_rows = _attach_opp_teamname(team_rows)
    rng = np.random.default_rng(seed)
    picks = team_rows.groupby("gameid", sort=True).sample(n=1, random_state=rng)
    return picks.reset_index(drop=True)


def validate(df: pd.DataFrame, tier_map: dict[str, str]) -> None:
    n = len(df)

    assert n == df["gameid"].nunique(), \
        "duplicate gameid: expected one row per game"

    assert (df["position"] == "team").all(), \
        "non-team position rows present — stage 2 must drop player rows"

    assert df["league"].isin(tier_map.keys()).all(), \
        "league without tier mapping present"

    blue = int((df["side"] == "Blue").sum())
    red = int((df["side"] == "Red").sum())
    assert blue + red == n, \
        f"unexpected side values present: {sorted(set(df['side']) - {'Blue', 'Red'})}"

    diff = abs(blue - red)
    tol = 5.0 * math.sqrt(n)
    assert diff <= tol, \
        f"side imbalance: blue={blue}, red={red}, diff={diff}, 5σ tolerance={tol:.1f}"

    assert df["opp_teamname"].notna().all(), "null opp_teamname"
    assert (df["opp_teamname"] != df["teamname"]).all(), \
        "opp_teamname equals teamname on some row"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=INPUT_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--tier-map", default=TIER_MAP_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print("Loading filtered rows...")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"  loaded {len(df):,} rows ({df['gameid'].nunique():,} games)")

    tier_map = load_tier_map(args.tier_map)

    print("Picking one team row per game...")
    out = run(df, seed=args.seed)
    blue = int((out["side"] == "Blue").sum())
    red = int((out["side"] == "Red").sum())
    print(f"  kept {len(out):,} rows  (blue={blue} {blue/len(out):.1%}, red={red} {red/len(out):.1%})")

    print("Validating...")
    validate(out, tier_map)
    print("  ok")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

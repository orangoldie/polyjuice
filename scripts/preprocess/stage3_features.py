"""
Stage 3: feature engineering.

  - Join `tier` onto each row from the Liquipedia league-tier map.
  - Engineer `killdiffat{T}` and `assistdiffat{T}` (Oracle ships gold/cs
    diffs but not kill/assist diffs).
  - Select exactly the 63-column locked v1 feature set, drop the rest.

Does NOT filter rows and does NOT split by checkpoint. The per-checkpoint
row filter (games shorter than T minutes) and file split happen in stage 4.

Input:  data/processed/02_one_row.csv
Output: data/processed/03_features.csv
"""
import argparse
import os

import pandas as pd

from scripts.preprocess.stage1_load_and_filter import TIER_MAP_PATH, load_tier_map


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "02_one_row.csv")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "03_features.csv")

CHECKPOINTS = (10, 15, 20, 25)

STATIC_FEATURES = ["side", "tier"]
FIRST_FLAGS = ["firstblood", "firstdragon", "firsttower", "firstbaron"]
IDENTIFIERS = ["gameid", "date", "year", "league", "patch", "teamname", "opp_teamname"]
GAMELENGTH = ["gamelength"]
LABEL = ["result"]

CHECKPOINT_FEATURES: list[str] = []
for _T in CHECKPOINTS:
    CHECKPOINT_FEATURES += [
        f"goldat{_T}", f"opp_goldat{_T}", f"golddiffat{_T}",
        f"csat{_T}", f"opp_csat{_T}", f"csdiffat{_T}",
        f"killsat{_T}", f"opp_killsat{_T}", f"killdiffat{_T}",
        f"assistsat{_T}", f"opp_assistsat{_T}", f"assistdiffat{_T}",
    ]

OUTPUT_COLUMNS = IDENTIFIERS + STATIC_FEATURES + CHECKPOINT_FEATURES + FIRST_FLAGS + GAMELENGTH + LABEL

ALWAYS_FILLED = IDENTIFIERS + STATIC_FEATURES + FIRST_FLAGS + GAMELENGTH + LABEL


def run(df: pd.DataFrame, tier_map: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    df["tier"] = df["league"].map(tier_map)
    for T in CHECKPOINTS:
        df[f"killdiffat{T}"] = df[f"killsat{T}"] - df[f"opp_killsat{T}"]
        df[f"assistdiffat{T}"] = df[f"assistsat{T}"] - df[f"opp_assistsat{T}"]
    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    cols = set(df.columns)
    expected = set(OUTPUT_COLUMNS)
    missing = expected - cols
    extra = cols - expected
    assert not missing, f"missing columns: {sorted(missing)}"
    assert not extra, f"extra columns: {sorted(extra)}"

    for col in ALWAYS_FILLED:
        assert df[col].notna().all(), f"null {col} present"

    assert df["side"].isin(["Blue", "Red"]).all(), \
        f"side outside {{Blue, Red}}: {sorted(set(df['side']) - {'Blue', 'Red'})}"
    assert df["tier"].isin(["S", "A", "B"]).all(), \
        f"tier outside {{S, A, B}}: {sorted(set(df['tier']) - {'S', 'A', 'B'})}"
    assert df["result"].isin([0, 1]).all(), \
        "result outside {0, 1}"
    for col in FIRST_FLAGS:
        assert df[col].isin([0, 1]).all(), f"{col} outside {{0, 1}}"

    for T in CHECKPOINTS:
        for stat in ("kill", "assist"):
            expected_diff = df[f"{stat}sat{T}"] - df[f"opp_{stat}sat{T}"]
            actual_diff = df[f"{stat}diffat{T}"]
            both_nan = expected_diff.isna() & actual_diff.isna()
            match = (expected_diff == actual_diff) | both_nan
            assert match.all(), \
                f"{stat}diffat{T} does not equal {stat}sat{T} − opp_{stat}sat{T}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=INPUT_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--tier-map", default=TIER_MAP_PATH)
    args = parser.parse_args()

    print("Loading stage-2 output...")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"  loaded {len(df):,} rows, {len(df.columns)} columns")

    tier_map = load_tier_map(args.tier_map)

    print("Engineering features...")
    out = run(df, tier_map)
    print(f"  output: {len(out):,} rows × {len(out.columns)} columns")

    print("Validating...")
    validate(out)
    print("  ok")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

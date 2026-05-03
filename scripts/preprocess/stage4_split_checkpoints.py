"""
Stage 4: split the stage-3 feature frame into 4 per-checkpoint datasets.

For each T in {10, 15, 20, 25}:
  - drop games without a populated checkpoint frame (any of this T's 12 stats
    null); gamelength is not the filter — games ending at ~24:30-24:59 with a
    fuzzy-boundary 25-min snapshot are still valid training rows, and games
    ending at exactly T*60 with no frame are not,
  - keep only that checkpoint's stats + static features + gated firsts
    + identifiers + gamelength + label,
  - write data/processed/checkpoint_{T}.csv and validate it.

First-event gates (realigned 2026-05-03 against direct livestats timing
measurements — see docs/specs/livestats-firsts-timing.md and
data/exploration/32_livestats_firsts_timing.json):
  - firstblood, firstdragon at every checkpoint,
  - firsttower at 20/25 only (78% leakage at min 15),
  - firstbaron excluded entirely (81% leakage at min 25).
The per-checkpoint table and the calibration tradeoff measured against the
2026 holdout are documented in docs/specs/preprocessing-pipeline.md.

Input:  data/processed/03_features.csv
Output: data/processed/checkpoint_{10,15,20,25}.csv
"""
import argparse
import os

import pandas as pd


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "03_features.csv")
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "processed")

CHECKPOINTS = (10, 15, 20, 25)

IDENTIFIERS = ["gameid", "date", "year", "league", "patch", "teamname", "opp_teamname"]
STATIC_FEATURES = ["side", "tier"]
GAMELENGTH = ["gamelength"]
LABEL = ["result"]

FIRST_GATES: dict[int, list[str]] = {
    10: ["firstblood", "firstdragon"],
    15: ["firstblood", "firstdragon"],
    20: ["firstblood", "firstdragon", "firsttower"],
    25: ["firstblood", "firstdragon", "firsttower"],
}


def checkpoint_stat_columns(T: int) -> list[str]:
    return [
        f"goldat{T}", f"opp_goldat{T}", f"golddiffat{T}",
        f"csat{T}", f"opp_csat{T}", f"csdiffat{T}",
        f"killsat{T}", f"opp_killsat{T}", f"killdiffat{T}",
        f"assistsat{T}", f"opp_assistsat{T}", f"assistdiffat{T}",
    ]


def output_columns(T: int) -> list[str]:
    return (
        IDENTIFIERS
        + STATIC_FEATURES
        + checkpoint_stat_columns(T)
        + FIRST_GATES[T]
        + GAMELENGTH
        + LABEL
    )


def run(df: pd.DataFrame, T: int) -> pd.DataFrame:
    cols = output_columns(T)
    kept = df[df[checkpoint_stat_columns(T)].notna().all(axis=1)]
    return kept[cols].reset_index(drop=True)


def validate(df: pd.DataFrame, T: int) -> None:
    expected = output_columns(T)
    cols = set(df.columns)
    missing = set(expected) - cols
    extra = cols - set(expected)
    assert not missing, f"missing columns for T={T}: {sorted(missing)}"
    assert not extra, f"extra columns for T={T}: {sorted(extra)}"

    if len(df) == 0:
        return

    for col in expected:
        assert df[col].notna().all(), f"null {col} present for T={T}"

    assert df["side"].isin(["Blue", "Red"]).all(), \
        f"side outside {{Blue, Red}} for T={T}"
    assert df["tier"].isin(["S", "A", "B"]).all(), \
        f"tier outside {{S, A, B}} for T={T}"
    assert df["result"].isin([0, 1]).all(), f"result outside {{0, 1}} for T={T}"
    for flag in FIRST_GATES[T]:
        assert df[flag].isin([0, 1]).all(), f"{flag} outside {{0, 1}} for T={T}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=INPUT_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    print("Loading stage-3 output...")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"  loaded {len(df):,} rows, {len(df.columns)} columns")

    os.makedirs(args.output_dir, exist_ok=True)
    for T in CHECKPOINTS:
        print(f"Checkpoint T={T}...")
        out = run(df, T)
        print(f"  kept {len(out):,} / {len(df):,} rows ({len(out)/len(df):.1%})")
        validate(out, T)
        print(f"  validated ({len(out.columns)} columns)")

        path = os.path.join(args.output_dir, f"checkpoint_{T}.csv")
        out.to_csv(path, index=False)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()

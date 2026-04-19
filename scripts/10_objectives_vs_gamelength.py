"""
Exploration script: how much objective accumulation is "baked in" by each checkpoint?

Idea: for games that ENDED at minute T, all objectives were necessarily taken by T.
Bucketing end-of-game objective counts by gamelength gives a natural proxy for
"objectives_by_minute_T" — games that ended at 25 min tell us what a game looks
like up to 25 min.

We aggregate at the GAME level (not team-row level). Each game contributes one
data point per bucket:
  - dragons_per_game: total drakes taken across both teams
  - barons_per_game: total barons across both teams
  - first_X_rate: % of games in bucket where event X happened (either team got it)

Uses 2020-2025 training window. Baron spawn time was 20:00 in 2020-2024, moved
to 25:00 in 2025, and returns to 20:00 in 2026 — so the aggregate 2020-2025
firstbaron-by-minute-T rates have 2025 skewing them slightly down. 2026 is
excluded from this analysis (held-out test set).

Outputs: data/exploration/10_objectives_vs_gamelength.json
"""
import csv
import glob
import json
import os
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "lol-historical-data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "exploration")

TRAIN_YEARS = {"2020", "2021", "2022", "2023", "2024", "2025"}
OBJECTIVE_COLS = ["dragons", "barons", "elders"]
FIRST_COLS = ["firstblood", "firstdragon", "firsttower", "firstbaron"]


def to_float(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


# Per-game aggregates: gameid -> {gamelength_min, obj_totals, first_flags}
games = {}  # gameid -> dict

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
for filepath in files:
    year_str = os.path.basename(filepath).split("_")[0]
    if year_str not in TRAIN_YEARS:
        continue
    print(f"Reading {os.path.basename(filepath)}...")
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("datacompleteness") != "complete":
                continue
            if row.get("position") != "team":
                continue
            gl = to_float(row.get("gamelength"))
            if gl is None or gl <= 0:
                continue
            gameid = row.get("gameid", "")
            if not gameid:
                continue

            if gameid not in games:
                games[gameid] = {
                    "bucket": int(gl // 60),
                    "obj_totals": defaultdict(float),
                    "obj_has": defaultdict(int),
                    "first_flags": defaultdict(int),  # 1 if any team got it
                }
            g = games[gameid]

            for col in OBJECTIVE_COLS:
                v = to_float(row.get(col))
                if v is not None:
                    g["obj_totals"][col] += v
                    g["obj_has"][col] += 1
            for col in FIRST_COLS:
                v = to_float(row.get(col))
                if v == 1.0:
                    g["first_flags"][col] = 1

# Bucket aggregation
bucket_game_count = defaultdict(int)
bucket_obj_sum = defaultdict(lambda: defaultdict(float))
bucket_obj_games = defaultdict(lambda: defaultdict(int))  # games with both team rows for this col
bucket_first_count = defaultdict(lambda: defaultdict(int))

for gameid, g in games.items():
    b = g["bucket"]
    bucket_game_count[b] += 1
    for col in OBJECTIVE_COLS:
        if g["obj_has"][col] == 2:  # both team rows contributed
            bucket_obj_sum[b][col] += g["obj_totals"][col]
            bucket_obj_games[b][col] += 1
    for col in FIRST_COLS:
        if g["first_flags"][col] == 1:
            bucket_first_count[b][col] += 1

rows = []
for bucket in sorted(bucket_game_count.keys()):
    if bucket < 15 or bucket > 50:
        continue
    games_in_bucket = bucket_game_count[bucket]
    entry = {"gamelength_min": bucket, "games": games_in_bucket}
    for col in OBJECTIVE_COLS:
        n = bucket_obj_games[bucket][col]
        entry[f"{col}_per_game"] = round(bucket_obj_sum[bucket][col] / n, 3) if n else None
    for col in FIRST_COLS:
        entry[f"{col}_pct"] = round(bucket_first_count[bucket][col] / games_in_bucket * 100, 1) \
            if games_in_bucket else None
    rows.append(entry)

total_games = len(games)

# Overall per-game means across all buckets
overall_obj_sum = defaultdict(float)
overall_obj_games = defaultdict(int)
for b in bucket_obj_sum:
    for col in OBJECTIVE_COLS:
        overall_obj_sum[col] += bucket_obj_sum[b][col]
        overall_obj_games[col] += bucket_obj_games[b][col]
overall_means = {
    col: round(overall_obj_sum[col] / overall_obj_games[col], 3) if overall_obj_games[col] else None
    for col in OBJECTIVE_COLS
}
overall_first = {
    col: round(sum(1 for g in games.values() if g["first_flags"][col] == 1) / total_games * 100, 1)
    for col in FIRST_COLS
}

summary = {
    "years": sorted(TRAIN_YEARS),
    "games_total": total_games,
    "overall_means_per_game": overall_means,
    "overall_first_event_rate_pct": overall_first,
    "per_minute_buckets": rows,
    "interpretation": (
        "Each row represents games that ENDED at minute T. "
        "dragons_per_game / barons_per_game = mean total across both teams for games in that bucket — "
        "since the game ended at T, this is an upper-bound for 'objectives_by_minute_T'. "
        "first_X_pct = % of games in bucket where event X happened at all. "
        "Used for estimating leakage magnitude when using end-of-game columns at a checkpoint-T model."
    ),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
outpath = os.path.join(OUTPUT_DIR, "10_objectives_vs_gamelength.json")
with open(outpath, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"\n=== Objectives by gamelength (2020-2025, {total_games} games) ===")
print(f"Overall per-game means (end of game): {overall_means}")
print(f"Overall first-event rates: {overall_first}\n")
header = f"{'min':>4}{'games':>7}{'drag':>8}{'baron':>8}{'elder':>8}"
header += f"{'FB%':>7}{'FD%':>7}{'FT%':>7}{'FBa%':>7}"
print(header)
for e in rows:
    print(
        f"{e['gamelength_min']:>4}"
        f"{e['games']:>7}"
        f"{e['dragons_per_game']:>8.2f}"
        f"{e['barons_per_game']:>8.2f}"
        f"{e['elders_per_game']:>8.3f}"
        f"{e['firstblood_pct']:>6.1f}%"
        f"{e['firstdragon_pct']:>6.1f}%"
        f"{e['firsttower_pct']:>6.1f}%"
        f"{e['firstbaron_pct']:>6.1f}%"
    )

print(f"\nWritten to {outpath}")

"""
Exploration script: average dragons/barons/elders per game for 2026 so far.
Complete-only filter. Team rows only. Per-team averages and per-game (both teams) totals.

Outputs: data/exploration/09_objectives_2026.json
"""
import csv
import json
import os
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "lol-historical-data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "exploration")
FILEPATH = os.path.join(DATA_DIR, "2026_LoL_esports_match_data_from_OraclesElixir.csv")

OBJECTIVE_COLS = ["dragons", "barons", "elders", "atakhans", "heralds", "void_grubs"]


def to_float(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


# Accumulators
# Per-team totals (one entry per team row, i.e. 2 per game)
team_sums = defaultdict(float)
team_counts = defaultdict(int)          # rows with non-null, non-empty value
team_rows_total = 0

# Per-game totals (sum across both teams) — gameid -> {col -> total across teams}
per_game = defaultdict(lambda: defaultdict(float))
per_game_has = defaultdict(lambda: defaultdict(int))  # how many team rows contributed

# Track nonzero elder games specifically
elder_games_nonzero = 0

# Track per-league, per-column averages
per_league_team_sums = defaultdict(lambda: defaultdict(float))
per_league_team_counts = defaultdict(lambda: defaultdict(int))
per_league_games = defaultdict(set)

with open(FILEPATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("datacompleteness") != "complete":
            continue
        if row.get("position") != "team":
            continue

        gameid = row.get("gameid", "")
        league = row.get("league", "")
        team_rows_total += 1
        per_league_games[league].add(gameid)

        for col in OBJECTIVE_COLS:
            val = to_float(row.get(col))
            if val is not None:
                team_sums[col] += val
                team_counts[col] += 1
                per_game[gameid][col] += val
                per_game_has[gameid][col] += 1
                per_league_team_sums[league][col] += val
                per_league_team_counts[league][col] += 1

num_games = len(per_game)

# Per-team averages (only over rows where the column was populated)
team_avgs = {
    col: round(team_sums[col] / team_counts[col], 3) if team_counts[col] else None
    for col in OBJECTIVE_COLS
}

# Per-game totals: only average across games where BOTH team rows had the value
per_game_totals = defaultdict(list)
for gameid, cols in per_game.items():
    for col in OBJECTIVE_COLS:
        if per_game_has[gameid][col] == 2:
            per_game_totals[col].append(cols[col])

per_game_avgs = {
    col: {
        "games": len(per_game_totals[col]),
        "avg_total_per_game": round(sum(per_game_totals[col]) / len(per_game_totals[col]), 3)
            if per_game_totals[col] else None,
        "pct_games_nonzero": round(
            sum(1 for v in per_game_totals[col] if v > 0) / len(per_game_totals[col]) * 100, 2
        ) if per_game_totals[col] else None,
    }
    for col in OBJECTIVE_COLS
}

# Per-league breakdown (top leagues by game count)
league_breakdown = []
for league, games in sorted(per_league_games.items(), key=lambda x: -len(x[1])):
    entry = {"league": league, "games": len(games)}
    for col in OBJECTIVE_COLS:
        cnt = per_league_team_counts[league][col]
        s = per_league_team_sums[league][col]
        entry[f"{col}_per_team"] = round(s / cnt, 3) if cnt else None
    league_breakdown.append(entry)

summary = {
    "year": 2026,
    "games_total": num_games,
    "team_rows_total": team_rows_total,
    "per_team_averages": team_avgs,
    "per_game_averages": per_game_avgs,
    "top_leagues": league_breakdown[:15],
    "notes": (
        "per_team_averages: mean objective count from the perspective of one team (one row). "
        "per_game_averages.avg_total_per_game: mean sum across both teams per game (dragons contested). "
        "null/empty values excluded from each column's mean."
    ),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
outpath = os.path.join(OUTPUT_DIR, "09_objectives_2026.json")
with open(outpath, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"\n=== 2026 objectives summary ({num_games} complete games) ===\n")
print(f"{'column':<14}{'per-team avg':>15}{'per-game avg':>15}{'games w/ val':>15}{'% games w/ >0':>16}")
for col in OBJECTIVE_COLS:
    pt = team_avgs[col]
    pg = per_game_avgs[col]["avg_total_per_game"]
    n = per_game_avgs[col]["games"]
    pct = per_game_avgs[col]["pct_games_nonzero"]
    print(f"{col:<14}"
          f"{pt if pt is not None else 'N/A':>15}"
          f"{pg if pg is not None else 'N/A':>15}"
          f"{n:>15}"
          f"{pct if pct is not None else 'N/A':>15}%")

print(f"\nTop leagues by game count (per-team avgs):")
for entry in league_breakdown[:10]:
    print(f"  {entry['league']:<20} games={entry['games']:<5} "
          f"drag={entry['dragons_per_team']} baron={entry['barons_per_team']} "
          f"elder={entry['elders_per_team']} atakhan={entry['atakhans_per_team']}")

print(f"\nWritten to {outpath}")

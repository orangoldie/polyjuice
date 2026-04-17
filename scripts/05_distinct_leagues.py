"""
Exploration script: distinct league names across all years.
Outputs results to data/exploration/05_distinct_leagues.json
"""
import csv
import glob
import json
import os
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "lol-historical-data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "exploration")

# league -> set of years it appears in
league_years = defaultdict(set)
# league -> game count
league_games = defaultdict(int)

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

for filepath in files:
    filename = os.path.basename(filepath)
    print(f"Processing {filename}...")

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("datacompleteness") != "complete":
                continue
            if row.get("position") != "team":
                continue
            if row.get("side") != "Blue":
                continue  # count each game once

            league = row.get("league", "unknown")
            year = row.get("year", "unknown")

            league_years[league].add(year)
            league_games[league] += 1

# Build output sorted by game count descending
leagues = []
for league in sorted(league_games, key=lambda l: -league_games[l]):
    leagues.append({
        "league": league,
        "games": league_games[league],
        "years": sorted(league_years[league]),
        "year_span": f"{min(league_years[league])}-{max(league_years[league])}",
    })

results = {
    "total_leagues": len(leagues),
    "leagues": leagues,
}

output_path = os.path.join(OUTPUT_DIR, "05_distinct_leagues.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# Print summary
print(f"\n=== {len(leagues)} distinct leagues ===\n")
print(f"{'League':<30} {'Games':>6}  {'Years Active'}")
print("-" * 65)
for entry in leagues:
    print(f"{entry['league']:<30} {entry['games']:>6}  {entry['year_span']} ({len(entry['years'])} yrs)")

print(f"\nFull results written to {output_path}")

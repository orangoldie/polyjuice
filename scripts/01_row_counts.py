"""
Exploration script: row counts per year and per league/tier.
Outputs results to data/exploration/01_row_counts.json
"""
import csv
import glob
import json
import os
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "lol-historical-data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "exploration")

results = {
    "per_year": {},
    "per_league": {},
    "per_year_league": {},
}

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

for filepath in files:
    filename = os.path.basename(filepath)
    print(f"Processing {filename}...")

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("datacompleteness") != "complete":
                continue

            year = row.get("year", "unknown")
            league = row.get("league", "unknown")
            position = row.get("position", "unknown")

            if year not in results["per_year"]:
                results["per_year"][year] = {"total_rows": 0, "team_rows": 0, "player_rows": 0, "games": 0}
            results["per_year"][year]["total_rows"] += 1

            if position == "team":
                results["per_year"][year]["team_rows"] += 1
            else:
                results["per_year"][year]["player_rows"] += 1

            if league not in results["per_league"]:
                results["per_league"][league] = {"total_rows": 0, "team_rows": 0}
            results["per_league"][league]["total_rows"] += 1
            if position == "team":
                results["per_league"][league]["team_rows"] += 1

            key = f"{year}_{league}"
            if key not in results["per_year_league"]:
                results["per_year_league"][key] = {"total_rows": 0, "team_rows": 0}
            results["per_year_league"][key]["total_rows"] += 1
            if position == "team":
                results["per_year_league"][key]["team_rows"] += 1

# Compute game counts (each game = 2 team rows)
for year_data in results["per_year"].values():
    year_data["games"] = year_data["team_rows"] // 2

for league_data in results["per_league"].values():
    league_data["games"] = league_data["team_rows"] // 2

for yl_data in results["per_year_league"].values():
    yl_data["games"] = yl_data["team_rows"] // 2

# Sort keys
results["per_year"] = dict(sorted(results["per_year"].items()))
results["per_league"] = dict(sorted(results["per_league"].items(), key=lambda x: -x[1]["games"]))
results["per_year_league"] = dict(sorted(results["per_year_league"].items()))

output_path = os.path.join(OUTPUT_DIR, "01_row_counts.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# Print summary
print("\n=== Games per year ===")
for year, data in results["per_year"].items():
    print(f"  {year}: {data['games']} games ({data['total_rows']} rows)")

print(f"\n=== Top 20 leagues by game count ===")
for i, (league, data) in enumerate(results["per_league"].items()):
    if i >= 20:
        break
    print(f"  {league}: {data['games']} games")

print(f"\nTotal leagues: {len(results['per_league'])}")
print(f"Full results written to {output_path}")

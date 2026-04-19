"""
Exploration script: Blue/Red side win rates by year and by league.
Outputs results to data/exploration/03_side_winrate.json
"""
import csv
import glob
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "lol-historical-data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "exploration")

per_year = {}
per_league = {}
per_year_league = {}

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

            year = row.get("year", "unknown")
            league = row.get("league", "unknown")
            side = row.get("side", "unknown")
            result = row.get("result", "")

            if result not in ("0", "1") or side not in ("Blue", "Red"):
                continue

            won = int(result)

            # Per year
            if year not in per_year:
                per_year[year] = {"blue_wins": 0, "red_wins": 0, "total": 0}
            per_year[year]["total"] += 1
            if side == "Blue":
                per_year[year]["blue_wins"] += won
            else:
                per_year[year]["red_wins"] += won

            # Per league
            if league not in per_league:
                per_league[league] = {"blue_wins": 0, "red_wins": 0, "total": 0}
            per_league[league]["total"] += 1
            if side == "Blue":
                per_league[league]["blue_wins"] += won
            else:
                per_league[league]["red_wins"] += won

            # Per year+league
            key = f"{year}_{league}"
            if key not in per_year_league:
                per_year_league[key] = {"blue_wins": 0, "red_wins": 0, "total": 0}
            per_year_league[key]["total"] += 1
            if side == "Blue":
                per_year_league[key]["blue_wins"] += won
            else:
                per_year_league[key]["red_wins"] += won


def add_rates(d):
    for data in d.values():
        games = data["total"] // 2  # 2 team rows per game
        data["games"] = games
        if games > 0:
            data["blue_wr"] = round(data["blue_wins"] / games * 100, 2)
            data["red_wr"] = round(data["red_wins"] / games * 100, 2)


add_rates(per_year)
add_rates(per_league)
add_rates(per_year_league)

results = {
    "per_year": dict(sorted(per_year.items())),
    "per_league": dict(sorted(per_league.items(), key=lambda x: -x[1]["games"])),
    "per_year_league": dict(sorted(per_year_league.items())),
}

output_path = os.path.join(OUTPUT_DIR, "03_side_winrate.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# Print summary
print("\n=== Blue side win rate by year ===")
for year, data in results["per_year"].items():
    print(f"  {year}: Blue {data['blue_wr']}% / Red {data['red_wr']}%  ({data['games']} games)")

print(f"\n=== Blue side win rate by league (top 20) ===")
for i, (league, data) in enumerate(results["per_league"].items()):
    if i >= 20:
        break
    print(f"  {league}: Blue {data['blue_wr']}% / Red {data['red_wr']}%  ({data['games']} games)")

print(f"\nFull results written to {output_path}")

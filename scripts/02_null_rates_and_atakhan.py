"""
Exploration script: null/empty rates for key columns across years + atakhan presence check.
Outputs results to data/exploration/02_null_rates.json and 02_atakhan.json
"""
import csv
import glob
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "lol-historical-data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "exploration")

# Key columns to check — mix of time-dependent and game-level
KEY_COLUMNS = [
    "datacompleteness", "league", "patch", "side", "position", "result", "gamelength",
    # Objectives
    "dragons", "barons", "heralds", "void_grubs", "atakhans", "opp_atakhans",
    "towers", "inhibitors", "elders",
    # Time-stamped (10 min)
    "goldat10", "xpat10", "csat10", "golddiffat10", "killsat10",
    # Time-stamped (15 min)
    "goldat15", "xpat15", "csat15", "golddiffat15", "killsat15",
    # Time-stamped (20 min)
    "goldat20", "xpat20", "csat20", "golddiffat20", "killsat20",
    # Time-stamped (25 min)
    "goldat25", "xpat25", "csat25", "golddiffat25", "killsat25",
]

# Track: per year, per column -> {total, empty}
null_rates = {}
# Track: atakhan non-empty values per year (team rows only)
atakhan_by_year = {}

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
            position = row.get("position", "")
            is_team_row = position == "team"

            if year not in null_rates:
                null_rates[year] = {}
            if year not in atakhan_by_year:
                atakhan_by_year[year] = {
                    "team_rows_total": 0,
                    "atakhans_present": 0,
                    "atakhans_nonzero": 0,
                    "opp_atakhans_present": 0,
                    "opp_atakhans_nonzero": 0,
                }

            for col in KEY_COLUMNS:
                if col not in null_rates[year]:
                    null_rates[year][col] = {"total": 0, "empty": 0, "team_total": 0, "team_empty": 0}
                val = row.get(col, "")
                null_rates[year][col]["total"] += 1
                if val == "" or val is None:
                    null_rates[year][col]["empty"] += 1
                if is_team_row:
                    null_rates[year][col]["team_total"] += 1
                    if val == "" or val is None:
                        null_rates[year][col]["team_empty"] += 1

            # Atakhan tracking (team rows only)
            if is_team_row:
                atakhan_by_year[year]["team_rows_total"] += 1
                atakhan_val = row.get("atakhans", "")
                opp_atakhan_val = row.get("opp_atakhans", "")
                if atakhan_val != "" and atakhan_val is not None:
                    atakhan_by_year[year]["atakhans_present"] += 1
                    if atakhan_val not in ("0", "0.0"):
                        atakhan_by_year[year]["atakhans_nonzero"] += 1
                if opp_atakhan_val != "" and opp_atakhan_val is not None:
                    atakhan_by_year[year]["opp_atakhans_present"] += 1
                    if opp_atakhan_val not in ("0", "0.0"):
                        atakhan_by_year[year]["opp_atakhans_nonzero"] += 1

# Compute null percentages (team rows only, since that's what we'll use)
null_summary = {}
for year in sorted(null_rates.keys()):
    null_summary[year] = {}
    for col in KEY_COLUMNS:
        d = null_rates[year][col]
        team_pct = (d["team_empty"] / d["team_total"] * 100) if d["team_total"] > 0 else None
        all_pct = (d["empty"] / d["total"] * 100) if d["total"] > 0 else None
        null_summary[year][col] = {
            "team_null_pct": round(team_pct, 2) if team_pct is not None else None,
            "all_null_pct": round(all_pct, 2) if all_pct is not None else None,
        }

atakhan_by_year = dict(sorted(atakhan_by_year.items()))

# Save
null_path = os.path.join(OUTPUT_DIR, "02_null_rates.json")
with open(null_path, "w", encoding="utf-8") as f:
    json.dump(null_summary, f, indent=2)

atakhan_path = os.path.join(OUTPUT_DIR, "02_atakhan.json")
with open(atakhan_path, "w", encoding="utf-8") as f:
    json.dump(atakhan_by_year, f, indent=2)

# Print highlights
print("\n=== Null rates for team rows (selected columns, % empty) ===")
print(f"{'Column':<20}", end="")
for year in sorted(null_summary.keys()):
    print(f"{year:>8}", end="")
print()
for col in KEY_COLUMNS:
    print(f"{col:<20}", end="")
    for year in sorted(null_summary.keys()):
        val = null_summary[year][col]["team_null_pct"]
        print(f"{val:>7.1f}%" if val is not None else f"{'N/A':>8}", end="")
    print()

print("\n=== Atakhan presence by year (team rows) ===")
for year, data in atakhan_by_year.items():
    total = data["team_rows_total"]
    present = data["atakhans_present"]
    nonzero = data["atakhans_nonzero"]
    print(f"  {year}: {present}/{total} rows have atakhan data ({nonzero} nonzero)")

print(f"\nResults written to {null_path} and {atakhan_path}")

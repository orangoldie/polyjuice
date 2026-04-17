"""
Exploration script: compute per-year metrics to help decide the minimum useful
training-year cutoff. Surfaces meta-shift signals (drake pool, objectives) and
tempo/pace trends alongside known Riot patch events.

Outputs data/exploration/08_training_year_cutoff.json
"""
import csv
import glob
import json
import os
import statistics
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "lol-historical-data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "exploration")

# Known meta events — annotated alongside the empirical numbers so a reader
# can see which cross-year deltas are explained by a Riot change.
META_EVENTS = {
    "2016": "Rift Herald added",
    "2019": "Elemental drake rework (soul, map transforms)",
    "2022": "Chemtech + Hextech drakes added (chemtech disabled part of year)",
    "2024": "Void Grubs replace first Rift Herald",
    "2025": "Atakhan + Feats of Strength introduced; Grubs consolidated",
    "2026": "Atakhan + Feats removed; Baron at 12:00; only 2 elemental drakes pre-soul",
}

# Per-year accumulators over team rows.
# Each year collects lists that we later reduce to mean/median.
year_stats = defaultdict(lambda: {
    "team_rows": 0,
    "games": set(),
    "gamelength_sec": [],
    "teamkills": [],
    "dragons": [],
    "barons": [],
    "heralds": [],
    "void_grubs": [],
    "atakhans": [],
    "elders": [],
    "infernals": [],
    "mountains": [],
    "clouds": [],
    "oceans": [],
    "chemtechs": [],
    "hextechs": [],
    "dragons_type_unknown": [],
    "killsat10": [],
    "killsat15": [],
    "goldat10": [],
    "goldat15": [],
    # Null-presence counters for era-gated columns
    "null_counts": defaultdict(int),
    "nonnull_counts": defaultdict(int),
})

ERA_GATED = [
    "heralds", "void_grubs", "atakhans",
    "chemtechs", "hextechs",
    "infernals", "mountains", "clouds", "oceans",
    "dragons (type unknown)",
]

def to_int(s):
    if s is None or s == "":
        return None
    try:
        return int(s)
    except ValueError:
        return None

def to_float(s):
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

for filepath in files:
    filename = os.path.basename(filepath)
    print(f"Loading {filename}...")
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("datacompleteness") != "complete":
                continue
            if row.get("position") != "team":
                continue
            year = row.get("year", "")
            if not year:
                continue
            s = year_stats[year]
            s["team_rows"] += 1
            s["games"].add(row.get("gameid", ""))

            # Null-presence tracking for era-gated columns
            for col in ERA_GATED:
                key = "dragons_type_unknown" if col == "dragons (type unknown)" else col
                raw = row.get(col, "")
                if raw == "" or raw is None:
                    s["null_counts"][key] += 1
                else:
                    s["nonnull_counts"][key] += 1

            # Numeric accumulators
            for col in ["gamelength", "teamkills", "dragons", "barons", "heralds",
                        "void_grubs", "atakhans", "elders", "infernals", "mountains",
                        "clouds", "oceans", "chemtechs", "hextechs",
                        "killsat10", "killsat15", "goldat10", "goldat15"]:
                v = to_int(row.get(col, ""))
                if v is None:
                    continue
                key = "gamelength_sec" if col == "gamelength" else col
                s[key].append(v)

            v = to_int(row.get("dragons (type unknown)", ""))
            if v is not None:
                s["dragons_type_unknown"].append(v)


def summarize(values):
    if not values:
        return None
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
    }


results = {"meta_events": META_EVENTS, "years": {}}

for year in sorted(year_stats.keys()):
    s = year_stats[year]
    n_team_rows = s["team_rows"]
    n_games = len(s["games"])

    # gamelength: convert seconds to minutes for readability
    gl_minutes = [v / 60.0 for v in s["gamelength_sec"]]

    year_out = {
        "games": n_games,
        "team_rows": n_team_rows,
        "gamelength_min": summarize(gl_minutes),
        "teamkills_per_team": summarize(s["teamkills"]),
        "dragons_per_team": summarize(s["dragons"]),
        "barons_per_team": summarize(s["barons"]),
        "heralds_per_team": summarize(s["heralds"]),
        "void_grubs_per_team": summarize(s["void_grubs"]),
        "atakhans_per_team": summarize(s["atakhans"]),
        "elders_per_team": summarize(s["elders"]),
        "drake_pool_mean": {
            "infernals": summarize(s["infernals"]),
            "mountains": summarize(s["mountains"]),
            "clouds": summarize(s["clouds"]),
            "oceans": summarize(s["oceans"]),
            "chemtechs": summarize(s["chemtechs"]),
            "hextechs": summarize(s["hextechs"]),
            "type_unknown": summarize(s["dragons_type_unknown"]),
        },
        "early_tempo": {
            "killsat10": summarize(s["killsat10"]),
            "killsat15": summarize(s["killsat15"]),
            "goldat10": summarize(s["goldat10"]),
            "goldat15": summarize(s["goldat15"]),
        },
        "column_availability_pct": {},
    }

    # Compute non-null % for era-gated columns (over team rows this year)
    for col in ERA_GATED:
        key = "dragons_type_unknown" if col == "dragons (type unknown)" else col
        nn = s["nonnull_counts"][key]
        total = nn + s["null_counts"][key]
        pct = round(100.0 * nn / total, 1) if total else 0.0
        year_out["column_availability_pct"][key] = pct

    if year in META_EVENTS:
        year_out["meta_event"] = META_EVENTS[year]

    results["years"][year] = year_out


output_path = os.path.join(OUTPUT_DIR, "08_training_year_cutoff.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# Compact console summary
print(f"\n=== Per-year summary (team-level) ===")
print(f"{'Year':<6}{'Games':>7}{'GLmin(med)':>12}{'Kills(mn)':>11}{'Drk(mn)':>9}{'Bar(mn)':>9}{'G@15(mn)':>11}  Event")
for year in sorted(results["years"].keys()):
    y = results["years"][year]
    gl = y["gamelength_min"]["median"] if y["gamelength_min"] else "-"
    tk = y["teamkills_per_team"]["mean"] if y["teamkills_per_team"] else "-"
    dr = y["dragons_per_team"]["mean"] if y["dragons_per_team"] else "-"
    ba = y["barons_per_team"]["mean"] if y["barons_per_team"] else "-"
    g15 = y["early_tempo"]["goldat15"]["mean"] if y["early_tempo"]["goldat15"] else "-"
    ev = y.get("meta_event", "")
    print(f"{year:<6}{y['games']:>7}{str(gl):>12}{str(tk):>11}{str(dr):>9}{str(ba):>9}{str(g15):>11}  {ev}")

print(f"\nFull results written to {output_path}")

"""
Exploration script: validate data integrity across the full dataset.
Outputs results to data/exploration/04_data_integrity.json
"""
import csv
import glob
import json
import os
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "lol-historical-data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "exploration")

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

# Collect all rows per game
games = defaultdict(list)

for filepath in files:
    filename = os.path.basename(filepath)
    print(f"Loading {filename}...")

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("datacompleteness") != "complete":
                continue
            gameid = row.get("gameid", "")
            games[gameid].append(row)

print(f"\nLoaded {len(games)} complete games. Running validations...\n")

issues = {
    "not_12_rows": [],
    "not_2_team_rows": [],
    "not_10_player_rows": [],
    "missing_side": [],
    "not_exactly_1_winner": [],
    "invalid_result_values": [],
    "opp_gold_mismatch_at10": [],
    "opp_gold_mismatch_at15": [],
    "golddiff_mismatch_at10": [],
    "golddiff_mismatch_at15": [],
    "team_kills_mismatch": [],
    "team_deaths_mismatch": [],
    "team_assists_mismatch": [],
    "time_data_before_gamelength": [],
}

MAX_EXAMPLES = 5  # store up to 5 example game IDs per issue

for gameid, rows in games.items():
    team_rows = [r for r in rows if r.get("position") == "team"]
    player_rows = [r for r in rows if r.get("position") != "team"]

    # 1. Row count checks
    if len(rows) != 12 and len(issues["not_12_rows"]) < MAX_EXAMPLES:
        issues["not_12_rows"].append({"gameid": gameid, "row_count": len(rows)})

    if len(team_rows) != 2 and len(issues["not_2_team_rows"]) < MAX_EXAMPLES:
        issues["not_2_team_rows"].append({"gameid": gameid, "team_row_count": len(team_rows)})

    if len(player_rows) != 10 and len(issues["not_10_player_rows"]) < MAX_EXAMPLES:
        issues["not_10_player_rows"].append({"gameid": gameid, "player_row_count": len(player_rows)})

    if len(team_rows) != 2:
        continue  # can't validate further without exactly 2 team rows

    # Identify sides
    blue_team = None
    red_team = None
    for tr in team_rows:
        if tr.get("side") == "Blue":
            blue_team = tr
        elif tr.get("side") == "Red":
            red_team = tr

    # 2. Both sides present
    if blue_team is None or red_team is None:
        if len(issues["missing_side"]) < MAX_EXAMPLES:
            issues["missing_side"].append({"gameid": gameid})
        continue

    # 3. Exactly 1 winner
    blue_result = blue_team.get("result", "")
    red_result = red_team.get("result", "")

    if blue_result not in ("0", "1") or red_result not in ("0", "1"):
        if len(issues["invalid_result_values"]) < MAX_EXAMPLES:
            issues["invalid_result_values"].append({"gameid": gameid, "blue": blue_result, "red": red_result})
    elif int(blue_result) + int(red_result) != 1:
        if len(issues["not_exactly_1_winner"]) < MAX_EXAMPLES:
            issues["not_exactly_1_winner"].append({"gameid": gameid, "blue": blue_result, "red": red_result})

    # 4. opp_* mirror check (gold at 10 and 15)
    for suffix, issue_key in [("10", "opp_gold_mismatch_at10"), ("15", "opp_gold_mismatch_at15")]:
        b_gold = blue_team.get(f"goldat{suffix}", "")
        r_gold = red_team.get(f"goldat{suffix}", "")
        b_opp = blue_team.get(f"opp_goldat{suffix}", "")
        r_opp = red_team.get(f"opp_goldat{suffix}", "")
        if b_gold and r_gold and b_opp and r_opp:
            if b_gold != r_opp or r_gold != b_opp:
                if len(issues[issue_key]) < MAX_EXAMPLES:
                    issues[issue_key].append({
                        "gameid": gameid,
                        "blue_gold": b_gold, "red_opp_gold": r_opp,
                        "red_gold": r_gold, "blue_opp_gold": b_opp,
                    })

    # 5. golddiff consistency
    for suffix, issue_key in [("10", "golddiff_mismatch_at10"), ("15", "golddiff_mismatch_at15")]:
        for team_row, label in [(blue_team, "blue"), (red_team, "red")]:
            gold = team_row.get(f"goldat{suffix}", "")
            opp_gold = team_row.get(f"opp_goldat{suffix}", "")
            diff = team_row.get(f"golddiffat{suffix}", "")
            if gold and opp_gold and diff:
                try:
                    expected = int(gold) - int(opp_gold)
                    actual = int(diff)
                    if expected != actual and len(issues[issue_key]) < MAX_EXAMPLES:
                        issues[issue_key].append({
                            "gameid": gameid, "side": label,
                            "gold": gold, "opp_gold": opp_gold,
                            "expected_diff": expected, "actual_diff": actual,
                        })
                except ValueError:
                    pass

    # 6. Team kills/deaths/assists == sum of player kills/deaths/assists
    for side_name, team_row in [("Blue", blue_team), ("Red", red_team)]:
        side_players = [r for r in player_rows if r.get("side") == side_name]
        for stat, issue_key in [("kills", "team_kills_mismatch"), ("deaths", "team_deaths_mismatch"), ("assists", "team_assists_mismatch")]:
            team_val = team_row.get(stat, "")
            if not team_val or not side_players:
                continue
            try:
                player_sum = sum(int(p.get(stat, "0")) for p in side_players if p.get(stat, ""))
                team_int = int(team_val)
                if player_sum != team_int and len(issues[issue_key]) < MAX_EXAMPLES:
                    issues[issue_key].append({
                        "gameid": gameid, "side": side_name,
                        "team_value": team_int, "player_sum": player_sum,
                    })
            except ValueError:
                pass

    # 7. Time data vs gamelength
    gamelength_str = blue_team.get("gamelength", "")
    if gamelength_str:
        try:
            gl = int(gamelength_str)
            checkpoints = [(600, "10"), (900, "15"), (1200, "20"), (1500, "25")]
            for min_seconds, suffix in checkpoints:
                if gl < min_seconds:
                    val = blue_team.get(f"goldat{suffix}", "")
                    if val and val != "" and len(issues["time_data_before_gamelength"]) < MAX_EXAMPLES:
                        issues["time_data_before_gamelength"].append({
                            "gameid": gameid,
                            "gamelength_sec": gl,
                            "has_data_at": f"{suffix}min",
                        })
        except ValueError:
            pass

# Count totals for each issue (not just examples)
issue_counts = {}
# Re-scan for counts
count_trackers = {k: 0 for k in issues}

for gameid, rows in games.items():
    team_rows = [r for r in rows if r.get("position") == "team"]
    player_rows = [r for r in rows if r.get("position") != "team"]

    if len(rows) != 12:
        count_trackers["not_12_rows"] += 1
    if len(team_rows) != 2:
        count_trackers["not_2_team_rows"] += 1
    if len(player_rows) != 10:
        count_trackers["not_10_player_rows"] += 1

    if len(team_rows) != 2:
        continue

    blue_team = None
    red_team = None
    for tr in team_rows:
        if tr.get("side") == "Blue":
            blue_team = tr
        elif tr.get("side") == "Red":
            red_team = tr

    if blue_team is None or red_team is None:
        count_trackers["missing_side"] += 1
        continue

    blue_result = blue_team.get("result", "")
    red_result = red_team.get("result", "")
    if blue_result not in ("0", "1") or red_result not in ("0", "1"):
        count_trackers["invalid_result_values"] += 1
    elif int(blue_result) + int(red_result) != 1:
        count_trackers["not_exactly_1_winner"] += 1

    for suffix, issue_key in [("10", "opp_gold_mismatch_at10"), ("15", "opp_gold_mismatch_at15")]:
        b_gold = blue_team.get(f"goldat{suffix}", "")
        r_gold = red_team.get(f"goldat{suffix}", "")
        b_opp = blue_team.get(f"opp_goldat{suffix}", "")
        r_opp = red_team.get(f"opp_goldat{suffix}", "")
        if b_gold and r_gold and b_opp and r_opp:
            if b_gold != r_opp or r_gold != b_opp:
                count_trackers[issue_key] += 1

    for suffix, issue_key in [("10", "golddiff_mismatch_at10"), ("15", "golddiff_mismatch_at15")]:
        for team_row in [blue_team, red_team]:
            gold = team_row.get(f"goldat{suffix}", "")
            opp_gold = team_row.get(f"opp_goldat{suffix}", "")
            diff = team_row.get(f"golddiffat{suffix}", "")
            if gold and opp_gold and diff:
                try:
                    if int(gold) - int(opp_gold) != int(diff):
                        count_trackers[issue_key] += 1
                except ValueError:
                    pass

    for side_name, team_row in [("Blue", blue_team), ("Red", red_team)]:
        side_players = [r for r in player_rows if r.get("side") == side_name]
        for stat, issue_key in [("kills", "team_kills_mismatch"), ("deaths", "team_deaths_mismatch"), ("assists", "team_assists_mismatch")]:
            team_val = team_row.get(stat, "")
            if not team_val or not side_players:
                continue
            try:
                player_sum = sum(int(p.get(stat, "0")) for p in side_players if p.get(stat, ""))
                if player_sum != int(team_val):
                    count_trackers[issue_key] += 1
            except ValueError:
                pass

    gamelength_str = blue_team.get("gamelength", "")
    if gamelength_str:
        try:
            gl = int(gamelength_str)
            for min_seconds, suffix in [(600, "10"), (900, "15"), (1200, "20"), (1500, "25")]:
                if gl < min_seconds and blue_team.get(f"goldat{suffix}", ""):
                    count_trackers["time_data_before_gamelength"] += 1
                    break
        except ValueError:
            pass

results = {
    "total_games": len(games),
    "checks": {}
}

for key in issues:
    results["checks"][key] = {
        "count": count_trackers[key],
        "examples": issues[key],
    }

output_path = os.path.join(OUTPUT_DIR, "04_data_integrity.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# Print summary
print(f"Total complete games: {len(games)}\n")
print("=== Validation Results ===")
for key, data in results["checks"].items():
    status = "PASS" if data["count"] == 0 else f"FAIL ({data['count']} games)"
    print(f"  {key}: {status}")

print(f"\nFull results written to {output_path}")

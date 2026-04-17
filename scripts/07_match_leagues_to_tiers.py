"""
Match Oracle's Elixir league names to Liquipedia tier classifications.
Uses exact matches against Liquipedia tournament link bases from the API.
Unmatched leagues are left for manual review.
Reads: 05_distinct_leagues.json, 06_liquipedia_tiers.json
Outputs: 07_league_tier_matching.json
"""
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "exploration")

# Load data
with open(os.path.join(DATA_DIR, "05_distinct_leagues.json")) as f:
    our_leagues = json.load(f)["leagues"]

with open(os.path.join(DATA_DIR, "06_liquipedia_tiers.json")) as f:
    tiers = json.load(f)

# Build lookup: liquipedia name -> tier (S takes priority over A over B)
lp_to_tier = {}
for tier in ["B", "A", "S"]:
    for name in tiers[tier]:
        lp_to_tier[name] = tier

# Obvious aliases: abbreviations or renames where the mapping is unambiguous
ALIASES = {
    # S-tier: international events and league renames
    "MSI": "Mid-Season Invitational",
    "WLDs": "World Championship",
    "EWC": "Esports World Cup",
    "FST": "First Stand Tournament",
    "MSC": "Mid-Season Cup",
    "NA LCS": "LCS",       # renamed 2019
    "EU LCS": "LEC",       # renamed 2019
    "LTA N": "LTA",        # sub-region
    "LTA S": "LTA",        # sub-region
    "OGN": "Champions",    # old LCK name (pre-2016), confirmed
    # A-tier: development leagues and abbreviations
    "LCKC": "LCK CL",
    "NACL": "North American Challengers League",
    "LCSA": "North American Challengers League",  # LCS Academy, rebranded to NACL
    "EU CS": "Elite Challenger Series",
    "NA CS": "North American Challengers League",
    "EM": "EMEA Masters",
    "EUM": "European Masters",
    "KeSPA": "KeSPA Cup",
    "DCup": "Demacia Cup",
    "IWCI": "International Wildcard Invitational",
    "GPL": "Garena Premier League",
    "Asia Master": "Asia Masters",
    "LAS": "Liga Regional Sur",    # Latin America South, same lineage
    "LRN": "Liga Regional Norte",
    "LRS": "Liga Regional Sur",
    "CD": "Circuito Desafiante",
    "NEST": "National Electronic Sports Tournament",
    "ASI": "Asia Star Challengers Invitational",
    # B-tier: ERLs
    "PRM": "Prime League",
    "HM": "Hitpoint Masters",
    "AL": "Arabian League",
    "RL": "Rift Legends",
    "ROL": "Road of Legends",
}

# Direct tier assignments for leagues not on Liquipedia but with known tier
DIRECT_TIER = {
    # ERLs (EMEA Regional Leagues) — all B-tier
    "UL": "B",       # Ultraliga (Poland)
    "LVP SL": "B",   # LVP SuperLiga (Spain)
    "GLL": "B",       # Greek Legends League
    "PGN": "B",       # PG Nationals (Italy)
    "ESLOL": "B",     # Elite Series (Benelux)
    "SLO": "B",       # SuperLiga Orange (Spain, old name for LVP SL)
    "DL": "B",        # Dutch League (Benelux ERL)
    "BL": "B",        # Belgian League (Benelux ERL)
    "UKLC": "B",      # UK League Championship (merged into NLC)
}

# Match each league — exact name match, then alias, then direct tier
matched = []
unmatched = []

for entry in our_leagues:
    name = entry["league"]
    games = entry["games"]
    years = entry["year_span"]

    tier = lp_to_tier.get(name)
    matched_to = name if tier else None

    # Try alias if no exact match
    if not tier and name in ALIASES:
        lp_name = ALIASES[name]
        tier = lp_to_tier.get(lp_name)
        if tier:
            matched_to = lp_name

    # Try direct tier assignment
    if not tier and name in DIRECT_TIER:
        tier = DIRECT_TIER[name]
        matched_to = f"{name} (direct)"

    result = {
        "league": name,
        "games": games,
        "years": years,
        "tier": tier,
        "matched_to": matched_to,
    }

    if tier:
        matched.append(result)
    else:
        unmatched.append(result)

# Summary
total_games = sum(e["games"] for e in our_leagues)
matched_games = sum(e["games"] for e in matched)
unmatched_games = sum(e["games"] for e in unmatched)

summary = {
    "total_our_leagues": len(our_leagues),
    "matched_count": len(matched),
    "unmatched_count": len(unmatched),
    "total_games": total_games,
    "matched_games": matched_games,
    "unmatched_games": unmatched_games,
    "matched_pct": round(100 * matched_games / total_games, 1),
    "by_tier": {},
}

for tier in ["S", "A", "B"]:
    tier_entries = [e for e in matched if e["tier"] == tier]
    summary["by_tier"][tier] = {
        "leagues": len(tier_entries),
        "games": sum(e["games"] for e in tier_entries),
    }

results = {
    "summary": summary,
    "matched": sorted(matched, key=lambda e: (-{"S": 3, "A": 2, "B": 1}[e["tier"]], -e["games"])),
    "unmatched": sorted(unmatched, key=lambda e: -e["games"]),
}

output_path = os.path.join(DATA_DIR, "07_league_tier_matching.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

# Print
print("=== Matching Summary ===")
print(f"  Our leagues: {len(our_leagues)}")
print(f"  Matched:     {len(matched)} ({matched_games} games, {summary['matched_pct']}%)")
print(f"  Unmatched:   {len(unmatched)} ({unmatched_games} games)")
print()

for tier in ["S", "A", "B"]:
    t = summary["by_tier"][tier]
    print(f"  Tier {tier}: {t['leagues']} leagues, {t['games']} games")

print(f"\n=== Matched Leagues ===")
for e in results["matched"]:
    alias = f" (via {e['matched_to']})" if e["matched_to"] != e["league"] else ""
    print(f"  [{e['tier']}] {e['league']:<25} {e['games']:>5} games  {e['years']}{alias}")

print(f"\n=== Unmatched Leagues ({len(unmatched)}) ===")
for e in results["unmatched"]:
    print(f"  {e['league']:<25} {e['games']:>5} games  {e['years']}")

print(f"\nResults written to {output_path}")

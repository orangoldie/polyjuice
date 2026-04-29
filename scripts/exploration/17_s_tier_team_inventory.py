"""
Enumerate every distinct team in 2024+ OE data for the leagues GPR rates,
with game counts. Cross-reference against the GPR-side team list from
`data/gpr/snapshots.csv` to surface candidates that need an alias-table
entry.

Output: data/exploration/17_s_tier_team_inventory.json with three lists:
  - oe_teams:     [{teamname, league, games, years}] for OE teams in
                  2024+ S-tier (+ 2024 VCS) data, sorted by games desc.
  - gpr_teams:    [{team_name, home_league_slug, snapshots, team_id}] from
                  snapshots.csv.
  - exact_matches:        OE teamname == GPR team_name (case-insensitive).
  - oe_unmatched_geq_20:  OE teams with >=20 games and no exact GPR match.
                          These are the rows that need a hand-mapped alias
                          (or an explicit "uncovered" flag) per spec
                          Step 2's structural validator.

The exact-match list is a head start for `data/gpr/team_aliases.csv`, not
an authoritative table — name collisions across leagues, rebrands, and
regional variants still need human review.

Usage: python -m scripts.exploration.17_s_tier_team_inventory
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "lol-historical-data"
GPR_SNAPSHOTS = REPO_ROOT / "data" / "gpr" / "snapshots.csv"
OUT_PATH = REPO_ROOT / "data" / "exploration" / "17_s_tier_team_inventory.json"

# S-tier leagues per data/exploration/07_league_tier_matching.json. These are
# the OE-side league names — note "EU LCS"/"NA LCS" predate 2024 so don't
# appear in our window, but listed for completeness against the tier file.
S_TIER_LEAGUES = {
    "LCK", "LPL", "LEC", "LCS", "CBLOL", "LCP",
    "LTA N", "LTA S", "LTA",
    "WLDs", "MSI", "EWC", "FST", "MSC", "Americas Cup",
}
# GPR rated VCS in 2024 only (VCS-as-a-league merged into LCP for 2025+).
VCS_LEAGUE = "VCS"
VCS_RATED_YEARS = {"2024"}
START_YEAR = 2024


def _included(row: dict) -> bool:
    """One row per game, in our window, in a GPR-rated league."""
    if row.get("datacompleteness") != "complete":
        return False
    if row.get("position") != "team":
        return False
    if row.get("side") != "Blue":
        return False
    try:
        year = int(row.get("year") or 0)
    except ValueError:
        return False
    if year < START_YEAR:
        return False
    league = row.get("league", "")
    if league in S_TIER_LEAGUES:
        return True
    if league == VCS_LEAGUE and (row.get("year") or "") in VCS_RATED_YEARS:
        return True
    return False


def collect_oe_teams() -> list[dict]:
    """Return [{teamname, league, games, years}] across in-window OE rows."""
    by_team: dict[tuple[str, str], dict] = {}
    files = sorted(RAW_DIR.glob("*.csv"))
    for fp in files:
        print(f"  scanning {fp.name}...")
        with fp.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not _included(row):
                    continue
                team = row.get("teamname") or ""
                league = row.get("league") or ""
                year = row.get("year") or ""
                if not team:
                    continue
                key = (team, league)
                entry = by_team.setdefault(key, {
                    "teamname": team,
                    "league": league,
                    "games": 0,
                    "years": set(),
                })
                entry["games"] += 1
                entry["years"].add(year)

    out = []
    for entry in by_team.values():
        out.append({
            "teamname": entry["teamname"],
            "league": entry["league"],
            "games": entry["games"],
            "years": sorted(entry["years"]),
        })
    out.sort(key=lambda e: -e["games"])
    return out


def collect_gpr_teams() -> list[dict]:
    """Return [{team_name, home_league_slug, snapshots, team_id}] from snapshots.csv."""
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    with GPR_SNAPSHOTS.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["team_name"], row["home_league_slug"], row["team_id"])
            counts[key] += 1
    out = [
        {"team_name": tn, "home_league_slug": ls, "team_id": tid, "snapshots": n}
        for (tn, ls, tid), n in counts.items()
    ]
    out.sort(key=lambda e: -e["snapshots"])
    return out


def main() -> int:
    print("Collecting OE teams (2024+, S-tier + 2024 VCS)...")
    oe_teams = collect_oe_teams()
    print(f"  {len(oe_teams)} (teamname, league) rows")

    print("Collecting GPR teams from snapshots.csv...")
    gpr_teams = collect_gpr_teams()
    print(f"  {len(gpr_teams)} GPR teams")

    gpr_names_lower = {g["team_name"].lower(): g for g in gpr_teams}
    exact_matches = []
    oe_unmatched = []
    for oe in oe_teams:
        m = gpr_names_lower.get(oe["teamname"].lower())
        if m is not None:
            exact_matches.append({
                "oe_teamname": oe["teamname"],
                "oe_league": oe["league"],
                "oe_games": oe["games"],
                "gpr_team_name": m["team_name"],
                "gpr_home_league": m["home_league_slug"],
                "gpr_team_id": m["team_id"],
            })
        elif oe["games"] >= 20:
            oe_unmatched.append(oe)

    payload = {
        "summary": {
            "oe_team_league_rows": len(oe_teams),
            "gpr_teams": len(gpr_teams),
            "exact_matches": len(exact_matches),
            "oe_unmatched_geq_20_games": len(oe_unmatched),
        },
        "oe_teams": oe_teams,
        "gpr_teams": gpr_teams,
        "exact_matches": exact_matches,
        "oe_unmatched_geq_20": oe_unmatched,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")

    print("\n=== Summary ===")
    for k, v in payload["summary"].items():
        print(f"  {k:<35} {v}")

    print(f"\n=== OE teams >=20 games with no exact GPR match ({len(oe_unmatched)}) ===")
    for t in oe_unmatched:
        print(f"  {t['teamname']:<35} {t['league']:<10} games={t['games']:>4}  yrs={','.join(t['years'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Build the initial `data/gpr/team_aliases.csv` from the inventory and a
hand-curated mapping for the OE teams that don't exact-match a GPR team.

The mapping is encoded in `HAND_MAPPINGS` below — kept in code rather than
CSV so the rationale for each non-trivial decision is reviewable in one
place. Re-run anytime the OE data extends or the GPR scrape adds new
teams; existing exact matches stay; hand mappings stay; new cases surface
as 'unmapped' and need a HAND_MAPPINGS entry or an explicit UNCOVERED
flag.

Output schema (per spec): `gpr_team_name, oe_teamname, notes`.
  - Exact name matches (114): notes blank.
  - Hand-mapped: notes documents why (rebrand, sponsor change, etc.).
  - Uncovered (no GPR snapshot exists for this OE team): gpr_team_name
    is empty, notes starts with 'UNCOVERED:'.

Usage:
  python -m scripts.exploration.17_s_tier_team_inventory   # produce inventory
  python -m scripts.gpr.build_aliases                       # write team_aliases.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY = REPO_ROOT / "data" / "exploration" / "17_s_tier_team_inventory.json"
ALIASES_CSV = REPO_ROOT / "data" / "gpr" / "team_aliases.csv"


# Hand-curated mappings for OE teams with >=20 games and no exact GPR match.
# Each entry: oe_teamname -> (gpr_team_name | None, notes).
# A None gpr_team_name marks the team as uncovered (no GPR snapshot exists);
# stage 5 will treat its games as has_gpr=0.
HAND_MAPPINGS: dict[str, tuple[str | None, str]] = {
    # Case/spacing variants of LCK names.
    "Nongshim RedForce":   ("NONGSHIM RED FORCE", "case + spacing variant"),
    "Gen.G":               ("Gen.G Esports",       "GPR uses full org name"),

    # Sponsor / rebrand variants. The GPR team_id is the same across the
    # rebrand, so its history covers OE rows under both names.
    "DRX":                 ("KIWOOM DRX",          "sponsor: KIWOOM DRX is GPR's current name"),
    "Team Liquid":         ("Team Liquid Alienware", "sponsor: Alienware. Applies to LCS and LTA N rows"),
    "RED Canids":          ("RED Canids Kalunga",  "sponsor: Kalunga. Applies to CBLOL and LTA S rows"),
    "Cloud9":              ("Cloud9 Kia",          "sponsor: Kia. Applies to LCS and LTA N rows"),
    "NRG":                 ("NRG Kia",             "sponsor: Kia"),
    "KaBuM! Esports":      ("KaBuM!",              "OE uses long-form, GPR uses short-form"),
    "Chiefs Esports Club": ("The Chiefs Esports Club", "GPR has leading 'The'"),
    "Team Flash":          ("TP.HCM Team Flash",   "GPR uses regional prefix TP.HCM"),
    "Los Grandes":         ("LOS",                 "GPR uses 3-letter code"),
    "Leviatan":            ("LEVIATÁN",            "accent + casing"),
    "MGN Vikings Esports": ("MGN Blue Esports",    "MGN org rebrand Blue→Vikings; same GPR team_id covers VCS '24 and LCP '25+"),
    "MAD Lions KOI":       ("Movistar KOI",        "rebrand: MAD Lions KOI '24 → Movistar KOI '25+; same GPR team_id covers both"),
    # Genuinely uncovered: GPR did not publish snapshots for these teams in
    # the years they played. Stage 5 treats their games as has_gpr=0.
    "Team BDS":            (None, "UNCOVERED: BDS played LEC '24-'25 but no GPR record; cause unknown (relegation? scrape gap?). Re-investigate if scrape is refreshed or if BDS appears in any LEC GPR push later"),
    "Rainbow Warriors":    (None, "UNCOVERED: VCS '24 had only 4 GPR-rated teams; this one fell below GPR's qualification threshold"),
    "Team Secret":         (None, "UNCOVERED: VCS '24. Saigon-Secret-as-rebrand was considered but rejected on review — they are not the same team"),
}


def main() -> int:
    data = json.loads(INVENTORY.read_text(encoding="utf-8"))
    oe_teams = data["oe_teams"]
    gpr_teams = {g["team_name"].lower(): g for g in data["gpr_teams"]}

    rows: list[dict] = []
    unmapped: list[dict] = []

    seen_oe_names: set[str] = set()
    for oe in oe_teams:
        name = oe["teamname"]
        if name in seen_oe_names:
            continue  # same team across leagues — one alias row covers them
        seen_oe_names.add(name)

        m = gpr_teams.get(name.lower())
        if m is not None:
            rows.append({
                "gpr_team_name": m["team_name"],
                "oe_teamname": name,
                "notes": "",
            })
            continue

        if name in HAND_MAPPINGS:
            gpr_name, note = HAND_MAPPINGS[name]
            rows.append({
                "gpr_team_name": gpr_name or "",
                "oe_teamname": name,
                "notes": note,
            })
            continue

        if oe["games"] >= 20:
            unmapped.append(oe)

    if unmapped:
        print(f"!! {len(unmapped)} OE teams >=20 games are unmapped — add to HAND_MAPPINGS:")
        for t in unmapped:
            print(f"   {t['teamname']!r:<35} {t['league']:<8} games={t['games']}")
        return 1

    rows.sort(key=lambda r: (r["gpr_team_name"] == "", r["oe_teamname"].lower()))

    ALIASES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with ALIASES_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gpr_team_name", "oe_teamname", "notes"])
        w.writeheader()
        w.writerows(rows)

    n_mapped = sum(1 for r in rows if r["gpr_team_name"])
    n_uncovered = sum(1 for r in rows if not r["gpr_team_name"])
    print(f"Wrote {ALIASES_CSV}")
    print(f"  total rows: {len(rows)}")
    print(f"  mapped:     {n_mapped}")
    print(f"  uncovered:  {n_uncovered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

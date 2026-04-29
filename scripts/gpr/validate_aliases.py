"""
Structural validator for `data/gpr/team_aliases.csv`.

Per the GPR spec (Step 2): every OE team with >=20 games in 2024+ S-tier
(plus 2024 VCS) data must be either:
  (a) mapped to a real GPR team_name (one that appears in snapshots.csv), or
  (b) explicitly flagged as uncovered with a documented reason.

This script asserts that and surfaces violations. Run after refreshing the
scrape, regenerating the inventory, or editing the alias CSV by hand.

Usage:
  python -m scripts.gpr.validate_aliases
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY = REPO_ROOT / "data" / "exploration" / "17_s_tier_team_inventory.json"
ALIASES_CSV = REPO_ROOT / "data" / "gpr" / "team_aliases.csv"
GPR_SNAPSHOTS = REPO_ROOT / "data" / "gpr" / "snapshots.csv"

THRESHOLD = 20  # spec's structural-mapping threshold


def load_aliases() -> list[dict]:
    with ALIASES_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_gpr_team_names() -> set[str]:
    with GPR_SNAPSHOTS.open(encoding="utf-8") as f:
        return {r["team_name"] for r in csv.DictReader(f)}


def main() -> int:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    aliases = load_aliases()
    gpr_team_names = load_gpr_team_names()

    alias_by_oe = {a["oe_teamname"]: a for a in aliases}

    errors: list[str] = []

    # Duplicate OE name in alias table
    seen: set[str] = set()
    for a in aliases:
        if a["oe_teamname"] in seen:
            errors.append(f"duplicate oe_teamname: {a['oe_teamname']!r}")
        seen.add(a["oe_teamname"])

    # Every OE team with >= THRESHOLD games is mapped or flagged
    qualifying_oe = [t for t in inventory["oe_teams"] if t["games"] >= THRESHOLD]
    qualifying_names = {t["teamname"] for t in qualifying_oe}
    missing = qualifying_names - alias_by_oe.keys()
    for name in sorted(missing):
        errors.append(f"OE team {name!r} has >={THRESHOLD} games but no alias entry")

    # Every gpr_team_name in the alias CSV exists in snapshots.csv
    # (catches typos in HAND_MAPPINGS).
    for a in aliases:
        gpr_name = a["gpr_team_name"]
        if not gpr_name:
            if not a["notes"].startswith("UNCOVERED:"):
                errors.append(
                    f"alias for {a['oe_teamname']!r} has empty gpr_team_name "
                    f"but notes does not start with 'UNCOVERED:': {a['notes']!r}"
                )
            continue
        if gpr_name not in gpr_team_names:
            errors.append(
                f"alias gpr_team_name {gpr_name!r} (for OE {a['oe_teamname']!r}) "
                f"is not present in snapshots.csv"
            )

    # Summary
    n_mapped = sum(1 for a in aliases if a["gpr_team_name"])
    n_uncovered = sum(1 for a in aliases if not a["gpr_team_name"])
    print(f"Aliases:    {len(aliases)} total — {n_mapped} mapped, {n_uncovered} uncovered")
    print(f"OE teams >={THRESHOLD} games: {len(qualifying_oe)}")
    print(f"GPR snapshots cover {len(gpr_team_names)} unique team_name values")

    # Coverage diagnostics: how many in-window games end up has_gpr=0 due
    # to uncovered teams? Useful to know vs the spec's <5% acceptance bar.
    uncovered_oe = {a["oe_teamname"] for a in aliases if not a["gpr_team_name"]}
    uncovered_games = sum(
        t["games"] for t in inventory["oe_teams"] if t["teamname"] in uncovered_oe
    )
    total_in_window_games = sum(t["games"] for t in inventory["oe_teams"])
    pct = 100.0 * uncovered_games / total_in_window_games if total_in_window_games else 0.0
    print(
        f"Uncovered games: {uncovered_games} / {total_in_window_games} "
        f"({pct:.2f}% of in-window 2024+ S-tier+VCS24 OE rows)"
    )

    if errors:
        print(f"\n!! {len(errors)} validation error(s):", file=sys.stderr)
        for e in errors:
            print(f"   {e}", file=sys.stderr)
        return 1

    print("\nOK — alias table is structurally valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

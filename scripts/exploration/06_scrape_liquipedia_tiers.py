"""
Scrape Liquipedia S/A/B/C/D tier tournament pages to extract league names per tier.
Uses the MediaWiki API to get parsed HTML content.
Outputs results to data/exploration/06_liquipedia_tiers.json
"""
import json
import os
import re
import time

import requests
from bs4 import BeautifulSoup

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "exploration")

API_URL = "https://liquipedia.net/leagueoflegends/api.php"
HEADERS = {
    "User-Agent": "PolyjuiceBot/1.0 (esports research project; contact: oran.goldie@gmail.com)",
    "Accept-Encoding": "gzip",
}

TIER_PAGES = {
    "S": "S-Tier_Tournaments",
    "A": "A-Tier_Tournaments",
    "B": "B-Tier_Tournaments",
    "C": "C-Tier_Tournaments",
    "D": "D-Tier_Tournaments",
}

# Team org names to skip (these appear as winner/runner-up links, not tournaments)
# We identify tournaments by their link title structure: "LeagueName/Year/..." or "EventName/Year"
TOURNAMENT_TITLE_PATTERN = re.compile(r"^([^/]+)/\d{4}")


def fetch_parsed_html(page_title: str) -> str:
    """Fetch parsed HTML for a Liquipedia page via the MediaWiki API."""
    params = {
        "action": "parse",
        "page": page_title,
        "format": "json",
        "prop": "text",
    }
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()["parse"]["text"]["*"]


def extract_league_names(html: str) -> set[str]:
    """Extract unique base league/tournament names from parsed HTML."""
    soup = BeautifulSoup(html, "html.parser")
    leagues = set()

    for a in soup.find_all("a"):
        title = a.get("title", "")
        href = a.get("href", "")

        # Skip nav/meta links
        if not href.startswith("/leagueoflegends/"):
            continue

        # Match tournament pattern: "LeagueName/Year" or "LeagueName/Year/Split"
        m = TOURNAMENT_TITLE_PATTERN.match(title)
        if m:
            league_name = m.group(1).strip()
            # Skip generic/meta page self-references
            if league_name.lower() not in ("portal", "category",
                                           "s-tier tournaments", "a-tier tournaments",
                                           "b-tier tournaments", "c-tier tournaments",
                                           "d-tier tournaments"):
                leagues.add(league_name)

    return leagues


results = {}

for tier, page in TIER_PAGES.items():
    print(f"\n=== Tier {tier} ({page}) ===")
    html = fetch_parsed_html(page)
    leagues = sorted(extract_league_names(html))
    results[tier] = leagues

    print(f"  Found {len(leagues)} unique leagues:")
    for l in leagues:
        print(f"    - {l}")

    # Be polite to Liquipedia API
    time.sleep(2)

# Post-processing: force TCL to B-tier only (appears in both A and B on Liquipedia)
if "TCL" in results.get("A", []):
    results["A"] = [l for l in results["A"] if l != "TCL"]
if "TCL" not in results.get("B", []):
    results["B"] = sorted(results.get("B", []) + ["TCL"])

# Save
output_path = os.path.join(OUTPUT_DIR, "06_liquipedia_tiers.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nResults written to {output_path}")

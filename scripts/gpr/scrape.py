"""
Scrape Riot's published GPR (Global Power Rankings) snapshots from
lolesports.com year pages.

The pages are server-rendered with the Apollo SSR cache embedded directly
in the HTML as `(window[Symbol.for("ApolloSSRDataTransport")] ??= []).push({...})`
calls. Each push payload contains `events[*].value.data.teamGPR`, a list of
TeamGPR records carrying both a current snapshot and a `teamGPRHistory` array
at ~10-day granularity.

We fetch each year page once, cache the raw HTML under `data/gpr/raw/`, parse
the Apollo pushes, flatten every (team, snapshot) pair, dedupe across pages,
and write `data/gpr/snapshots.csv`.

Usage:
  python -m scripts.gpr.scrape           # use cached HTML when present
  python -m scripts.gpr.scrape --refresh # re-fetch all pages
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
GPR_DIR = REPO_ROOT / "data" / "gpr"
RAW_DIR = GPR_DIR / "raw"
SNAPSHOTS_CSV = GPR_DIR / "snapshots.csv"

# Year pages to scrape. The 2024 and 2024/worlds pages overlap heavily
# (worlds is the same year window with the additional Worlds K-factor games);
# we pull both and dedupe on (team_id, dateCalculated) so the scrape is
# robust to whichever one Riot keeps live.
PAGES = [
    ("2024", "2024"),
    ("2024-worlds", "2024/worlds"),
    ("2025", "2025"),
    ("2026", "2026"),
]

BASE_URL = "https://lolesports.com/en-GB/gpr/"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Matches the literal opening of an Apollo SSR transport push.
PUSH_OPEN_RE = re.compile(
    r'window\[Symbol\.for\("ApolloSSRDataTransport"\)\] \?\?= \[\]\)\.push\(',
)

SNAPSHOT_FIELDS = [
    "team_id",
    "team_code",
    "team_name",
    "home_league_slug",
    "snapshot_date",
    "power_score",
    "team_elo",
    "rank",
]


def fetch_page(slug: str) -> str:
    url = BASE_URL + slug
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def load_or_fetch(label: str, slug: str, refresh: bool) -> str:
    """Return raw HTML for a year page, using disk cache when available."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cached = RAW_DIR / f"{label}.html"
    if cached.exists() and not refresh:
        return cached.read_text(encoding="utf-8", errors="replace")
    print(f"  fetching {BASE_URL + slug} ...", file=sys.stderr)
    html = fetch_page(slug)
    cached.write_text(html, encoding="utf-8")
    return html


def extract_apollo_pushes(html: str) -> Iterator[dict]:
    """Yield each parsed JSON payload from the Apollo SSR transport pushes.

    The push form is `(window[...] ??= []).push(<json>)` where <json> can
    contain nested parens inside string literals. We balance parens manually
    while respecting string quoting + backslash escaping.
    """
    for m in PUSH_OPEN_RE.finditer(html):
        i = m.end()
        depth = 1
        j = i
        in_str = False
        esc = False
        while j < len(html) and depth > 0:
            c = html[j]
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = not in_str
            elif not in_str:
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
            j += 1
        payload = html[i : j - 1]
        # Apollo SSR pushes are JS source, not JSON — they can contain bare
        # `undefined` literals (e.g. `"nextFetchPolicy":undefined`). Substitute
        # `null` before parsing. Safe because `undefined` is not a JSON token,
        # so it can only appear in the JS-only positions.
        payload = re.sub(r"(?<![A-Za-z0-9_])undefined(?![A-Za-z0-9_])", "null", payload)
        try:
            yield json.loads(payload)
        except json.JSONDecodeError as e:
            print(f"  parse error: {e}; payload starts {payload[:120]!r}", file=sys.stderr)


def _iter_data_dicts(payload: dict) -> Iterator[dict]:
    """Yield each `data` dict from a push payload.

    Two shapes appear in the wild:
      - `events[*].value.data` — pushes that stream Apollo events
      - `rehydrate[<opaque key>].data` — pushes that carry pre-rendered SSR
        cache entries directly
    Both can hold `teamGPR` / `leagueELO`. We yield from both.
    """
    for ev in payload.get("events") or []:
        data = (ev.get("value") or {}).get("data")
        if isinstance(data, dict):
            yield data
    rehydrate = payload.get("rehydrate") or {}
    if isinstance(rehydrate, dict):
        for entry in rehydrate.values():
            if isinstance(entry, dict):
                data = entry.get("data")
                if isinstance(data, dict):
                    yield data


def iter_team_records(payloads: Iterable[dict]) -> Iterator[dict]:
    """Yield each TeamGPR dict from a stream of Apollo SSR push payloads."""
    for payload in payloads:
        for data in _iter_data_dicts(payload):
            teams = data.get("teamGPR")
            if isinstance(teams, list):
                yield from teams


def flatten_to_snapshots(team_records: Iterable[dict]) -> Iterator[dict]:
    """Emit one row per (team, history-snapshot) pair.

    Each TeamGPR has a `teamGPRHistory` array; we also include the
    `currentTeamGPR` snapshot since the page sometimes carries it
    independently of the history list.
    """
    for t in team_records:
        team = t.get("team") or {}
        team_id = team.get("id")
        team_code = team.get("code")
        team_name = team.get("name")
        home_league = team.get("homeLeague") or {}
        home_league_slug = home_league.get("slug")

        snapshots = []
        hist = t.get("teamGPRHistory") or []
        snapshots.extend(hist)
        cur = t.get("currentTeamGPR")
        if isinstance(cur, dict):
            snapshots.append(cur)

        for s in snapshots:
            if not isinstance(s, dict):
                continue
            yield {
                "team_id": team_id,
                "team_code": team_code,
                "team_name": team_name,
                "home_league_slug": home_league_slug,
                "snapshot_date": s.get("dateCalculated"),
                "power_score": s.get("gprScore"),
                "team_elo": s.get("elo"),
                "rank": s.get("rank"),
            }


def dedupe(rows: Iterable[dict]) -> list[dict]:
    """Dedupe on (team_id, snapshot_date). Same key across pages must agree."""
    by_key: dict[tuple[str, str], dict] = {}
    conflicts = 0
    for r in rows:
        if r["team_id"] is None or r["snapshot_date"] is None:
            continue
        key = (r["team_id"], r["snapshot_date"])
        prior = by_key.get(key)
        if prior is None:
            by_key[key] = r
            continue
        # Same (team, date) seen twice across pages — they must agree on the
        # numeric snapshot. If they don't, the published rating diverged for
        # the same publication timestamp, which would be a data-quality flag
        # worth surfacing rather than silently overwriting.
        if (
            prior["power_score"] != r["power_score"]
            or prior["team_elo"] != r["team_elo"]
            or prior["rank"] != r["rank"]
        ):
            conflicts += 1
            print(
                f"  conflict at {key}: "
                f"prior={prior['power_score']}/{prior['team_elo']}/{prior['rank']} "
                f"new={r['power_score']}/{r['team_elo']}/{r['rank']}",
                file=sys.stderr,
            )
    if conflicts:
        print(f"  {conflicts} cross-page conflicts on (team_id, snapshot_date)", file=sys.stderr)
    out = list(by_key.values())
    out.sort(key=lambda r: (r["team_id"], r["snapshot_date"]))
    return out


def write_snapshots(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SNAPSHOT_FIELDS)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch year pages from lolesports.com (otherwise use cached HTML)",
    )
    args = ap.parse_args(argv)

    all_rows: list[dict] = []
    per_page_counts: list[tuple[str, int, int]] = []  # (label, teams, snapshots)

    for label, slug in PAGES:
        html = load_or_fetch(label, slug, refresh=args.refresh)
        teams = list(iter_team_records(extract_apollo_pushes(html)))
        # Each page serializes its query result twice (Apollo events stream +
        # SSR rehydrate cache), so the same team appears in both. Dedupe on
        # team id for the per-page count so the summary matches what's on
        # lolesports.com.
        unique_team_ids = {(t.get("team") or {}).get("id") for t in teams}
        unique_team_ids.discard(None)
        page_rows = list(flatten_to_snapshots(teams))
        per_page_counts.append((label, len(unique_team_ids), len(page_rows)))
        all_rows.extend(page_rows)

    deduped = dedupe(all_rows)
    write_snapshots(deduped, SNAPSHOTS_CSV)

    print("=== Scrape summary ===")
    for label, n_teams, n_snap in per_page_counts:
        print(f"  {label:<12} teams={n_teams:>4}  snapshots={n_snap:>5}")
    print(f"  --- total raw snapshots: {sum(n for _, _, n in per_page_counts)}")
    print(f"  --- after dedupe:        {len(deduped)}")
    n_unique_teams = len({r["team_id"] for r in deduped})
    print(f"  --- unique teams:        {n_unique_teams}")
    print(f"  wrote {SNAPSHOTS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

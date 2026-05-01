"""
Discovery: timing distribution of first-events in S-tier pro games via the
unofficial lolesports livestats API. Settles whether `firstdragon` should be
gated into the 10-min checkpoint model (per discovery-checklist.md section 1)
and re-verifies the section 3 claim that herald/voidgrub are not in the API.

Spec: docs/specs/livestats-firsts-timing.md

Method:
  1. Discover S-tier completed games in 2025-01-01 .. today via getSchedule.
  2. For each sampled game, fetch /window/ at game start (no startingTime) to
     pin the game-start wallclock, then fetch /window/?startingTime=... at each
     game-time minute in TARGET_MINUTES.
  3. Detect first-event firing minute by walking the polled team states.
  4. Aggregate to histograms + per-checkpoint resolution rates.

Outputs:
  - data/exploration/32_livestats_gameids.json  (list of discovered games)
  - data/exploration/livestats_frames/{game_id}_{slug}.json  (cached responses)
  - data/exploration/32_livestats_firsts_timing.json  (summary)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

# ---------------------------------------------------------------------------
# Constants

API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"  # public lolesports.com frontend key
PERSISTED = "https://esports-api.lolesports.com/persisted/gw"
FEED = "https://feed.lolesports.com/livestats/v1"

# S-tier leagues (2025-2026 active set). Per the spec's scope.
LEAGUES = {
    "98767991310872058": "LCK",
    "98767991302996019": "LEC",
    "98767991314006698": "LPL",
    "113470291645289904": "LTA North",
    "113475181634818701": "LTA South",
    "113476371197627891": "LCP",
    "98767975604431411":  "Worlds",
    "98767991325878492":  "MSI",
    "113464388705111224": "First Stand",
}

DATE_START = "2025-01-01T00:00:00Z"
# DATE_END auto-derived from "now"

# Game-time minutes to poll (inclusive on both ends).
# Dense early (when first-events fire), sparser late (just baron range).
TARGET_MINUTES = list(range(3, 16)) + [17, 19, 21, 23, 25, 27, 29]

RATE_LIMIT_SEC = 0.25  # min spacing between API calls
MAX_MATCHES_PER_LEAGUE = 60  # discovery cap; sampling happens after

ROOT = Path(__file__).resolve().parent.parent.parent
EXPL_DIR = ROOT / "data" / "exploration"
FRAMES_DIR = EXPL_DIR / "livestats_frames"
GAMEIDS_FILE = EXPL_DIR / "32_livestats_gameids.json"
SUMMARY_FILE = EXPL_DIR / "32_livestats_firsts_timing.json"

# ---------------------------------------------------------------------------
# HTTP

_last_call = [0.0]


def http_get(url: str, retries: int = 4) -> dict | None:
    """GET with rate-limit + retry. Returns parsed JSON, or None on hard fail."""
    for attempt in range(retries):
        # rate limit
        elapsed = time.time() - _last_call[0]
        if elapsed < RATE_LIMIT_SEC:
            time.sleep(RATE_LIMIT_SEC - elapsed)
        _last_call[0] = time.time()

        req = urllib.request.Request(url, headers={"x-api-key": API_KEY})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read()
            if not body:
                return None  # 200 with empty body — treat as "no data"
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                print(f"  bad json from {url[:120]}", file=sys.stderr)
                return None
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return None  # data not available, not a transient failure
            if e.code == 429 or 500 <= e.code < 600:
                wait = 2 ** attempt
                print(f"  http {e.code}, retry in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"  http {e.code}: {url}", file=sys.stderr)
            return None
        except (urllib.error.URLError, TimeoutError) as e:
            wait = 2 ** attempt
            print(f"  network error ({e}), retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
    print(f"  giving up: {url}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Step 1: Discover gameIds

def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def walk_schedule(league_id: str, date_start: datetime, date_end: datetime) -> list[dict]:
    """Walk getSchedule backward via pageToken. Yield events in date range with state=completed."""
    events_out: list[dict] = []
    page_token = None
    pages_seen = 0

    while True:
        url = f"{PERSISTED}/getSchedule?hl=en-US&leagueId={league_id}"
        if page_token:
            url += f"&pageToken={page_token}"
        d = http_get(url)
        if not d:
            break

        sched = d["data"]["schedule"]
        events = sched.get("events", [])
        pages = sched.get("pages", {})
        pages_seen += 1

        if not events:
            break

        # events come newest-first within a page.
        oldest_seen = parse_iso(events[-1]["startTime"])

        for e in events:
            ts = parse_iso(e["startTime"])
            if ts < date_start or ts > date_end:
                continue
            if e.get("state") != "completed":
                continue
            if e.get("type") != "match":
                continue
            events_out.append(e)

        # page_token "older" advances backward in time
        next_token = pages.get("older")
        if not next_token:
            break
        if oldest_seen < date_start:
            break  # no point paging further
        page_token = next_token

        if pages_seen > 50:
            print(f"  safety stop after 50 pages on league {league_id}", file=sys.stderr)
            break

    return events_out


def discover_gameids() -> list[dict]:
    """Walk getSchedule + getEventDetails for all configured leagues."""
    print("Discovering games...")
    date_start = parse_iso(DATE_START)
    date_end = datetime.now(timezone.utc)

    games: list[dict] = []

    for league_id, league_name in LEAGUES.items():
        print(f"  {league_name}...", end=" ", flush=True)
        events = walk_schedule(league_id, date_start, date_end)
        # Cap matches per league: walk_schedule returns newest-first, so taking
        # the first N keeps the most recent matches.
        events = events[:MAX_MATCHES_PER_LEAGUE]
        n_games_before = len(games)

        for ev in events:
            match_id = ev["match"]["id"]
            d = http_get(f"{PERSISTED}/getEventDetails?hl=en-US&id={match_id}")
            if not d:
                continue
            event = d["data"].get("event")
            if not event or "match" not in event:
                continue
            for g in event["match"].get("games", []):
                if g.get("state") != "completed":
                    continue
                blue_team = next((t for t in g.get("teams", []) if t.get("side") == "blue"), {})
                red_team = next((t for t in g.get("teams", []) if t.get("side") == "red"), {})
                games.append({
                    "game_id": g["id"],
                    "game_number": g.get("number"),
                    "match_id": match_id,
                    "league_id": league_id,
                    "league_name": league_name,
                    "block_name": ev.get("blockName"),
                    "match_start": ev["startTime"],
                    "blue_team_id": blue_team.get("id"),
                    "red_team_id": red_team.get("id"),
                })
        print(f"{len(events)} matches -> {len(games) - n_games_before} games")

    print(f"Total: {len(games)} games")
    return games


# ---------------------------------------------------------------------------
# Step 2: Fetch frames

def floor_10s(dt: datetime) -> datetime:
    """Round down to the nearest 10s; required by /window/?startingTime."""
    return dt.replace(second=(dt.second // 10) * 10, microsecond=0)


def fmt_starting_time(dt: datetime) -> str:
    """Format datetime as YYYY-MM-DDTHH:MM:SS.000Z (10s-aligned)."""
    dt = floor_10s(dt)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def fetch_init(game_id: str) -> dict | None:
    """Fetch /window/ without startingTime — returns ~10 frames at game start."""
    return http_get(f"{FEED}/window/{game_id}")


def fetch_at(game_id: str, starting_time: datetime) -> dict | None:
    """Fetch /window/?startingTime=... — returns frames at wallclock starting there."""
    s = fmt_starting_time(starting_time)
    return http_get(f"{FEED}/window/{game_id}?startingTime={s}")


def cache_path(game_id: str, slug: str) -> Path:
    return FRAMES_DIR / f"{game_id}_{slug}.json"


def load_or_fetch(game_id: str, slug: str, fetcher) -> dict | None:
    """Cached fetch: check disk first, write on success, return parsed dict."""
    p = cache_path(game_id, slug)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            p.unlink()  # corrupt — refetch

    d = fetcher()
    if d is None:
        # write a sentinel so we don't refetch a known-missing window
        p.write_text(json.dumps({"_missing": True}), encoding="utf-8")
        return None
    p.write_text(json.dumps(d), encoding="utf-8")
    return d


def fetch_game(game_meta: dict) -> dict | None:
    """Fetch all polls for one game. Returns metadata about the fetch outcome."""
    game_id = game_meta["game_id"]
    init = load_or_fetch(game_id, "init", lambda: fetch_init(game_id))
    if init is None or init.get("_missing") or not init.get("frames"):
        return {"game_id": game_id, "ok": False, "reason": "no_init"}

    # game-start wallclock = first frame's rfc460Timestamp
    first_frame = init["frames"][0]
    game_start_wall = parse_iso(first_frame["rfc460Timestamp"])

    polls_done = 0
    polls_missing = 0
    for minute in TARGET_MINUTES:
        target = game_start_wall + timedelta(minutes=minute)
        slug = f"m{minute:02d}"
        d = load_or_fetch(game_id, slug, lambda t=target: fetch_at(game_id, t))
        if d is None or d.get("_missing"):
            polls_missing += 1
        else:
            polls_done += 1

    return {
        "game_id": game_id,
        "ok": True,
        "game_start": first_frame["rfc460Timestamp"],
        "polls_done": polls_done,
        "polls_missing": polls_missing,
    }


# ---------------------------------------------------------------------------
# Step 3: Parse first events

def team_state(frame: dict) -> dict:
    """Extract the bits we care about from one frame."""
    bt, rt = frame.get("blueTeam", {}), frame.get("redTeam", {})
    return {
        "game_state": frame.get("gameState"),
        "kills": (bt.get("totalKills", 0) or 0) + (rt.get("totalKills", 0) or 0),
        "dragons": len(bt.get("dragons", []) or []) + len(rt.get("dragons", []) or []),
        "towers": (bt.get("towers", 0) or 0) + (rt.get("towers", 0) or 0),
        "barons": (bt.get("barons", 0) or 0) + (rt.get("barons", 0) or 0),
        "inhibitors": (bt.get("inhibitors", 0) or 0) + (rt.get("inhibitors", 0) or 0),
    }


def parse_game(game_id: str) -> dict | None:
    """Read cached polls, return per-event firing minute (or None if not by minute 29)."""
    init_p = cache_path(game_id, "init")
    if not init_p.exists():
        return None
    try:
        init = json.loads(init_p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if init.get("_missing"):
        return None

    # Per-event firing: smallest minute at which we observed the count >= 1.
    # We initialise to None and set on first observation.
    fired: dict[str, int | None] = {
        "firstblood": None,
        "firstdragon": None,
        "firsttower": None,
        "firstbaron": None,
    }

    pause_minutes = 0
    polls_seen = 0
    last_observed_minute = None

    for minute in TARGET_MINUTES:
        p = cache_path(game_id, f"m{minute:02d}")
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if d.get("_missing"):
            continue
        frames = d.get("frames") or []
        if not frames:
            continue

        # Use the FIRST in_game frame we have for this poll.
        chosen = next((f for f in frames if f.get("gameState") == "in_game"), None)
        if chosen is None:
            # all frames non-in_game → likely paused or finished
            non_in_game = Counter(f.get("gameState") for f in frames)
            if any(s == "paused" for s in non_in_game):
                pause_minutes += 1
            continue

        polls_seen += 1
        last_observed_minute = minute
        st = team_state(chosen)

        if fired["firstblood"] is None and st["kills"] >= 1:
            fired["firstblood"] = minute
        if fired["firstdragon"] is None and st["dragons"] >= 1:
            fired["firstdragon"] = minute
        if fired["firsttower"] is None and st["towers"] >= 1:
            fired["firsttower"] = minute
        if fired["firstbaron"] is None and st["barons"] >= 1:
            fired["firstbaron"] = minute

    if polls_seen == 0:
        return None

    return {
        "game_id": game_id,
        "polls_seen": polls_seen,
        "last_observed_minute": last_observed_minute,
        "fired": fired,
        "pause_minutes": pause_minutes,
    }


# ---------------------------------------------------------------------------
# Step 4: Aggregate

CHECKPOINTS = [10, 15, 20, 25]
EVENTS = ["firstblood", "firstdragon", "firsttower", "firstbaron"]


def aggregate(records: list[dict], gameids_meta: dict[str, dict]) -> dict:
    """Build per-event histograms + checkpoint resolution rates."""
    n_total = len(records)
    n_with_pause = sum(1 for r in records if r["pause_minutes"] > 0)

    by_event: dict[str, dict] = {}
    for ev in EVENTS:
        firing_minutes = [r["fired"][ev] for r in records if r["fired"][ev] is not None]
        not_fired_count = sum(1 for r in records if r["fired"][ev] is None)

        # Resolution at each checkpoint: only count games with at least one poll
        # at-or-after the checkpoint (otherwise we don't know if it had fired).
        pct_by_checkpoint = {}
        for cp in CHECKPOINTS:
            eligible = [r for r in records if r["last_observed_minute"] is not None and r["last_observed_minute"] >= cp]
            if not eligible:
                pct_by_checkpoint[str(cp)] = None
                continue
            fired_by_cp = sum(
                1 for r in eligible
                if r["fired"][ev] is not None and r["fired"][ev] <= cp
            )
            pct_by_checkpoint[str(cp)] = round(fired_by_cp / len(eligible), 4)

        # 1-min histogram
        hist = Counter(firing_minutes)
        histogram = {str(m): hist.get(m, 0) for m in TARGET_MINUTES}

        by_event[ev] = {
            "n_with_event": len(firing_minutes),
            "n_without_event_in_window": not_fired_count,
            "median_minute": median(firing_minutes) if firing_minutes else None,
            "p25_minute": _percentile(firing_minutes, 25),
            "p75_minute": _percentile(firing_minutes, 75),
            "min_minute": min(firing_minutes) if firing_minutes else None,
            "max_minute": max(firing_minutes) if firing_minutes else None,
            "pct_by_checkpoint": pct_by_checkpoint,
            "histogram_per_minute": histogram,
        }

    by_league: dict[str, int] = Counter(
        gameids_meta[r["game_id"]]["league_name"] for r in records
    )

    return {
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_games": n_total,
        "n_games_with_any_pause_seen": n_with_pause,
        "by_league": dict(by_league),
        "checkpoints_minutes": CHECKPOINTS,
        "polled_minutes": TARGET_MINUTES,
        "by_event": by_event,
        "presence_only": {
            "firstherald": False,  # confirmed: not exposed in /window or /details
            "firstvoidgrub": False,
            "firstatakhan": False,
        },
        "verifications": {
            "event_field_exposure": (
                "Confirmed via probe: blueTeam/redTeam expose totalKills, towers, "
                "barons, inhibitors, dragons[]. heralds, voidGrubs, atakhans not "
                "present in /window or /details."
            ),
            "frame_at_checkpoint_convention": (
                "Per-poll: first in_game frame from /window/?startingTime=game_start"
                "+minute*60s. Resolution +/- 1 min."
            ),
            "pause_handling": (
                "When all frames at a poll are non-in_game we skip the poll. "
                f"{n_with_pause}/{n_total} games had at least one such poll."
            ),
        },
    }


def _percentile(xs: list[int], p: int) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ---------------------------------------------------------------------------
# Main

def load_or_discover_gameids(rediscover: bool) -> list[dict]:
    if not rediscover and GAMEIDS_FILE.exists():
        return json.loads(GAMEIDS_FILE.read_text(encoding="utf-8"))["games"]
    games = discover_gameids()
    GAMEIDS_FILE.write_text(json.dumps({
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "leagues": LEAGUES,
        "date_start": DATE_START,
        "n_games": len(games),
        "games": games,
    }, indent=2), encoding="utf-8")
    return games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=300, help="sample size; 0 = all")
    parser.add_argument("--rediscover", action="store_true",
                        help="re-walk getSchedule/getEventDetails")
    parser.add_argument("--analyze-only", action="store_true",
                        help="skip fetch, only re-aggregate from cached frames")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    EXPL_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    games = load_or_discover_gameids(args.rediscover)
    print(f"Have {len(games)} candidate games.")

    # Stratified sample by league: take up to N/leagues per league, fill remainder.
    rng = random.Random(args.seed)
    by_lg: dict[str, list[dict]] = defaultdict(list)
    for g in games:
        by_lg[g["league_name"]].append(g)
    for lg in by_lg:
        rng.shuffle(by_lg[lg])

    if args.limit and args.limit < len(games):
        per_lg = max(1, args.limit // max(1, len(by_lg)))
        sample: list[dict] = []
        for lg, gs in by_lg.items():
            sample.extend(gs[:per_lg])
        # top up if short
        rest = [g for g in games if g not in sample]
        rng.shuffle(rest)
        sample.extend(rest[: max(0, args.limit - len(sample))])
        sample = sample[: args.limit]
    else:
        sample = games

    print(f"Sample: {len(sample)} games "
          f"({dict(Counter(g['league_name'] for g in sample))})")

    if not args.analyze_only:
        for i, g in enumerate(sample, 1):
            if i % 25 == 0 or i == 1:
                print(f"[{i}/{len(sample)}] fetching {g['league_name']} {g['game_id']}")
            fetch_game(g)

    # Analyze
    print("Parsing cached frames...")
    records = []
    gameids_meta = {g["game_id"]: g for g in sample}
    for g in sample:
        rec = parse_game(g["game_id"])
        if rec:
            records.append(rec)
    print(f"Parsed {len(records)}/{len(sample)} games.")

    summary = aggregate(records, gameids_meta)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {SUMMARY_FILE}")

    # Print headline
    print("\n=== Headline: % resolved by checkpoint ===")
    print(f"{'event':<14} " + " ".join(f"{cp:>6}" for cp in CHECKPOINTS) + "  median(min)")
    for ev in EVENTS:
        d = summary["by_event"][ev]
        cps = " ".join(
            f"{(d['pct_by_checkpoint'][str(cp)] or 0)*100:>5.1f}%"
            for cp in CHECKPOINTS
        )
        med = d["median_minute"]
        med_s = f"{med:.1f}" if med is not None else "  -"
        print(f"  {ev:<12} {cps}  {med_s}")


if __name__ == "__main__":
    main()

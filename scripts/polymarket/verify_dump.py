"""Sanity-check the polymarket dump without loading whole files into memory.

Streams the CSVs and prints a one-screen summary:
  - row counts and unique-key counts per file
  - duplicate detection (the parallel-process race could leave dupes)
  - referential integrity (every condition_id in trades.csv exists in markets.csv etc.)
  - per-market trade-count agreement (markets.csv n_trades_fetched vs trades.csv reality)
  - ledger status breakdown
  - slug-parse coverage on events.csv
  - one live spot-check: re-fetch one random ok market and compare row count
"""
from __future__ import annotations

import csv
import json
import random
import sys
import urllib.request
from collections import Counter
from pathlib import Path

OUT_DIR = Path("data/polymarket")
EVENTS_CSV = OUT_DIR / "events.csv"
MARKETS_CSV = OUT_DIR / "markets.csv"
TRADES_CSV = OUT_DIR / "trades.csv"
LEDGER_CSV = OUT_DIR / "_fetch_ledger.csv"


def stream_rows(path: Path):
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def section(title: str):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def check_events():
    section("events.csv")
    n = 0
    ids = set()
    dup_ids = 0
    null_team_a = 0
    null_match_date = 0
    for row in stream_rows(EVENTS_CSV):
        n += 1
        eid = row.get("event_id") or ""
        if eid in ids:
            dup_ids += 1
        ids.add(eid)
        if not (row.get("team_a_short") or ""):
            null_team_a += 1
        if not (row.get("match_date") or ""):
            null_match_date += 1
    print(f"  rows                    : {n}")
    print(f"  unique event_id         : {len(ids)}")
    print(f"  duplicate event_id rows : {dup_ids}  (should be 0)")
    print(f"  null team_a_short       : {null_team_a}/{n}  (slugs not matching pattern)")
    print(f"  null match_date         : {null_match_date}/{n}")
    return ids


def check_markets():
    section("markets.csv")
    n = 0
    market_ids = set()
    cond_to_status = {}
    cond_to_n_trades = {}
    cond_to_event = {}
    dup_market_ids = 0
    status_counts = Counter()
    game_counts = Counter()
    for row in stream_rows(MARKETS_CSV):
        n += 1
        mid = row.get("market_id") or ""
        if mid in market_ids:
            dup_market_ids += 1
        market_ids.add(mid)
        cid = row.get("condition_id") or ""
        cond_to_status[cid] = row.get("fetch_status") or ""
        try:
            cond_to_n_trades[cid] = int(row.get("n_trades_fetched") or 0)
        except ValueError:
            cond_to_n_trades[cid] = 0
        cond_to_event[cid] = row.get("event_id") or ""
        status_counts[row.get("fetch_status") or ""] += 1
        try:
            game_counts[int(row.get("game_number") or 0)] += 1
        except ValueError:
            pass
    print(f"  rows                    : {n}")
    print(f"  unique market_id        : {len(market_ids)}")
    print(f"  duplicate market_id rows: {dup_market_ids}  (should be 0)")
    print(f"  unique condition_id     : {len(cond_to_status)}")
    print(f"  fetch_status breakdown  : {dict(status_counts)}")
    print(f"  game_number breakdown   : {dict(sorted(game_counts.items()))}")
    return cond_to_status, cond_to_n_trades, cond_to_event


def check_trades(known_conditions: set[str]):
    section("trades.csv")
    n = 0
    cond_counter = Counter()
    seen_keys = set()
    dup_rows = 0
    null_price = 0
    null_size = 0
    for row in stream_rows(TRADES_CSV):
        n += 1
        cid = row.get("condition_id") or ""
        cond_counter[cid] += 1
        # Dedupe key matches the original probe in script 30: tx + ts + price + size + side
        key = (row.get("transaction_hash"), row.get("timestamp_unix"),
               row.get("price"), row.get("size"), row.get("side"),
               row.get("asset"))
        if key in seen_keys:
            dup_rows += 1
        else:
            seen_keys.add(key)
        if not (row.get("price") or ""):
            null_price += 1
        if not (row.get("size") or ""):
            null_size += 1
    orphan_conds = set(cond_counter.keys()) - known_conditions
    print(f"  rows                    : {n}")
    print(f"  unique condition_id     : {len(cond_counter)}")
    print(f"  duplicate fill rows     : {dup_rows}  (should be 0; race-induced if >0)")
    print(f"  null price              : {null_price}")
    print(f"  null size               : {null_size}")
    print(f"  orphan condition_ids    : {len(orphan_conds)}  (in trades but not markets — should be 0)")
    if orphan_conds:
        for c in list(orphan_conds)[:5]:
            print(f"    e.g. {c}")
    return cond_counter


def check_agreement(cond_to_status, cond_to_n_trades, cond_counter):
    section("trade count agreement (markets.csv n_trades_fetched vs trades.csv reality)")
    mismatches = []
    missing_in_trades = []
    for cid, status in cond_to_status.items():
        if status != "ok":
            continue  # empty / error markets shouldn't have trades
        expected = cond_to_n_trades.get(cid, 0)
        actual = cond_counter.get(cid, 0)
        if expected != actual:
            mismatches.append((cid, expected, actual))
        if actual == 0 and expected > 0:
            missing_in_trades.append(cid)
    print(f"  ok markets                  : {sum(1 for s in cond_to_status.values() if s == 'ok')}")
    print(f"  count mismatches            : {len(mismatches)}  (should be 0)")
    if mismatches:
        for cid, exp, act in mismatches[:5]:
            print(f"    {cid}  expected={exp} actual={act}")
    print(f"  ok markets with 0 trades    : {len(missing_in_trades)}  (should be 0; status='empty' is the right marker for those)")


def check_ledger(cond_to_status):
    section("_fetch_ledger.csv")
    n = 0
    latest = {}
    for row in stream_rows(LEDGER_CSV):
        n += 1
        cid = row.get("condition_id") or ""
        latest[cid] = row.get("status") or ""
    counts = Counter(latest.values())
    print(f"  rows (incl. retries)    : {n}")
    print(f"  unique condition_id     : {len(latest)}")
    print(f"  latest-status breakdown : {dict(counts)}")
    # Markets table 'fetch_status' should agree with ledger latest 'status'
    disagree = 0
    not_in_ledger = 0
    for cid, mstatus in cond_to_status.items():
        lstatus = latest.get(cid)
        if lstatus is None:
            not_in_ledger += 1
        elif lstatus != mstatus:
            disagree += 1
    print(f"  markets.csv vs ledger disagreement: {disagree}  (should be 0)")
    print(f"  markets without ledger entry      : {not_in_ledger}  (should be 0)")


def live_spot_check(cond_to_status, cond_counter):
    section("live spot check (one random ok market, refetched)")
    ok_conds = [c for c, s in cond_to_status.items() if s == "ok" and cond_counter.get(c, 0) > 0]
    if not ok_conds:
        print("  no ok markets to sample")
        return
    cid = random.choice(ok_conds)
    print(f"  sampling condition_id   : {cid}")
    # Recount via API (just count, don't store)
    n_api = 0
    page = 0
    while True:
        url = (f"https://data-api.polymarket.com/trades?market={cid}"
               f"&limit=500&offset={page * 500}")
        req = urllib.request.Request(url, headers={"User-Agent": "polyjuice-verify"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                rows = json.loads(r.read())
        except Exception as e:
            print(f"  API fetch failed: {e}")
            return
        if not isinstance(rows, list) or not rows:
            break
        n_api += len(rows)
        if len(rows) < 500:
            break
        page += 1
    n_local = cond_counter.get(cid, 0)
    print(f"  local trade count       : {n_local}")
    print(f"  API trade count         : {n_api}")
    print(f"  delta                   : {n_api - n_local}  (0 means perfect agreement)")


def main() -> int:
    if not OUT_DIR.exists():
        print(f"no output dir at {OUT_DIR}", file=sys.stderr)
        return 1
    event_ids = check_events()
    cond_to_status, cond_to_n_trades, cond_to_event = check_markets()
    cond_counter = check_trades(set(cond_to_status.keys()))
    check_agreement(cond_to_status, cond_to_n_trades, cond_counter)
    check_ledger(cond_to_status)
    live_spot_check(cond_to_status, cond_counter)
    # markets-to-events referential integrity
    section("markets -> events referential integrity")
    orphan_market_events = {e for e in cond_to_event.values() if e and e not in event_ids}
    print(f"  markets pointing to missing event_id: {len(orphan_market_events)}  (should be 0)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

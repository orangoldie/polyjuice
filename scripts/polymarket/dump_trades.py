"""Dump every fill on every settled LoL "Game N Winner" market to CSV.

See docs/specs/polymarket-trade-dump.md for design rationale.

Outputs (under data/polymarket/):
  events.csv         one row per kept event
  markets.csv        one row per kept "Game N Winner" market
  trades.csv         one row per fill
  _fetch_ledger.csv  append-only per-market completion log
  _staging/          per-market staging files for crash-safe commits
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"
DATA_API_TRADES = "https://data-api.polymarket.com/trades"
USER_AGENT = "polyjuice-trade-dump"

OUT_DIR = Path("data/polymarket")
EVENTS_CSV = OUT_DIR / "events.csv"
MARKETS_CSV = OUT_DIR / "markets.csv"
TRADES_CSV = OUT_DIR / "trades.csv"
LEDGER_CSV = OUT_DIR / "_fetch_ledger.csv"
STAGING_DIR = OUT_DIR / "_staging"
INVENTORY_CACHE = OUT_DIR / "_events_inventory.json"
INVENTORY_TTL_HOURS = 12

EVENT_VOL_FLOOR = 100.0
TRADES_PAGE_SIZE = 500
MAX_PAGES_PER_MARKET = 200  # 100k fills cap; far above any observed value
SLEEP_BETWEEN_PAGES_S = 0.10
SLEEP_BETWEEN_MARKETS_S = 0.25
RETRY_ATTEMPTS = 8
RETRY_BACKOFF_BASE_S = 1.5

GAME_N_PAT = re.compile(r"\bgame\s+(\d+)\s+winner\b", re.IGNORECASE)
SLUG_PAT = re.compile(r"^lol-([a-z0-9]+)-([a-z0-9]+)-(\d{4}-\d{2}-\d{2})$")

EVENTS_COLS = [
    "event_id", "event_slug", "event_title", "start_date", "end_date",
    "closed", "volume_usd", "n_markets_total",
    "team_a_short", "team_b_short", "match_date", "fetched_at",
]
MARKETS_COLS = [
    "market_id", "condition_id", "event_id", "question", "game_number",
    "outcome_0", "outcome_1", "token_0", "token_1",
    "market_start", "market_end", "market_closed", "volume_usd",
    "n_trades_fetched", "fetch_status", "fetched_at",
]
TRADES_COLS = [
    "condition_id", "timestamp_unix", "price", "size", "side",
    "asset", "outcome_index", "transaction_hash", "proxy_wallet",
]
LEDGER_COLS = [
    "condition_id", "run_started_at", "status", "n_pages_seen",
    "last_page_size", "n_trades_written", "error",
]

log = logging.getLogger("polymarket.dump_trades")


def http_get_json(url: str, *, timeout: float = 30.0) -> object:
    """GET JSON with retry/backoff on 429/5xx and transient network errors."""
    last_exc: Exception | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code in (429, 500, 502, 503, 504):
                wait = RETRY_BACKOFF_BASE_S ** attempt
                log.warning("HTTP %s on %s — backoff %.1fs (attempt %d/%d)",
                            e.code, url, wait, attempt + 1, RETRY_ATTEMPTS)
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_exc = e
            wait = RETRY_BACKOFF_BASE_S ** attempt
            log.warning("network error on %s: %s — backoff %.1fs (attempt %d/%d)",
                        url, e, wait, attempt + 1, RETRY_ATTEMPTS)
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_slug(slug: str) -> tuple[str | None, str | None, str | None]:
    m = SLUG_PAT.match(slug or "")
    if not m:
        return None, None, None
    return m.group(1), m.group(2), m.group(3)


def _inventory_cache_fresh() -> bool:
    if not INVENTORY_CACHE.exists():
        return False
    mtime = dt.datetime.fromtimestamp(INVENTORY_CACHE.stat().st_mtime, dt.timezone.utc)
    age = dt.datetime.now(dt.timezone.utc) - mtime
    return age < dt.timedelta(hours=INVENTORY_TTL_HOURS)


def list_settled_lol_events(use_cache: bool = True) -> list[dict]:
    """Walk the Gamma events endpoint for closed LoL events. Cached on disk."""
    if use_cache and _inventory_cache_fresh():
        with INVENTORY_CACHE.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        log.info("inventory: using cache (%d events, age < %dh)",
                 len(cached), INVENTORY_TTL_HOURS)
        return cached

    seen: set[str] = set()
    out: list[dict] = []
    offset = 0
    while True:
        url = (f"{GAMMA_EVENTS}?series_slug=league-of-legends"
               f"&limit=200&offset={offset}&closed=true")
        page = http_get_json(url)
        if not isinstance(page, list) or not page:
            break
        new = [e for e in page if e.get("id") and e["id"] not in seen]
        for e in new:
            seen.add(e["id"])
        out.extend(new)
        log.info("inventory: offset=%d returned=%d new=%d running_total=%d",
                 offset, len(page), len(new), len(out))
        if len(page) < 200:
            break
        offset += 200

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with INVENTORY_CACHE.open("w", encoding="utf-8") as f:
        json.dump(out, f)
    log.info("inventory: cached %d events to %s", len(out), INVENTORY_CACHE)
    return out


def fetch_event_full(slug: str) -> dict | None:
    """Pull the full event JSON (markets[] is only present here, not in list-mode)."""
    page = http_get_json(f"{GAMMA_EVENTS}?slug={slug}")
    if isinstance(page, list) and page:
        return page[0]
    return None


def is_game_n_winner(question: str | None) -> int | None:
    if not question:
        return None
    m = GAME_N_PAT.search(question)
    if not m:
        return None
    return int(m.group(1))


def fetch_all_trades(condition_id: str) -> tuple[list[dict], int, int]:
    """Paginate the data-api trades endpoint until exhaustion.

    Returns (trades, n_pages_seen, last_page_size). Raises if the loop hits
    MAX_PAGES_PER_MARKET without seeing a short page — that signals an unexpected
    pagination cap and the caller should record the market as `error`.
    """
    out: list[dict] = []
    pages = 0
    last_len = 0
    for page_idx in range(MAX_PAGES_PER_MARKET):
        url = (f"{DATA_API_TRADES}?market={condition_id}"
               f"&limit={TRADES_PAGE_SIZE}&offset={page_idx * TRADES_PAGE_SIZE}")
        r = http_get_json(url)
        pages += 1
        if not isinstance(r, list):
            raise RuntimeError(f"unexpected response shape for {url}: {type(r).__name__}")
        last_len = len(r)
        out.extend(r)
        if last_len < TRADES_PAGE_SIZE:
            return out, pages, last_len
        time.sleep(SLEEP_BETWEEN_PAGES_S)
    raise RuntimeError(
        f"hit MAX_PAGES_PER_MARKET={MAX_PAGES_PER_MARKET} for condition_id={condition_id}"
    )


def trade_to_row(t: dict) -> dict:
    return {
        "condition_id": t.get("conditionId"),
        "timestamp_unix": t.get("timestamp"),
        "price": t.get("price"),
        "size": t.get("size"),
        "side": t.get("side"),
        "asset": t.get("asset"),
        "outcome_index": t.get("outcomeIndex"),
        "transaction_hash": t.get("transactionHash"),
        "proxy_wallet": t.get("proxyWallet"),
    }


def ensure_csv_header(path: Path, columns: list[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=columns).writeheader()


def append_rows(path: Path, columns: list[str], rows: list[dict]) -> None:
    """Append rows then flush+fsync. Caller must have written header first."""
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        for r in rows:
            w.writerow({k: r.get(k) for k in columns})
        f.flush()
        os.fsync(f.fileno())


def upsert_event(event_row: dict) -> None:
    """Insert or replace the row for this event_id in events.csv. Best-effort:
    a metadata-write failure must not abort trade fetching, since trade data is
    the irreplaceable bit and the ledger is the authoritative completion record."""
    try:
        ensure_csv_header(EVENTS_CSV, EVENTS_COLS)
        rewrite_csv_with_replacement(EVENTS_CSV, EVENTS_COLS, "event_id", event_row)
    except (PermissionError, FileNotFoundError, OSError) as e:
        log.warning("upsert_event failed for event_id=%s: %s — skipping metadata write",
                    event_row.get("event_id"), e)


def upsert_market(market_row: dict) -> None:
    """Same best-effort semantics as upsert_event."""
    try:
        ensure_csv_header(MARKETS_CSV, MARKETS_COLS)
        rewrite_csv_with_replacement(MARKETS_CSV, MARKETS_COLS, "market_id", market_row)
    except (PermissionError, FileNotFoundError, OSError) as e:
        log.warning("upsert_market failed for market_id=%s: %s — skipping metadata write",
                    market_row.get("market_id"), e)


def _atomic_replace(src: Path, dst: Path, attempts: int = 10,
                    initial_wait_s: float = 0.1) -> None:
    """os.replace with retry. On Windows this can fail two ways:
      - PermissionError (WinError 32): another process is holding dst.
      - FileNotFoundError (WinError 2): src vanished between fsync and replace,
        usually because antivirus quarantined the freshly-written tmp file.
    Retry both."""
    wait = initial_wait_s
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except (PermissionError, FileNotFoundError) as e:
            if i == attempts - 1:
                raise
            log.warning("replace %s -> %s failed (%s); retrying in %.1fs",
                        src, dst, e, wait)
            time.sleep(wait)
            wait *= 2


def _retry_unlink(path: Path, attempts: int = 10,
                  initial_wait_s: float = 0.1) -> None:
    """unlink with retry. Same Windows scanner-holds-the-file failure mode as
    _atomic_replace. Missing file is fine."""
    wait = initial_wait_s
    for i in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError as e:
            if i == attempts - 1:
                raise
            log.warning("unlink %s blocked (%s); retrying in %.1fs",
                        path, e, wait)
            time.sleep(wait)
            wait *= 2


def rewrite_csv_with_replacement(path: Path, columns: list[str],
                                 pk: str, new_row: dict) -> None:
    """Replace the existing row with same pk if any, else append. Atomic via tempfile."""
    pk_value = str(new_row.get(pk))
    tmp = path.with_suffix(path.suffix + ".tmp")
    replaced = False
    with path.open("r", newline="", encoding="utf-8") as src, \
         tmp.open("w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=columns)
        writer.writeheader()
        for row in reader:
            if str(row.get(pk)) == pk_value:
                writer.writerow({k: new_row.get(k) for k in columns})
                replaced = True
            else:
                writer.writerow({k: row.get(k) for k in columns})
        if not replaced:
            writer.writerow({k: new_row.get(k) for k in columns})
        dst.flush()
        os.fsync(dst.fileno())
    _atomic_replace(tmp, path)


def write_staging_trades(condition_id: str, trades: list[dict]) -> Path:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    path = STAGING_DIR / f"{condition_id}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRADES_COLS)
        w.writeheader()
        for t in trades:
            w.writerow(trade_to_row(t))
        f.flush()
        os.fsync(f.fileno())
    return path


def commit_staged_trades(staging_path: Path) -> int:
    """Append the staging file's body (sans header) to trades.csv. Returns row count."""
    ensure_csv_header(TRADES_CSV, TRADES_COLS)
    n = 0
    with staging_path.open("r", newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        rows = list(reader)
        n = len(rows)
        if rows:
            with TRADES_CSV.open("a", newline="", encoding="utf-8") as dst:
                w = csv.DictWriter(dst, fieldnames=TRADES_COLS)
                for r in rows:
                    w.writerow({k: r.get(k) for k in TRADES_COLS})
                dst.flush()
                os.fsync(dst.fileno())
    return n


def append_ledger(condition_id: str, run_started_at: str, status: str,
                  n_pages_seen: int, last_page_size: int,
                  n_trades_written: int, error: str = "") -> None:
    ensure_csv_header(LEDGER_CSV, LEDGER_COLS)
    append_rows(LEDGER_CSV, LEDGER_COLS, [{
        "condition_id": condition_id,
        "run_started_at": run_started_at,
        "status": status,
        "n_pages_seen": n_pages_seen,
        "last_page_size": last_page_size,
        "n_trades_written": n_trades_written,
        "error": error,
    }])


def read_ok_conditions() -> set[str]:
    """Conditions whose latest ledger entry is `ok` or `empty`."""
    if not LEDGER_CSV.exists():
        return set()
    latest: dict[str, str] = {}
    with LEDGER_CSV.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row.get("condition_id") or ""
            latest[cid] = row.get("status") or ""
    return {cid for cid, st in latest.items() if st in ("ok", "empty")}


def cleanup_on_startup() -> None:
    """Recover from a previous crashed run.

    1. Any leftover staging files = uncommitted markets — discard.
    2. Any trades.csv rows whose condition_id is not in the ok-set = orphaned
       partial appends from a crash between commit and ledger-write — strip them.
    """
    if STAGING_DIR.exists():
        leftover = list(STAGING_DIR.glob("*.csv"))
        for p in leftover:
            _retry_unlink(p)
        if leftover:
            log.warning("cleanup: discarded %d leftover staging file(s)", len(leftover))

    if not TRADES_CSV.exists() or TRADES_CSV.stat().st_size == 0:
        return

    ok = read_ok_conditions()
    seen_in_trades: set[str] = set()
    with TRADES_CSV.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row.get("condition_id") or ""
            seen_in_trades.add(cid)
    orphans = seen_in_trades - ok
    if not orphans:
        return

    log.warning("cleanup: %d orphan condition(s) in trades.csv — rewriting",
                len(orphans))
    tmp = TRADES_CSV.with_suffix(TRADES_CSV.suffix + ".tmp")
    kept = 0
    dropped = 0
    with TRADES_CSV.open("r", newline="", encoding="utf-8") as src, \
         tmp.open("w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=TRADES_COLS)
        writer.writeheader()
        for row in reader:
            if (row.get("condition_id") or "") in orphans:
                dropped += 1
            else:
                writer.writerow({k: row.get(k) for k in TRADES_COLS})
                kept += 1
        dst.flush()
        os.fsync(dst.fileno())
    _atomic_replace(tmp, TRADES_CSV)
    log.warning("cleanup: kept %d rows, dropped %d orphan rows", kept, dropped)


def process_market(event: dict, market: dict, run_started: str) -> tuple[int, str]:
    """Fetch trades for one market, write staging → trades.csv → ledger. Returns (n_trades, status)."""
    cid = market.get("conditionId")
    if not cid:
        return 0, "error"

    try:
        trades, pages, last_len = fetch_all_trades(cid)
    except Exception as e:
        log.error("fetch failed for cid=%s: %s", cid, e)
        append_ledger(cid, run_started, "error", 0, 0, 0, error=str(e)[:200])
        return 0, "error"

    if not trades:
        append_ledger(cid, run_started, "empty", pages, last_len, 0)
        return 0, "empty"

    staging = write_staging_trades(cid, trades)
    n_written = commit_staged_trades(staging)
    append_ledger(cid, run_started, "ok", pages, last_len, n_written)
    _retry_unlink(staging)
    return n_written, "ok"


def process_event(event_summary: dict, run_started: str,
                  ok_conditions: set[str]) -> dict:
    """Walk one event: parse, write events row, walk markets. Returns summary stats."""
    slug = event_summary.get("slug") or ""
    full = fetch_event_full(slug)
    if not full:
        log.warning("event %s: full fetch returned nothing — skipping", slug)
        return {"event_slug": slug, "skipped": True, "n_markets": 0, "n_trades": 0}

    markets_all = full.get("markets") or []
    game_markets = [(m, is_game_n_winner(m.get("question")))
                    for m in markets_all]
    game_markets = [(m, n) for m, n in game_markets if n is not None]
    if not game_markets:
        log.info("event %s: no Game N Winner markets — skipping", slug)
        return {"event_slug": slug, "skipped": True, "n_markets": 0, "n_trades": 0}

    team_a, team_b, match_date = parse_slug(slug)
    upsert_event({
        "event_id": full.get("id"),
        "event_slug": slug,
        "event_title": full.get("title"),
        "start_date": full.get("startDate"),
        "end_date": full.get("endDate"),
        "closed": full.get("closed"),
        "volume_usd": full.get("volume"),
        "n_markets_total": len(markets_all),
        "team_a_short": team_a,
        "team_b_short": team_b,
        "match_date": match_date,
        "fetched_at": now_iso(),
    })

    total_trades = 0
    for m, game_n in game_markets:
        cid = m.get("conditionId")
        if not cid:
            continue
        if cid in ok_conditions:
            log.debug("skip cid=%s (already ok)", cid)
            continue

        try:
            outcomes = json.loads(m.get("outcomes") or "[]")
        except (TypeError, json.JSONDecodeError):
            outcomes = []
        try:
            tokens = json.loads(m.get("clobTokenIds") or "[]")
        except (TypeError, json.JSONDecodeError):
            tokens = []

        # Provisional markets row with status=pending; rewritten after fetch.
        market_row_base = {
            "market_id": m.get("id"),
            "condition_id": cid,
            "event_id": full.get("id"),
            "question": m.get("question"),
            "game_number": game_n,
            "outcome_0": outcomes[0] if len(outcomes) > 0 else None,
            "outcome_1": outcomes[1] if len(outcomes) > 1 else None,
            "token_0": tokens[0] if len(tokens) > 0 else None,
            "token_1": tokens[1] if len(tokens) > 1 else None,
            "market_start": m.get("startDate"),
            "market_end": m.get("endDate"),
            "market_closed": m.get("closed"),
            "volume_usd": m.get("volume"),
            "fetched_at": now_iso(),
        }

        n_trades, status = process_market(full, m, run_started)
        total_trades += n_trades
        upsert_market({
            **market_row_base,
            "n_trades_fetched": n_trades,
            "fetch_status": status,
        })
        ok_conditions.add(cid) if status in ("ok", "empty") else None

        log.info("  market cid=%s game=%s status=%s trades=%d",
                 cid, game_n, status, n_trades)
        time.sleep(SLEEP_BETWEEN_MARKETS_S)

    return {"event_slug": slug, "skipped": False,
            "n_markets": len(game_markets), "n_trades": total_trades}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-events", type=int, default=None,
                        help="Stop after processing N events (for smoke testing).")
    parser.add_argument("--vol-floor", type=float, default=EVENT_VOL_FLOOR,
                        help=f"Drop events with lifetime volume <= this (default {EVENT_VOL_FLOOR}).")
    parser.add_argument("--refresh-inventory", action="store_true",
                        help="Force a fresh inventory pull, ignoring the cache.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    run_started = now_iso()
    log.info("run started %s", run_started)

    cleanup_on_startup()

    log.info("listing settled LoL events ...")
    events = list_settled_lol_events(use_cache=not args.refresh_inventory)
    log.info("inventory: %d settled events found", len(events))

    kept = [e for e in events if (float(e.get("volume") or 0) > args.vol_floor)]
    log.info("inventory: %d events after volume floor > $%.0f",
             len(kept), args.vol_floor)
    kept.sort(key=lambda e: e.get("endDate") or "", reverse=True)

    if args.limit_events is not None:
        kept = kept[:args.limit_events]
        log.info("limit: processing first %d events", len(kept))

    ok_conditions = read_ok_conditions()
    log.info("resume: %d condition(s) already marked ok/empty", len(ok_conditions))

    n_events_done = 0
    n_events_skipped = 0
    n_markets_done = 0
    n_trades_done = 0

    for i, e in enumerate(kept, 1):
        log.info("[%d/%d] event %s (vol=$%.0f, end=%s)",
                 i, len(kept), e.get("slug"), float(e.get("volume") or 0),
                 (e.get("endDate") or "")[:10])
        try:
            summary = process_event(e, run_started, ok_conditions)
        except Exception as exc:
            log.exception("event %s: unhandled error: %s", e.get("slug"), exc)
            continue
        if summary["skipped"]:
            n_events_skipped += 1
        else:
            n_events_done += 1
            n_markets_done += summary["n_markets"]
            n_trades_done += summary["n_trades"]

    log.info("DONE: events_processed=%d events_skipped=%d markets=%d trades=%d",
             n_events_done, n_events_skipped, n_markets_done, n_trades_done)
    return 0


if __name__ == "__main__":
    sys.exit(main())

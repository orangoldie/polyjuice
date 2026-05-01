# Spec: Polymarket LoL trade-history dump

Status: implemented and verified
Date: 2026-04-29 (drafted), 2026-04-30 (implemented and aligned post-run)

## Objective

Produce a local, idempotent raw-data dump of every Polymarket fill on every settled "Game N Winner" market across all LoL events in the Gamma inventory, written to three CSVs (`events.csv`, `markets.csv`, `trades.csv`). The dump is the foundation for later edge / calibration / backtest work — this spec covers acquisition only. Joining to Oracle's Elixir games, computing derived `p_team_X` columns, and any modeling work are out of scope.

## Background

- Polymarket exposes a Gamma events API (`gamma-api.polymarket.com/events`) and a data-API trades endpoint (`data-api.polymarket.com/trades?market={conditionId}&limit=500&offset=...`) that paginates per-fill history. Both endpoints are public and unauthenticated.
- Inventory probe (`scripts/exploration/29_polymarket_event_inventory.py`) found ~1,104 LoL events under `series_slug=league-of-legends`, of which ~976 are settled with lifetime volume > $100. Each event carries 20–40 sub-markets ("Game N Winner", "Match Winner", handicaps, totals, etc.). We want only the `Game N Winner` markets.
- A spike on one event (`scripts/exploration/30_polymarket_threshold_anchor.py`) confirmed pagination works as expected and returns per-fill rows with `{timestamp, price, size, side, asset, transactionHash, ...}`.
- Trade-count probe (`scripts/exploration/31_polymarket_g1_trade_counts.py`) sampled 12 recent settled events: G1-winner trade counts ranged from 7 to ~2,900 fills (median ~1.3k for tier-1 matchups, tens-to-low-hundreds for regional games). Total corpus is on the order of low millions of fills across all G1+ markets and all events.

## Scope

**In scope:**
- All Polymarket events with `series_slug=league-of-legends` and `closed=true`.
- Drop "junk" events: lifetime event-level `volume <= $100`.
- Within each kept event, every market whose `question` matches `Game N Winner` for any integer N — all games in the series, not just G1. Captures G1, G2, G3, G4, G5 alike.
- Fetch every trade for each kept market via paginated `data-api.polymarket.com/trades`.

**Out of scope (this spec):**
- Open / live events (re-run later for forward testing).
- "Match Winner" / handicap / totals / first-blood markets.
- `prices-history` minute curves (separate dump if needed).
- On-chain/block-level enrichment.
- Derived columns like `p_outcome_0` or `p_team_X`.
- Team alias resolution / joining to Oracle's Elixir games.
- Parquet / partitioned storage. Single CSV per data type for v1.

## Output schema

Three CSVs under `data/polymarket/`:

### `events.csv` — one row per kept event
| column | source | notes |
|---|---|---|
| `event_id` | `events[].id` | PK |
| `event_slug` | `events[].slug` | e.g. `lol-we-blg-2026-04-29` |
| `event_title` | `events[].title` | |
| `start_date` | `events[].startDate` | ISO8601 UTC |
| `end_date` | `events[].endDate` | ISO8601 UTC |
| `closed` | `events[].closed` | always `true` for v1 |
| `volume_usd` | `events[].volume` | event-level lifetime volume |
| `n_markets_total` | `len(events[].markets)` | all sub-markets, not just G winners |
| `team_a_short` | parsed from slug | 2nd token of `lol-{a}-{b}-{date}` |
| `team_b_short` | parsed from slug | 3rd token; alias resolution deferred |
| `match_date` | parsed from slug | last token, `YYYY-MM-DD` |
| `fetched_at` | now() | ISO8601 UTC, when this row was written |

### `markets.csv` — one row per kept "Game N Winner" market
| column | source | notes |
|---|---|---|
| `market_id` | `markets[].id` | PK |
| `condition_id` | `markets[].conditionId` | used as the trades-API key |
| `event_id` | parent event | FK to `events.csv` |
| `question` | `markets[].question` | |
| `game_number` | regex from `question` | 1, 2, 3, ... |
| `outcome_0` | `markets[].outcomes[0]` | team name as Polymarket has it |
| `outcome_1` | `markets[].outcomes[1]` | |
| `token_0` | `markets[].clobTokenIds[0]` | |
| `token_1` | `markets[].clobTokenIds[1]` | |
| `market_start` | `markets[].startDate` | |
| `market_end` | `markets[].endDate` | |
| `market_closed` | `markets[].closed` | |
| `volume_usd` | `markets[].volume` | NB: zeroed-out post-settlement by API for many markets — keep raw value |
| `n_trades_fetched` | computed | number of fill rows written for this market in this run |
| `fetch_status` | enum | `ok` / `empty` / `error` (see "Idempotency") |
| `fetched_at` | now() | ISO8601 UTC |

### `trades.csv` — one row per fill
| column | source | notes |
|---|---|---|
| `condition_id` | parent market | FK to `markets.csv` |
| `timestamp_unix` | `trades[].timestamp` | unix seconds, UTC |
| `price` | `trades[].price` | float in (0,1) |
| `size` | `trades[].size` | shares, $1 notional each |
| `side` | `trades[].side` | `BUY` / `SELL` |
| `asset` | `trades[].asset` | clobTokenId — 77-digit uint256 string. **Loaders must read as string** (Excel/pandas auto-coerce silently destroys precision; load with `dtype={"asset": str, "condition_id": str, "transaction_hash": str, "proxy_wallet": str}`). |
| `outcome_index` | `trades[].outcomeIndex` | 0 or 1 — added during implementation; the API returns it for free, saves a token_0/token_1 lookup downstream |
| `transaction_hash` | `trades[].transactionHash` | Polygon tx |
| `proxy_wallet` | `trades[].proxyWallet` if present | nullable |

Every other field returned by the trades endpoint is dropped for v1 to keep the file lean. If a needed field surfaces later, it can be added to the schema and the affected markets re-fetched (cheap — see idempotency).

## Idempotency strategy

A long run *will* fail mid-flight (network, transient 5xx, rate-limit). We need to be able to re-run and pick up where we left off, **without** falsely marking an incomplete market as done.

Approach:

1. **Per-market progress is recorded in a separate ledger file**, not inferred from `trades.csv` row counts. Inferring "done" from "rows exist for condition_id X" is unsafe — a partially-paginated market also has rows.
2. Ledger: `data/polymarket/_fetch_ledger.csv` with one row per `(condition_id, run_started_at, status, n_pages_seen, last_page_size, n_trades_written, error)`. Append-only.
3. Status semantics, set only at the very end of a market's fetch:
   - `ok` — pagination terminated by a short page (`len < 500`) **and** all fills were successfully appended to `trades.csv`.
   - `empty` — first page returned `[]`. Considered complete.
   - `error` — any exception, HTTP non-2xx, or the loop hit `max_pages` without a short page (suggesting a cap or infinite loop). Not considered complete.
4. On startup the script reads the ledger, computes the set of `condition_id` with at least one `ok` or `empty` status, and skips those. Markets last seen as `error` are retried.
5. **Crash safety while writing trades:** the implementation uses a per-market staging file under `data/polymarket/_staging/{condition_id}.csv`. Flow per market: fetch all pages into memory → write staging file (flush + fsync) → append staging body to `trades.csv` (flush + fsync) → append `ok` ledger row (flush + fsync) → unlink staging. The append-to-trades and ledger-append happen sequentially, not atomically; the residual crash window is closed by **cleanup-on-startup**: any leftover staging files are deleted, and any `condition_id` present in `trades.csv` without a matching `ok`/`empty` ledger entry has its rows stripped. This makes the dataset eventually-consistent across crashes.
6. **Best-effort metadata writes.** `upsert_event` / `upsert_market` rewrite the events/markets CSV via tempfile + `os.replace`. On Windows this can fail intermittently with `WinError 32` (file in use by AV/indexer/IDE) or `WinError 2` (tmp quarantined). The implementation retries with backoff and, if all retries fail, logs a warning and continues — a metadata-write hiccup must not abort trade fetching, since trades are the irreplaceable bit and the ledger is the authoritative completion record. Any missed metadata rows are picked up on the next resume.
7. **Single-process discipline is required.** The script holds no lockfile. Two concurrent processes will race on the staging files, ledger appends, and tmp paths, and corrupt the dataset (observed during initial implementation). Always confirm there is exactly one `python.exe` running.
8. Closed markets are immutable (no new fills after settlement), so once `ok` they never need refresh.

Two consequences worth calling out:
- A market with millions of fills could blow memory. Not a real concern for LoL (largest observed = ~3k fills ≈ tens of KB), but if we ever point this at a high-volume market we'd switch to per-page append + a `pages_completed` counter in the ledger and resume from offset.
- `events.csv` and `markets.csv` rows are written *before* trade fetching begins for that event. If a market's trade fetch errors, its `markets.csv` row still exists with `fetch_status=error`. Re-runs upsert (overwrite by `market_id`) so the row gets corrected to `ok` on success.

## Run mechanics

- Inventory source: pull via `gamma-api.polymarket.com/events?series_slug=league-of-legends&closed=true&limit=200&offset=...`. The Gamma list endpoint returns intermittent 500s on deep offsets (1000+) under filtered queries — server-side, not our problem to solve, but the retry budget needs to handle it. The implementation caches the result to `data/polymarket/_events_inventory.json` with a 12-hour TTL so resumes don't re-pay the inventory cost; pass `--refresh-inventory` to bust the cache.
- Filter: `volume > 100` at event level (override via `--vol-floor`).
- Per-event: pull full event JSON via `?slug={slug}` (the inventory list omits the `markets[]` array), filter to those whose question matches `r"\bgame\s+(\d+)\s+winner\b"` (case-insensitive, substring search). The actual question text is `LoL: <Team A> vs <Team B> - Game N Winner`, so a substring match — not a prefix match — is correct.
- Per-market: paginate trades with `limit=500`, advance offset by page until a short page or empty page or `max_pages` (set to 200 = 100k fills, far above anything expected).
- Throttling: 100ms sleep between trade pages, 250ms between markets. Exponential backoff on 429/5xx with up to 8 retries (bumped from spec's original 5 to absorb the deep-offset Gamma 500s).
- HTTP 400s from `data-api.polymarket.com/trades` are treated as terminal (no retry) — they signal a Polymarket-side data inconsistency for that specific `conditionId`, not a transient failure.
- Logging: one INFO line per market (`event_slug, condition_id, n_trades, status`), one summary line per event, one final summary at end of run.
- Parsing: extract `team_a_short`, `team_b_short`, `match_date` from slug with `r"^lol-([a-z0-9]+)-([a-z0-9]+)-(\d{4}-\d{2}-\d{2})$"`. Slugs that don't match get NULLs for all three columns and are still kept — alias resolution is deferred anyway. (Observed in practice: 0/880 slugs failed to match for settled vol>$100 events; the regex covers everything.)
- CLI flags: `--limit-events N` (smoke testing), `--vol-floor F`, `--refresh-inventory`, `--verbose`.

## Code layout

- `scripts/polymarket/dump_trades.py` — main entry point (CLI, reads no config, writes the three CSVs + ledger).
- `scripts/polymarket/verify_dump.py` — diagnostic. Streams the CSVs (never reads them whole into memory), prints a one-screen summary covering row/PK counts, duplicate detection, referential integrity, trade-count agreement between markets.csv and trades.csv, ledger status breakdown, and one live API spot-check on a random `ok` market. Run after every dump or whenever the dataset's integrity is in question.
- `scripts/polymarket/__init__.py`
- Reuses the `fetch_all_trades` pattern from `scripts/exploration/30_polymarket_threshold_anchor.py:18-46`, modified to (a) accept a max_pages guard that distinguishes "terminated by short page" from "hit cap", and (b) raise on any HTTP error rather than printing and continuing.

## Acceptance check (post-run)

Verified by `verify_dump.py`. The post-implementation run achieved all green:

| Check | Pass criterion | Actual |
|---|---|---|
| events.csv | settled vol>$100 events count | 880 |
| markets.csv | 1×–5× event count | 2,038 (≈2.3×) |
| trades.csv | per-market local count == live API count on a random sample | 26 == 26, delta 0 |
| Duplicate fill rows | 0 | 0 |
| Trade-count mismatches (markets.csv vs trades.csv) | 0 | 0 |
| Orphan condition_ids in trades.csv | 0 | 0 |
| markets→events referential integrity | 0 mismatches | 0 |
| Latest-status ledger breakdown | only `ok`/`empty` | 2,028 ok + 10 empty |
| Re-run with all CSVs in place is a near-no-op | exits in seconds, 0 fetches | confirmed |
| Surgical re-fetch (delete one cid's trades + ledger row → only that cid is refetched) | confirmed | confirmed |

## Known gaps

- **9 markets that consistently 400'd from `data-api.polymarket.com/trades`** (their parent 6 events) were dropped from the dataset and recorded as a known limitation. These are server-side data inconsistencies on Polymarket — not transient — so retrying gains nothing. Total impact: 0.4% of markets, dropped cleanly along with their parent events.
- **10 `Game N Winner` markets are genuinely empty** (zero fills in their lifetime). These are unplayed Game 4/5 markets in BO5 series that ended early (3-0, 3-1). They're kept in `markets.csv` with `fetch_status=empty` for completeness; downstream consumers should filter on `fetch_status=ok` if they only want markets with actual trade data.
- **Game-number distribution** in markets.csv reflects BO-format reality: 879 G1, 879 G2, 174 G3, 106 G4. ~1k 5-game series didn't reach G3+, consistent with LoL being BO3-heavy outside of finals/Worlds.

## Future work (not this spec)

- **Sudden-drop / mean-reversion edge detection.** Per-fill granularity (vs minute buckets) is what makes this tractable: an ML model trained on short windows of fill-level features (price, size, side, recency, microstructure of the last K fills) could flag transient mispricings — e.g. a thin book getting walked by one impatient seller, with price snapping back over the next minute. The fill-level dump is the prerequisite dataset for that line of work.
- Add `prices-history` per market (separate file).
- Re-run periodically against `closed=false` events for forward / live tracking.
- Build the team-alias map from `team_a_short` / `team_b_short` values — likely heavy overlap with `data/gpr/team_aliases.csv`.
- Convert to Parquet partitioned by `match_date` year-month if scan time gets painful.
- Join to per-game Oracle's Elixir rows to enable model-vs-market edge computation.

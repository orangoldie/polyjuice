# Chat Summary — 2026-04-16 (updated)

## Project: Polyjuice
Statistical arbitrage on Polymarket for LoL esports. Build a win-probability model from historical data, compare to Polymarket odds, bet when edge exceeds threshold.

## Files Created

**Project docs:**
- `CLAUDE.md` — project context for Claude (auto-loaded each conversation)
- `discovery-checklist.md` — all open questions, exploration tasks, and decisions

**Scripts (all in `scripts/`):**
- `01_row_counts.py` — games per year and league
- `02_null_rates_and_atakhan.py` — null rates for key columns + atakhan presence
- `03_side_winrate.py` — Blue/Red side win rates
- `04_data_integrity.py` — data validation (12 rows/game, mirroring, aggregation)
- `05_distinct_leagues.py` — all 121 league names with game counts and year spans
- `06_scrape_liquipedia_tiers.py` — scrapes Liquipedia S/A/B/C/D tier pages via MediaWiki API
- `07_match_leagues_to_tiers.py` — matches our leagues to Liquipedia tiers (exact + aliases + direct)

**Outputs (all in `data/exploration/`):**
- `01_row_counts.json`, `02_null_rates.json`, `02_atakhan.json`, `03_side_winrate.json`, `04_data_integrity.json`
- `05_distinct_leagues.json`, `06_liquipedia_tiers.json`, `07_league_tier_matching.json`

## Key Findings

1. **~85K complete games** across 2014-2026, 121 leagues. LPL loses ~68% of data after complete-only filter.
2. **All scripts filter to `datacompleteness == "complete"`** — this gives 0% nulls on 10/15 min columns.
3. **25-min columns** have 3-10% nulls (games ending before 25 min — expected).
4. **Atakhan is 2025-only** — added in patch 25.S1.1, removed in patch 26.1.
5. **void_grubs from 2024, heralds from 2016** — newer features limit training years if included.
6. **Blue side wins 52-58%** consistently across all years and leagues. Real signal.
7. **Data integrity is solid** — only 2 games with no winner (filter out), and ~2K games with fuzzy 25-min checkpoint boundary.
8. **Null vs 0 matters** for features that didn't exist in older years — noted for pre-processing pipeline design.
9. **League tier mapping**: using Liquipedia S/A/B classification. 70/121 leagues matched (80.7% of games). S-tier: major pro leagues + internationals (20 leagues, 21K games). A-tier: minor pro + development (28 leagues, 28K games). B-tier: ERLs/national leagues (22 leagues, 20K games). 51 leagues unmatched (~16K games) — mostly LATAM nationals, academy leagues, small events. No official Riot tier system exists; Liquipedia's is community-maintained.
10. **TCL classified as B-tier** (appears in both A and B on Liquipedia, we chose B).

## Decisions Made
- Filter to `datacompleteness == "complete"` everywhere
- Inline results per exploration task (not a separate findings section)
- No tier classification until we check Polymarket coverage

## Remaining Exploration Tasks
- [ ] Determine minimum useful starting year
- [ ] Check which tournaments/leagues Polymarket covers (manual)
- [ ] Team name consistency across years
- [ ] Resolve remaining 51 unmatched leagues if needed (LATAM, academy, unknowns)

## Open Design Questions (Sections 1-3 in checklist)
Columns/time-frames, team vs player rows, champion/draft data, patch handling, Elo prior, roster changes, target variable definition, one model vs per-tier, baseline model, XGBoost vs alternatives, train/test split, calibration, Polymarket integration, live API, bankroll management — all still open.

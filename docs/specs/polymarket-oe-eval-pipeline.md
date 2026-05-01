# Spec: Polymarket → Oracle's Elixir eval pipeline

Status: implemented (retroactive spec — written 2026-05-01 covering work done 2026-04-30 → 2026-05-01)
Date: 2026-05-01

## Objective

Take the historical Polymarket "Game N Winner" trade dump and join it to Oracle's Elixir (OE) per-game data so that, for every settled market, we can read the unified market price and the production GPR model's win-probability at each modeling checkpoint (T = 10 / 15 / 20 / 25 in-game minutes). The deliverable is a single eval table at `data/polymarket/eval_table.csv` that downstream calibration / edge-detection work consumes.

This spec covers the join + eval-table build only. Calibration of the implied edge, leakage verification, and live-trading readiness are out of scope (open questions are listed under "Known gaps").

## Background

Inputs already in place at the start of this work:

- `data/polymarket/{events,markets,trades}.csv` — produced by `scripts/polymarket/dump_trades.py` per `polymarket-trade-dump.md`. ~2,000 ok markets, millions of fills.
- `data/lol-historical-data/*.csv` — Oracle's Elixir per-game CSVs, 2014–2026.
- `data/processed/checkpoint_{10,15,20,25}_gpr.csv` — per-checkpoint feature tables produced by `scripts/build_dataset.py`, used by the production model.
- `data/models/baseline_logistic_gpr/production/checkpoint_{T}/{model.pkl,calibrator.pkl,metadata.json}` — production GPR model artifacts per `baseline-logistic.md` + `gpr-feature.md`.

Open question: for past games where we know the OE truth and the model can re-score, what did the Polymarket market price say at each checkpoint? This is the dataset that lets us measure model-vs-market edge.

## Scope

**In scope:**
- Resolve Polymarket team-name strings (free-text `outcome_0` / `outcome_1`) to canonical OE `teamname` values.
- Join each `fetch_status=ok` Polymarket Game-N-Winner market to a unique OE `gameid`.
- For each (market, checkpoint) pair where the join succeeded, attach: market price at the checkpoint, GPR model probability at the checkpoint, OE realized result, and diagnostics (staleness, fill density, gamelength).
- Write a flat eval table.

**Out of scope (this spec):**
- Edge calibration / reliability of the model-vs-market signal (next spec).
- Train/eval leakage verification (assumed clean — model trained on ≤2025, eval set is ~85% 2026 — but not formally checked here).
- Live-data feature path (livestats integration, see `livestats-firsts-timing.md`).
- Order-book depth / executable price / slippage modeling.
- Bet sizing, fees, costs.

## Pipeline

Three stages, all run from project root.

### Stage 1: build the alias map

`scripts/polymarket/build_alias_map.py` → `data/polymarket/team_aliases.csv`

Polymarket carries free-text team names in `markets.outcome_0` / `outcome_1` (e.g. `"Gen.G"`, `"FURIA Esports"`). OE uses canonical `teamname` strings (e.g. `"Gen.G"`, `"FURIA"`). The alias map is the lookup that bridges the two.

Strategy:
1. Pull every distinct Polymarket name across all `fetch_status=ok` markets (218 distinct names as of 2026-05-01).
2. Normalize both sides (lowercase + alnum-only) and exact-match against the union of OE `teamname` values from 2023–2026 — resolves 198/218.
3. The remaining 20 names are handled by two hardcoded dicts inside the script:
   - `MANUAL_OVERRIDES` — 16 fuzzy resolutions (e.g. `"FURIA Esports"` → `"FURIA"`, `"BIG"` → `"Berlin International Gaming"`). Curated by inspection via the review TSV from script 36 (see "Probes" below).
   - `DROPPED` — 4 names left intentionally unmapped because the right answer is genuinely uncertain (`{"DN Freecs", "OKSavingsBank BRION", "GIANTX PRIDE", "devils.one x KMT"}`). These will surface as `team_unmapped` misses downstream.
4. Output schema: `pm_name`, `oe_teamname`, `source` (`exact` | `manual`).

The script is idempotent: rerunning produces the same file. To revise the map, edit `MANUAL_OVERRIDES` / `DROPPED` and re-run.

### Stage 2: join markets to OE gameids

`scripts/polymarket/join_oe.py` → `data/polymarket/markets_oe_joined.csv`

For every `fetch_status=ok` market, look up the OE gameid by:

1. Resolve `outcome_0` / `outcome_1` via `team_aliases.csv` to OE teamnames.
2. Build a key `(match_date, frozenset({oe_team_a, oe_team_b}), game_number)` and look it up in an OE index over team rows from 2022–2026.
3. If exactly one OE gameid matches → `status=ok`.
4. Otherwise → `status=miss` with one of these `miss_reason` buckets:
   - `team_unmapped:<names>` — alias map didn't resolve one or both teams.
   - `oe_pair_present_but_game_num_off_have=[X]_want=Y` — series exists but at a different game number.
   - `oe_neither_team_on_date` / `oe_missing_team:<name>` — OE has no record of the team on that date.
   - `oe_both_teams_played_but_not_each_other` — both teams had matches that day, but not against each other.
   - `oe_ambiguous_N` — multiple OE gameids match the key.
   - `pm_missing_date_or_gamenum` — Polymarket row missing the join key.

Output columns: `condition_id`, `event_id`, `event_slug`, `match_date`, `game_number`, `pm_outcome_0`, `pm_outcome_1`, `oe_team_a`, `oe_team_b`, `oe_gameid`, `oe_league`, `status`, `miss_reason`.

### Stage 3: build the eval table

`scripts/polymarket/build_eval_table.py` → `data/polymarket/eval_table.csv`

For every joined `status=ok` market and every checkpoint T ∈ {10, 15, 20, 25}, emit one row if all of the following hold:

- The checkpoint is reachable: `T*60 < gamelength_s`.
- There is at least one Polymarket fill at-or-before `oe_date + T*60` in the trades stream.
- The most recent such fill is no more than `MAX_STALENESS_S = 5 minutes` old.
- The production GPR model has a probability for `(checkpoint=T, gameid, teamname=oe_team_a)` — i.e. the gameid is present in `data/processed/checkpoint_T_gpr.csv`.

`team_a` is defined as the team mapped from `outcome_0` (so `market_p_team_a` is the unified P(outcome_0 wins) and `realized_team_a_won` is OE `result` for that team).

Market price at the checkpoint is computed by streaming `trades.csv` in chunks (the file is too big to load whole), filtering to our cids, and unifying onto outcome_0: `p0 = price if outcome_index == 0 else 1 - price`. The eval price is the `p0` of the last fill at-or-before checkpoint time.

Model probability at the checkpoint is computed by loading the per-checkpoint scaler / model / calibrator / mean-imputer payload, slicing `checkpoint_T_gpr.csv` to our joined gameids, calling `prepare_Xy` + `predict_proba` from `scripts.models.baseline_logistic_gpr`, then applying the saved calibrator if present.

## Output schema: `data/polymarket/eval_table.csv`

| column | type | notes |
|---|---|---|
| `oe_gameid` | str | OE primary key |
| `condition_id` | str | Polymarket market PK (FK to `markets.csv`) |
| `event_slug` | str | for human-readable spot checks |
| `league` | str | OE league code |
| `checkpoint` | int | one of {10, 15, 20, 25} |
| `checkpoint_t_unix` | int | absolute unix seconds at the checkpoint |
| `oe_team_a`, `oe_team_b` | str | canonical OE teamnames; team_a := mapped from `outcome_0` |
| `market_p_team_a` | float | unified P(team_a wins) at the checkpoint |
| `staleness_s` | int | seconds between checkpoint and the eval-fill timestamp (≤ 300) |
| `n_fills_pm60s` | int | fill density: fills in [checkpoint − 60s, checkpoint + 60s] (diagnostic only) |
| `p_model_team_a` | float | calibrated GPR model probability for team_a at this checkpoint |
| `p_model_calibrator` | str | the calibrator kind from metadata.json (or `""` if none) |
| `realized_team_a_won` | int | 0/1 from OE `result` for team_a |
| `gamelength_s` | int | for filtering / sanity |

## Probes / supporting scripts

These ran during discovery and answered the questions that shaped the pipeline. They live in `scripts/exploration/` and write summaries to `data/exploration/`.

| Script | Question answered | Key finding |
|---|---|---|
| `33_pmoe_oe_timezone_check.py` | What timezone is OE's `date` column in? | UTC. Per-league hour-of-day distribution matches the expected UTC mapping for LCK/LEC/LCS/LPL kickoff times. |
| `34_pmoe_shortcode_inventory_obsolete.py` | Can we join via the 2–3 letter slug shortcodes alone? | No — too ambiguous (e.g. `genga` maps to both `Gen.G` and `Oxygen Gaming`). Kept as `_obsolete` for the record. Drove the decision to use the full team-name strings from `outcome_0/1` instead. |
| `35_pmoe_outcome_inventory.py` | Are `outcome_0` / `outcome_1` full team names that we can match? | Yes. 218 distinct values, mostly recognizable team names, large overlap with OE on exact normalized match. |
| `36_pmoe_alias_map_review.py` | Build a fuzzy-match TSV for human review of the names that didn't exact-match. | Output: `data/exploration/36_polymarket_oe_alias_map.tsv`. The reviewer's resolutions were transcribed into `MANUAL_OVERRIDES` / `DROPPED` in `build_alias_map.py`. |
| `37_pmoe_oe_team_lookup.py` | Quick OE-side lookup tool for resolving fuzzy candidates. | Used while curating the manual overrides. |
| `40_pmoe_spot_check.py` | Are joins correct? Pick 2 random `status=ok` rows and print both sides side-by-side. | Confirmed: dates, teams, and game numbers line up. |
| `41_pmoe_oe_game_column_check.py` | Does OE's `game` column reliably number 1..N within a series? | Yes. Drove the decision to use `(date, frozenset(teams), game)` as the join key. |
| `42_pmoe_date_anchor_check.py` | Is OE `date` actually the in-game kickoff timestamp? | Yes (within seconds). On 8 high-volume markets, `oe_date + gamelength` lands within ~30s of the first market price-pin (resolution moment) for 6/8; the other two are off by ~2 minutes — close enough that "OE date is the kickoff time" is the right model. This is what makes `checkpoint_t_unix = oe_date + T*60` valid. |
| `44_pmoe_no_model_prob_diag.py` | Why are ~840 joined gameids missing model probabilities per checkpoint? | They live in lower-tier leagues that the preprocessing pipeline filtered out. Dropped from eval — not a bug, just out of model scope. |

The two scripts that actually feed the spec inputs (numbered 31, 32) live in `scripts/exploration/` and are documented elsewhere (script 31's findings are in `polymarket-trade-dump.md`; script 32 is its own spec, `livestats-firsts-timing.md`).

## Run mechanics

All three stages run from project root and are idempotent on stable input data:

```
python scripts/polymarket/build_alias_map.py     # writes team_aliases.csv
python scripts/polymarket/join_oe.py             # writes markets_oe_joined.csv
python scripts/polymarket/build_eval_table.py    # writes eval_table.csv
```

Re-running stage 1 or 2 with the same inputs produces the same output. Stage 3 is sensitive to model artifact changes — refresh after retraining.

No CLI flags. The scripts are intentionally stupid: hardcoded paths, hardcoded year ranges (`2022–2026` for OE, `2023–2026` for the alias map), no config. If you want to change the year window, edit the constant at the top of the script.

Runtime: stage 1 ≈ 30s (loads OE CSVs), stage 2 ≈ 60s (loads OE + builds index + iterates markets), stage 3 ≈ 2–3 min (loads 4 model checkpoints + streams `trades.csv`).

## Code layout

- `scripts/polymarket/build_alias_map.py` — stage 1.
- `scripts/polymarket/join_oe.py` — stage 2.
- `scripts/polymarket/build_eval_table.py` — stage 3. Imports `prepare_Xy`, `predict_proba`, `numeric_cols_in` from `scripts.models.baseline_logistic_gpr`.
- `scripts/exploration/3{3,4,5,6,7}_pmoe_*.py`, `4{0,1,2,4}_pmoe_*.py` — supporting probes (table above).
- `scripts/exploration/_tee.py` — small helper that tees stdout to `data/exploration/<script>.txt` so probe output is captured.

## Acceptance / current numbers

Run on 2026-05-01:

| Stage | Output | Numbers |
|---|---|---|
| 1 | `team_aliases.csv` | 214 names mapped (198 exact + 16 manual), 4 dropped, 0 unresolved |
| 2 | `markets_oe_joined.csv` | 1,832 ok / 2,028 total markets (90.3% join rate). Misses dominated by `team_unmapped` (intentional drops + early-2022 OE gap) and `oe_neither_team_on_date` (mostly community / amateur events not in OE). |
| 3 | `eval_table.csv` | ~5–6k rows across 4 checkpoints. Dominant skip reasons: `no_model_prob_T*` (~3.3k total — lower-tier leagues outside model scope), `too_stale_T*` (~660 — sparse early-game order books), `no_fill_before_T*` (~475 — markets that didn't open fills before the checkpoint). |

Edge distribution (sanity, not a verdict): mean |edge| ≈ 0.12, ~63% of rows have |edge| > 0.05, ~43% > 0.10. **This is not yet evidence of profitable edge — see "Known gaps".**

## Known gaps

- **No reliability check on the edge signal.** We have model-calibration on the original holdout (per `baseline-logistic.md`), but we have not bucketed `(p_model − p_market)` and checked whether the realized win-rate moves with the predicted edge. Until that's done, the edge percentiles above are descriptive, not actionable.
- **No formal leakage check.** The production GPR model was trained on data through 2025; the eval set is mostly 2026 markets, so leakage is unlikely. Not formally verified — no script asserts every eval `gameid` is post the model's training cutoff.
- **OE-date precision.** Script 42 found 6/8 high-volume markets have `oe_date + gamelength` matching the resolution pin within ~30s, but 2/8 were off by ~2 minutes. If those offsets are systematic (vs. specific to which fill was picked as "first pin"), `checkpoint_t_unix` could be biased by tens of seconds, which matters most at T=10 where books are thinnest.
- **`market_p_team_a` is last-fill mid, not executable price.** No spread / depth / slippage. A live trading layer would need the order book.
- **Coverage:** ~840 joined gameids per checkpoint are dropped because their league isn't in the modeling pipeline (see `44_pmoe_no_model_prob_diag.txt` for the breakdown). Not a bug — those leagues were filtered upstream — but it limits the bettable universe to the leagues the model was trained on.
- **9 markets dropped at the trade-dump stage** (Polymarket-side data inconsistencies, see `polymarket-trade-dump.md` known gaps) carry through here.

## Future work (not this spec)

- Reliability diagram of `(p_model − p_market)` vs realized win-rate, bucketed by checkpoint and league. This is the calibration step that decides whether the edge is real.
- Sell-side framing: the user's actual goal is not to predict the winner but to capture price drift — for each row, also compute `market_p_team_a` at later checkpoints (15 → 20 → 25) and look at the distribution of realized price moves conditional on the edge sign at the entry checkpoint.
- Live-readiness checklist: livestats coverage of the GPR feature set, live game ↔ market matching, order-book depth, fees, sizing.

# Chat Summary — full preprocessing pipeline live (2026-04-20)

## Project
Statistical arbitrage on Polymarket for LoL esports. Build a win-probability model from historical data, compare to Polymarket odds, bet when edge exceeds threshold.

## Status
**Discovery complete across all 5 sections of `discovery-checklist.md`.** Full preprocessing pipeline (stages 1–4 + `scripts/build_dataset.py` orchestrator) implemented TDD-style with full test coverage + run-green on real data. **Next: model training / experiment harness.**

Current datasets:
- `data/processed/03_features.csv` — 46,972 games × 63 columns (post-stage 3).
- `data/processed/checkpoint_{10,15,20,25}.csv` — 46,972 / 46,971 / 46,865 / 44,118 rows (post-stage 4).

Column order at stage 3: `gameid, date, year, league, patch, teamname, opp_teamname, side, tier, <checkpoint XatY group>, firstblood..firstbaron, gamelength, result`.

## v1 shape
- **v1 = in-game checkpoint model** at 10/15/20/25 min. v2 = pre-game / Elo / draft.
- **Training window:** 2020-2025; 2026 held out as test set.
- **Live data source:** unofficial lolesports livestats API (verified working).
- **Model:** XGBoost, per-checkpoint (4 models), unified across tiers with `tier` as a feature. Logistic regression baseline.
- **Manual execution for v1:** model emits edge signal, user places bet on polymarket.com.

---

## Section 1 — data & features

### Row filters (locked)
- `datacompleteness == "complete"` only.
- Drop 2 no-winner games (integrity check identified them).
- Drop games without a populated checkpoint frame at stage 4 (any `*at{T}` stat null). Replaces the earlier `gamelength < T*60` plan — Oracle's frame capture doesn't perfectly align with gamelength near the boundary: ~1.25K games at T=25 end ~24:30-24:59 but have a populated `*at25` snapshot (valid — kept), and a handful of games end at exactly T*60 with no frame (invalid — dropped).
- Drop unmapped-tier games (~16K, mostly LATAM nationals/academy). Re-open if v1 needs more data.
- Drop games with a null `teamname` on either team row (~900 games, mostly LCK Challengers / LDL / academy) — needed to derive `opp_teamname` at stage 2.
- Drop academy self-match games where both team rows share a `teamname` (8 gameids, LAS 2020-21) — outcome is structurally meaningless.
- Drop games with any null first-event flag (`firstblood` / `firstdragon` / `firsttower` / `firstbaron`) on either team row (~230 games, all in data-quality-flagged leagues: LPL 2020-21, LDL). Games are long enough that the nulls mean "data missing" not "event never happened", so treating them as unresolved would be a train/inference mismatch.
- Year range 2020-2026 loaded; 2020-2025 train/dev, 2026 held out.

### v1 feature list (locked)

**A. Game context (constant across checkpoints):** `side`, `tier` (S/A/B from Liquipedia mapping).

**B. Checkpoint stats at minute T (T ∈ {10, 15, 20, 25}) — raw + opp + diff for each:**
- Gold: `goldat{T}`, `opp_goldat{T}`, `golddiffat{T}` (CSV).
- CS: `csat{T}`, `opp_csat{T}`, `csdiffat{T}` (CSV).
- Kills: `killsat{T}`, `opp_killsat{T}`, `killdiffat{T}` (engineered).
- Assists: `assistsat{T}`, `opp_assistsat{T}`, `assistdiffat{T}` (engineered).

**C. First-event flags (per A3 gates — strict checkpoint leakage control):**
- `firstblood` at all checkpoints.
- `firstdragon`, `firsttower` at 15/20/25 (~7% residual train/inference mismatch at 15-min accepted for signal gain).
- `firstbaron` at 25 only.

**D. Identifiers (NOT fed to model, kept for joins/splits/analysis):** `gameid`, `date`, `league`, `patch`, `teamname`, `opp_teamname`. (Preferred `teamname` over `teamid` during pipeline build: human-readable and materially cleaner — ~21 null teamnames vs ~1.1K null teamids in the filtered window. `opp_teamname` is derived at stage 2 by pivoting each gameid's two team rows on `side`.)

**Explicitly excluded (and why):**
- `atakhan` — not at inference (removed 2026 patch 26.1). Training on a feature absent in production wastes capacity and risks miscalibration. (Reversed earlier "keep with nulls" decision.)
- All cumulative objective counts (`dragons`, `barons`, `towers`, `inhibitors`, `turretplates`, `chemtechs`, `hextechs`, etc.) — CSV columns are end-of-game aggregates → leakage at any checkpoint before game end. Measured: 45%+ dragon overcount at 20-min model, 98% baron overcount.
- `heralds`, `void_grubs` — not in livestats API → train/inference mismatch.
- `xp*at{T}` — livestats exposes `level`, not XP. Level→XP heuristic skipped for v1 (correlated with gold/CS, marginal independent signal).
- `deathsat{T}` — redundant with `opp_killsat{T}` (<1% divergence from solo deaths to environment).
- `league` — redundant with `tier`; 121 sparse categories add noise. Kept as identifier.
- `patch`, `split`, `playoffs` — weak signal; in-game state already captures what matters.
- Player-row features (champions, per-player stats) — one-row-per-game decision + draft modeling deferred to v2.

### Why raw + opp + diff for every stat
The diff is mathematically collinear with the raws, but XGBoost's axis-aligned splits can't synthesize it cheaply — exposing the diff directly saves split-selection depth. Raws still carry pace/scaling info (a 30k/27k game ≠ a 22k/19k game even at the same diff). Information-theoretically redundant, modeling-practically useful.

### Per-checkpoint vs unified model → per-checkpoint
4 separate XGBoost models, one per T. Each checkpoint is a different prediction task (golddiff=2000 means very different things at min 10 vs min 25). One training script loops `for T in [10,15,20,25]: train_model(T)`.

### One model per tier → unified with `tier` as a feature
XGBoost splits on `tier` early and learns tier-specific subtrees natively. Per-tier specialization (3 × 4 = 12 models) deferred to v1.5 experiment.

### Row perspective at stage 2
Random side per game (`np.random.default_rng(seed=42)`). Balances `side` ~50/50, makes it a learnable feature, matches inference use case (predict P(team X wins) from team X's perspective directly).

### Pre-processing pipeline structure

```
scripts/preprocess/
  stage1_load_and_filter.py   → data/processed/01_filtered.csv            [DONE]
  stage2_one_row_per_game.py  → data/processed/02_one_row.csv             [DONE]
  stage3_features.py          → data/processed/03_features.csv            [DONE]
  stage4_split_checkpoints.py → data/processed/checkpoint_{10,15,20,25}.csv [DONE]
scripts/build_dataset.py (orchestrator, runs 1→4)                          [DONE]
```

CSV format (observability — eyeball intermediates, git-diffable); move to parquet if size/speed becomes an issue.

### TDD / testing conventions
- `tests/preprocess/test_stageN_*.py` mirrors `scripts/preprocess/stageN_*.py`.
- Root `conftest.py` prepends the repo to `sys.path` so `from scripts.preprocess... import` works from tests.
- Each stage module exposes `run(df, ...) -> DataFrame` (pure transform) and `validate(df, ...) -> None` (raises on failure). Tests hit both with small synthetic fixtures; no real-CSV slices in tests.
- Stage N trusts stage N-1's validator — tests at stage N don't re-verify stage N-1 invariants.
- Stage 4 exposes `run(df, T) -> DataFrame` and `validate(df, T) -> None`; tests parameterize over T ∈ {10,15,20,25}.
- Current count: **123 tests green** (stage 1: 23, stage 2: 16, stage 3: 22, stage 4: 62).

### Stage-boundary validations (implemented)
- **Stage 1:** `datacompleteness` all "complete"; `year` ∈ 2020-2026; league present in tier map; exactly 12 rows per gameid; non-null `result`/`side`/`gamelength`/`teamname`/`firstblood`/`firstdragon`/`firsttower`/`firstbaron` on team rows; exactly 1 winner per gameid; exactly 2 distinct teamnames per gameid.
- **Stage 2:** unique gameids, all team rows, tier-mapped league, `side` ∈ {Blue, Red}, `side` balance within 5σ binomial tolerance, `opp_teamname` non-null and != `teamname`.
- **Stage 3:** exact 63-column locked schema (no missing, no extras); no NaN in `ALWAYS_FILLED` (identifiers, statics, firsts, gamelength, result); `side` ∈ {Blue, Red}; `tier` ∈ {S, A, B}; `result` / `firstX` ∈ {0, 1}; engineered `killdiffat{T}` and `assistdiffat{T}` equal `raw − opp_raw` (NaN-tolerant for short-game `*at25` columns).
- **Stage 4:** per-T schema exact (no missing, no extras); no NaN in any kept column (all 12 `*at{T}` stats populated, plus identifiers/statics/firsts/gamelength/result); `side`, `tier`, `result`, gated-`firstX` domain checks.

### Real-data run — stages 1-4 (2026-04-20)
- Stage 1: 826,284 raw rows (2020-2026) → **563,664 rows / 46,972 games** kept.
- Stage 2: **46,972 rows** (one per game); side balance Blue=23,439 (49.9%) / Red=23,533 (50.1%).
- Stage 3: **46,972 × 63**; all validators green.
- Stage 4 (populated-frame filter):
  - T=10: 46,972 kept (100.0%)
  - T=15: 46,971 kept (100.0%)
  - T=20: 46,865 kept (99.8%)
  - T=25: 44,118 kept (93.9%)

**(year, tier) distribution on stage 3 output:**

| year | S | A | B | Total |
|---|---|---|---|---|
| 2020 | 1,960 | 2,983 | 2,417 | 7,360 |
| 2021 | 1,829 | 2,986 | 2,657 | 7,472 |
| 2022 | 1,480 | 2,711 | 2,475 | 6,666 |
| 2023 | 1,481 | 3,305 | 2,594 | 7,380 |
| 2024 | 1,369 | 3,009 | 2,484 | 6,862 |
| 2025 | 1,789 | 3,488 | 3,167 | 8,444 |
| 2026 | 757 | 1,028 | 1,003 | 2,788 |
| **Total** | **10,665** | **19,510** | **16,797** | **46,972** |

2020-2025 train/dev pool: ~44.2K games. 2026 held-out test: 2,788 (year-to-date). After the per-checkpoint `gamelength ≥ T*60` filter in stage 4, expect ~10% loss at T=25 → ~39.8K train, ~2.5K test at the hardest checkpoint. Fine for XGBoost; S-tier (10.7K across 7 years) is the thinnest slice and will bite on a v1.5 per-tier-model experiment, not on v1.

**Data loss vs raw baseline (`01_row_counts.json`, 2020-2026):** 61,586 raw → 46,972 kept = 14,614 games dropped (23.7%). Dominated by the unmapped-tier filter (~16K pre-registered in the checklist, mostly LATAM nationals / academy / other leagues without a Liquipedia tier). Null-teamname / self-match / null-firsts / incomplete-datacompleteness / no-winner filters combined contribute single-digit-thousands. Loss is heaviest in 2020-2022 (24-36%) where Oracle shipped more tier-2/3 regional coverage; by 2025-26 drops to 5-10% as Oracle coverage narrowed to leagues in the tier map.

---

## Section 2 — model design

- **Target:** in-game checkpoint winner (v1). Pre-game / Elo model = v2.
- **Baseline:** logistic regression on the same feature set per checkpoint. If XGBoost doesn't clearly beat it, investigate bug/leakage.
- **Primary model:** XGBoost. Mature, sparsity-aware, calibrates cleanly with isotonic. LightGBM unnecessary at ~70K rows.
- **Hyperparameter tuning:** Optuna Bayesian search, 50-100 trials per checkpoint, objective = val log-loss. Tune `max_depth` (3-10), `learning_rate` (0.01-0.3), `n_estimators` via early stopping, `min_child_weight`, `subsample`/`colsample_bytree`, `reg_alpha`/`lambda`. No nested CV — 2025 val (~9K games) is large enough.
- **Train/val/test split:** temporal.
  | Split | Years | Purpose |
  |---|---|---|
  | Train (dev) | 2020-2024 | Fit during dev |
  | Val (dev) | 2025 | HP tuning, early stopping |
  | Test | 2026 | Final held-out — touched exactly once |
  | Walk-forward CV | 2020-2025 (3 folds) | Separate exercise — meta-drift estimate |
  | Production model | Refit on 2020-2025 | Deploy for live betting |

  Temporal avoids Bo5 series leakage + matches production reality. Walk-forward folds = (20-22→23), (20-23→24), (20-24→25).
- **Calibration:** Brier score + reliability diagram (10 bins) per checkpoint. Correct with isotonic regression on val if miscalibrated. Critical for edge calculation (miscalibrated 0.70 that's really 0.60 inflates Kelly sizing).
- **Class balance:** ~50/50 by construction (every game has 1 winner, filters are winner-agnostic). Verify with one-liner after pipeline runs.

---

## Section 3 — Polymarket integration & execution

- **Poly odds as model input:** NO. Keeps strategy honest ("model disagrees with Poly → bet").
- **Live data:** lolesports livestats API (`feed.lolesports.com/livestats/v1/window/{gameId}` + `/details/{gameId}`). Verified against live LCK game 2026-04-16. Free, sub-second cadence. Intersection feature set with Oracle's Elixir training columns (gold, kills, CS, derived firsts).
- **Poly programmatic API:** deferred to v2 (`py-clob-client` + funded Polygon wallet + API key). Manual execution for v1.
- **Liquidity:** sufficient ($7K-$51K per game on B-tier ERLs; more on S-tier). Spread ~1% on good games → practical minimum edge: 2-3%.
- **Latency:** Poly reacts fast to discrete events (baron/drake/ace), unquantified on gradual state. Edge likely lives in gradual state — the thesis is "our model values a 3k gold lead at min 15 more correctly than the crowd", not "we react faster to baron".
- **Bankroll:** quarter-Kelly (k=0.25) with 2% per-bet hard cap. Kelly formula `f* = (b·p − q)/b`, stake = `0.25 · f* · bankroll`. Revisit cap after ~100 live bets.

---

## Section 4 — data exploration (reference)

All scripts `01_*.py` through `10_*.py` run; outputs in `data/exploration/*.json`. Drove these decisions:
- **85,313 complete games, 2014-2026.** Core features (result, side, 10/15 min stats) 0% null across all years.
- **Blue side wins 52-58%** across all years/leagues — real, consistent signal.
- **Atakhan** only exists in 2025-2026; 2026 has 246 pre-patch-26.1 games still populated.
- **2 no-winner games** identified (filter them out).
- **Objectives leakage confirmed:** dragons/barons are end-of-game aggregates — unusable at any pre-game-end checkpoint. First-event resolution rates measured per minute bucket.
- **Training cutoff:** 2020 chosen (first full Dragon Soul year, modern gamelength/kill pace). 2022 alternative not pursued further.
- **Tier matching:** 70 of 121 leagues matched (80.7% of games). S: 21K games, A: 28K, B: 20K. 51 unmatched (16K games, mostly LATAM nationals/academy).

---

## Section 5 — open questions (resolved)
- **External data sources** (champion WR, draft meta, Elo): deferred to v2.
- **Experiment tracking:** simple CSV log — `run_id, timestamp, checkpoint, hparams_json, val_logloss, val_brier, val_auc, notes`. MLflow only if iteration rate justifies it in v2.
- **Poly coverage:** all tiers depending on season. No modeling changes needed.
- **Team/org mapping Oracle's Elixir ↔ Polymarket:** v1 irrelevant (manual execution); v2 blocker (build mapping table seeded from Liquipedia canonical names).

---

## 2026 meta context (corrected this session)

Baron timing was wrongly documented as "12:00" in the checklist, `08_training_year_cutoff.json/.py`, and `10_objectives_vs_gamelength.py`. Corrected:
- **2020-2024:** Baron@20:00 (traditional).
- **2025:** Baron@25:00 (pushed back to make room for Atakhan@20 + Feats of Strength).
- **2026:** Baron back to 20:00 — matches 2020-2024. Atakhan + Feats removed in patch 26.1. Added: role quests, item rework, only 2 elemental drakes pre-soul.

Implication: 2026 is structurally closer to 2020-2024 than to 2025 in baron terms. The ~6% gold/kill uptick at 10/15 min in 2026 is from role quests + item rework + 2-drakes-pre-soul, not baron timing. Normal-magnitude meta drift.

**Feats of Strength residual concern:** 2025-only mechanic (removed 2026) that granted free boot upgrades + stat boost to the first team reaching 3-of-{firstblood, first tower, first epic}. Its effects leak into `goldat20/25` and `killsat20/25` via compounding power advantage — not a CSV column we can drop. ~50% of 2025 games earned feats. Accepted as risk for v1; monitor calibration on 2026 test; if 20/25-min models are specifically off, revisit via sample-weighting or excluding 2025.

---

## Immediate next step
Pipeline is done. Next chunk is the **model training + experiment harness**:
- Temporal split: train 2020-2024, val 2025, test 2026 (touched exactly once).
- Logistic-regression baseline per T, then XGBoost per T (4 models total).
- Optuna (50-100 trials per checkpoint) on val log-loss, early stopping.
- Isotonic calibration on val if Brier / reliability diagram says it's needed.
- Experiment log: simple CSV (`run_id, timestamp, checkpoint, hparams_json, val_logloss, val_brier, val_auc, notes`).

## Key files
- `docs/preprocessing-pipeline.md` — pipeline spec (what it does, locked schema, invariants).
- `discovery-checklist.md` — authoritative *decisions and reasoning* across all sections.
- `scripts/preprocess/stage{1,2,3,4}_*.py` + `tests/preprocess/test_stage{1,2,3,4}_*.py` — implemented pipeline + tests.
- `scripts/build_dataset.py` — stages 1→4 orchestrator.
- `data/processed/0{1,2,3}_*.csv` + `data/processed/checkpoint_{10,15,20,25}.csv` — current pipeline outputs.
- `data/exploration/*.json` — baseline exploration outputs (scripts 01-10).
- `scripts/exploration/*.py` — exploration scripts (moved under `scripts/exploration/` — they are not pipeline code).
- `data/columns.txt` — Oracle's Elixir column list.
- `conftest.py` — pytest rootdir anchor.

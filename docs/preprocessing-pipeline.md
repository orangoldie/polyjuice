# Preprocessing pipeline spec

Transform raw Oracle's Elixir CSVs into 4 per-checkpoint training datasets for the v1 in-game win-probability model.

Rationale for every decision below lives in `discovery-checklist.md` — this doc is the *what*, not the *why*.

## Pipeline shape

```
data/lol-historical-data/*.csv
        │  stage 1 — load + row filters
        ▼
data/processed/01_filtered.csv          (all 12 rows per surviving game, all columns)
        │  stage 2 — collapse to one row per game
        ▼
data/processed/02_one_row.csv           (one team-row per game, random perspective)
        │  stage 3 — feature engineering + column pruning + tier join
        ▼
data/processed/03_features.csv          (46,972 × 63, locked schema)
        │  stage 4 — per-checkpoint split
        ▼
data/processed/checkpoint_{10,15,20,25}.csv
```

Orchestrator: `scripts/build_dataset.py` runs stages 1→4 in order.

All intermediates are CSV for observability. Move to parquet only if size/speed becomes a problem.

## Invariants

Each stage N exposes `run(...) -> DataFrame` (pure transform) and `validate(...) -> None` (raises on failure). Stage N trusts stage N-1's validator; stage N tests do not re-verify stage N-1 invariants.

Validators run at the end of each stage's main, before writing output. A failing validator stops the pipeline at the point of corruption rather than letting drift propagate.

## Stage 1 — load + row filters

**Module:** `scripts/preprocess/stage1_load_and_filter.py`

**Input:** all `{year}_*.csv` under `data/lol-historical-data/` for years 2020–2026.

**Output:** `data/processed/01_filtered.csv` — all 12 rows per surviving game, all source columns preserved.

**Row filters (applied in order):**

1. `datacompleteness == "complete"`.
2. `year` ∈ 2020..2026.
3. `league` present in the Liquipedia tier map (`data/exploration/07_league_tier_matching.json`).
4. Exactly 2 team rows and 10 player rows per `gameid`.
5. On team rows: non-null `result`, `side`, `gamelength`, `teamname`, `firstblood`, `firstdragon`, `firsttower`, `firstbaron`.
6. Exactly 1 winner per `gameid` (drops the 2 documented no-winner games).
7. Exactly 2 distinct `teamname`s per `gameid` (drops academy self-matches).

**Validator checks:** every filter above holds post-hoc on the output.

## Stage 2 — collapse to one row per game

**Module:** `scripts/preprocess/stage2_one_row_per_game.py`

**Input:** `data/processed/01_filtered.csv`.

**Output:** `data/processed/02_one_row.csv` — one team-row per `gameid`.

**Transform:**

- Drop the 10 player rows per game.
- For each `gameid`, randomly pick one of the two team rows using `np.random.default_rng(seed=42)`. Fixed seed for reproducibility.
- Derive `opp_teamname` for the kept row by looking at the sibling team row of the same `gameid`.

**Rationale summary:** random perspective (vs. always-blue) makes `side` a genuinely learnable feature (~50/50 balance) and matches the inference framing — at bet time we predict P(Team X wins), feeding Team X's stats as "team" and opponent as "opp".

**Validator checks:** unique `gameid`, all rows are team rows, every league in tier map, `side` ∈ {Blue, Red}, `side` balance within 5σ binomial tolerance, `opp_teamname` non-null and != `teamname`.

## Stage 3 — feature engineering + schema lock

**Module:** `scripts/preprocess/stage3_features.py`

**Input:** `data/processed/02_one_row.csv`.

**Output:** `data/processed/03_features.csv` — exactly 63 columns in a locked order.

**Transform:**

- Join `tier` (S/A/B) from the Liquipedia mapping onto each row via `league`.
- Engineer `killdiffat{T}` = `killsat{T} − opp_killsat{T}` for each T ∈ {10, 15, 20, 25}. Same for `assistdiffat{T}`. Oracle's Elixir ships gold/CS diffs but not kill/assist diffs.
- Select exactly the 63-column locked schema; drop everything else.

**Locked schema (63 columns):**

- **Identifiers (7)** — `gameid, date, year, league, patch, teamname, opp_teamname`.
- **Static features (2)** — `side, tier`.
- **Checkpoint stats (48)** — for each T ∈ {10, 15, 20, 25}, twelve columns: `goldat{T}, opp_goldat{T}, golddiffat{T}, csat{T}, opp_csat{T}, csdiffat{T}, killsat{T}, opp_killsat{T}, killdiffat{T}, assistsat{T}, opp_assistsat{T}, assistdiffat{T}`.
- **First-event flags (4)** — `firstblood, firstdragon, firsttower, firstbaron`.
- **Other (2)** — `gamelength, result`.

**NaN policy:** NaN tolerated on the 12 `*at{T}` stat columns (short games may be missing late-checkpoint frames). All other columns are always filled.

**Excluded from training** (rationale in discovery-checklist):

- `atakhan` — not at inference (removed in 2026 patch 26.1).
- All cumulative objective counts (`dragons`, `barons`, `towers`, `inhibitors`, `turretplates`, `chemtechs`, `hextechs`) — end-of-game aggregates, leakage at any pre-game-end checkpoint.
- `heralds`, `void_grubs` — not exposed by the livestats API.
- `xp*at{T}` — livestats exposes `level`, not XP.
- `deathsat{T}` — ≈ `opp_killsat{T}` in pro play (<1% divergence from solo deaths to environment).
- `league`, `patch`, `split`, `playoffs` as features — redundant with `tier` or weak signal; `league` kept as identifier.
- Player-row features (champion picks/bans, per-player stats) — draft modeling deferred to v2.

**Validator checks:** exact 63-column schema (no missing, no extras); no NaN in always-filled columns (identifiers, statics, firsts, gamelength, result); `side` ∈ {Blue, Red}; `tier` ∈ {S, A, B}; `result`, `first*` ∈ {0, 1}; engineered `killdiffat{T}` and `assistdiffat{T}` equal `raw − opp_raw` (NaN-tolerant).

## Stage 4 — per-checkpoint split

**Module:** `scripts/preprocess/stage4_split_checkpoints.py`

**Input:** `data/processed/03_features.csv`.

**Output:** `data/processed/checkpoint_{10,15,20,25}.csv` — one file per T, with only that T's columns.

**Transform per T:**

1. **Row filter:** drop rows where any of the twelve `*at{T}` stat columns is null. This is the "populated checkpoint frame" filter — gamelength is *not* the gate, because Oracle's frame capture doesn't perfectly align with gamelength near the boundary:
   - Games ending ~24:30-24:59 with a populated `*at25` snapshot are valid (kept).
   - Games ending at exactly T*60 with no frame emitted are invalid (dropped).
2. **Column selection:** identifiers (7) + static features (2) + this T's 12 stat columns + gated firsts + `gamelength` + `result`.

**First-event gates:**

| Feature      | T=10 | T=15 | T=20 | T=25 |
|--------------|:----:|:----:|:----:|:----:|
| `firstblood` |  ✅  |  ✅  |  ✅  |  ✅  |
| `firstdragon`|  ❌  |  ✅  |  ✅  |  ✅  |
| `firsttower` |  ❌  |  ✅  |  ✅  |  ✅  |
| `firstbaron` |  ❌  |  ❌  |  ❌  |  ✅  |

Gates reflect per-checkpoint resolution rates — e.g. firstdragon is only ~50% resolved by min 10 (train/inference mismatch), firstbaron isn't 50%+ resolved until min 25.

**Validator checks:** per-T schema exact (no missing, no extras); no NaN in any kept column; `side`, `tier`, `result`, gated `firstX` domain checks.

## Data volumes (real run, 2026-04-20)

- Stage 1: 826,284 raw rows → 563,664 rows / 46,972 games.
- Stage 2: 46,972 rows (Blue 49.9% / Red 50.1%).
- Stage 3: 46,972 × 63.
- Stage 4: T=10 → 46,972 · T=15 → 46,971 · T=20 → 46,865 · T=25 → 44,118.

## Testing

- Tests mirror layout: `tests/preprocess/test_stageN_*.py` ↔ `scripts/preprocess/stageN_*.py`.
- Each test hits both `run()` and `validate()` with small synthetic fixtures. No real-CSV slices in tests.
- Root `conftest.py` prepends the repo to `sys.path` so `from scripts.preprocess... import` works from tests.
- Stage 4 tests parameterize over T ∈ {10, 15, 20, 25}.
- Current count: 123 tests green (stage 1: 23, stage 2: 16, stage 3: 22, stage 4: 62).

## How to run

```bash
# full pipeline
python scripts/build_dataset.py

# individual stages (module form, so sibling imports resolve)
python -m scripts.preprocess.stage1_load_and_filter
python -m scripts.preprocess.stage2_one_row_per_game
python -m scripts.preprocess.stage3_features
python -m scripts.preprocess.stage4_split_checkpoints

# tests
python -m pytest tests/
```

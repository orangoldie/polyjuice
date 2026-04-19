# Discovery Checklist

Questions to answer and explorations to run before we can write a proper system spec.

Status: `[ ]` = open, `[~]` = in progress, `[x]` = resolved

When a question is resolved, add a `> **Decision:** ...` block under it with the reasoning and conclusion.

---

## 1. Data & Feature Engineering

- [x] **Which columns to include and how to structure time-frames?**
  Do we build separate feature sets per checkpoint (10/15/20/25 min), or keep everything in one flat row with all time-frame columns together? What are the tradeoffs?
  > **Decision:** Resolved by the 1.7 pre-processing pipeline work. **Per-checkpoint** — 4 separate datasets (and 4 separate XGBoost models), each containing only its own checkpoint's stats + static columns + gated firsts. Full v1 feature list locked in 1.7. See 1.7 for the per-checkpoint vs unified reasoning.

- [x] **Team rows only or player rows too?**
  Do we need the 10 player rows per game, or are the 2 team-aggregate rows enough?
  > **Decision:** **Team rows only, one row per game** (randomly chosen side, see 1.7). Player rows dropped in pipeline stage 2. Player-level features (champion picks/bans, per-player stats) are excluded from v1 — draft modeling deferred; per-player stats are dominated by the team aggregates we already use. Re-open if v2 adds draft features.

- [x] **How to handle the atakhan column?**
  Verify it only exists in 2025+ data. How do we treat it for earlier years — drop it? Keep with nulls?
  > **Decision:** **Drop `atakhan` from training entirely.** Atakhan was removed in 2026 (patch 26.1) so it will always be null at inference. Training on a feature that's absent in production wastes model capacity (XGBoost learns splits that route every live game through the default direction) and risks miscalibration — training accuracy looks good, live accuracy degrades silently.
  >
  > **Reverses the earlier "keep atakhan, leave nulls" decision** in section 4. The original rationale was "keeps 2025 games in training" — but that's wrong: we can include 2025 games without the atakhan column. Dropping the column does not drop the rows.
  >
  > **Cost:** slight accuracy loss on the 25-min model (atakhan spawns at 20 min, only relevant to the 25-min checkpoint). Acceptable — train/inference parity is more important than squeezing one feature.

- [x] **Do we want to incorporate champion/draft data?**
  Champions are in the data (picks, bans). Is draft modeling worth the complexity, or do we leave it out? If yes, how to encode it (one-hot per champ? win rates? composition archetypes?).
  > **Decision: defer to v2.** No champion or draft features in v1.
  >
  > **Why:**
  > - v1 is an in-game checkpoint model. By min 10, the draft's effect is already realized in `goldat10` / `csat10` / `killsat10` — draft is a *cause* of the stats we already feed in, not independent signal.
  > - Encoding is expensive and brittle: ~170 champions × 10 slots = huge sparse space. One-hot is too sparse; champion WR lookups go stale with every patch; composition archetypes need manual tagging. Each option is a data-quality project on its own.
  > - Pulling champion data back in partially reverses the one-row-per-game decision, since champions live on player rows.
  >
  > **Re-open path:** if v1 calibration is poor at the 10-min checkpoint (where in-game signal is weakest and draft's relative value is highest), champions are the obvious next lever. Revisit as a v2 feature.

- [x] **How to handle patch/meta shifts?**
  The game changes significantly between patches. Do we include patch as a feature? Group into eras? Or does the model learn this implicitly from the data?
  > **Decision:** **Do not include patch as a feature.** In-game state (gold, CS, kills, objectives) already captures the outcomes of meta shifts — a 3k gold lead at 15min predicts a win regardless of whether chemtech or hextech is in the pool. Patch as a categorical feature would add hundreds of sparse levels and mostly noise.
  >
  > **If meta drift hurts calibration, address via sample weighting** toward recent patches (exponential decay on `date`) rather than a patch feature. Walk-forward CV (see train/test split decision) will surface whether drift is actually a problem. Era grouping (pre-soul / soul / grub-era / etc.) explicitly rejected — arbitrary cutoffs that throw away continuous variation.

- [x] **How to encode Blue/Red side?**
  Blue side has a known historical WR advantage in pro. How significant is it in our data?
  > **Decision:** **`side` is a binary categorical feature (Blue/Red)** fed to the model directly. Signal verified — Blue wins 52-58% consistently across years and leagues (`03_side_winrate.json`). Pipeline stage 2 randomly picks blue or red row per game (fixed seed) so `side` has ~50/50 balance and is genuinely learnable rather than a constant. See 1.7 for the random-perspective reasoning.

- [x] **How to handle non-time-dependent columns across time frames?**
  Columns like dragon/baron counts are 0 at early checkpoints. Does including them cause any issues or is it harmless?
  > **Decision:** **Drop all cumulative objective counts from checkpoint models. Use `first*` columns with strict checkpoint gates.**
  >
  > **The core problem:** CSV columns `dragons`, `barons`, `elders`, `heralds`, `void_grubs`, `atakhans` are **end-of-game aggregates** — there are no `dragonsat10` / `dragonsat15` equivalents. Using them at a checkpoint-T model trains on future info (the dragons taken after minute T).
  >
  > **Leakage magnitudes measured** (script `10_objectives_vs_gamelength.py`, 2020-2025, 58,784 games — games ending at minute T bucket the `by-minute-T` upper bound):
  > | Feature | End-of-min-20 bucket | End-of-min-25 bucket | Full-game avg | Overcount at 25-min model |
  > |---|---:|---:|---:|---:|
  > | dragons / game | 2.44 | 3.49 | 4.49 | **~1.0 drake (22%)** |
  > | barons / game | 0.02 | 0.83 | 1.32 | **~0.5 baron (37%)** |
  > | elders / game | 0.00 | 0.00 | 0.11 | ~0.1 |
  >
  > Leakage at 20-min model is far worse (45%+ for dragons, ~98% for barons) — cumulative counts are unusable at any early checkpoint.
  >
  > **First-event resolution rates** (% of games where event happened at all, by end of a T-minute game):
  > | Event | By min 15 | By min 20 | By min 25 | Lifetime |
  > |---|---:|---:|---:|---:|
  > | firstblood | 100% | 100% | 100% | 99.9% |
  > | firstdragon | 93% | 100% | 100% | 100% |
  > | firsttower | 93% | 100% | 100% | 100% |
  > | firstbaron | 0% (can't spawn) | 2% | 83% | 92.4% |
  >
  > **Final feature gates for checkpoint models:**
  > | Feature | 10 min | 15 min | 20 min | 25 min |
  > |---|---|---|---|---|
  > | `firstblood` | ✅ | ✅ | ✅ | ✅ |
  > | `firstdragon` | ❌ | ✅ | ✅ | ✅ |
  > | `firsttower` | ❌ | ✅ | ✅ | ✅ |
  > | `firstbaron` | ❌ | ❌ | ❌ | ✅ (83% resolved by min 25) |
  > | `firstherald` / `firstgrub` | ❌ | ❌ | ❌ | ❌ (not in live API) |
  > | All cumulative counts (`dragons`, `barons`, `elders`, etc.) | ❌ | ❌ | ❌ | ❌ |
  >
  > **Why drop `firstdragon` at 10-min** (debated): dragon spawns at 5, first contest resolves min 6-12 → only ~50% of games have firstdragon by min 10. CSV gives end-of-game firstdragon, creating a train/inference mismatch — model trains on `firstdragon=1` for games where event hadn't fired by 10, but in production at min 10 those games show `firstdragon=0`. The mismatch is the exact leakage mode that looks good in backtest and fails live. Dropped to preserve calibration.
  >
  > **Why the signal loss is acceptable:** teams don't take uncontested dragons/barons — those contests show up in `golddiffat{N}` and `killsat{N}` anyway. We lose the direct count but keep the effect.
  >
  > **Re-open path:** if v1 underperforms, re-source historical frames from the livestats API to build real `dragonsat{N}` / `baronsat{N}` features (Option C from section 3 discussion).

- [x] **Filter to datacompleteness == "complete" only**
  Do we use all rows or only complete ones?
  > **Decision:** Filter to `datacompleteness == "complete"` in all scripts and in the final pipeline. Drops ~11% of games but gives us 0% nulls on time-stamped columns (10/15 min). Notable: LPL loses ~68% of its games to this filter. All exploration scripts already apply this filter.

- [~] **Pre-processing pipeline design**
  Filtering (incomplete data, missing columns), encoding (categorical features), normalization — what does each step look like?

  **Null-vs-zero policy (resolved — moot):** the question only matters for era-missing features, and none survive into the v1 training set:
  - `atakhan` — dropped (not at inference; see section above)
  - `void_grubs`, `heralds` — dropped (not in livestats API)
  - `chemtechs`, `hextechs` — dropped under the A3 cumulative-count rule (same leakage class as `dragons`)

  **Final v1 feature list (locked):**

  *A. Game context (constant across checkpoints):*
  - `side` (Blue/Red, categorical)
  - `tier` (S/A/B from Liquipedia mapping)

  *B. Checkpoint stats at minute T (T ∈ {10, 15, 20, 25}) — raw + opp + diff for each:*
  - Gold: `goldat{T}`, `opp_goldat{T}`, `golddiffat{T}` (CSV)
  - CS: `csat{T}`, `opp_csat{T}`, `csdiffat{T}` (CSV)
  - Kills: `killsat{T}`, `opp_killsat{T}`, `killdiffat{T}` (engineered)
  - Assists: `assistsat{T}`, `opp_assistsat{T}`, `assistdiffat{T}` (engineered)

  **Deaths dropped:** in a 2-team game, team A's deaths ≈ team B's kills against team A, so `deathsat{T}` ≈ `opp_killsat{T}` (and vice versa). The only divergence is solo deaths to environment (tower/minions/drowning), which are <1% in pro play. Keeping deaths would add 3 redundant columns per checkpoint that are near-perfectly correlated with kills — wastes split-selection cycles, no signal gained. Assists are NOT redundant (multiple players can assist on one kill, total_assists ≠ total_kills) — kept.

  **Why all three (raw + opp + diff) for each stat:** the diff is mathematically collinear with the raws (`diff = raw - opp_raw`), so for a linear model it would be redundant. But XGBoost is a tree model and splits on one feature per node — to express "golddiff > 3000" from raws alone it needs a deep cascade of nested splits that approximates a diagonal threshold with axis-aligned cuts. Pre-computing the diff exposes the concept directly so the tree can split on it in a single node. The raws still carry pace/scaling info (a 30k/27k game ≠ a 22k/19k game even with the same diff). Information-theoretically redundant, modeling-practically useful.

  *C. First-event flags (per A3 gates):*
  - `firstblood` at all checkpoints
  - `firstdragon`, `firsttower` at 15/20/25
  - `firstbaron` at 25 only

  *D. Identifiers (kept for joins/splits/analysis, NOT fed to model):*
  - `gameid`, `date`, `league`, `patch`, `teamid`, `opp_teamid`

  **Explicitly excluded** (and why):
  - `xp*at{T}` — livestats only exposes `level`, not XP. Level→XP heuristic skipped for v1: highly correlated with gold/CS, marginal independent signal, not worth the complexity. Re-add only if calibration analysis shows it's missing.
  - `league` (as feature) — redundant with `tier`; 121 sparse categories add noise. Kept as identifier only.
  - `patch`, `split`, `playoffs` — weak signal; in-game state already captures what matters. If meta drift hurts accuracy, address via sample weighting toward recent patches, not as a feature.
  - All cumulative objective counts (`dragons`, `barons`, `towers`, `inhibitors`, `turretplates`, `chemtechs`, `hextechs`, etc.) — A3 leakage rule
  - `heralds`, `void_grubs`, `atakhan` — not at inference
  - `dpm`, `damagetakenperminute`, `visionscore`, ward stats — not in livestats
  - Player-row features (champion picks/bans, per-player stats) — A2 (one row per game) + draft modeling deferred
  - `team kpm`, `ckpm`, `firstmidtower`, `firsttothreetowers` — derived from end-of-game or not in livestats

  **Row filters (decided so far):**
  - `datacompleteness == "complete"` (existing)
  - Drop 2 no-winner games (from data integrity check)
  - Drop games shorter than the checkpoint minute (e.g. for 25-min model, drop games ending before 25 min)
  - **Drop games whose league has no tier mapping** (~16K games, mostly LATAM nationals/academy). Re-open if v1 needs more data.
  - Year range: 2020-2025 train; 2026 held out (decided in section 4)

  **Per-checkpoint vs unified model (decided): per-checkpoint.** Build 4 separate XGBoost models, one per T ∈ {10, 15, 20, 25}. Each model trains on its own dataset (one row per game, only that checkpoint's features + static columns + gated firsts). At inference, livestats game time routes to the appropriate model.

  *Why per-checkpoint over unified:*
  - Each checkpoint is genuinely a different prediction task — golddiff=2000 at min 10 means something very different from golddiff=2000 at min 25. Different baseline win-prob distributions, different feature semantics.
  - A unified model would have to learn this as an interaction with `checkpoint_minute` — possible for trees but requires depth and is fragile.
  - Calibration must be verified per-checkpoint regardless of architecture; specialized models are easier to calibrate than one model serving 4 regimes.
  - The "shared learning" benefit of the unified model is overstated: ~70K games per checkpoint is plenty for XGBoost.
  - Clean data shape: no nulls from gated firsts (firstdragon at 10-min row, etc.), no IID violation from 4 correlated rows per game.

  *Cost is fake:* the "4 artifacts to maintain" objection collapses when you write one training script that loops `for T in [10, 15, 20, 25]: train_model(T)`. Same code, 4 serialized models. Late-checkpoint models do train on fewer games (games ending before T are excluded — ~10% loss for 25-min model), acceptable.

  **Pipeline structure (decided): 4 staged scripts, each writes its intermediate output as CSV.**

  ```
  scripts/preprocess/
    01_load_and_filter.py    → data/processed/01_filtered.csv
    02_one_row_per_game.py   → data/processed/02_one_row.csv
    03_features.py           → data/processed/03_features.csv
    04_split_checkpoints.py  → data/processed/checkpoint_{10,15,20,25}.csv
  ```

  Plus a thin orchestrator (`scripts/build_dataset.py`) that runs them in order.

  *Stage responsibilities:*
  1. **Load + filter rows** — read raw CSVs, apply `datacompleteness == "complete"`, drop 2 no-winner games, drop unmapped-tier games, year range 2020-2026 (2026 kept for held-out test set, separated downstream).
  2. **Collapse to one row per game** — drop the 10 player rows; for each gameid, randomly keep one of the 2 team rows (perspective choice, see below). Fixed RNG seed for reproducibility.
  3. **Feature engineering** — drop excluded columns, compute engineered diffs (`killdiffat{T}`, `assistdiffat{T}`), join in `tier` from Liquipedia mapping.
  4. **Split by checkpoint** — emit 4 files, each with that checkpoint's features + static columns + gated firsts + identifiers + label. Drop games shorter than the checkpoint at this stage.

  *Format:* CSV for now (observability — eyeball intermediates, git-diffable). Move to parquet if size/speed becomes an issue.

  **Row perspective at stage 2 (decided): random side per game with fixed seed.**

  For each gameid, randomly keep blue or red team row (`np.random.default_rng(seed=42)`). Result: `side` is balanced ~50/50 and becomes a learnable categorical feature.

  *Why random over always-blue:*
  - **`side` is a validated signal** — `03_side_winrate.json` showed Blue wins 52-58% across all years/leagues. Always-blue would bake this into the model's baseline (no `side` feature possible since it'd be constant); random-perspective lets the model split on `side` explicitly and capture side × other-feature interactions if they exist.
  - **Inference framing matches betting use case.** Polymarket markets resolve "Team X wins?". With random-perspective the model predicts from Team X's perspective directly: feed Team X's stats as "team", opponent as "opp", feed `side`. Output is P(Team X wins). With always-blue we'd predict P(blue wins) and invert when Team X is on red — works but extra plumbing.
  - **Cost is trivial:** one RNG call per game in stage 2, fixed seed for reproducibility.

  *Gotcha for future work:* if we ever augment by keeping both rows per game, both rows must go to the same train/test split (same gameid → group leakage otherwise). Not an issue with current one-row-per-game decision.

  **Stage-boundary validations (decided):** each stage writes its CSV and then runs a validator that asserts post-conditions. Fail fast with a clear error rather than letting silent drift propagate to training.

  - **After stage 1:** only complete rows, year range 2020-2026, expected row count (~12 × complete-games), no null `result` / `side` / `gamelength` on team rows, the 2 no-winner games dropped.
  - **After stage 2:** one row per gameid, row count == unique gameid count, `side` is ~50/50 (binomial check, not exactly 50/50), all mapped to a known tier.
  - **After stage 3:** exact locked feature-column set present (nothing extra, nothing missing), no NaN in model features, engineered diffs match `raw − opp_raw` to zero rounding error, `tier` ∈ {S, A, B}.
  - **After stage 4:** each checkpoint file has that checkpoint's features + static + gated firsts + identifiers + label and nothing else; no rows with `gamelength < T*60`; per-file row counts logged so regressions are visible.

  Implemented as assertions in each stage script (not a separate pass) so the stage fails at the point of corruption rather than at the next stage's read.

  Remaining sub-items: none — pre-processing pipeline design is now fully specified.

- [x] **Incorporate team strength / Elo prior?**
  Should we compute a rolling Elo/Glicko rating from historical results and use it as a feature? How would we build it?
  > **Decision: defer to v2.** No Elo / team-rating features in v1.
  >
  > **Why:**
  > - v1 is in-game. Team strength is already bleeding into `goldat10` / `csat10` / `killsat10` — a stronger team opens a lead more often. Elo is most valuable as a *pre-game prior*, which is exactly where v2 lives.
  > - Building Elo right is its own project: K-factor tuning, cold-start for new teams, decay across splits, cross-region calibration. A sloppy version gives a feature worse than `tier`.
  > - Couples tightly to the unresolved roster-change question — Elo on the org is noisy when starters change between splits; Elo on the roster is sparse.
  > - `tier` (S/A/B) already captures coarse strength. Fine-grained intra-tier ranking is the marginal value Elo adds — worth the build cost in v2, not v1.
  >
  > **Re-open path:** v2 pre-game model will need this as its backbone. If v1 calibration is specifically weak on the 10-min checkpoint for intra-tier matchups, pull the v2 work forward.

- [x] **Handle roster changes?**
  Same org can have different players mid-split. Does this matter? If we use team identity as a feature, how do we account for this?
  > **Decision: not applicable to v1.** `teamid` is an identifier, not a model feature (see 1.7). Becomes a design question for v2 pre-game/Elo model — candidates: lineup-hash ratings with regression-to-org-mean, or Elo partial-reset when ≥3 starters change.

---

## 2. Model Design

- [x] **Define target variable clearly**
  The `result` column (0/1) is our label. But what are we actually predicting and when?
  - **Pre-game model**: predict winner before the match starts, using only team identity, Elo, side, draft, tier, etc. No in-game stats.
  - **In-game checkpoint models**: predict winner at 10/15/20/25 min marks using live game state (gold diff, kills, objectives, etc.).
  - Do we build one or both? Are these separate models or one model with a "time" feature? The Polymarket use case should drive this — does Poly offer live in-game betting or only pre-match?
  > **Decision:** **v1 = in-game checkpoint model.** **v2 = pre-game / team-rating model.**
  >
  > **Why in-game first:**
  > - **Uses our actual data advantage.** Oracle's Elixir checkpoint columns (gold/xp/kills/objectives at 10/15/20/25 min) are the asset most bettors don't have clean access to. Pre-game team identity is reproducible by anyone with Liquipedia access.
  > - **Signal density is much higher.** `goldat15` alone is a strong win-probability signal (well-documented in LoL literature). Pre-game features (team, side, draft) carry far less information per row.
  > - **Edge is more likely to exist in live markets.** Pre-match markets attract more sharp bettors and public data; in-play markets on ERL/B-tier games update mechanically with latency and retail noise. Poly liquidity spans S/A/B tiers (see section 4 finding) — we chase inefficiency where it lives.
  > - **Fail-fast logic.** If the rich feature set can't beat Polymarket, the sparser pre-game feature set definitely won't. In-game failing kills the project early and cheaply; pre-game failing tells us nothing about the harder question.
  >
  > **Why pre-game / team-rating is v2 (not skipped):**
  > - Team rating / Elo naturally becomes a **feature** of the in-game model (prior on which team was favored before the game started), so the v2 work plugs into v1's pipeline rather than competing with it.
  > - Pre-match markets likely have higher Polymarket liquidity than in-play; once the system is proven on in-game, adding pre-game extends coverage to more bettable markets.
  >
  > **Model structure (open sub-question):** one model with a `checkpoint_minute` feature vs. one model per checkpoint (10/15/20/25). Tracked in the "One model per tier or unified" item below — decide empirically.
  >
  > **Critical-path implication:** Section 3 "Live game data API" becomes a v1 blocker. Needs to move up in priority — candidates to research: Riot Esports API, GRID, Bayes Esports, PandaScore.

- [x] **One model per tier or one unified model?**
  More data (combined) vs. cleaner signal (per-tier) — what are the tradeoffs? Need to explore both.
  > **Decision: v1 = unified across tiers with `tier` as a categorical feature** (4 models total — one per checkpoint, each trained on all tiers combined). Per-tier specialization (3 × 4 = 12 models) deferred to v1.5 as an experiment.
  >
  > **Why unified-with-tier-feature first:**
  > - **Fits tree behavior natively.** XGBoost splits on tier early and learns tier-specific subtrees within one model. "Feature X behaves differently in tier Y" is exactly what tree splits express. We get most of the per-tier specialization benefit without fragmenting the data.
  > - **Data efficiency.** ~70K games split into 21K S / 28K A / 20K B per checkpoint. Plenty for XGBoost, but no reason to fragment when the model can learn the conditional structure itself.
  > - **Reversibility.** Starting unified costs nothing if per-tier wins later — same features, same checkpoints, just retrain three subsets and compare AUC/calibration. Going per-tier first locks the pipeline in and is more expensive to undo.
  >
  > **v1.5 experiment:** once we have a unified baseline, retrain per-tier (3 models per checkpoint, no tier feature) and compare. Decision criterion: per-tier wins only if it improves calibration or AUC by a margin that justifies the artifact count and ongoing maintenance.

- [x] **Baseline model**
  What's the simplest model we can build to sanity-check the data before going to XGBoost?
  > **Decision: logistic regression** on the same feature set per checkpoint. Cheap, interpretable, gives a lower bound — if XGBoost doesn't clearly beat it, something's wrong (bug, leakage, or features have no nonlinear structure).

- [x] **XGBoost vs. alternatives**
  Is XGBoost the right choice? How does it compare to logistic regression, LightGBM, or other options for this type of problem?
  > **Decision: XGBoost** for v1. Mature, battle-tested on tabular, sparsity-aware splits, calibrates cleanly with isotonic post-processing. LightGBM is faster but ~70K rows doesn't need it. Logistic regression is the baseline (above), not an alternative. Revisit only if XGBoost can't beat baseline.

- [x] **Hyperparameter tuning strategy**
  Which hyperparameters matter most (learning rate, max_depth, n_estimators, etc.)? How do we tune them?
  > **Decision: Optuna (Bayesian search), ~50-100 trials per checkpoint, tuned on the 2025 validation set.**
  >
  > **Knobs to tune:**
  > - `max_depth` (3-10) — tree depth; deeper = more complex.
  > - `learning_rate` / `eta` (0.01-0.3) — shrinkage per tree; lower = more trees, better generalization.
  > - `n_estimators` — set high, pick via **early stopping** on val log-loss.
  > - `min_child_weight` — min samples per leaf; higher = more conservative splits.
  > - `subsample` (0.6-1.0) — row subsampling per tree; regularization.
  > - `colsample_bytree` (0.6-1.0) — feature subsampling per tree.
  > - `reg_alpha` (L1), `reg_lambda` (L2) — leaf-weight penalties.
  >
  > **No nested CV for v1.** 2025 val set (~9K games) is large enough, and 2026 held-out test provides the honest final number. Saves compute. Objective = val log-loss (aligns with calibration goal, not just accuracy).

- [x] **Train/test split strategy**
  Which years to train on, which to validate, which to test? How many games do we need? Do we split by tier or not?
  > **Decision: temporal split, with walk-forward CV as a separate confidence-check exercise.**
  >
  > | Split | Years | Purpose |
  > |---|---|---|
  > | **Train (dev)** | 2020-2024 | Fit the model during development |
  > | **Val (dev)** | 2025 | Hyperparameter tuning, early stopping |
  > | **Test** | 2026 | Final held-out evaluation, calibration check — touched exactly once |
  > | **Walk-forward CV** | 2020-2025 (multiple folds) | Separate exercise — measure meta-drift across years |
  > | **Production model** | Refit on 2020-2025 with chosen hyperparams | Deploy for live 2026 betting |
  >
  > **Why temporal:** matches production reality (we predict future games from past data), naturally avoids Bo5 series leakage (no series spans years), and respects no-future-leakage. Random splits would scatter sibling Bo5 games across train/test, inflating apparent accuracy because the model effectively gets told the answer through context-rich sibling games (same teams, same day, same patch, often correlated drafts).
  >
  > **Why walk-forward as a separate exercise:** folds = (train 2020-2022 → val 2023), (train 2020-2023 → val 2024), (train 2020-2024 → val 2025). Average performance across folds gives a realistic accuracy estimate across meta shifts (chemtech 2022, grubs 2024, atakhan 2025). Used as a basis for Kelly stake sizing and a sanity check on whether accuracy survives meta drift — does not change the production model.
  >
  > **2026 discipline:** evaluated exactly once at the end. Any hyperparameter or architecture change made in response to 2026 numbers contaminates the test set and we lose our honest measurement. If we need to iterate after seeing 2026 results, we need a new held-out window.
  >
  > **Per-tier splits:** not done — `tier` is a feature of the unified model (per the "one model per tier or one unified model" decision above). Per-tier specialization is a v1.5 experiment.
  >
  > **Class balance:** every game has one winner and one loser, so raw class balance is exactly 50/50. Filters applied (datacompleteness, no-winner drop, unmapped-tier drop, game-too-short-for-checkpoint) shouldn't materially skew this since they're winner-agnostic. Verify empirically once the dataset is built.

- [x] **Calibration verification**
  The model needs to output calibrated probabilities — "70% confident" should mean it's right ~70% of the time. How do we measure and ensure this? (Brier score, calibration curves, etc.)
  > **Decision: measure with Brier score + reliability diagram per checkpoint; correct with isotonic regression if needed.**
  >
  > **Measurement:**
  > - **Brier score** on val set — overall calibration + sharpness, lower = better.
  > - **Reliability diagram** — 10 bins, predicted vs actual WR, should hug the diagonal.
  > - Per-checkpoint (each model calibrates independently).
  >
  > **Correction (if miscalibrated):** **isotonic regression** fit on val set. Chosen over Platt scaling because isotonic is non-parametric / flexible and we have enough data (~9K val games per checkpoint) to fit it without overfitting. Platt (sigmoid) is the fallback if isotonic misbehaves on sparse regions.
  >
  > **Why this matters for betting:** edge calculation is `model_prob − poly_implied_prob`. A miscalibrated 0.70 that's really 0.60 inflates apparent edge and blows up Kelly sizing.

- [x] **Class balance check**
  After filtering the data (removing incomplete rows, specific tiers, etc.), is the win/loss ratio still roughly 50/50? If not, does it affect model training? Every game has a winner and a loser, so it should be balanced in raw form — but filtering could skew it.
  > **Decision:** Should be ~50/50 by construction — every game has exactly one winner and one loser, and all filters (datacompleteness, no-winner drop, unmapped-tier drop, game-too-short-for-checkpoint) are winner-agnostic. The random-perspective stage also can't skew it (independent of `result`). **Verify with one line after pipeline runs**; if materially off, it's a pipeline bug, not a modeling problem.

---

## 3. Polymarket Integration & Live Execution

- [x] **Should Polymarket odds be a model input?**
  What are the pros and cons of feeding Poly's pre-game odds into our model vs. keeping our model fully independent and only comparing after?
  > **Decision: keep the model independent.** Poly odds are not a feature. The strategy is "our model disagrees with Poly → bet"; if Poly is an input, the model learns to agree with it and the edge collapses by construction. Poly odds are used only post-hoc as the comparison target for edge calculation.

- [x] **Live game data API**
  What API can we use to get real-time game state? Riot Games API? Third-party? Does it expose the same features we train on (gold, kills, objectives at checkpoints)?
  > **Decision:** For v1 we use the **unofficial lolesports livestats API** (`feed.lolesports.com/livestats/v1/window/{gameId}` and `/details/{gameId}`). Discovery of gameIds via the `esports-api.lolesports.com/persisted/gw/getSchedule` + `getEventDetails` endpoints with the well-known public `x-api-key`. Verified working against live LCK game `115548128962906252` (2026-04-16).
  >
  > **Why this source:**
  > - **Free and working today.** Paid providers (PandaScore $2-10K/mo; GRID commercial) are unjustifiable before we've validated the edge.
  > - **GRID Open Access does not cover LoL** — currently CS2/Dota 2 only, "more titles coming soon."
  > - **Same pipeline as Oracle's Elixir.** Our training data comes through Riot's LoL Esports Data Portal (Bayes → GRID), meaning reasonable feature parity with what we train on.
  > - **Sub-second frame cadence.** Real-time enough for in-play betting, not polling-based.
  > - **Coverage.** All Riot-broadcast leagues appear on lolesports.com (LCK/LPL/LEC/LCS/Worlds/MSI). ERLs and TCL coverage still to be verified but likely present since they're on the lolesports site.
  >
  > **Feature coverage vs. Oracle's Elixir training columns:**
  > - ✅ **Directly available:** `goldat{X}`, `killsat{X}`, `csat{X}` (sum player creepScore), `golddiffat{X}` (derived), `dragons` with type (chemtech/hextech/infernal/etc.), `barons`, `towers`, `inhibitors`, per-player K/D/A
  > - ❌ **Missing — not exposed in `/window` or `/details`:** `xpat{X}` (only `level` is available, not XP), `heralds`, `void_grubs`, `turretplates`, `atakhans` (moot for 2026), full `visionscore`, `dpm`/`damagetakenperminute`
  > - ⚠️ **Derivable from frame deltas:** `firstblood`, `firstdragon`, `firstbaron`, `firstherald`, `firsttower`
  >
  > **Implication for training feature set:** build the v1 model **only on the intersection feature set** (gold, kills, CS, dragons+type, barons, towers, inhibitors + derived firsts). Drop XP, heralds, void grubs from training — they have some signal but not enough to justify a mismatch between training and production feature availability. If walk-forward validation shows the intersection model works, we ship. If it doesn't, we reassess before investing in paid data.
  >
  > **Risks we're accepting:**
  > - **Unofficial API** — no SLA, can break or be rate-limited any time. Acceptable for v1 prototype; not for scaled production.
  > - **ToS ambiguity** — reverse-engineered endpoint being used for a betting-adjacent product. Low-volume research use is likely below the radar; Poly integration changes the calculus if we scale.
  > - **Elder dragon encoding** — `dragons` array may or may not include `"elder"` as a type; needs confirmation from a late-game sample.
  >
  > **Open sub-questions for later:**
  > - Does an events endpoint exist that exposes herald/grub kills? (5-min probe if we decide these features matter.)
  > - ERL/TCL coverage — do those games appear in the schedule endpoint and have livestats frames?
  > - Fallback plan: if Riot tightens the unofficial API, pivot to GRID commercial quote or PandaScore.

- [x] **Polymarket API for programmatic betting**
  Can we place bets via API? What are the docs, rate limits, and requirements?
  > **Decision: not needed for v1 — manual execution.** Model generates edge signals; user places bets manually on polymarket.com. Programmatic betting (CLOB API via `py-clob-client`, funded Polygon wallet, API key) deferred to v2 once edge is proven and volume justifies automation.

- [x] **Market liquidity & spread**
  How much volume do LoL markets on Polymarket typically have? What's the spread? Is there enough liquidity to actually execute on the edge we find?
  > **Finding:** Liquidity is sufficient. Volume seen at $7K-$51K per game on B-tier ERLs (section 4 manual research), larger on S-tier. **Spread ~1% on good games** (user-observed from manual trading). Edge threshold must clear spread + any Poly fees; 1% spread means a ~2-3% modeled edge is the practical minimum worth betting. Systematic liquidity/spread study deferred to v2 (when automation starts).

- [x] **Latency: game state change -> odds update**
  How quickly do Polymarket odds react to in-game events? This defines our edge window.
  > **Finding:** Poly reacts **fast to discrete high-salience events** — dragon kills, baron kills, aces (user-observed). Reaction to gradual state (accumulated gold lead, CS diff, tower damage without a take) is unquantified.
  >
  > **Implication for v1:** edge likely lives in gradual state rather than discrete events — by the time Poly has adjusted to a baron, our 25-min model saw it too. The bet thesis is "our model values a 3k gold lead at min 15 more correctly than the crowd", not "we react to baron faster than Poly". Manual execution is fine for this.
  >
  > Systematic latency measurement deferred to v2.

- [x] **Bankroll management**
  How do we size bets? Kelly criterion? Fractional Kelly? What's the strategy?
  > **Decision: quarter-Kelly (k=0.25) with a 2% per-bet hard cap.**
  >
  > **Kelly formula (binary bet):** `f* = (b·p − q) / b` where `p` = model prob, `q = 1−p`, `b` = net-odds-to-1 (for a $0.60 YES contract, `b = 0.40/0.60 ≈ 0.67`). Stake = `0.25 · f* · bankroll`.
  >
  > **Why fractional over full Kelly:** full Kelly assumes exact `p`. Our `p` is a noisy model output. Quarter-Kelly gives ~88% of full-Kelly growth rate with ~6% of variance — asymmetric tradeoff in our favor when edge estimates are uncertain.
  >
  > **Why the 2% cap on top:** circuit-breaker against bugs. If a data glitch makes the model output `p=0.95` against Poly's `0.50`, quarter-Kelly still sizes ~10% of bankroll — the cap caps the blast radius of a bad edge estimate at 2%. Trades a tiny bit of growth in rare legitimate-huge-edge cases for bounded tail risk.
  >
  > **Revisit cap after ~100 live bets** with calibration matching backtest. Consider raising to 3-4% only if live edge is confirmed.

---

## 4. Data Exploration Tasks

Concrete scripts to write during discovery.
All script outputs should be saved to `data/exploration/` as JSON (or CSV where appropriate) so we can reference results quickly without re-running.

- [x] Load full dataset, report row counts per year and per league/tier
  > **Output** (`01_row_counts.json`): ~85K total games across 2014-2026 (complete only). Ramps from 814 (2014) to 10K+/year (2020-2025). 2026 has ~3K (partial year). 121 unique leagues. Top: LCK (5,138), LDL (2,577), TCL (2,498), VCS (2,429), LCKC (2,344), LPL (2,263). **LPL data quality issue**: drops from 6,971 to 2,263 after complete-only filter (~68% loss).

- [x] Check null rates for key columns across years
  > **Output** (`02_null_rates.json`): 10/15 min columns: 0% null. 20 min: <1% (short games). 25 min: 3-10% (games ending before 25 min). Core columns (result, side, gamelength, dragons, barons, towers): 0%. Feature availability by era: heralds from 2016 (patchy until 2018), void_grubs from 2024, atakhans 2025 only. If we use newer features, training data is limited to recent years. Core features are clean across all years.

- [x] Confirm atakhan column presence/absence by year
  > **Output** (`02_atakhan.json`): Atakhan data exists only in 2025-2026. Added in 2025 Season 1 (patch 25.S1.1), **removed in 2026** (patch 26.1). 246 nonzero values in 2026 are from pre-removal patch games. Effectively a 2025-only feature.
- [x] Compute Blue/Red side win rate by year and league
  > **Output** (`03_side_winrate.json`): Blue side wins 52-58% across all years (most recent years ~52-53%). Consistent across all top 20 leagues — no league has Red side advantage. VCS closest at 51.3% Blue. Side is a real, consistent signal worth including as a feature.
- [x] Validate data integrity across full dataset
  > **Output** (`04_data_integrity.json`): 85,313 complete games. 12 rows per game, sides, result values, opp_* mirroring, golddiff computation, team kills/deaths/assists == player sums: all PASS. Two minor issues: (1) 2 games with no winner (both result=0) — filter out. (2) 2,040 games (~2.4%) have checkpoint data slightly beyond gamelength — all near the 25-min boundary (games ending at ~24:30-24:59 with `*at25` data). Not a real issue, just fuzzy checkpoint timing.
- [x] List all distinct leagues in our dataset
  > **Output** (`05_distinct_leagues.json`): 121 unique leagues with game counts and year spans. Top: LCK (5,138), LDL (2,577), TCL (2,498), VCS (2,429).
- [x] Scrape Liquipedia tier classifications (S/A/B/C/D)
  > **Output** (`06_liquipedia_tiers.json`): Scraped via MediaWiki API. S-tier: 17 leagues (Worlds, MSI, major pro leagues). A-tier: 64 leagues (minor pro + development). B-tier: 20 leagues (ERLs/national). C-tier: 17. D-tier: 11. TCL forced to B-tier only (appeared in both A and B).
- [~] Map each league to its tier
  > **Output** (`07_league_tier_matching.json`): 70 of 121 leagues matched (80.7% of games). S: 20 leagues/21K games, A: 28 leagues/28K games, B: 22 leagues/20K games. 51 leagues unmatched (16K games) — mostly LATAM nationals, academy leagues, and small/unknown tournaments. Matching uses: exact Liquipedia name match, confirmed abbreviation aliases, and direct ERL assignments. Unmatched leagues left for future investigation if needed.
- [x] Check which tournaments/leagues appear on Polymarket (manual research)
  > **Finding**: Polymarket covers all tiers depending on the season. Active markets seen on B-tier ERLs (LES, LIT, Prime League, Road of Legends, TCL) with $7K-$51K volume per game, plus S-tier (Esports World Cup). Liquidity is real and spans S/A/B tiers.
- [x] Determine minimum useful starting year for training data — were there major game/meta/tournament structure changes that make older data irrelevant? Or does including all years help?
  > **Decision:** Training cutoff will be **2020 or 2022** — we'll train both and compare empirically before committing. See `08_training_year_cutoff.json`.
  >
  > **Why these two:**
  > Both candidates train through 2025 inclusive (see "Handling of 2025/2026" below); 2026 is held out.
  > - **2020-2025** (~59K games): First full year of the Dragon Soul era (soul added preseason 2020, patch 9.23). Matches Riot/AWS's own Win Probability model cutoff (patch 10.4). Gamelength and kill pace have stabilized at modern values (~31min median, ~14 kills/team).
  > - **2022-2025** (~38K games): Adds Chemtech + Hextech drakes to the pool, so the `chemtechs`/`hextechs` columns are meaningful throughout. Caveat: chemtech was disabled for part of 2022 — still a cleaner feature space than mixing 2020-2021 (pool of 4) with later years (pool of 6).
  >
  > **Why not earlier:**
  > - **2014-2015: broken data.** `dragons` and `opp_dragons` are zero-filled even when `firstdragon=1` (verified in 2015 sample). Elemental-type columns populated with 0 despite the elemental system not existing yet. Unusable.
  > - **2016-2018: different game.** No dragon soul, no map transforms (2016-2018 preseason), herald data patchy until 2018, median gamelength 33-35min vs 31min modern. Pace and objective incentives are structurally different.
  > - **2019: transitional.** Elemental Rift map transforms existed but **no Dragon Soul** — the single biggest modern objective incentive is missing. Including 2019 mixes soul-era and pre-soul games under one label.
  >
  > **Why not later (2024+ or 2025+):**
  > - Drops below ~20K games — tight for per-tier modeling and calibration.
  > - 2025 is atakhan-era with Baron pushed to 25:00 and Feats of Strength active (all one-year-only mechanics); 2026 removes atakhan/feats, returns Baron to 20:00 (matching 2020-2024), and adds role quests + item rework + only 2 elemental drakes pre-soul. 2025 is the most anomalous year relative to 2020-2024 and 2026.
  >
  > **Handling of 2025/2026:** **include 2025 in training** (cutoff-through-2025) — adds ~9K recent games. Atakhan/heralds/void_grubs/chemtechs/hextechs are all dropped from the v1 feature set (per section 1 decisions — drop features not at inference, drop cumulative counts for leakage), so the era-missing-column handling is moot. **Hold out 2026 as validation / deployment context** — it's a structurally different patch (no atakhan/feats, Baron back to 20:00, only 2 elemental drakes pre-soul, role quests + item rework) and is our actual betting target, so honest validation belongs there.
  >
  > **Residual concern — Feats of Strength in 2025 training data:** Feats of Strength (2025-only mechanic, removed 2026) granted free boot upgrades + stat boost to the first team reaching 3-of-{firstblood, first tower, first epic}. Its effects bleed into `goldat20`, `goldat25`, `killsat20/25` via compounding power advantage — not a CSV column we can drop. ~50% of 2025 games earned feats (~4-5K games with subtly different feature distributions vs 2020-2024 and 2026). Accepted risk for v1; monitor on 2026 test set; if calibration is specifically off in the 20/25-min models, revisit via sample-weighting or excluding 2025.

---

## 5. Open Questions

- [x] **Any external data sources worth adding?**
  Champion win rates, draft meta stats, historical betting odds, team rankings — what's available and impactful?
  > **Decision: defer to v2.** No external sources in v1. For v2, ordered by value: (1) team ratings / Elo — backbone of v2 pre-game model, (2) champion WR / draft meta — for draft modeling, (3) historical odds from other books — validation only.

- [x] **Experiment tracking approach?**
  How do we compare model iterations? MLflow? Simple CSV log? Something else?
  > **Decision: simple CSV log for v1.** Columns: `run_id, timestamp, checkpoint, hparams_json, val_logloss, val_brier, val_auc, notes`. ~4 models × ~50 Optuna trials × a few iterations = hundreds of runs, not thousands — MLflow is overkill and adds infra. Upgrade to MLflow in v2 if iteration rate justifies it.

- [x] **Which tournaments does Polymarket actually cover?**
  Need to check Polymarket directly. No point modeling leagues that don't have markets.
  > **Finding (from section 4 manual research):** Polymarket covers **all tiers depending on season** — active markets seen on B-tier ERLs (LES, LIT, Prime League, Road of Legends, TCL) with $7K-$51K volume, plus S-tier (Esports World Cup, Worlds). Coverage justifies modeling across S/A/B tiers as decided in 1.7. No modeling changes needed.

- [x] **Team/org links across data sources**
  Do team IDs in Oracle's Elixir map cleanly to the team names Polymarket uses? Any name mismatches to handle?
  > **Decision: not a v1 concern, v2 blocker.**
  >
  > **v1:** irrelevant. Model consumes `gameid` from Oracle's Elixir (training) and livestats `gameId` (inference). User reads Poly market manually and opens the matching livestats game. No name resolution needed.
  >
  > **v2:** required. Automated bet placement needs to match "this Poly market" ↔ "this livestats fixture" by league + scheduled time + team names. Expect mismatches (abbreviations, "SKT T1" vs "T1", org rebrands, EU-style name variants). Plan: build a mapping table seeded from Liquipedia's canonical names, with a manual-review step for unmatched Poly markets. Treat as a dedicated v2 discovery task.


# Discovery Checklist

Questions to answer and explorations to run before we can write a proper system spec.

Status: `[ ]` = open, `[~]` = in progress, `[x]` = resolved

When a question is resolved, add a `> **Decision:** ...` block under it with the reasoning and conclusion.

---

## 1. Data & Feature Engineering

- [ ] **Which columns to include and how to structure time-frames?**
  Do we build separate feature sets per checkpoint (10/15/20/25 min), or keep everything in one flat row with all time-frame columns together? What are the tradeoffs?

- [ ] **Team rows only or player rows too?**
  Do we need the 10 player rows per game, or are the 2 team-aggregate rows enough?

- [ ] **How to handle the atakhan column?**
  Verify it only exists in 2025+ data. How do we treat it for earlier years — drop it? Keep with nulls?

- [ ] **Do we want to incorporate champion/draft data?**
  Champions are in the data (picks, bans). Is draft modeling worth the complexity, or do we leave it out? If yes, how to encode it (one-hot per champ? win rates? composition archetypes?).

- [ ] **How to handle patch/meta shifts?**
  The game changes significantly between patches. Do we include patch as a feature? Group into eras? Or does the model learn this implicitly from the data?

- [ ] **How to encode Blue/Red side?**
  Blue side has a known historical WR advantage in pro. How significant is it in our data?

- [ ] **How to handle non-time-dependent columns across time frames?**
  Columns like dragon/baron counts are 0 at early checkpoints. Does including them cause any issues or is it harmless?

- [x] **Filter to datacompleteness == "complete" only**
  Do we use all rows or only complete ones?
  > **Decision:** Filter to `datacompleteness == "complete"` in all scripts and in the final pipeline. Drops ~11% of games but gives us 0% nulls on time-stamped columns (10/15 min). Notable: LPL loses ~68% of its games to this filter. All exploration scripts already apply this filter.

- [ ] **Pre-processing pipeline design**
  Filtering (incomplete data, missing columns), encoding (categorical features), normalization — what does each step look like?
  Note: pay attention to null vs 0 distinction. For features that didn't exist in older game versions (atakhan pre-2025, void_grubs pre-2024, heralds pre-2016), null means "not collected" while 0 means "didn't happen." Filling nulls with 0 is truthful here (nobody slayed an atakhan in 2020), but XGBoost treats nulls and 0 differently — need to decide which approach to use.

- [ ] **Incorporate team strength / Elo prior?**
  Should we compute a rolling Elo/Glicko rating from historical results and use it as a feature? How would we build it?

- [ ] **Handle roster changes?**
  Same org can have different players mid-split. Does this matter? If we use team identity as a feature, how do we account for this?

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

- [ ] **One model per tier or one unified model?**
  More data (combined) vs. cleaner signal (per-tier) — what are the tradeoffs? Need to explore both.

- [ ] **Baseline model**
  What's the simplest model we can build to sanity-check the data before going to XGBoost?

- [ ] **XGBoost vs. alternatives**
  Is XGBoost the right choice? How does it compare to logistic regression, LightGBM, or other options for this type of problem?

- [ ] **Hyperparameter tuning strategy**
  Which hyperparameters matter most (learning rate, max_depth, n_estimators, etc.)? How do we tune them?

- [ ] **Train/test split strategy**
  Which years to train on, which to validate, which to test? How many games do we need? Do we split by tier or not? Needs investigation once we see the data distribution.
  Note: training cutoff decided in section 4 (2020 or 2022, through 2025 inclusive; 2026 held out). Open sub-questions: per-tier splits, class-balance effects after filtering.
  **Consider walk-forward validation as a pre-deployment confidence check** — train on 2020-N, test on N+1, stepping forward (e.g. train 2020-2023 → test 2024; train 2020-2024 → test 2025). Purpose: measure how accuracy/calibration degrade across meta shifts (chemtech 2022, grubs 2024, atakhan 2025) to get a realistic prior for 2026 performance before betting real money. Does not change the production model (which trains on everything 2020-2025) — purely a sanity-check on whether the hypothesis holds and a basis for Kelly stake sizing.

- [ ] **Calibration verification**
  The model needs to output calibrated probabilities — "70% confident" should mean it's right ~70% of the time. How do we measure and ensure this? (Brier score, calibration curves, etc.)

- [ ] **Class balance check**
  After filtering the data (removing incomplete rows, specific tiers, etc.), is the win/loss ratio still roughly 50/50? If not, does it affect model training? Every game has a winner and a loser, so it should be balanced in raw form — but filtering could skew it.

---

## 3. Polymarket Integration & Live Execution

- [ ] **Should Polymarket odds be a model input?**
  What are the pros and cons of feeding Poly's pre-game odds into our model vs. keeping our model fully independent and only comparing after?

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

- [ ] **Polymarket API for programmatic betting**
  Can we place bets via API? What are the docs, rate limits, and requirements?

- [ ] **Market liquidity & spread**
  How much volume do LoL markets on Polymarket typically have? What's the spread? Is there enough liquidity to actually execute on the edge we find?

- [ ] **Latency: game state change -> odds update**
  How quickly do Polymarket odds react to in-game events? This defines our edge window.

- [ ] **Bankroll management**
  How do we size bets? Kelly criterion? Fractional Kelly? What's the strategy?

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
  > - 2025 is atakhan-era (feature only available one year); 2026 removes atakhan and shifts Baron to 12:00. Both are anomalies relative to 2020-2024.
  >
  > **Handling of 2025/2026:** **include 2025 in training** (cutoff-through-2025) — adds ~9K recent games. **Keep the `atakhan` column as a feature, leave nulls as nulls** (do not 0-fill) — XGBoost's sparsity-aware splits learn a default direction for missing values natively, and 0-filling would collapse the semantic difference between "feature didn't exist" (2020-2024) and "team took zero atakhans" (2025). Atakhan won't fire in production since 2026 has no atakhan, but it's a cheap inclusion and unlocks the 2025 games. **Hold out 2026 as validation / deployment context** — it's a structurally different patch (no atakhan, Baron at 12:00, only 2 elemental drakes pre-soul) and is our actual betting target, so honest validation belongs there.

---

## 5. Open Questions

- [ ] **Any external data sources worth adding?**
  Champion win rates, draft meta stats, historical betting odds, team rankings — what's available and impactful?

- [ ] **Experiment tracking approach?**
  How do we compare model iterations? MLflow? Simple CSV log? Something else?

- [ ] **Which tournaments does Polymarket actually cover?**
  Need to check Polymarket directly. No point modeling leagues that don't have markets.
  Links to investigate:
  - Polymarket: search for "League of Legends" or "LoL" markets
  - Oracle's Elixir leagues list: cross-reference with what Poly covers

- [ ] **Team/org links across data sources**
  Do team IDs in Oracle's Elixir map cleanly to the team names Polymarket uses? Any name mismatches to handle?


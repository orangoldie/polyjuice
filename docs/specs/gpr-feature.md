# Spec: GPR team-strength feature for in-game logistic model

Status: draft, pre-implementation
Date: 2026-04-24

## Objective

Inject Riot's Global Power Rankings (GPR) team-strength rating into the in-game checkpoint logistic baseline as a **fixed-coefficient logit offset** built from Riot's published expected-score formula. Test whether it (a) improves overall log-loss on the 2026 S-tier holdout and (b) closes the T=10 S-tier band gap identified in `scripts/exploration/15_band_gap_bootstrap.py`.

## Background

- The in-game checkpoint model uses snapshot features (gold diff, XP diff, towers, dragons, etc.) at T ∈ {10, 15, 20, 25}. It has no notion of pre-game team strength.
- Phase 1 calibration diagnostics found one borderline-real miscalibration in the 70–75% betting band: T=10 S-tier is ~9pp overconfident (n=102, CI barely excludes zero). The hypothesized mechanism is that strong teams come back from min-10 gold deficits more often than the model expects — team strength is doing work the current features don't capture.
- Riot's GPR is an Elo-based team rating (80% team Elo + 20% league Elo). Per-match updates on outcomes only; 2-year window for team Elo, 3-year window for league Elo. Covers LCK, LPL, LEC, LCS/LTA, LCP, CBLOL from 2024 onward — i.e. S-tier only. Does NOT cover VCS. Source: [dev diary](https://lolesports.com/en-GB/news/dev-diary-unveiling-the-global-power-rankings).
- Riot publishes the expected-score formula explicitly: `W_e = 1 / (10^(−dr/600) + 1)`, where `dr` is the difference of *Power Scores* (the 0.8·team + 0.2·league composite). They report **65% predictive accuracy on S-tier** — directly comparable to our holdout slice.
- GPR is preferred over a homegrown Oracle's Elixir Elo as the starting point because (a) league Elo / regional strength is non-trivial to replicate, (b) zero engineering cost on the rating itself, (c) it's what the market references. Homegrown Elo remains the fallback if this experiment fails and as v2 groundwork.

## Success criteria

The experiment succeeds if **both** of the following hold on the 2026 S-tier holdout:

1. **Aggregate log-loss improves across all 4 checkpoints**, paired-bootstrap CI for `(baseline+GPR) − baseline` excludes zero (improvement direction).
   - Resample unit: `gameid`.
   - Per-game statistic: mean of per-checkpoint log-loss over the checkpoints that game reaches (mean, not sum, so game length doesn't warp weighting).
   - Pairing: `(baseline_score[gameid], baseline+gpr_score[gameid])`.
2. **T=10 S-tier in-band ECE narrows.** Re-run `scripts/exploration/15_band_gap_bootstrap.py` on the new model and check that the [0.65, 0.80) S-tier T=10 gap shrinks vs the current baseline's −9.4pp. Required to be directionally correct (gap moves toward zero); the bootstrap CI on the new gap need not exclude zero given n=102.

There is no learned coefficient on the GPR signal under the chosen design (the offset is fixed-coefficient), so no coefficient sign-check.

Secondary (does not gate ship/no-ship, but informs next step):
- AUC delta per checkpoint.
- ECE in [0.65, 0.80) per checkpoint at T={15, 20, 25}.
- Effect on overall reliability diagrams.

If success: ship, then consider XGBoost re-run as a follow-up (new feature creates fresh interaction candidates that didn't exist when XGB v1 failed).

If failure: document as negative result. Candidates then become homegrown Elo, trajectory features (`golddiffat15 − golddiffat10`), or accepting the T=10 S-tier gap as n=102 noise.

## Scope

- **Leagues:** S-tier only (LCK, LPL, LEC, LCS/LTA N+S, LCP, CBLOL). VCS excluded from the *experiment scope* — GPR doesn't cover it. VCS 2024+ rows remain in the training set with `has_gpr=0, gpr_diff=0` (i.e. they pool into the unrated population); documented here so nobody re-litigates it — the `has_gpr` coefficient is not interpreted.
- **Checkpoints:** all four, T ∈ {10, 15, 20, 25}.
- **Training window:** 2020–2025 inclusive (matches baseline production fit). GPR feature populates per the year/anchor map in the data plan; rows outside it carry `has_gpr=0, gpr_diff=0`.
- **Holdout:** 2026, S-tier only.

## Non-goals

- Homegrown Oracle's Elixir Elo.
- Pre-game v2 model.
- XGBoost re-run (sequenced after logistic confirms the feature is useful).
- Calibration overhaul; isotonic stays as-is.
- VCS coverage via alternate sources.
- Per-tier calibrator (deprioritized per chat-summary).

## Data plan

### Scrape

- Source: `lolesports.com/en-GB/gpr/{year}`. The page is JS-hydrated; the real API lives behind AWS and is inspectable via devtools.
- What's actually available:
  - **Past years (2024, 2025):** only the final per-team rating at year end. No per-match history is retained or published. One value per team per completed year.
  - **Current year (2026):** the live rating, which updates per-match. Re-scraping at any point gives a current snapshot; we do not get a back-history of intra-year updates.
- Output shape: one row per `(team, year)` with `team_name, year, power_score, team_elo, league_elo, league`. For 2024 and 2025, `power_score` is the year-end final. For 2026, it's whatever the live rating was at scrape time (re-scrape on a cadence as the year progresses).
- Implementation: direct API call preferred; Playwright fallback if JS hydration is required.

### Team name normalization

- Primary risk in the whole experiment. Join key between GPR and Oracle's Elixir is team name, which diverges (SKT vs T1, Samsung vs Gen.G, branding changes mid-season, regional variants).
- Build a manual alias table: `data/gpr/team_aliases.csv` mapping `gpr_team_name → oe_team_name`.
- Validation is structural, not sampled: enumerate every distinct `teamname` in 2024+ S-tier OE data, count games per team, and require every team with ≥20 games to be mapped (either to a GPR name or explicitly flagged as uncovered, e.g. VCS). Then a 50-random-game spot-check on top as a sanity pass.
- Unmatched teams → log and treat as `has_gpr=0` for that game. Acceptable loss if <5% of games; re-examine if higher.

### Join to games — year/anchor map

Each per-match GPR update is `R_new = R_prev + K · I · (W − We)` (per the dev diary). The rating *as of date D* is therefore a path integral over every match in the rolling window before D — by construction the rating "knows" the result of any in-window game. Using a snapshot to predict a game that contributed to that snapshot is leakage; the rolling 2-year team-Elo window changes which past matches feed the rating, not the temporal direction.

This pins the year/anchor map:

| Game year | Anchor | Why |
|---|---|---|
| ≤ 2023 | none — `has_gpr=0` | GPR didn't exist; nothing to anchor to |
| 2024 | none — `has_gpr=0` | end-2024 is leaky for these games; nothing earlier exists |
| 2025 | end-2024, held constant for the year | computed before any 2025 match — clean. Up to 12 months stale by Dec |
| 2026 | end-2025, held constant for the year | clean. Up to 12 months stale by Dec |

What we have available but are deliberately *not* using:
- **Live 2026 rating** (re-scrapable). A single mid-year live snapshot applied uniformly across 2026 reintroduces leakage on earlier 2026 games. A strict per-game `snapshot_date < game_date` join would be clean but requires per-game snapshots we don't have. The live snapshot's natural home is bet-time inference (predicting one upcoming match with the rating "as of now"), which is downstream of this experiment and not in scope.

2024 rows stay in the training pool with `has_gpr=0` (rather than being dropped) — they help the non-GPR coefficients regardless, and `has_gpr` cleanly absorbs the population shift between rated and unrated rows.

## Feature construction

Stage 5 emits two new columns on the per-checkpoint feature matrix:

| Column | Definition | Notes |
|---|---|---|
| `gpr_diff` | `power_score[teamname] − power_score[opp_teamname]`, 0 when missing | Per the year/anchor map, populated for 2025 (end-2024 anchor) and 2026 (end-2025 anchor); 0 elsewhere. Each game appears as two rows (one per team's perspective); computed per row. |
| `has_gpr` | 1 if both teams have a valid snapshot per the year/anchor map, else 0 | Population indicator — flags the rated subset so the model can pick up any mean-level shift in the unrated population without it bleeding into the offset. |

`logit_p_riot` is **not** a stored column — it's computed at model-train/predict time from `gpr_diff` (see "Riot's prior as a fixed-coefficient logit offset" below). Keeping `gpr_diff` as the persisted column means the path-A fallback (learned coefficient on `gpr_diff` directly) is a one-line model-side revert without re-running stage 5.

Deliberately not including `league_elo_diff` as a separate column in v1 — high collinearity with `gpr_diff` (it's a component of the power score), adds a coefficient, complicates interpretation. Revisit if v1 succeeds and we want to tease apart team vs regional strength.

### Riot's prior as a fixed-coefficient logit offset (chosen design)

GPR enters the model as a **fixed-coefficient logit offset** built from Riot's published expected-score formula:

```
p_riot      = 1 / (1 + 10^(−gpr_diff / 600))      # Riot's W_e, applied to row's gpr_diff
logit_p_riot = log(p_riot / (1 − p_riot))
logit P     = α + β·X + 1 · logit_p_riot          # offset with coefficient fixed at 1
```

For rows with `has_gpr = 0`, `gpr_diff = 0` → `p_riot = 0.5` → `logit_p_riot = 0`, so the offset contributes nothing on unrated rows (as intended).

Why this design rather than learning a coefficient on `gpr_diff`:

- **Riot's prior is calibrated.** They report 65% predictive accuracy on S-tier matches — directly the slice we're operating on. A learned coefficient would re-discover what Riot has already baked in, from a thinner slice (~4–5K leakage-clean games).
- **Leakage of GPR matters less.** Under the year/anchor map there's no leakage at training time anyway, but with a fixed-coefficient offset the model cannot overfit GPR even if the input were noisier than expected — there's no coefficient knob to overfit through.
- **Divisor is published, not estimated.** Riot's `W_e` divisor is 600 (flatter than canonical Elo's 400 — at a 200-point Power Score gap Riot says 68%, canonical Elo says 76%; reasonable for noisier esports ratings). One less hyperparameter to fit, one less unknown.
- **Cleaner conceptual framing.** "Given Riot's prior, what does in-game state add?" is the question we actually want to answer.

#### Prior sanity check (gates the fit)

The published 65% is Riot's number on their evaluation set, not ours. Before fitting the offset model, compute `p_riot` from end-2024 anchors on 2025 S-tier games and check accuracy + log-loss + Brier of `p_riot` alone (as a standalone predictor, no in-game features). Pass criteria:

- Accuracy ≥ 62% (a few pp tolerance below Riot's 65% — ours is one calendar year, ~4–5K games, with non-trivial sample noise).
- Log-loss < log(2) ≈ 0.693 by a clear margin (otherwise `p_riot` is no better than uninformative).

If either fails, Riot's prior doesn't transfer to our slice and we revert to **path A fallback** (see below). This is the only B → A pivot trigger.

#### Fallback: path A (learned coefficient on `gpr_diff`)

If the prior sanity check fails, run path A: `gpr_diff` enters the logistic as a normal feature with a learned coefficient. Single-line model-side change because `gpr_diff` is already the persisted stage-5 column. Under path A, the success criteria add a coefficient sign check:

- `β_gpr` strictly positive at T=10 with `|β/SE| > 2`.
- Non-negative at T={15, 20, 25}.

This is unchanged from prior versions of the spec; documented here so the fallback path is fully specified and we don't reopen the design discussion if we end up taking it.

## Training & evaluation

- Same split and filters as production baseline logistic (see `docs/baseline-logistic.md`).
- Refit logistic with `logit_p_riot` added as a *fixed-coefficient offset* (coefficient = 1, not learned), plus `has_gpr` as a normal binary feature with a learned coefficient. The remaining feature set is unchanged.
- `sklearn.LogisticRegressionCV` does not natively support an offset term. Implementation will use either statsmodels (`GLM(..., family=Binomial(), offset=...).fit_regularized(...)` with a small custom 5-fold CV loop for `C` selection) or a tiny custom MLE in `scipy.optimize`. Decision deferred to implementation; tests must compare both ends of the same minimum.
- One fit per checkpoint.
- Metrics on 2026 S-tier holdout:
  - Log-loss (per checkpoint + aggregate, paired bootstrap CI vs current baseline).
  - AUC (per checkpoint).
  - ECE in [0.65, 0.80) band (per checkpoint).
  - Re-run script 15 bootstrap on T=10 S-tier band gap.
- Report `β_has_gpr` and the intercept per checkpoint (the only two coefficients sensitive to the GPR injection). No `β_gpr` to report under the chosen design.

## Risks

- **Team-name join drift.** Silent joins to wrong teams corrupt the feature — and unlike path A, there's no learned coefficient on GPR that would surface a sign flip. Mitigation leans harder on the structural alias-table validation and the prior sanity check (a wrong join would tank Riot-prior accuracy on the 2025 slice well below 62%).
- **Snapshot access.** If the API doesn't expose end-2024, end-2025, and current 2026 finals, the spec is shelved. Resolved by the scrape spike (step 1).
- **Prior doesn't transfer to our slice.** Riot's 65% is on their S-tier evaluation set; ours is 2025 LCK/LPL/LEC/LCS/LTA/LCP/CBLOL with end-2024 anchor. If accuracy on our slice falls clearly below 62%, the offset is biased and we fall back to path A (learned coefficient on `gpr_diff`). The sanity check is cheap and runs before any model fit.
- **Offset-coefficient assumption (β=1).** Fixing the offset coefficient at 1 assumes Riot's logit is calibrated as-is for our slice. If it's well-calibrated in shape but mis-scaled (e.g., systematically too confident), the offset will inject mis-scaled information and the in-game `β` will partially compensate — but only partially, since it can't selectively offset just the GPR component. If diagnostics suggest this (offset-only model has good ranking but poor calibration on the 2025 sanity slice), the cleanest move is path A; allowing β_offset to be learned would reintroduce all the leakage / thin-fit concerns A has.
- **VCS exclusion.** S-tier set loses VCS games for this experiment. Documented, not fixed.
<!-- leakage risk addressed — the year/anchor map in the data plan uses only end-of-prior-year snapshots for training/eval, which are computed before any in-window game. Live 2026 ratings are deliberately excluded for this reason. -->

## Code & data layout

GPR is treated as an external data source, structurally parallel to `data/lol-historical-data/`. Its ingestion is independent of the main preprocessing pipeline. Its *join* onto features is appended as the **last** stage of the main pipeline — placed there (rather than mid-pipeline, before the checkpoint split) so the stage can be removed cleanly if the experiment fails, without disturbing the earlier stages' outputs. The cost is that the join runs four times (once per checkpoint CSV) instead of once at per-game level; this is acceptable given the small join size.

### Directory layout (additions only — no reorg of existing tree)

```
data/
  lol-historical-data/                  # existing external source
  gpr/                                  # NEW — scraped external source
    snapshots.csv                       # (team, snapshot_date, event, power_score, team_elo, league_elo, league)
    team_aliases.csv                    # hand-maintained: gpr_name → oe_name
  processed/
    01_filtered.csv                     # unchanged
    02_one_row.csv                      # unchanged
    03_features.csv                     # unchanged
    checkpoint_{10,15,20,25}.csv        # unchanged — produced by stage 4
    checkpoint_{10,15,20,25}_gpr.csv    # NEW — produced by stage 5, adds gpr_diff + has_gpr

scripts/
  build_dataset.py                      # add stage 5 to STAGES list
  gpr/                                  # NEW — independent ingestion
    __init__.py
    scrape.py                           # hits lolesports API, writes snapshots.csv
    aliases.py                          # alias resolution utilities
  preprocess/
    stage1_load_and_filter.py           # unchanged
    stage2_one_row_per_game.py          # unchanged
    stage3_features.py                  # unchanged
    stage4_split_checkpoints.py         # unchanged (name + output)
    stage5_join_gpr.py                  # NEW — reads checkpoint_{T}.csv + data/gpr/snapshots.csv, writes checkpoint_{T}_gpr.csv

tests/
  # add tests for stage5 join + scripts/gpr/aliases
```

### Pipeline behavior

- `build_dataset.py` runs stages 1 → 5. Stage 5 is purely offline (reads local CSVs only); the scrape is NOT part of `build_dataset.py`.
- Scraping is a separate, user-invoked step with its own cadence:
  ```
  python -m scripts.gpr.scrape      # refresh data/gpr/snapshots.csv
  python scripts/build_dataset.py   # rebuild processed/ from raw + gpr
  ```
  Mirrors how raw OE CSVs are also refreshed manually.
- Stage 5 writes to `checkpoint_{T}_gpr.csv` rather than overwriting `checkpoint_{T}.csv`. This preserves the non-GPR checkpoint outputs, so removing stage 5 from the pipeline is a one-line change with zero ripple.

### Model wiring

- Models that want to use GPR load `checkpoint_{T}_gpr.csv`.
- Models that don't (existing baseline, existing XGBoost v1) keep loading `checkpoint_{T}.csv` unchanged.
- No shared join logic in model code — the join is done once, in stage 5.

## Sequencing

Step 1 is a true gate: if the API doesn't expose end-2024, end-2025, and current 2026 per-team finals, this spec is shelved and we reopen with a wayback-based plan or fall back to homegrown Elo. Per-match intra-year history is *not* required — the year/anchor map only consumes year-end snapshots. Steps 2–9 are contingent on step 1 succeeding.

1. **Scrape spike (gate)** — confirm the API exposes end-2024, end-2025, and current 2026 per-team finals.
2. **Team alias table** — manual + structural validation against 2024+ S-tier `teamname`s (≥20-game threshold) + 50-game spot-check.
3. **`scripts/gpr/scrape.py`** — produces `data/gpr/snapshots.csv`.
4. **`stage5_join_gpr.py`** — join logic + tests, applying the year/anchor map (≤2023 / 2024 → `has_gpr=0`; 2025 ← end-2024; 2026 ← end-2025). Emits `gpr_diff` and `has_gpr`; does *not* compute `logit_p_riot`.
5. **Wire stage 5 into `build_dataset.py`.**
6. **Riot-prior sanity check (gate).** Standalone exploration script. Compute `p_riot = 1 / (1 + 10^(−gpr_diff / 600))` from end-2024 anchor on 2025 S-tier games; report accuracy / log-loss / Brier vs constant-0.5 baseline. Pass = accuracy ≥ 62% and log-loss clearly below 0.693. If pass, proceed to step 7. If fail, fall back to path A (single-line model-side change; rest of the spec applies).
7. **`baseline_logistic_gpr.py`** — model with `logit_p_riot` as a fixed-coefficient offset (β=1) and `has_gpr` as a learned binary feature. Implementation via statsmodels GLM with offset, or custom `scipy.optimize` MLE — decided at code time.
8. **Evaluate on 2026 S-tier holdout** — log-loss / AUC / ECE per checkpoint, paired-bootstrap CI on aggregate log-loss vs current baseline.
9. **Re-run `scripts/exploration/15_band_gap_bootstrap.py`** on the new model to check whether the T=10 S-tier [0.65, 0.80) gap closes.
10. **Decision point:** ship, XGBoost follow-up, or negative-result writeup.

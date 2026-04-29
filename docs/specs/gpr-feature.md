# Spec: GPR team-strength feature for in-game logistic model

Status: draft, scrape spike done — pre-implementation
Date: 2026-04-26

## Objective

Add Riot's Global Power Rankings (GPR) team-strength rating as **four new columns (`gpr_team`, `gpr_opp_team`, `gpr_diff`, `has_gpr`) on the per-checkpoint feature matrix**, refit the logistic baseline with `{gpr_team, gpr_opp_team, has_gpr}` as model inputs (`gpr_diff` persisted but dropped at fit time, per the baseline's `*diffat{T}` convention), and test whether it (a) improves overall log-loss on the 2026 S-tier holdout and (b) closes the T=10 S-tier band gap identified in `scripts/exploration/15_band_gap_bootstrap.py`. The headline test is on the linear contrast `β_team − β_opp` per checkpoint.

## Background

- The in-game checkpoint model uses snapshot features (gold diff, XP diff, towers, dragons, etc.) at T ∈ {10, 15, 20, 25}. It has no notion of pre-game team strength.
- Phase 1 calibration diagnostics found one borderline-real miscalibration in the 70–75% betting band: T=10 S-tier is ~9pp overconfident (n=102, CI barely excludes zero). The hypothesized mechanism is that strong teams come back from min-10 gold deficits more often than the model expects — team strength is doing work the current features don't capture.
- Riot's GPR is an Elo-based team rating (80% team Elo + 20% league Elo). Per-match updates on outcomes only; 2-year window for team Elo, 3-year window for league Elo. Covers LCK, LPL, LEC, LCS/LTA, LCP, CBLOL from 2024 onward — i.e. S-tier. Also covers VCS in 2024 (the spike confirmed 108 snapshots for VCS teams that year). The VCS league itself was merged into LCP for the 2025 reorg, so there is no VCS-as-a-league post-2024 to cover. Source: [dev diary](https://lolesports.com/en-GB/news/dev-diary-unveiling-the-global-power-rankings).
- Riot publishes the expected-score formula explicitly: `W_e = 1 / (10^(−dr/600) + 1)`, where `dr` is the difference of *Power Scores* (the 0.8·team + 0.2·league composite). They report **65% predictive accuracy on S-tier** — useful as a context number for what GPR alone can do, even though we don't bake the formula into the model directly under the chosen design (see "Future option: fixed-coefficient logit offset" at the end of feature construction).
- GPR is preferred over a homegrown Oracle's Elixir Elo as the starting point because (a) league Elo / regional strength is non-trivial to replicate, (b) zero engineering cost on the rating itself, (c) it's what the market references. Homegrown Elo remains the fallback if this experiment fails and as v2 groundwork.

### Scrape spike findings (2026-04-26)

The original spec was written under the assumption that only year-end finals were retrievable for past years. The spike disproved that:

- The `lolesports.com/en-GB/gpr/{year}` pages embed the SSR'd Apollo cache directly in the rendered HTML. A single page fetch returns full team-level GPR data — no Playwright, no separate API call required.
- Each team's response includes a `teamGPRHistory` array with **per-team snapshots at ~10-day granularity** (the page's `step:10` parameter), each carrying `{gprScore, elo, rank, dateCalculated}`.
- Coverage observed:
  - 2024 (`/gpr/2024/worlds`): 80 teams × 27 history points covering Feb 10 → Nov 2.
  - 2025 (`/gpr/2025`): 60 teams × 26 history points covering Jan 22 → Nov 9.
  - 2026 (`/gpr/2026`): 58 teams × 10 history points so far covering Jan 26 → Apr 26, growing every ~10 days.
- T1's history trajectory across 2024-2025 matches lived reality (rank 1 early 2024 → rank 4-5 by Worlds 2024; rank 3 early 2025 → rank 1 by Worlds Final 2025), confirming these are the published-at-the-time ratings rather than retrocomputed values.
- Discovered GraphQL schema (visible in the page's pre-hydration query strings):
  ```
  teamGPR(step: Int, startDate: String, endDate: String, hl: Locale!, sport: Sport): [TeamGPR]
  TeamGPR {
    id, averageOpponentGPR, teamMatchRecord, teamGameRecord,
    currentTeamGPR  { gprScore, elo, rank, dateCalculated },
    previousTeamGPR { gprScore, elo, rank, dateCalculated },
    teamGPRHistory  [{ gprScore, elo, rank, dateCalculated }],
    tournaments,
    team { id, code, name, slug, image, homeLeague { id, name, slug, image } }
  }
  ```
  The endpoint URL itself was not located in the spike (the page's Apollo SSR transport bypasses the runtime client). It's not blocking for v1 — page scrape is sufficient — but worth running down later if we want finer than 10-day granularity or live 2026 inference past whatever date the page last ran.
- Practical effect: per-game leakage-free joins (`snapshot_date < game_date`, take most recent qualifying snapshot) become workable with median ~5-day staleness, max ~10. The original year/anchor map is obsolete; see "Join to games — per-game cutoff" below.

## Success criteria (joint test)

The experiment succeeds if **all** of the following hold on the 2026 S-tier holdout:

1. **Aggregate log-loss improves across all 4 checkpoints**, paired-bootstrap CI for `(baseline+GPR) − baseline` excludes zero (improvement direction).
   - Resample unit: `gameid`.
   - Per-game statistic: mean of per-checkpoint log-loss over the checkpoints that game reaches (mean, not sum, so game length doesn't warp weighting).
   - Pairing: `(baseline_score[gameid], baseline+gpr_score[gameid])`.
2. **GPR rating-gap contrast is signed correctly** (higher `gpr_team − gpr_opp_team` → higher P(`teamname` wins)). The test is on the linear contrast `β_team − β_opp`, with z-stat computed from the full coefficient covariance: `Var(β_team − β_opp) = Var(β_team) + Var(β_opp) − 2·Cov(β_team, β_opp)`.
   - At T=10: `β_team − β_opp` strictly positive, `|contrast / SE| > 2`.
   - At T={15, 20, 25}: `β_team − β_opp ≥ 0`. Late checkpoints are dominated by in-game features, so statistical insignificance there is expected and not a failure.
3. **T=10 S-tier in-band gap narrows.** Re-run `scripts/exploration/15_band_gap_bootstrap.py` on the new model and check that the [0.65, 0.80) S-tier T=10 gap shrinks vs the current baseline's −9.4pp. Required to be directionally correct (gap moves toward zero); the bootstrap CI on the new gap need not exclude zero given n=102.

Secondary (does not gate ship/no-ship, but informs next step):
- AUC delta per checkpoint.
- ECE in [0.65, 0.80) per checkpoint at T={15, 20, 25}.
- Effect on overall reliability diagrams.

If success: ship, then consider XGBoost re-run as a follow-up (new feature creates fresh interaction candidates that didn't exist when XGB v1 failed).

If failure: document as negative result. Candidates then become homegrown Elo, trajectory features (`golddiffat15 − golddiffat10`), or accepting the T=10 S-tier gap as n=102 noise.

## Scope

- **Leagues:** S-tier (LCK, LPL, LEC, LCS/LTA N+S, LCP, CBLOL) plus VCS where covered. The per-game-cutoff join applies uniformly — any team with a qualifying snapshot gets `has_gpr=1` regardless of league. In practice this means 2024 VCS games join cleanly (GPR rated VCS in 2024); 2025+ has no VCS-as-a-league to consider since it was merged into LCP. No special-case exclusion needed.
- **Checkpoints:** all four, T ∈ {10, 15, 20, 25}.
- **Training window:** 2020–2025 inclusive (matches baseline production fit). GPR feature populates for any row whose game date falls strictly after both teams' most-recent `teamGPRHistory` snapshot — in practice ≈ all S-tier rows from Feb 10 2024 onward, ~10K games. Earlier rows carry `has_gpr=0, gpr_diff=0`.
- **Holdout:** 2026, S-tier only.

## Non-goals

- Homegrown Oracle's Elixir Elo.
- Pre-game v2 model.
- XGBoost re-run (sequenced after logistic confirms the feature is useful).
- Calibration overhaul; isotonic stays as-is.
- Per-tier calibrator (deprioritized per chat-summary).

## Data plan

### Scrape

- **Source.** `lolesports.com/en-GB/gpr/{year}` for past years (`2024`, `2024/worlds`, `2025`) and current (`2026`). The page is server-rendered with the Apollo SSR cache embedded in the HTML.
- **What we actually pull from each page** (via plain `curl` + HTML parse, no Playwright):
  - One `TeamGPR` entry per team listed on the page, containing:
    - `currentTeamGPR`: latest snapshot at scrape time.
    - `previousTeamGPR`: snapshot before that.
    - `teamGPRHistory`: array of historical snapshots at ~10-day granularity for the year window the page queried.
    - `team`: id, code, name, slug, image, `homeLeague { id, name, slug, image }`.
    - `averageOpponentGPR`, `teamMatchRecord`, `teamGameRecord`, `tournaments` (kept for diagnostics; not joined to game features).
  - Each snapshot carries `{ gprScore, elo, rank, dateCalculated }`. `gprScore` is the published Power Score (= 0.8·team_elo + 0.2·league_elo); `elo` is the team-Elo-only component; `dateCalculated` is the publication timestamp.
  - One `LeagueELO` entry per league with that league's Elo time series (kept for diagnostics; not joined to game features in v1 — see "Deliberately not including `league_elo_diff`" below).
- **Output shape.** Two CSVs under `data/gpr/`:
  - `snapshots.csv` — one row per `(team_id, dateCalculated)` with `team_id, team_code, team_name, home_league_slug, snapshot_date, power_score, team_elo, rank`. This is the table stage 5 joins against.
  - `team_aliases.csv` — hand-maintained `gpr_team_name → oe_team_name` mapping.
- **Implementation.** Plain HTTP fetch + regex/JSON extraction of the `(window[Symbol.for("ApolloSSRDataTransport")] ??= []).push(...)` blocks; parse the embedded payloads; flatten to the snapshots schema. No Playwright, no API auth.
- **Calling the GraphQL endpoint directly is out of scope for v1.** The page-scrape gives us everything we need; the endpoint URL would only matter if we ever wanted finer than 10-day granularity, ratings as of arbitrary endDates outside what the page already requested, or live mid-year 2026 snapshots past today's scrape. Filed as a future option, not blocking.

### Team name normalization

- Primary risk in the whole experiment. Join key between GPR and Oracle's Elixir is team name, which diverges (SKT vs T1, Samsung vs Gen.G, branding changes mid-season, regional variants).
- Build a manual alias table: `data/gpr/team_aliases.csv` mapping `gpr_team_name → oe_team_name`.
- Validation is structural, not sampled: enumerate every distinct `teamname` in 2024+ OE data for the leagues GPR rates (S-tier + 2024 VCS), count games per team, and require every team with ≥20 games to be mapped (either to a GPR name or explicitly flagged as uncovered for a documented reason — e.g. an S-tier team that played fewer games than GPR's qualification threshold). Then a 50-random-game spot-check on top as a sanity pass.
- Unmatched teams → log and treat as `has_gpr=0` for that game. Acceptable loss if <5% of games; re-examine if higher.

### Join to games — per-game cutoff

For each `(game, team)` row, attach the most recent `teamGPRHistory` snapshot for that team with `dateCalculated < game_date`. Strict less-than enforces leakage prevention by construction: each per-match Elo update is `R_new = R_prev + I · (W − W_e)`, so the rating *as of date D* is a path integral over matches strictly before D — it cannot encode the outcome of a game played at or after D. The 2-year team-Elo window changes which past matches feed the rating, not the temporal direction.

The join rule:

| Condition | Outcome |
|---|---|
| Both `teamname` and `opp_teamname` have a snapshot dated `< game_date` | `has_gpr=1`, `gpr_diff = power_score[teamname] − power_score[opp_teamname]` |
| Either team is missing a qualifying snapshot | `has_gpr=0`, `gpr_diff=0` |

When multiple snapshots qualify, take the latest. Strict `<` (not `≤`): `dateCalculated == game_date` to-the-second is treated as same-day-as-or-after-game and excluded.

**Date type normalization.** OE `date` is a UTC datetime ("YYYY-MM-DD HH:MM:SS"); Riot `dateCalculated` is an ISO-8601 string with timezone (typically Z / UTC). Stage 5 parses both to tz-aware `pandas.Timestamp` in UTC before comparison, and asserts no naive timestamps slip through (a naive-vs-aware comparison would raise; a both-naive comparison with mismatched implicit zones would silently introduce up to ~1 day of phantom leakage near regional broadcast windows). The strict-`<` cutoff is then applied on the UTC datetimes.

Coverage in practice (from the scrape spike):

- **Pre-2024-02-10:** no snapshots exist; `has_gpr=0`.
- **2024-02-10 onward:** rated-team rows in S-tier leagues qualify. Median staleness ~5 days, max ~10 within an active season.
- **Cross-year continuity:** a January 2025 game whose only qualifying snapshot is the 2024 year-end snapshot (e.g. 2024-11-02) joins to that snapshot. Correct — that snapshot reflects everything Riot knew about the team going into the offseason, and predates the 2025 game by definition.
- **2026 (live):** every page re-scrape extends the 2026 history by one ~10-day step, so anchors for 2026 games stay within ~10 days of the game date as the year progresses.

2020–2023 rows and unrated leagues stay in the training pool with `has_gpr=0` (rather than being dropped) — they help the non-GPR coefficients regardless, and `has_gpr` cleanly absorbs the population shift between rated and unrated rows.

## Feature construction

Stage 5 emits four new columns on the per-checkpoint feature matrix, mirroring how stage 3 emits the `{raw, opp, diff}` triple for in-game stats. All four are persisted in `checkpoint_{T}_gpr.csv`; the linear-model layer chooses which to consume (see "Why drop `gpr_diff` at model input" below).

| Column | Definition | Notes |
|---|---|---|
| `gpr_team` | `power_score[teamname]` from the per-game-cutoff join (see Data plan), 0 when `has_gpr=0` | The team's own published Power Score (= 0.8·team_elo + 0.2·league_elo) as of the most recent qualifying snapshot. Each game appears as two rows (one per team's perspective); computed per row. |
| `gpr_opp_team` | `power_score[opp_teamname]` from the per-game-cutoff join, 0 when `has_gpr=0` | Opponent's Power Score, same join. |
| `gpr_diff` | `gpr_team − gpr_opp_team` | Convenience column, exact arithmetic of the two above. Persisted for parity with stage 3's `*diffat{T}` and for tree-model follow-ups; **dropped at linear-model input time** (see below). |
| `has_gpr` | 1 if both teams had a qualifying snapshot per the per-game-cutoff join, else 0 | Population indicator. Absorbs the mean-level shift between rated and unrated rows so the GPR coefficients are fit cleanly on the covered subset. The `has_gpr` coefficient itself is a nuisance — not interpreted, since its population is a mix of pre-2024 games, post-2024 unrated leagues, and the small early-season pre-coverage window. |

**Why drop `gpr_diff` at linear-model input** (same argument as the `*diffat{T}` drop in `docs/specs/baseline-logistic.md`): stage 5 ensures `gpr_diff == gpr_team − gpr_opp_team` exactly, so the triple is perfectly collinear. For a linear model with l2, keeping all three would split weight across collinear columns and make `coefficients.csv` meaningless (ridge spreads weight arbitrarily across exactly-collinear features). `{gpr_team, gpr_opp_team}` spans the same 2D space as `{gpr_diff, gpr_sum}`, so it's strictly more expressive than `gpr_diff` alone — it captures both a **rating-gap** effect (recovered as `β_team − β_opp`) and a **rating-level** effect (recovered as `β_team + β_opp`). The level effect *should* be ~0 by Elo construction (equal Power Scores → equal expected win prob regardless of level), but if it isn't we want to see that, and `gpr_diff` alone hides it.

Tree-based follow-ups (XGBoost re-run) keep all three columns by the same axis-aligned-splits argument that motivated keeping `*diffat{T}` for XGBoost in the baseline.

Encoding choice (raw vs log-ratio vs rank-within-league): raw Power Score. Elo math says equal diffs map to equal logit shifts in `W_e`, so the linear logit on raw scores is well-matched to how the rating was constructed. Revisit only if the gap-coefficient behaves oddly at evaluation (wrong sign, non-monotone residuals vs `gpr_diff` bins).

**Future option: key the alias table on `gpr_team_id`** (instead of `gpr_team_name`). GPR's `team.id` is stable across rebrands; team names are not (DWG KIA → Dplus KIA, SKT → T1, etc.), so a name-keyed table needs a new row per rebrand event. Not done in v1 because OE has no compatible ID — `oe:team:{hash}` is a different namespace, and `teamid` was rejected as a join key in the discovery checklist due to ~1.1K nulls (vs ~21 for `teamname`). So even with `gpr_team_id` on the GPR side, the OE side stays on `teamname`. Mapping shape becomes `gpr_team_id → oe_teamname` instead of `gpr_team_name → oe_teamname`, eliminating GPR-side rebrand churn at no cost. Filed for v2.

Deliberately not including `league_elo_diff` as a separate column in v1 — high collinearity with `gpr_team`/`gpr_opp_team` (the league component is 0.2 of each), adds two coefficients, complicates interpretation. Revisit if v1 succeeds and we want to tease apart team vs regional strength.

Why the per-game-cutoff join makes a learned coefficient the natural call (vs. the earlier "fixed-coefficient logit offset" design — see Future Options below):

- **Training set is large enough now.** Under the new join we get ~10K rated S-tier games (2024-02-10 onward) instead of the ~4–5K that the old year/anchor map produced. Plenty for fitting two coefficients per checkpoint.
- **Anchors are fresh.** Median ~5-day staleness, max ~10. The model sees the rating that was actually published right before the game, not a year-end summary up to 12 months stale. The signal-to-noise on the rating gap should be much closer to "what GPR was designed to predict" than the offset-against-stale-anchor framing.
- **Interactions get learned.** Coefficients let the model implicitly down-weight the GPR effect as T grows and in-game state takes over. An offset can't do that without becoming path-A-with-extra-steps.
- **Diagnostics are richer.** Coefficients with SEs and a contrast z-stat are a real test of whether GPR carries signal on our slice; an offset has nothing analogous.

### Future option: fixed-coefficient logit offset (Riot's `W_e` baked in)

Kept as a documented option, not the chosen design. If at evaluation time `gpr_diff`'s coefficient lands wrong-signed or noisy, or if a future v2 wants to test a more Bayesian construction, the alternative is:

```
p_riot       = 1 / (1 + 10^(−gpr_diff / 600))          # Riot's W_e
logit_p_riot = log(p_riot / (1 − p_riot))
logit P      = α + β·X + 1 · logit_p_riot              # coefficient fixed at 1
```

`gpr_diff` here is read from the persisted stage-5 column (or equivalently computed as `gpr_team − gpr_opp_team`); `has_gpr=0` rows get `gpr_diff=0 → p_riot=0.5 → logit_p_riot=0`, so the offset contributes nothing on unrated rows. The case for it: Riot reports 65% predictive accuracy on S-tier and publishes the divisor (600), so we'd be baking in a calibrated S-tier prior with no estimated parameters; the model fits residuals on top. The case against (and why we're not starting here): we'd need a prior-sanity-check on our slice before fitting, the implementation requires bypassing `LogisticRegressionCV`'s no-offset constraint (statsmodels GLM with `offset=...` or a custom `scipy.optimize` MLE), and the path-A learned-coefficient fit is now well-supported enough that the extra mechanism doesn't pay for itself.

## Training & evaluation

- Same split, scaler, regularization grid, and fitter as production baseline logistic (see `docs/specs/baseline-logistic.md`). `LogisticRegressionCV` is reused as-is — no offset machinery, no statsmodels detour for the production fit.
- Refit logistic with `gpr_team`, `gpr_opp_team`, `has_gpr` added to the feature matrix. `gpr_diff` is persisted in `checkpoint_{T}_gpr.csv` but **dropped at model-input time** in `prepare_Xy`, mirroring how `baseline-logistic.md` drops `*diffat{T}`. `gpr_team` and `gpr_opp_team` are numerics and go through the same `StandardScaler` block; `has_gpr` is a binary passthrough alongside `side_is_blue` and the gated firsts.
- **Standardization of GPR columns is fit on rated rows only** (i.e. rows where `has_gpr=1` in the training set), and unrated rows have `gpr_team` / `gpr_opp_team` imputed to that rated-population mean before scaling — so post-scaling, unrated rows sit at 0 on both columns. Without this, `StandardScaler` fit on the union pulls the mean toward zero (since unrated rows are zero-coded), and `has_gpr=0` rows land at a distinctly negative standardized value. Under l2 the regularizer would then split the resulting mean-shift between `has_gpr` and `β_team`/`β_opp`, biasing the headline contrast `β_team − β_opp` that success criterion #2 is computed on. With the rated-only fit + impute-to-rated-mean, `has_gpr` is the only column carrying the rated-vs-unrated signal, and `β_team`/`β_opp` are estimated cleanly on the rated subpopulation. Implementation: in `prepare_Xy` (or the fit step), compute `μ = X_train.loc[has_gpr==1, ['gpr_team','gpr_opp_team']].mean()`, set `X_train.loc[has_gpr==0, ['gpr_team','gpr_opp_team']] = μ`, then fit `StandardScaler` on the resulting numeric block as usual. Same `μ` is reused at val/test time.
- One fit per checkpoint.
- Metrics on 2026 S-tier holdout:
  - Log-loss (per checkpoint + aggregate, paired bootstrap CI vs current baseline).
  - AUC (per checkpoint).
  - ECE in [0.65, 0.80) band (per checkpoint).
  - Re-run script 15 bootstrap on T=10 S-tier band gap.
- Report `β_team`, `β_opp`, `β_has_gpr`, and the **`β_team − β_opp` contrast** with its SE and z-stat per checkpoint. SEs and the coefficient covariance come from a small statsmodels post-fit on identical features at the chosen `C` (sklearn's `LogisticRegressionCV` doesn't expose either) — diagnostic-only; the production model stays sklearn.

## Risks

- **Team-name join drift.** Silent joins to wrong teams corrupt the feature. Mitigation: structural alias-table validation (every team with ≥20 games in 2024+ S-tier OE data must be mapped or explicitly flagged), 50-game spot-check, and the success-criterion #2 sign-check on `β_gpr` — a wrong join would manifest as a wrong-signed or near-zero coefficient.
- **Snapshot access.** Resolved by the scrape spike. The page-scrape path is robust to anything except the lolesports site changing its SSR cache embedding format; if that happens, the GraphQL endpoint discovery becomes the next move (filed as a future option above).
- **Snapshot granularity.** ~10-day step is fine for the rating-as-prior use case; a single 10-day window doesn't include enough new matches to materially shift Elo for established teams. For roster-shock cases (mid-split lineup changes, surprise upsets) the rating may lag the team's true strength by up to 10 days. Mitigation: none in v1 — accepted noise floor.
- **Cross-year-boundary edge case.** Some early-2025 / early-2026 games anchor against the prior year-end snapshot (no in-year history exists yet). That's fine for leakage but means the rating is up to ~3 months stale at the start of each season — strictly worse than mid-season anchors, but still better than the original year/anchor map. Watch for any disproportionate `gpr_diff` residual concentration in Jan-Feb of each year.
- **Thin coefficient fit.** Under the per-game cutoff join, the GPR signal is fit through `β_team` and `β_opp` on ~10K rated S-tier games (2024-02-10 onward, minus the 2026 holdout). The criterion-2 test is on the contrast `β_team − β_opp`. If contrast z at T=10 lands in [1, 2] we treat that as a slice-thinness signal rather than declaring failure, and consider switching to the fixed-coefficient logit offset as the documented Future Option above.
<!-- leakage risk addressed by the strict per-game cutoff (snapshot dateCalculated < game_date) in the Data plan. -->

## Code & data layout

GPR is treated as an external data source, structurally parallel to `data/lol-historical-data/`. Its ingestion is independent of the main preprocessing pipeline. Its *join* onto features is appended as the **last** stage of the main pipeline — placed there (rather than mid-pipeline, before the checkpoint split) so the stage can be removed cleanly if the experiment fails, without disturbing the earlier stages' outputs. The cost is that the join runs four times (once per checkpoint CSV) instead of once at per-game level; this is acceptable given the small join size.

### Directory layout (additions only — no reorg of existing tree)

```
data/
  lol-historical-data/                  # existing external source
  gpr/                                  # NEW — scraped external source
    snapshots.csv                       # one row per (team_id, snapshot_date); cols: team_id, team_code, team_name, home_league_slug, snapshot_date, power_score, team_elo, rank
    team_aliases.csv                    # hand-maintained: gpr_team_name → oe_teamname
  processed/
    01_filtered.csv                     # unchanged
    02_one_row.csv                      # unchanged
    03_features.csv                     # unchanged
    checkpoint_{10,15,20,25}.csv        # unchanged — produced by stage 4
    checkpoint_{10,15,20,25}_gpr.csv    # NEW — produced by stage 5, adds gpr_team + gpr_opp_team + gpr_diff + has_gpr

scripts/
  build_dataset.py                      # add stage 5 to STAGES list
  gpr/                                  # NEW — independent ingestion
    __init__.py
    scrape.py                           # fetches /gpr/{year} pages, parses Apollo SSR cache from HTML, writes snapshots.csv
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

Step 1 (scrape spike) is **already complete** as of 2026-04-26 — see Background. The page-scrape returns SSR'd Apollo cache with team-level GPR and per-team `teamGPRHistory` at ~10-day granularity for 2024-02-10 onward. Steps 2–9 follow.

1. **~~Scrape spike (gate)~~** — done. Findings logged in Background; raw artifacts in `images/gpr-dev-notes.md` and the spike's exploratory dumps under `/tmp/gpr-spike/` (not committed).
2. **Team alias table** — `data/gpr/team_aliases.csv`, hand-maintained `gpr_team_name → oe_team_name`. Structural validation: every team with ≥20 games in 2024+ S-tier OE data must be mapped or explicitly flagged uncovered. Plus a 50-random-game spot-check.
3. **`scripts/gpr/scrape.py`** — fetches `lolesports.com/en-GB/gpr/{2024,2024/worlds,2025,2026}`, parses the embedded Apollo SSR cache, flattens to `data/gpr/snapshots.csv` (one row per `(team_id, dateCalculated)`). Ship a checked-in fixture (`tests/fixtures/gpr/sample_page.html` — one cached year page) and a parser test that asserts the schema (expected team count, snapshot fields present, `dateCalculated` parses, `power_score`/`team_elo`/`rank` non-null on a sampled team). The lolesports SSR-cache embedding format can change at any time; without the fixture the first signal of breakage is a silent empty `snapshots.csv` followed by a wholesale `has_gpr=0` corruption of the next pipeline run.
4. **`stage5_join_gpr.py`** — applies the per-game cutoff (`dateCalculated < game_date`, take latest qualifying snapshot per team), emits `checkpoint_{T}_gpr.csv` with `gpr_team`, `gpr_opp_team`, `gpr_diff`, `has_gpr` columns (asserting `gpr_diff == gpr_team − gpr_opp_team` exactly, mirroring stage 3's `*diff` invariant). Tests cover the strict-`<` boundary, both-teams-rated requirement, and cross-year-boundary anchoring.
5. **Wire stage 5 into `build_dataset.py`.**
6. **Riot prior sanity check (pre-fit diagnostic)** — before fitting, run a 5-line check on the rated training rows: compute `p_riot = 1 / (1 + 10^(−gpr_diff / 600))` per game (using the leakage-free `gpr_diff` from stage 5), bin into deciles of `gpr_diff`, plot empirical win rate vs `p_riot` per bin. Output: `data/exploration/16_gpr_prior_calibration.png` + a one-line summary metric (mean abs deviation, or empirical-WR slope vs `p_riot` slope of 1). If `W_e` calibrates sensibly on our slice (slope ≈ 1, no monotonicity violations), the learned-coefficient fit is well-motivated. If it doesn't, the offset future-option won't save us either, and the experiment's prior gets adjusted before we burn cycles on the refit.
7. **`baseline_logistic_gpr.py`** — refit of baseline logistic with `gpr_team` + `gpr_opp_team` + `has_gpr` added; `gpr_diff` dropped at `prepare_Xy` time per baseline convention. Same `LogisticRegressionCV` / split as baseline; scaler fit on rated rows only with unrated rows imputed to the rated mean (see Training & evaluation). Reports SE + z for each new coefficient and for the `β_team − β_opp` contrast (via a small statsmodels post-fit on identical features).
8. **Evaluate on 2026 S-tier holdout** — log-loss / AUC / ECE per checkpoint, paired-bootstrap CI on aggregate log-loss vs current baseline; re-run `scripts/exploration/15_band_gap_bootstrap.py` on the new model for the T=10 S-tier band-gap closure check.
9. **Decision point:** ship if the joint test passes; switch to the fixed-coefficient-offset Future Option if the `β_team − β_opp` contrast lands wrong-signed or borderline (z ∈ [1, 2]); write up as a negative result if it clearly fails; consider XGBoost re-run as a follow-up if it ships (which would consume `gpr_diff` directly per the baseline-XGBoost convention).

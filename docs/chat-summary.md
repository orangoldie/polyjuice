# Chat Summary — GPR feature: spec finalized + scraper shipped (2026-04-27)

## Where we are

The GPR (Riot Global Power Rankings) team-strength feature spec is locked and the first piece of infrastructure (the scraper) is checked in and verified end-to-end. We are mid-Step-3 of the spec's sequencing — see `docs/specs/gpr-feature.md` for the authoritative plan.

## What's shipped this session

- **`docs/specs/gpr-feature.md`** — finalized after a focused review pass. Notable changes from the prior draft:
  - **Standardization fit on rated rows only** (with unrated rows imputed to the rated mean before scaling). Stops `has_gpr` from having to absorb a mean-shift through l2-regularized coefficients, which would bias the headline `β_team − β_opp` contrast that success criterion #2 is computed on.
  - **Date-type normalization** in stage 5 — both `dateCalculated` and OE `date` parsed to tz-aware UTC `pandas.Timestamp` before the strict-`<` comparison. Guards against silent ~1-day phantom leakage from naive-vs-aware datetime mismatch.
  - **Scraper fixture + parser test** added to Step 3 (not yet implemented — see "what's next").
  - **Riot prior sanity check** added as Step 6 (between stage 5 and the refit). 5-line check: bin `gpr_diff`, plot empirical WR vs `W_e = 1/(1+10^(-gpr_diff/600))`, see if Riot's published prior actually calibrates on our slice. Cheap insurance — if it doesn't, the offset Future Option won't save us either.
  - **`gpr_team_id` future option** for the alias table key (more rebrand-stable than name; deferred to v2 because OE side has no compatible ID).
  - **VCS no longer special-cased.** The previous spec excluded VCS based on a stale "GPR doesn't cover VCS" claim from the prior chat-summary. The truth: VCS-as-a-league doesn't exist post-2025 (merged into LCP), but 2024 VCS *is* GPR-rated and we have 108 snapshots for VCS teams in the scraped data. The per-game-cutoff join now applies uniformly — any team with a qualifying snapshot gets `has_gpr=1` regardless of league.
  - **Objective rewrite** — was stale, claimed "two new columns (`gpr_diff`, `has_gpr`)" with `gpr_diff` as the fit coefficient. Actual design adds four columns (`gpr_team`, `gpr_opp_team`, `gpr_diff`, `has_gpr`) and drops `gpr_diff` at model input in favor of `{team, opp}` so the model can express both rating-gap and rating-level effects.

- **`scripts/gpr/scrape.py`** — implemented, working end-to-end. Fetches `lolesports.com/en-GB/gpr/{2024,2024/worlds,2025,2026}`, caches raw HTML under `data/gpr/raw/` (gitignored, ~10MB), parses Apollo SSR push payloads, dedupes on `(team_id, snapshot_date)`, writes `data/gpr/snapshots.csv`.
  - Two non-obvious things the scraper handles: (a) the SSR pushes contain bare `undefined` JS literals (invalid JSON) — substituted to `null` before parsing; (b) two payload shapes coexist on different pages (`events[*].value.data` vs `rehydrate[<key>].data`) — the spike's extractor only handled the first, missing 2 of 4 pages until we patched it.
  - Output: 4,279 unique snapshots, 86 unique teams, range 2024-02-10 → 2026-04-26, 64 distinct snapshot dates. Coverage by league looks complete (LCK 638, LPL 1023, LEC 630, CBLOL 532, LCP 514, LCS 449, plus VCS 108, PCS 135, LLA 81, LTA N/S, LJL).
  - Sanity-checked T1's trajectory matches lived reality (rank 1 early 2024 → 4-5 by Worlds 2024 → 1 by Worlds Final 2025 → 4-5 by spring 2026). Confirms snapshots are published-at-the-time, not retroactively recomputed.
  - Reproducible: re-run with `python -m scripts.gpr.scrape` (offline from cache) or `--refresh` to re-fetch.

- **`.gitignore`** — added `data/gpr/raw/` so cached HTML doesn't bloat the repo.

## State of v1 production model (unchanged from prior session)

- Baseline logistic per checkpoint, train 2020-2025, eval 2026.
- XGBoost v1 ties baseline within CI noise — no nonlinear structure to exploit at these checkpoints.
- Calibration diagnosis (scripts 14 + 15) found one borderline-real miscalibration: T=10 S-tier ~9pp overconfident in [0.65, 0.80), n=102, CI barely excludes zero. Aggregate band gaps are ±2pp at every T. The GPR feature is the targeted intervention for that T=10 S-tier gap.

## Immediate next step (what a fresh chat should pick up)

**Step 2 of the spec — the team alias table.** It's the highest-risk piece of the experiment and now unblocked because `data/gpr/snapshots.csv` exists.

Concrete deliverable:

1. Enumerate every distinct `teamname` in 2024+ OE data for the leagues GPR rates (S-tier + 2024 VCS) with game counts. New exploration script — `scripts/exploration/17_s_tier_team_inventory.py` or similar.
2. Enumerate every distinct `team_name` in `data/gpr/snapshots.csv` (already done implicitly — 86 teams).
3. Hand-create `data/gpr/team_aliases.csv` with columns `gpr_team_name, oe_teamname, notes`. Map every OE-side team with ≥20 games. Document any flagged-uncovered with a reason.
4. Validator: assert every OE team with ≥20 games is either mapped or flagged. Iterate until clean.
5. 50-random-game spot-check.

Estimated half day to a day depending on rebrand complexity.

## Subsequent steps (per spec)

4. **`stage5_join_gpr.py`** — per-game cutoff join, emits `checkpoint_{T}_gpr.csv`. Tests cover the strict-`<` boundary, both-teams-rated requirement, cross-year-boundary anchoring.
5. **Wire stage 5 into `build_dataset.py`.**
6. **Riot prior sanity check** — `scripts/exploration/16_gpr_prior_calibration.py`. Gates the refit; cheap to run.
7. **`baseline_logistic_gpr.py`** — refit with `{gpr_team, gpr_opp_team, has_gpr}`. Scaler fit on rated rows only, statsmodels post-fit for SE / coefficient covariance / contrast z-stat.
8. **Evaluate on 2026 S-tier holdout** — paired-bootstrap log-loss CI, AUC, ECE, re-run script 15 for the band-gap closure check.
9. **Decision point** — ship if joint test passes; switch to fixed-coefficient-offset Future Option if the contrast lands borderline; negative-result write-up if it clearly fails.

## Outstanding hygiene (not blocking but worth doing)

- **Scraper fixture + parser test** (`tests/fixtures/gpr/sample_page.html` + `tests/gpr/test_scrape.py`). Spec calls for it; deferred this session to keep momentum on the alias table. The lolesports SSR-cache embedding format can change at any time and the first signal of breakage is currently a silent empty `snapshots.csv`.
- **Per-page summary printing in scraper** prints unique team count per page now (post-fix) — was confusingly showing `events_teams + rehydrate_teams` (e.g. 160 instead of 80) before.

## Key references

- `docs/specs/gpr-feature.md` — authoritative spec for everything above.
- `docs/specs/baseline-logistic.md` — baseline this experiment refits on top of.
- `scripts/exploration/15_band_gap_bootstrap.py` — produces the T=10 S-tier band-gap number that success criterion #3 tests against.
- `scripts/gpr/scrape.py` + `data/gpr/snapshots.csv` — what we built this session.
- `discovery-checklist.md` — broader project-level decisions (preprocessing pipeline, feature lock, train/test split, etc.).

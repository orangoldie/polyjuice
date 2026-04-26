# Chat Summary — calibration diagnosis and next direction (2026-04-24)

## State of the project

- v1 in-game checkpoint model shipped for T ∈ {10, 15, 20, 25}. Baseline logistic (production: 2020–2025 train + optional isotonic) is the model currently scored on 2026.
- XGBoost v1 failed acceptance on all 4 checkpoints; ties the baseline within CI noise. No bug — features appear to have no nonlinear structure to exploit at these checkpoints.
- Project housekeeping this session: `scripts/models/` collapsed to 2 files; exploration CSVs moved out of `data/experiments/`.

## Phase 1 diagnostics ran (scripts 14 + 15)

Goal was to figure out *where* calibration is broken in the 70-75% betting band before adding anything.

**Three findings that push back on the earlier working hypotheses:**

1. **Dropping 2025 from training does NOT help calibration in the betting band — it mildly hurts.** Production (with 2025) beats sibling (without) in [0.65, 0.80) at every checkpoint. The Feats-of-Strength contamination concern doesn't manifest here.
2. **The 4.1% global ECE at T=20 is not a betting-band problem.** T=20 in [0.65, 0.80) is well-calibrated (+1.3pp). The ECE blowup lives in the low-prob bin and a noisy high-prob bin — neither matters for 70-75% betting.
3. **Per-tier point-estimate gaps from script 14 looked like real opposing biases (S overconfident at T=10, A underconfident at T=20). Bootstrap in script 15 says they're mostly noise.** Only T=10 S-tier's −9.4pp gap has a CI that excludes zero, and it barely does (CI [−19%, −0.1%], n=102).

**Honest status of calibration at 70-75%:** aggregate band gaps are ±2pp at every T. The only borderline-real miscalibration is T=10 S-tier being ~9pp overconfident.

## Current direction — S-tier focus + team-strength prior

Polymarket's current liquidity on LoL is concentrated on S-tier markets, so we're narrowing scope there anyway. That reframes the question:

- **Do we need to fit a general multi-tier calibration system?** No. Bet on S-tier, model on S-tier, calibrate on S-tier.
- **Is the T=10 S-tier overconfidence a real signal or a statistical ghost?** Borderline — needs another data point (more 2026 games, or an intervention that should close the gap if the mechanism is what we think).
- **What's the natural mechanism?** Tier-1 teams come back from min-10 gold deficits more often than the model expects — i.e. team strength is doing work that the current features don't capture.

Plan: **use Riot's GPR (Global Power Rankings) as a team-strength feature for S-tier games, see if T=10 S-tier gap closes.** GPR preferred over a homegrown Oracle's-Elixir Elo as the starting point: it's Riot's own validated rating, captures roster/region nuance we'd have to hand-engineer, and symbolically matches what we want the feature to represent (established pro strength, not a summary of historical results). Homegrown Elo stays on the table as v2 groundwork and as the fallback if GPR coverage / scrape proves unworkable.

## GPR build — decisions (see `docs/specs/gpr-feature.md`)

Open questions from the prior chat are now resolved in the spec:

- **Training-data coverage.** Keep 2020–2025 training window. Pre-2024 rows get `has_gpr=0, gpr_diff=0`; 2024+ rows get real values where both teams have a snapshot. Missingness-indicator pattern — `gpr_diff`'s coefficient is fit cleanly on the covered subset.
- **Scrape viability.** Gating unknown. Spec's step 1 is a scrape spike to confirm the underlying API exposes historical per-team snapshots. If it only returns current ratings, the spec is shelved and we reopen with a wayback-based plan or fall back to homegrown Elo.
- **Temporal resolution.** Per-`(team, snapshot_date)` if the API exposes it; event-snapshot granularity otherwise. Join rule: `snapshot_date < game_date`, take `max`. Strict `<` prevents leakage by construction.
- **League coverage.** LCK, LPL, LEC, LCS/LTA N+S, LCP, CBLOL covered. VCS not covered by GPR — excluded from this experiment.
- **Encoding.** Raw `gpr_diff = power_score[teamname] − power_score[opp_teamname]`. Elo math says equal diffs → equal logit shifts, so raw diff is the correct shape. Revisit only if the coefficient misbehaves.
- **Cold-start teams.** Fall through to `has_gpr=0, gpr_diff=0`. Acceptable if <5% of S-tier 2024+ games land here.

## Open questions on what we'd do with it

- **Does including Elo close the T=10 S-tier band gap?** This is the primary test. Re-run script 15 post-feature; if gap collapses, team strength was the missing piece.
- **Does it move AUC / log-loss materially?** Or only calibration-in-band? Different answers point to different model surgery.
- **Does it help pre-game (v2 model) enough to justify pulling v2 forward?** If Elo alone gives a usable pre-game WR prior, v2's groundwork is mostly done.
- **If T=10 S-tier gap doesn't close with GPR — then what?** That's a meaningful negative result. Candidates: fall back to homegrown Oracle's-Elixir Elo (more coverage, different construction), trajectory features (`golddiffat15 − golddiffat10`), stacking baseline + XGB, or accepting that the gap is just n=102 noise and waiting for more 2026 data.

## Things explicitly deprioritized (and why)

- **Drop-2025 ablation / Feats-of-Strength handling.** Script 14 showed it doesn't help in the band. Dead.
- **Per-tier calibrator.** Script 15 shows only 1 of 12 cells has a non-zero gap at 95% CI, and we're narrowing to S-tier anyway — no reason to maintain per-tier infrastructure.
- **Trajectory features, stacking, Riot's missing features (XP diff, herald/grub firsts).** All still on the table but downstream of the Elo experiment — no point adding more features if team strength alone resolves the only real signal.
- **Homegrown Oracle's-Elixir Elo.** Kept as fallback / v2 groundwork but not the starting point — GPR symbolically captures what we want ("established pro team strength") without us having to re-engineer roster/region/decay logic.

## References

- `scripts/exploration/14_calibration_per_tier_bucketed.py` — per-tier × per-T × per-bin calibration, production vs sibling.
- `scripts/exploration/15_band_gap_bootstrap.py` — 5000-iter bootstrap CIs on per-tier band gaps.
- `data/exploration/14_calibration_per_tier_bucketed.csv`, `data/exploration/15_band_gap_bootstrap.csv` — outputs.

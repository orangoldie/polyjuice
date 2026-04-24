# Chat Summary — v1 calibration plan (2026-04-24)

## State of the project

- v1 in-game checkpoint model shipped for T ∈ {10, 15, 20, 25}. Baseline logistic (production: 2020–2025 train + optional isotonic) is the model currently scored on 2026.
- XGBoost v1 was tuned (Optuna 100 trials/checkpoint) and **failed acceptance on all 4 checkpoints** — ties the baseline within CI noise. Verified reproducibly:
  - All 4 XGB artifacts re-predict their own metrics.json bit-for-bit.
  - `BASELINE_H2` constants in xgboost_v1.py match runs.csv.
  - Acceptance H2 slice (3,274 / 3,274 / 3,271 / 3,176 games) is identical between baseline and XGB — comparison is fair. XGB has slightly *more* training data (year-vs-date split boundary) and still loses. No bug; features have no nonlinear structure to exploit at these checkpoints.
- Project housekeeping completed this session:
  - `scripts/models/` collapsed from 6 files to 2: `baseline_logistic.py` + `xgboost_v1.py`. Baseline now matches xgboost's CLI pattern (`--fit-production`, `--evaluate-test`, `--evaluate-test-production`). 34/34 baseline tests + 47/47 xgboost tests pass. Deleted `baseline_logistic_h2_rerun.py` (output lived in the `BASELINE_H2` constant), `baseline_logistic_2026_eval.py`, `baseline_logistic_production.py`, `baseline_logistic_production_2026_eval.py`.
  - Exploration scripts 11 and 13's CSV outputs moved from `data/experiments/` to `data/exploration/`. `data/experiments/` now holds only `runs.csv`.
  - Root cleanup: deleted stray `3.0` empty file, moved `chat-summary.md` / `get-started.md` / `metrics-explanation.txt` under `docs/`.

## Problem to solve next — calibration at the 70-75% betting band

The user plans to bet on Polymarket specifically in the **70-75% predicted-win band**. Current production calibration on 2026:

| T  | ECE   | Brier | Status                |
|----|-------|-------|-----------------------|
| 10 | 0.022 | 0.190 | Borderline            |
| 15 | 0.019 | 0.160 | OK                    |
| 20 | **0.041** | 0.137 | **Not bettable — 4%** |
| 25 | 0.023 | 0.097 | OK                    |

Global ECE is a weak signal though — what actually matters is **per-bin calibration at 65-80%, per tier**. That's what Phase 1 diagnoses.

## Research findings (agent pull + Riot Dev Diary / GPR / Pinnacle / arXiv 2309.02449 / ProjektZero)

1. **GPR is Tier-1 only** (LCK/LPL/LEC/LTA N+S/LCP/VCS/CBLOL). ProjektZero-LoL-Model on GitHub already builds Elo on Oracle's Elixir and covers all tiers — that's our reference if we roll our own. URL pattern `lolesports.com/en-GB/gpr/{year}/{event}` works; page is JS-hydrated, real API lives behind AWS (inspect devtools if we want time-series).
2. **Elo prior helps calibration more than AUC, with biggest impact at early checkpoints.** Research specifically finds the gain concentrates in the probability band where the user wants to bet.
3. **Riot's own WP model has features we don't:** XP diff, alive count at checkpoint, active buff timers (baron/elder currently active, not just "first taken"), herald trinket in inventory.
4. **Draft effects are NOT fully subsumed by min-10 stats** — contradicts our discovery-checklist reasoning for deferring draft features. Scaling comp down 1k at 15 ≠ tempo comp down 1k. Still expensive to encode; deferring stays defensible.
5. **Herald + void grubs carry tempo signal separate from dragons** in the 2024+ meta. Current model collapses them entirely.

## Plan — phased, in priority order

### Phase 1 — Diagnose (hours). **Start here.**
Goal: figure out *where* calibration is actually broken before picking interventions.

- Per-bin calibration focused on the **65-80% bucket**, split per-tier (S/A/B) and per-T. Extension of `scripts/exploration/12_calibration_bins_2026.py`.
- Quick ablation: retrain baseline **without 2025** (Feats of Strength era), evaluate on 2026, compare ECE + per-bin calibration vs current. If T=20 ECE drops materially, Feats leakage is confirmed and we know how to handle it.
- Decision criterion: outcome of Phase 1 dictates whether we go (a) team-strength features, (b) Feats/era handling, (c) better calibrator, or some mix.

### Phase 2 — Team strength prior (1-2 days MVP, 1-2 weeks for full Elo)
Single biggest expected calibration lever.

- Start simple: **rolling win rate weighted by opponent tier** over a trailing window, computed offline from 2020-2025 Oracle's Elixir. Joined on `teamname` (identifiers already in pipeline output).
- Layer **GPR snapshots** as an additional feature for Tier-1 games where available (optional; don't block on it).
- Upgrade to full Elo/Glicko if simple version shows calibration gains but AUC gains are small (means we're on the right track but can sharpen).
- Requires re-opening v1's "no team identity as feature" decision from discovery-checklist — user has explicitly OK'd this.

### Phase 3 — Riot's missing features (few days)
- **XP diff** at each checkpoint (`xpat{T} - opp_xpat{T}`, Oracle's Elixir has it). Inference parity: livestats exposes `level` only — heuristic-convert via a fitted level↔XP mapping.
- **First herald** and **first void grub** flags, checkpoint-gated like firstdragon/firsttower. Need to confirm livestats exposure for these.
- Alive count / active-buff-timer features require livestats frame parsing — out of scope until bet pipeline is being built.

### Phase 4 — Sample weighting (only if Phase 1 says so)
- Exponential date-decay on training, or explicit 2025 downweight if Feats leakage is the demonstrated problem.

## User preferences captured this session

- Bet at 70-75% probability band — calibration analysis should focus there, not globally.
- OK re-opening discovery-checklist decisions (team identity, patch-as-feature, etc.) — checklist is a starting point, not a rulebook.
- Engineering appetite is fine — even multi-day features on the table.
- Wants to also quick-check: does dropping 2025 from training improve 2026 calibration? (Subsumed into Phase 1.)

## Recommended next action when resuming

Phase 1 diagnostics — write an exploration script (probably `scripts/exploration/14_calibration_per_tier_bucketed.py`) that:
1. Loads production baseline predictions on 2026 per T.
2. Reports per-tier (S/A/B) × per-bin reliability, with tighter bins around 65-80%.
3. Runs a second pass training on 2020-2024 only (sibling, already exists), scores on 2026, reports the same table — diff shows whether 2025 inclusion hurts at T=20.
4. Writes results to `data/exploration/14_*.csv` + a console table.

Outcome of this script decides Phase 2 vs Phase 4 priority.

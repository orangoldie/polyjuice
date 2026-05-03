# Session summary: 2026-05-03 — model-market blender

Started from the question "is the edge real or a calibration artifact?" — ended with a deployable v1 strategy.

---

## What was done

### 1. Edge reliability bins (`scripts/exploration/47_pmoe_edge_reliability_bins.py`)

Bucketed `eval_table.csv` by `(p_model − p_market)` and checked realized edge per bin. Result: directionally correct (sign of predicted edge matches sign of realized) but **magnitude inflated by ~1.7×** — when raw model says +20pp edge, you only get +12pp. EM league specifically anti-edge at every threshold; LCKC the cleanest. Headline: "bet when |edge| ≥ t" works (every threshold's CI excludes zero), but raw model edge cannot be trusted as a Kelly input.

### 2. Model-market blender (`48_pmoe_blender_fit_oof.py`)

Fit `logit(p_blend) = a + b·logit(p_model) + c·logit(p_market)` per checkpoint via group-k-fold by `oe_gameid`. Both inputs land with positive significant weight at every checkpoint — **the market carries information the model doesn't** (~50% weight at T=10/15, ~30% at T=20/25). Reliability bins now sit on the diagonal (no more 1.7× overstatement). Threshold-betting realized edge improved 30% (raw model thr≥0.10: +0.117; blender: +0.151). EM auto-fixed by the blender learning to defer to the market in regimes the model is wrong about.

### 3. OE-date timing audit (`49_pmoe_date_anchor_full.py`)

User flagged concern: we treat OE `date` as exact game start. Earlier script 42 had validated this on 8 markets; widened to all 1478 cleanly-resolved markets. **Median offset +62s** (typical games within ~1 min). p25/p75 = (−13s, +283s). S/A-tier majors (LCK/LPL/LCP/LCS/LEC) median offsets within ±20s — essentially perfect. LCKC median +33s, EM +108s. Long tail dominated by **join errors** (markets with offsets in hours-to-days) — same `event_slug` appearing twice with different gamelengths suggests a `game_number` ambiguity in same-day Bo3 joins. Worth fixing the join logic later.

### 4. Blender timing sweep (`50_pmoe_blender_timing_sweep.py`)

Sweep filter thresholds {∞, 300s, 180s, 60s} and refit blender at each. **Counterintuitive result:** as filter tightens, blender's edge over the market *grows* (Δ Brier −0.007 → −0.038; threshold-bet realized edge +0.15 → +0.22). Timing skew was *diluting* signal, not creating phantom edge. T=25 c/(b+c) shifted 0.33 → 0.54 with filtering — late-checkpoint coefficient was the most biased, in the predicted direction.

### 5. Production fit + persisted artifact (`51_pmoe_blender_fit_production.py`)

Selection rule: **S-tier league allowlist (matches GPR-eligible) ∩ |offset_s| ≤ 60s**. Population: 560 rows / 142 games / 8 leagues actually present (LCP 143, LCK 127, LEC 83, LCS 68, CBLOL 68, FST 32, WLDs 31, EWC 8). Persisted to `data/models/model_market_blender/coefficients.json` with metadata (fit date, allowlist, n_rows). README.txt has the inference snippet.

Per-checkpoint coefficients on the production fit:

| cp | a | b (model) | c (market) | c/(b+c) | OOF Brier blend | Δ vs market |
|---|---|---|---|---|---|---|
| 10 | +1.06 | 0.83 | 0.37 | 0.31 | 0.154 | −0.035 |
| 15 | +1.05 | 0.46 | 0.76 | **0.62** | 0.132 | −0.024 |
| 20 | +1.22 | 0.53 | 0.61 | 0.54 | 0.110 | −0.033 |
| 25 | +1.49 | 0.52 | 0.60 | 0.54 | 0.073 | −0.041 |

Headline T=25 OOF Brier of 0.073 looked too good — triggered Phase 6.

### 6. Temporal holdout + bootstrap CIs (`52_pmoe_blender_temporal_holdout.py`)

Fit on first 80% of games chronologically (through 2026-04-15), evaluate on last 20%. **Signal survives**: aggregate Δ blend − market = −0.036, CI [−0.062, −0.009] — excludes zero. T=25 specifically regressed from OOF 0.073 to holdout 0.092, but still beat market (0.135) by 0.043. Per-checkpoint CIs on full population: 3/4 of blend-vs-market and 3/4 of blend-vs-model deltas exclude zero. **The 0.073 was overoptimistic, but the signal is real.**

### 7. Staleness + endgame check (`53_pmoe_blender_staleness_check.py`)

Last "could-still-kill-it" concerns: was T=25 beating a stale market or sniping nearly-decided endgame? **Both ruled out.**

- **Staleness:** median 2-3s across all checkpoints, p90 ≤ 26s. Market is live.
- **Time-to-end at T=25:** median 469s (7.8 min) of game remaining. ~25% are within 5 min of end.
- **Crucial subset test:** restricted T=25 to snapshots with ≥5 min remaining (n=106) — Brier gap **grew** from −0.041 to −0.052. Blender's edge is biggest on genuinely-undecided late-game states, not on endgame snipes.

---

## Decisions made during deliberation (not separate scripts, but load-bearing)

- **Use the blender for sizing AND direction, not just direction.** The blender's `p_blend − p_market` is calibrated by construction (verified in 47/48's reliability bins) — unlike the raw model edge which inflates by ~1.7×. So it's safe as a Kelly input when combined with the existing risk controls.
- **Stay with `baseline_logistic_gpr` (not the non-GPR baseline).** ECE-on-S-tier favors the non-GPR baseline (T=10: 0.048 vs 0.063), but Brier and log-loss favor GPR at every checkpoint on every relevant slice (script 23: GPR LL is 4-7% better on 2025 S-tier rated; even bigger gain on stale-anchor games at +0.10 LL at T=10). ECE alone undervalues sharpness — GPR is sharper at the cost of a slightly higher ECE, which is the net-positive tradeoff for a betting model. Script 46's "baseline wins on production" was a cross-tier number; on the S-tier slice the blender uses, GPR wins.
- **T=10 is the riskiest live entry point.** Two reasons compound: raw GPR model has its weakest ECE at T=10 (0.063, with extreme bins miscalibrated), AND the blender places its highest model-weight at T=10 (b=0.83). The blender re-calibration absorbs most of this, but if pre-bet caution is wanted, restrict the first ~50 live bets to T=15+ and add T=10 only after live results validate the blender's T=10 confidence.
- **Don't bet directly on `p_model`, only via `p_blend`.** Blender absorbs raw-model miscalibration via its logistic re-fit; raw probabilities should not be used as Kelly inputs without that step.

---

## Where the trading thesis stands now

- **Production blender persisted** at `data/models/model_market_blender/coefficients.json`. Inference is sigmoid(a + b·logit(p_model) + c·logit(p_market)) per checkpoint.
- **Headline:** blender beats the market on temporal holdout with significant CI. T=25 OOF (0.073) was overoptimistic; realistic T=25 Brier ~0.09. Per-bet realized edge expectation: **~+0.10 to +0.15** (gross of spread/fees).
- **Operational rule (v1):**
  1. League ∈ S-tier allowlist (LCK, LPL, LCP, LCS, LEC, CBLOL, WLDs, etc. — full list in coefficients.json)
  2. Compute `p_model` from `baseline_logistic_gpr` for the appropriate checkpoint T.
  3. Compute `p_blend = sigmoid(a + b·logit(p_model) + c·logit(p_market))` using the persisted per-cp coefficients.
  4. `edge = p_blend − p_market`.
  5. Bet when `|edge| ≥ 0.10`. Quarter-Kelly with 2% bankroll cap (existing decision from discovery-checklist section 3).
  6. Kelly formula uses `p_blend` directly — its magnitude is calibrated, unlike the raw model edge which would over-bet by ~70%.
  7. Log every bet (`p_model`, `p_market`, `p_blend`, `edge_pred`, outcome) — re-check magnitude calibration after ~50 bets; if realized slope drifts from 1.0, shrink predicted edge before sizing.
  8. Optional pre-bet caution: skip T=10 for the first ~50 live bets (T=10 is where raw model is least calibrated AND blender weights model heaviest).

---

## Open issues / followups not yet done

- **Spread + fees not modeled.** All edge numbers gross. Quantifying this is now the largest gap between predicted +0.10–0.15 edge and live $-PnL. Use `trades.csv` to estimate per-checkpoint median spread.
- **Polymarket → OE join bugs.** Step 3 surfaced ~1% of markets with hours-to-days offsets, almost certainly wrong-game matches via `game_number` ambiguity in same-day Bo3s. Worth fixing in `scripts/polymarket/build_eval_table.py`.
- **Blender intercepts of +1.0 to +1.5** are larger than expected — partly base-rate skew (team_a wins 53.5%), partly population restriction. Re-check after more games accumulate; may indicate calibration tilt to correct.
- **Per-league counts in fit are uneven** (LCP+LCK = 48% of fit). If LCP/LCK are especially inefficient in our window, the blender may be partly memorizing that. Re-fit with more leagues balanced once data accumulates.
- **Re-fit cadence:** quarterly is probably right.

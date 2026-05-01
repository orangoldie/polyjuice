# Spec: Livestats first-event timing study

Status: draft, pre-implementation
Date: 2026-04-29

## Objective

Measure the empirical timing distribution of `firstblood`, `firstdragon`, `firsttower`, `firstherald`, `firstbaron` in S-tier pro games using the unofficial lolesports livestats API, and use those measurements to revisit the per-checkpoint feature gates locked in `discovery-checklist.md` section 1 ("How to handle non-time-dependent columns across time frames"). Specifically: decide whether `firstdragon` should be added to the 10-min model, and confirm or correct the gates for the other firsts at 15/20/25.

This is a one-shot discovery exercise. Output is a JSON in `data/exploration/` and an updated decision in `discovery-checklist.md`. No reusable fetcher is built; the script lives in `scripts/exploration/`.

## Background

- The current feature gates were set from indirect evidence (% of games with a flag set in the end-of-game CSV by gamelength bucket). That gives a *necessary-condition* upper bound — if `firstbaron=0` for games ending at min 20, baron didn't happen by min 20. But it can't tell us *when* within the gamelength the event happened, so the "by min 10" / "by min 15" rates are estimates.
- The firstdragon-at-10 case is the live debate: rejected in section 1 with rationale "dragon spawns at 5, first contest resolves min 6-12 → only ~50% of games have firstdragon by min 10". The 50% number is an estimate, not a measurement. User intuition is that pro firstdragon is usually <10 min.
- Section 3 also claims `firstherald` and `firstgrub` are "not in live API". This trip re-verifies that claim from real frame parsing, since the original probe only inspected one game.
- The livestats API (`feed.lolesports.com/livestats/v1/window/{gameId}`) serves both live and completed games. Frames at ~10s cadence. Discovery via `esports-api.lolesports.com/persisted/gw/getSchedule` → `getEventDetails` → `match.games[].id`. Public `x-api-key` (hardcoded in the lolesports.com frontend).

## Scope

- **Sample:** target ~300 S-tier games from 2025-01-01 to today (2026-04-29).
- **Leagues:** LCK, LEC, LCS, LPL splits (Spring + Summer 2025, Spring 2026), Worlds 2025, MSI 2025, First Stand 2026. If total fetched exceeds ~400, downsample uniformly; if under 200, expand to other 2025 international events.
- **Events measured:** `firstblood`, `firstdragon`, `firsttower`, `firstherald`, `firstbaron`. Plus a presence check for `firstvoidgrub` and `firstatakhan` for completeness (we don't model them, but knowing if they're exposed is cheap).
- **Filter:** drop remade / canceled games, drop games where frame coverage is incomplete (missing the first 5 minutes, or gaps > 60s in the first 25 min).

Out of scope (deferred or skipped per user direction):
- Train/inference parity check between Oracle CSV and livestats — skipped.
- Mapping livestats `gameId` to Oracle `gameid` — skipped.
- Coverage probe for non-S tiers — assume similar; defer to v2 if it matters.
- Live-inference latency — defer to v2.
- Building a reusable fetcher / cache — defer.

## Method

### Step 1: Discover gameIds

Walk the persisted API:

```
GET getLeagues                  → leagueIds for the listed leagues
GET getSchedule?leagueId=...    → events (matches), paginate via pageToken
GET getEventDetails?id=<match>  → match.games[].id  (one or more game IDs per Bo3/Bo5)
```

Filter events to `state == "completed"` and `startTime` in the 2025-01-01 → today window. Persist the resulting gameId list as `data/exploration/30_livestats_gameids.json` so the fetch step can be re-run independently.

**Open: schedule pagination depth.** Confirm `getSchedule` paginates back to early 2025 reliably. If it caps before that, expand the league set or shorten the date window.

### Step 2: Fetch frames

For each gameId, walk `/window/{gameId}?startingTime={iso8601}` from the game's start time forward in 10-frame (~100s) increments until the response stops yielding new frames. Cache raw frame JSON to `data/exploration/livestats_frames/{gameId}.json` so re-runs of the analysis don't re-hit the API.

Rate-limit assumption: 1 req/s with retries on 429. Conservative; tune up only if needed.

### Step 3: Detect first events

For each game, walk frames in time order and detect the first frame at which:

- `firstblood`: any team's `totalKills` transitions 0 → ≥1.
- `firstdragon`: any team's dragon count transitions 0 → ≥1. (Confirms field exists; per section 1 the type matters less than the timing.)
- `firsttower`: any team's tower kill count transitions 0 → ≥1. Will need to handle the inhibitor-tower / nexus-tower distinction — first *outer turret* is the canonical "first tower" in the CSV.
- `firstherald`: probe whether the field exists at all; if so, detect 0 → ≥1.
- `firstbaron`: any team's baron count transitions 0 → ≥1.
- `firstvoidgrub`, `firstatakhan`: presence-only check.

**Frame-at-checkpoint convention:** "the value at min T" = the last frame whose `gameTime` ≤ T·60s since gamestart. Same convention applies to checkpoint stats if we ever add them.

### Step 4: Verifications (Tier 2 questions #7, #8, #9 from prior discussion)

- **#7 — Event derivation cleanliness.** For each detected event, log the frame's gameTime *and* a plain reading of the response (e.g. log raw `dragons` array around the transition). Manually spot-check 10 games per event to confirm no false positives from respawns / reconnects / display flicker. Failures here mean the detection rule is wrong, not that the data is bad.
- **#8 — Pause behavior.** For ~5 games known to have had pauses (LCK Spring 2026 has at least two televised pauses — pick from there), check whether the frame `gameTime` advances during the pause or freezes. If it freezes, our timing measurements are clean (gameTime = real game time). If it advances, we need to subtract pause duration before bucketing — not hard, but needs to be done.
- **#9 — Schedule pagination.** While walking `getSchedule`, log how far back pageToken takes us per league. Document in the output.

### Step 5: Summarize

Write `data/exploration/30_livestats_firsts_timing.json`:

```json
{
  "fetched_at": "2026-04-29T...",
  "n_games": 312,
  "leagues": {"LCK Spring 2026": 87, "...": ...},
  "by_event": {
    "firstdragon": {
      "n_resolved": 312,
      "median_seconds": 412,
      "p25_seconds": 378, "p75_seconds": 455,
      "histogram_30s_bins": [...],
      "pct_by_checkpoint": {"10": 0.74, "15": 1.00, "20": 1.00, "25": 1.00}
    },
    "firstblood":   {...},
    "firsttower":   {...},
    "firstherald":  {"available_in_api": false, ...} or {...},
    "firstbaron":   {...}
  },
  "presence_only": {"firstvoidgrub": true, "firstatakhan": false},
  "verifications": {
    "event_detection_spot_checks": "10/10 clean for firstdragon, 9/10 firstbaron (one ambiguous frame logged at ...)",
    "pause_behavior": "gameTime freezes during pause (verified on 5 games)",
    "schedule_pagination": "getSchedule reaches 2024-12-15 for LCK before pageToken returns empty"
  }
}
```

## Decision criteria (gate updates)

For each event × checkpoint cell in section 1's gates table, update based on measured `pct_by_checkpoint` from the JSON:

- **≥ 95% resolved by checkpoint:** safe to include as a feature. Train/inference mismatch is small enough that the leakage class doesn't dominate signal.
- **70-95% resolved:** judgment call. Document the mismatch rate and consider including only if signal is strong (refit the baseline logistic with vs without and compare val log-loss).
- **< 70% resolved:** don't include. Mismatch rate too high; calibration risk dominates.

Headline cells to revisit:
- `firstdragon` @ 10 — currently ❌. If measured ≥ 95% by min 10 → flip to ✅.
- `firstherald`, `firstvoidgrub` @ all checkpoints — currently ❌ "not in live API". If presence check shows the field IS exposed → flip the rationale and apply the same threshold rule.
- `firstbaron` @ 25 — currently ✅ at 83% resolved. If measured rate is much lower than the section 1 estimate, may need to drop it.

## Deliverable checklist

- [ ] `scripts/exploration/30_livestats_firsts_timing.py` — single script doing steps 1-5.
- [ ] `data/exploration/30_livestats_gameids.json` — discovered gameIds.
- [ ] `data/exploration/livestats_frames/{gameId}.json` — cached raw frames (gitignored).
- [ ] `data/exploration/30_livestats_firsts_timing.json` — summary.
- [ ] Updated section 1 gates table in `discovery-checklist.md` with measured numbers and a `2026-04-29` provenance note.
- [ ] Updated section 3 herald/grub claim if the API actually exposes them.

## Risks

- **Unofficial API breakage.** If the hardcoded `x-api-key` is rotated or rate-limit tightens mid-fetch, the fetch step fails. Mitigation: cache aggressively (step 2), so a partial fetch is still analyzable.
- **Frame data missing for some games.** Frames may be unavailable for older 2025 games or non-broadcast games. Mitigation: drop games with incomplete coverage (already in scope filter) and report `n_games` actually used.
- **Event field naming drift.** Field names in the API response may differ from our assumptions (`dragons` vs `dragonKills` vs nested under a `team` object). Step 3 is written defensively; first run will likely require one round of schema correction.

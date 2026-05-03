"""Staleness + endgame check on the clean S-tier blender population.

For each checkpoint, look at:
  - staleness_s   (seconds since the last polymarket fill before the snapshot)
  - n_fills_pm60s (number of fills in ±60s window around the snapshot)
  - time_to_end   (gamelength_s - checkpoint*60; seconds of game left after snapshot)

If T=25 has high staleness, low n_fills, or low time_to_end, the apparent
T=25 blender win is partly "beating a stale market in the last minute of a
nearly-decided game" rather than real predictive edge.

Then: split each cp by time_to_end (>=300s vs <300s), refit per-cp blenders
on each subset, and compare Brier(blend) vs Brier(market). If the edge
disappears in the >=300s subset, that's the answer.
"""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from _tee import tee_stdout
tee_stdout('53_pmoe_blender_staleness_check.txt')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

EVAL = 'data/polymarket/eval_table.csv'
OFFSETS = 'data/exploration/49_pmoe_date_anchor_full.csv'
TIER_FILE = 'data/exploration/07_league_tier_matching.json'
EPS = 1e-4
N_FOLDS = 5
TIMING_THRESHOLD_S = 60


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def oof(X, y, groups):
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return np.full(len(y), np.nan)
    k = min(N_FOLDS, n_groups)
    gkf = GroupKFold(n_splits=k)
    out = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        lr = LogisticRegression(C=1e6, fit_intercept=True, solver='lbfgs',
                                 max_iter=1000)
        lr.fit(X[tr], y[tr])
        out[te] = lr.predict_proba(X[te])[:, 1]
    return out


def main():
    with open(TIER_FILE) as f:
        s_tier = sorted({m['league'] for m in json.load(f)['matched']
                          if m.get('tier') == 'S'})

    df = pd.read_csv(EVAL).dropna(subset=['p_model_team_a', 'market_p_team_a',
                                            'realized_team_a_won', 'oe_gameid',
                                            'checkpoint', 'league',
                                            'condition_id', 'gamelength_s'])
    off = pd.read_csv(OFFSETS)[['condition_id', 'offset_s']]
    df = df.merge(off, on='condition_id', how='left')
    sub = df[df['league'].isin(s_tier) & df['offset_s'].notna() &
             (df['offset_s'].abs() <= TIMING_THRESHOLD_S)].copy()
    sub['time_to_end'] = sub['gamelength_s'] - sub['checkpoint'] * 60
    print(f'clean S-tier population: {len(sub)} rows / '
          f'{sub["oe_gameid"].nunique()} games')

    # Distributions per checkpoint
    print(f'\n=== distributions per checkpoint ===')
    for col in ['staleness_s', 'n_fills_pm60s', 'time_to_end']:
        print(f'\n  {col}:')
        print(f'  {"cp":>4s} | {"n":>4s} | {"min":>6s} | {"p10":>6s} | '
              f'{"p25":>6s} | {"p50":>6s} | {"p75":>6s} | {"p90":>6s} | '
              f'{"max":>7s} | {"mean":>7s}')
        for cp in [10, 15, 20, 25]:
            sc = sub[sub['checkpoint'] == cp]
            x = sc[col].dropna().to_numpy()
            if len(x) == 0:
                continue
            print(f'  {cp:>4d} | {len(x):4d} | {x.min():6.0f} | '
                  f'{np.percentile(x,10):6.0f} | {np.percentile(x,25):6.0f} | '
                  f'{np.percentile(x,50):6.0f} | {np.percentile(x,75):6.0f} | '
                  f'{np.percentile(x,90):6.0f} | {x.max():7.0f} | '
                  f'{x.mean():7.0f}')

    # Subset analysis: time_to_end >= 300s vs < 300s
    print(f'\n=== per-cp blender on time_to_end subsets ===')
    print(f'  splits the population by whether the snapshot is >= 5 min '
          f'before game end')
    print(f'  {"cp":>4s} | {"subset":>10s} | {"n":>4s} | {"games":>5s} | '
          f'{"market":>7s} | {"blend":>7s} | {"Δ blend-mkt":>11s}')
    for cp in [10, 15, 20, 25]:
        sc = sub[sub['checkpoint'] == cp]
        for label, mask in [('full',     np.ones(len(sc), dtype=bool)),
                             ('end>=300s', (sc['time_to_end'] >= 300).to_numpy()),
                             ('end<300s',  (sc['time_to_end'] <  300).to_numpy())]:
            ssub = sc[mask]
            if len(ssub) < 30:
                print(f'  {cp:>4d} | {label:>10s} | {len(ssub):4d}  (skip — too small)')
                continue
            X = np.column_stack([logit(ssub['p_model_team_a']),
                                  logit(ssub['market_p_team_a'])])
            y = ssub['realized_team_a_won'].to_numpy()
            g = ssub['oe_gameid'].to_numpy()
            preds = oof(X, y, g)
            v = ~np.isnan(preds)
            bk = brier(ssub['market_p_team_a'].to_numpy()[v], y[v])
            bb = brier(preds[v], y[v])
            print(f'  {cp:>4d} | {label:>10s} | {len(ssub):4d} | '
                  f'{ssub["oe_gameid"].nunique():5d} | {bk:7.4f} | '
                  f'{bb:7.4f} | {bb-bk:+11.4f}')

    # Cross-tab staleness vs Brier on T=25 specifically
    print(f'\n=== T=25 only: staleness bins vs Brier ===')
    s25 = sub[sub['checkpoint'] == 25].copy()
    bins = [(0, 30), (30, 120), (120, 300), (300, 1e9)]
    for lo, hi in bins:
        sl = s25[(s25['staleness_s'] >= lo) & (s25['staleness_s'] < hi)]
        if len(sl) < 5:
            print(f'  staleness {lo:>4d}-{int(hi):>5d}s: n={len(sl):3d} (skip)')
            continue
        y = sl['realized_team_a_won'].to_numpy()
        bk = brier(sl['market_p_team_a'].to_numpy(), y)
        bm = brier(sl['p_model_team_a'].to_numpy(), y)
        # mean p_market and mean p_model — sanity
        print(f'  staleness {lo:>4d}-{int(hi):>5d}s: n={len(sl):3d}  '
              f'market Brier={bk:.4f}  model Brier={bm:.4f}  '
              f'mean(|p-0.5|)_mkt={np.abs(sl["market_p_team_a"]-0.5).mean():.3f}  '
              f'_mdl={np.abs(sl["p_model_team_a"]-0.5).mean():.3f}')


if __name__ == '__main__':
    main()

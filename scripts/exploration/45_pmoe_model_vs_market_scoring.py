"""Score p_model vs p_market against realized outcomes — Brier and log loss.

Per checkpoint and per top-row-count league. Paired bootstrap CI on
(market_score - model_score); positive means model wins (lower score = better).
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from _tee import tee_stdout
tee_stdout('45_pmoe_model_vs_market_scoring.txt')

import numpy as np
import pandas as pd

EPS = 1e-6
N_BOOT = 2000


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def logloss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def paired_bootstrap_diff(p_a, p_b, y, score_fn, n_boot=N_BOOT, seed=42):
    """CI on score(p_b) - score(p_a). >0 means p_a is better (lower loss)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = score_fn(p_b[idx], y[idx]) - score_fn(p_a[idx], y[idx])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def report(label, sub):
    p_model = sub['p_model_team_a'].to_numpy()
    p_mkt = sub['market_p_team_a'].to_numpy()
    y = sub['realized_team_a_won'].to_numpy()
    n = len(y)
    if n < 30:
        print(f'  {label:24s} n={n:5d}  (skip — too small)')
        return
    bm, bk = brier(p_model, y), brier(p_mkt, y)
    lm, lk = logloss(p_model, y), logloss(p_mkt, y)
    blo, bhi = paired_bootstrap_diff(p_model, p_mkt, y, brier)
    llo, lhi = paired_bootstrap_diff(p_model, p_mkt, y, logloss)
    # CI on (market - model). >0 with CI excluding 0 = model significantly better.
    sig_b = 'MODEL' if blo > 0 else ('market' if bhi < 0 else '~tie ')
    sig_l = 'MODEL' if llo > 0 else ('market' if lhi < 0 else '~tie ')
    print(f'  {label:24s} n={n:5d} | '
          f'Brier mdl={bm:.4f} mkt={bk:.4f} Δ={bk - bm:+.4f} [{blo:+.4f},{bhi:+.4f}] {sig_b} | '
          f'LL mdl={lm:.4f} mkt={lk:.4f} Δ={lk - lm:+.4f} [{llo:+.4f},{lhi:+.4f}] {sig_l}')


def main():
    df = pd.read_csv('data/polymarket/eval_table.csv')
    print(f'rows: {len(df)}  unique markets: {df["condition_id"].nunique()}  '
          f'unique games: {df["oe_gameid"].nunique()}')
    print(f'base rate (team_a wins): {df["realized_team_a_won"].mean():.4f}')

    print('\n=== overall ===')
    report('all', df)

    print('\n=== per checkpoint ===')
    for T in sorted(df['checkpoint'].unique()):
        report(f'T={T}', df[df['checkpoint'] == T])

    print('\n=== per league (top 10 by row count, across all checkpoints) ===')
    top_leagues = df['league'].value_counts().head(10).index.tolist()
    for lg in top_leagues:
        report(lg, df[df['league'] == lg])

    print('\n=== top-3 leagues × checkpoint ===')
    for lg in top_leagues[:3]:
        for T in sorted(df['checkpoint'].unique()):
            sub = df[(df['league'] == lg) & (df['checkpoint'] == T)]
            report(f'{lg} T={T}', sub)


if __name__ == '__main__':
    main()

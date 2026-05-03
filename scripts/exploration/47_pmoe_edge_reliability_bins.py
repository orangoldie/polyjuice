"""Edge reliability: when the model says +X edge over the market, does team_a
actually win at p_market + X?

Bucket eval rows by (p_model - p_market). For each bucket, compare mean
predicted edge to realized edge (= mean(realized) - mean(p_market)). If the
points sit on the diagonal, filtering by |edge| > threshold is a tradeable
strategy. If they're flat / inverted at the tails, big-|edge| picks are the
model's worst predictions and the filter loses money.

Also reports cumulative stats for |edge| >= threshold to make the
"only-bet-when-delta-is-big" question concrete.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from _tee import tee_stdout
tee_stdout('47_pmoe_edge_reliability_bins.txt')

import numpy as np
import pandas as pd

EVAL = 'data/polymarket/eval_table.csv'
N_BOOT = 2000

EDGES = [-1.01, -0.20, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 0.20, 1.01]
THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20]


def boot_mean_ci(x, n_boot=N_BOOT, seed=42):
    if len(x) < 2:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bin_report(df, label):
    print(f'\n=== reliability bins: {label} (n={len(df)}) ===')
    print(f'  {"bin":>16s} | {"n":>5s} | {"mean pred edge":>14s} | '
          f'{"realized edge":>14s} {"[2.5%, 97.5%]":>20s} | '
          f'{"slope-on-diag?":>14s}')
    df = df.copy()
    df['edge'] = df['p_model_team_a'] - df['market_p_team_a']
    df['realized_edge'] = df['realized_team_a_won'] - df['market_p_team_a']
    df['bin'] = pd.cut(df['edge'], bins=EDGES, include_lowest=True)
    rows = []
    for b, sub in df.groupby('bin', observed=True):
        n = len(sub)
        if n == 0:
            continue
        pred = sub['edge'].mean()
        real = sub['realized_edge'].to_numpy()
        real_mean = float(real.mean())
        lo, hi = boot_mean_ci(real)
        on_diag = '✅' if (lo <= pred <= hi) else '❌'
        print(f'  {str(b):>16s} | {n:5d} | {pred:+14.4f} | '
              f'{real_mean:+14.4f} [{lo:+.4f}, {hi:+.4f}] |     {on_diag}')
        rows.append((str(b), n, pred, real_mean, lo, hi))
    return rows


def threshold_report(df, label):
    """Cumulative: only bet when |edge| >= t. Reports n, hit rate, mean
    realized edge (signed in the direction of the bet), and naive PnL per
    $1 stake using the market price (decimal odds = 1/p_taken)."""
    print(f'\n=== bet-when-|edge|>=t: {label} (n={len(df)}) ===')
    print(f'  {"thr":>5s} | {"n_bets":>6s} | {"frac":>5s} | '
          f'{"mean |pred edge|":>16s} | {"mean realized edge":>18s} '
          f'{"[2.5%, 97.5%]":>20s} | {"naive $/bet":>11s}')
    df = df.copy()
    df['edge'] = df['p_model_team_a'] - df['market_p_team_a']
    # bet team_a if edge > 0, team_b if edge < 0
    # signed realized edge = direction * (realized - p_market_taken)
    # YES side: pay p_market_team_a, win 1 if team_a wins
    # NO side: pay 1 - p_market_team_a, win 1 if team_a loses
    p_a = df['market_p_team_a'].to_numpy()
    y_a = df['realized_team_a_won'].to_numpy()
    edge = df['edge'].to_numpy()
    # naive PnL per $1: bet on side model favors
    bet_a = edge > 0
    cost = np.where(bet_a, p_a, 1 - p_a)
    win = np.where(bet_a, y_a, 1 - y_a)  # 0/1
    pnl = win - cost                      # per $1 stake
    abs_edge = np.abs(edge)
    # signed realized edge in direction of bet
    realized_signed = np.where(bet_a, y_a - p_a, (1 - y_a) - (1 - p_a))
    n = len(df)
    for t in THRESHOLDS:
        m = abs_edge >= t
        nb = int(m.sum())
        if nb == 0:
            print(f'  {t:5.2f} | {nb:6d} | {nb/n:5.1%} |  (no bets)')
            continue
        rs = realized_signed[m]
        lo, hi = boot_mean_ci(rs)
        mean_pred = float(abs_edge[m].mean())
        mean_real = float(rs.mean())
        mean_pnl = float(pnl[m].mean())
        print(f'  {t:5.2f} | {nb:6d} | {nb/n:5.1%} | {mean_pred:16.4f} | '
              f'{mean_real:+18.4f} [{lo:+.4f}, {hi:+.4f}] | '
              f'{mean_pnl:+11.4f}')


def main():
    df = pd.read_csv(EVAL)
    needed = ['p_model_team_a', 'market_p_team_a', 'realized_team_a_won',
              'checkpoint', 'league']
    df = df.dropna(subset=needed).copy()
    print(f'rows: {len(df)}  unique markets: {df["condition_id"].nunique()}')
    print(f'base rate (team_a wins): {df["realized_team_a_won"].mean():.4f}')
    print(f'mean p_market_team_a: {df["market_p_team_a"].mean():.4f}')
    print(f'mean p_model_team_a:  {df["p_model_team_a"].mean():.4f}')
    print(f'mean (p_model - p_market): '
          f'{(df["p_model_team_a"] - df["market_p_team_a"]).mean():+.4f}')

    bin_report(df, 'all')
    threshold_report(df, 'all')

    for cp in sorted(df['checkpoint'].unique()):
        sub = df[df['checkpoint'] == cp]
        bin_report(sub, f'T={cp}')
        threshold_report(sub, f'T={cp}')

    # league cuts only for the leagues that mattered in script 45
    for lg in ['LCKC', 'EM']:
        sub = df[df['league'] == lg]
        if len(sub) >= 30:
            bin_report(sub, f'league={lg}')
            threshold_report(sub, f'league={lg}')


if __name__ == '__main__':
    main()

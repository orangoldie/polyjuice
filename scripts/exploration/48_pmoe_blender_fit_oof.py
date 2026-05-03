"""Fit a model-market blender: logit(p_blend) = a + b·logit(p_model) + c·logit(p_market).

Uses GroupKFold by oe_gameid so per-game checkpoint rows can't leak across
folds. Outputs OOF blended probabilities so the downstream threshold/
reliability analysis is honest (no row scored by a model fit on its own game).

Reports:
  - Fitted (b, c) coefficients per checkpoint and shared
  - OOF Brier(blend) vs Brier(model) vs Brier(market)
  - Reliability bins on (p_blend - p_market)
  - Threshold-bet PnL on (p_blend - p_market), with EM excluded variant
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from _tee import tee_stdout
tee_stdout('48_pmoe_blender_fit_oof.txt')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

EVAL = 'data/polymarket/eval_table.csv'
EPS = 1e-4
N_BOOT = 2000
N_FOLDS = 5

EDGES = [-1.01, -0.20, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 0.20, 1.01]
THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20]


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def boot_mean_ci(x, n_boot=N_BOOT, seed=42):
    if len(x) < 2:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fit_blender_oof(df, by_checkpoint=False):
    """Group k-fold by oe_gameid. Returns p_blend per row + per-fold coefs."""
    df = df.reset_index(drop=True).copy()
    X = np.column_stack([logit(df['p_model_team_a']),
                         logit(df['market_p_team_a'])])
    y = df['realized_team_a_won'].to_numpy()
    groups = df['oe_gameid'].to_numpy()
    p_blend = np.full(len(df), np.nan)

    if by_checkpoint:
        coefs = {}
        for cp, sub in df.groupby('checkpoint'):
            idx = sub.index.to_numpy()
            Xc = X[idx]; yc = y[idx]; gc = groups[idx]
            n_groups = len(np.unique(gc))
            k = min(N_FOLDS, n_groups)
            gkf = GroupKFold(n_splits=k)
            fold_coefs = []
            for tr, te in gkf.split(Xc, yc, gc):
                lr = LogisticRegression(C=1e6, fit_intercept=True,
                                        solver='lbfgs', max_iter=1000)
                lr.fit(Xc[tr], yc[tr])
                p_blend[idx[te]] = lr.predict_proba(Xc[te])[:, 1]
                fold_coefs.append((lr.intercept_[0], lr.coef_[0, 0],
                                   lr.coef_[0, 1]))
            coefs[cp] = np.mean(fold_coefs, axis=0)
        return p_blend, coefs

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_coefs = []
    for tr, te in gkf.split(X, y, groups):
        lr = LogisticRegression(C=1e6, fit_intercept=True,
                                solver='lbfgs', max_iter=1000)
        lr.fit(X[tr], y[tr])
        p_blend[te] = lr.predict_proba(X[te])[:, 1]
        fold_coefs.append((lr.intercept_[0], lr.coef_[0, 0], lr.coef_[0, 1]))
    return p_blend, np.mean(fold_coefs, axis=0)


def reliability_bins(df, edge_col, label):
    print(f'\n--- reliability bins ({edge_col}): {label} (n={len(df)}) ---')
    print(f'  {"bin":>16s} | {"n":>5s} | {"pred edge":>10s} | '
          f'{"realized":>10s} {"[2.5%, 97.5%]":>20s}')
    df = df.copy()
    df['edge'] = df[edge_col] - df['market_p_team_a']
    df['realized_edge'] = df['realized_team_a_won'] - df['market_p_team_a']
    df['bin'] = pd.cut(df['edge'], bins=EDGES, include_lowest=True)
    for b, sub in df.groupby('bin', observed=True):
        if len(sub) == 0:
            continue
        pred = sub['edge'].mean()
        real = sub['realized_edge'].to_numpy()
        lo, hi = boot_mean_ci(real)
        on_diag = '✅' if (lo <= pred <= hi) else '❌'
        print(f'  {str(b):>16s} | {len(sub):5d} | {pred:+10.4f} | '
              f'{real.mean():+10.4f} [{lo:+.4f}, {hi:+.4f}] {on_diag}')


def threshold_report(df, edge_col, label):
    print(f'\n--- bet-when-|edge|>=t ({edge_col}): {label} (n={len(df)}) ---')
    print(f'  {"thr":>5s} | {"n":>5s} | {"frac":>5s} | '
          f'{"|pred|":>8s} | {"realized":>10s} {"[2.5%, 97.5%]":>20s} | '
          f'{"$/bet":>8s}')
    p_a = df['market_p_team_a'].to_numpy()
    y_a = df['realized_team_a_won'].to_numpy()
    edge = (df[edge_col] - df['market_p_team_a']).to_numpy()
    bet_a = edge > 0
    cost = np.where(bet_a, p_a, 1 - p_a)
    win = np.where(bet_a, y_a, 1 - y_a)
    pnl = win - cost
    realized_signed = np.where(bet_a, y_a - p_a, (1 - y_a) - (1 - p_a))
    abs_edge = np.abs(edge)
    n = len(df)
    for t in THRESHOLDS:
        m = abs_edge >= t
        nb = int(m.sum())
        if nb == 0:
            print(f'  {t:5.2f} | {nb:5d} |  (no bets)')
            continue
        rs = realized_signed[m]
        lo, hi = boot_mean_ci(rs)
        print(f'  {t:5.2f} | {nb:5d} | {nb/n:5.1%} | '
              f'{abs_edge[m].mean():8.4f} | {rs.mean():+10.4f} '
              f'[{lo:+.4f}, {hi:+.4f}] | {pnl[m].mean():+8.4f}')


def main():
    df = pd.read_csv(EVAL)
    df = df.dropna(subset=['p_model_team_a', 'market_p_team_a',
                            'realized_team_a_won', 'checkpoint',
                            'league', 'oe_gameid']).copy()
    print(f'rows: {len(df)}  games: {df["oe_gameid"].nunique()}')

    # Shared blender
    p_shared, coefs_shared = fit_blender_oof(df, by_checkpoint=False)
    df['p_blend_shared'] = p_shared
    a, b, c = coefs_shared
    print(f'\n=== shared blender (avg fold coefs) ===')
    print(f'  intercept={a:+.4f}  b·logit(p_model)={b:+.4f}  '
          f'c·logit(p_market)={c:+.4f}')
    print(f'  ratio c/(b+c) = {c/(b+c):.3f}  (1.0 = use market only, '
          f'0.0 = use model only)')

    # Per-checkpoint blender
    p_cp, coefs_cp = fit_blender_oof(df, by_checkpoint=True)
    df['p_blend_cp'] = p_cp
    print(f'\n=== per-checkpoint blender (avg fold coefs) ===')
    print(f'  {"cp":>4s} | {"intercept":>10s} | {"b (model)":>10s} | '
          f'{"c (market)":>11s} | {"c/(b+c)":>8s}')
    for cp in sorted(coefs_cp):
        a, b, c = coefs_cp[cp]
        share = c / (b + c) if (b + c) > 0 else float('nan')
        print(f'  {cp:>4d} | {a:+10.4f} | {b:+10.4f} | {c:+11.4f} | '
              f'{share:8.3f}')

    # Brier comparison (overall + per checkpoint)
    print(f'\n=== Brier on OOF predictions ===')
    print(f'  {"slice":>14s} | {"n":>5s} | {"model":>7s} | {"market":>7s} | '
          f'{"blend_sh":>9s} | {"blend_cp":>9s}')
    def row(label, sub):
        y = sub['realized_team_a_won'].to_numpy()
        print(f'  {label:>14s} | {len(sub):5d} | '
              f'{brier(sub["p_model_team_a"], y):7.4f} | '
              f'{brier(sub["market_p_team_a"], y):7.4f} | '
              f'{brier(sub["p_blend_shared"], y):9.4f} | '
              f'{brier(sub["p_blend_cp"], y):9.4f}')
    row('all', df)
    for cp in sorted(df['checkpoint'].unique()):
        row(f'T={cp}', df[df['checkpoint'] == cp])
    for lg in ['LCKC', 'EM', 'LCK', 'LCP', 'LEC']:
        sub = df[df['league'] == lg]
        if len(sub) >= 30:
            row(f'lg={lg}', sub)

    # Reliability + threshold on the per-cp blend (more honest of the two)
    reliability_bins(df, 'p_blend_cp', 'all')
    threshold_report(df, 'p_blend_cp', 'all')

    df_no_em = df[df['league'] != 'EM']
    threshold_report(df_no_em, 'p_blend_cp', 'all minus EM')

    for cp in sorted(df['checkpoint'].unique()):
        sub = df[df['checkpoint'] == cp]
        threshold_report(sub, 'p_blend_cp', f'T={cp}')

    for lg in ['LCKC', 'EM']:
        sub = df[df['league'] == lg]
        if len(sub) >= 30:
            threshold_report(sub, 'p_blend_cp', f'league={lg}')

    # Compare raw-model edge vs blended edge head-to-head
    print(f'\n=== head-to-head: raw-model edge vs blended edge ===')
    print(f'  (signed realized edge in direction of bet, all rows)')
    for label, col in [('raw model', 'p_model_team_a'),
                       ('blend (shared)', 'p_blend_shared'),
                       ('blend (per-cp)', 'p_blend_cp')]:
        edge = (df[col] - df['market_p_team_a']).to_numpy()
        bet_a = edge > 0
        y_a = df['realized_team_a_won'].to_numpy()
        p_a = df['market_p_team_a'].to_numpy()
        rs = np.where(bet_a, y_a - p_a, (1 - y_a) - (1 - p_a))
        for t in [0.05, 0.10]:
            m = np.abs(edge) >= t
            if m.sum() == 0:
                continue
            lo, hi = boot_mean_ci(rs[m])
            print(f'  {label:>16s} thr={t:.2f}  n={m.sum():4d}  '
                  f'realized={rs[m].mean():+.4f} [{lo:+.4f}, {hi:+.4f}]')


if __name__ == '__main__':
    main()

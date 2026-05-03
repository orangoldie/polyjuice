"""Sweep timing-filter threshold and refit the model-market blender at each.

For each |offset_s| threshold in {inf, 300, 180, 60}:
  - Filter eval_table to markets passing that threshold.
  - Fit per-checkpoint blender (group k-fold by oe_gameid).
  - Report (a, b, c) per checkpoint and Brier(blend) vs Brier(model)/Brier(market).
  - Report threshold-betting realized edge at |edge|>=0.10, overall and LCKC-only.

If coefficients (especially c/(b+c)) shift sharply as the filter tightens, the
unfiltered blender is biased by timing-induced information asymmetry.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from _tee import tee_stdout
tee_stdout('50_pmoe_blender_timing_sweep.txt')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

EVAL = 'data/polymarket/eval_table.csv'
OFFSETS = 'data/exploration/49_pmoe_date_anchor_full.csv'
EPS = 1e-4
N_FOLDS = 5
N_BOOT = 2000
THRESHOLDS_S = [float('inf'), 300, 180, 60]
BET_THR = 0.10


def logit(p):
    return np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def boot_mean_ci(x, n_boot=N_BOOT, seed=42):
    if len(x) < 2:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    return (float(np.percentile(x[idx].mean(axis=1), 2.5)),
            float(np.percentile(x[idx].mean(axis=1), 97.5)))


def fit_blender_per_cp_oof(df):
    """Per-checkpoint blender via group k-fold on oe_gameid. Returns
    (df with p_blend column, dict cp -> (a, b, c) avg fold coefs)."""
    df = df.reset_index(drop=True).copy()
    p_blend = np.full(len(df), np.nan)
    coefs = {}
    for cp, sub in df.groupby('checkpoint'):
        idx = sub.index.to_numpy()
        X = np.column_stack([logit(sub['p_model_team_a']),
                              logit(sub['market_p_team_a'])])
        y = sub['realized_team_a_won'].to_numpy()
        groups = sub['oe_gameid'].to_numpy()
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            continue
        k = min(N_FOLDS, n_groups)
        gkf = GroupKFold(n_splits=k)
        fold_coefs = []
        for tr, te in gkf.split(X, y, groups):
            lr = LogisticRegression(C=1e6, fit_intercept=True,
                                    solver='lbfgs', max_iter=1000)
            lr.fit(X[tr], y[tr])
            p_blend[idx[te]] = lr.predict_proba(X[te])[:, 1]
            fold_coefs.append((lr.intercept_[0], lr.coef_[0, 0],
                                lr.coef_[0, 1]))
        coefs[cp] = np.mean(fold_coefs, axis=0)
    df['p_blend_cp'] = p_blend
    return df, coefs


def threshold_edge(df, edge_col, thr=BET_THR):
    p_a = df['market_p_team_a'].to_numpy()
    y_a = df['realized_team_a_won'].to_numpy()
    edge = (df[edge_col] - df['market_p_team_a']).to_numpy()
    abs_edge = np.abs(edge)
    bet_a = edge > 0
    rs = np.where(bet_a, y_a - p_a, (1 - y_a) - (1 - p_a))
    m = abs_edge >= thr
    if m.sum() == 0:
        return 0, float('nan'), float('nan'), float('nan')
    lo, hi = boot_mean_ci(rs[m])
    return int(m.sum()), float(rs[m].mean()), lo, hi


def main():
    eval_df = pd.read_csv(EVAL)
    eval_df = eval_df.dropna(subset=['p_model_team_a', 'market_p_team_a',
                                       'realized_team_a_won', 'oe_gameid',
                                       'checkpoint', 'league']).copy()
    off = pd.read_csv(OFFSETS)[['condition_id', 'offset_s']]
    eval_df = eval_df.merge(off, on='condition_id', how='left')
    has_off = eval_df['offset_s'].notna().sum()
    print(f'eval rows: {len(eval_df)}  with offset_s: {has_off}  '
          f'(no-offset rows kept only in inf-threshold pass)')

    # Per-threshold sweep
    rows = []
    for thr_s in THRESHOLDS_S:
        if thr_s == float('inf'):
            sub = eval_df.copy()
            label = 'inf'
        else:
            sub = eval_df[(eval_df['offset_s'].abs() <= thr_s)].copy()
            label = f'<={int(thr_s)}s'
        n_markets = sub['oe_gameid'].nunique()
        print(f'\n=================================================')
        print(f'== filter |offset|{label}  rows={len(sub)}  markets={n_markets}')
        print(f'=================================================')
        sub2, coefs = fit_blender_per_cp_oof(sub)

        print(f'\nPer-checkpoint blender coefs:')
        print(f'  {"cp":>4s} | {"n":>5s} | {"a":>8s} | {"b (model)":>10s} | '
              f'{"c (market)":>11s} | {"c/(b+c)":>8s}')
        for cp in sorted(coefs):
            a, b, c = coefs[cp]
            n_cp = (sub2['checkpoint'] == cp).sum()
            share = c / (b + c) if (b + c) > 0 else float('nan')
            print(f'  {cp:>4d} | {n_cp:5d} | {a:+8.4f} | {b:+10.4f} | '
                  f'{c:+11.4f} | {share:8.3f}')

        # Brier
        valid = sub2.dropna(subset=['p_blend_cp'])
        y = valid['realized_team_a_won'].to_numpy()
        bm = brier(valid['p_model_team_a'], y)
        bk = brier(valid['market_p_team_a'], y)
        bb = brier(valid['p_blend_cp'], y)
        print(f'\nBrier (n={len(valid)}):  model={bm:.4f}  market={bk:.4f}  '
              f'blend={bb:.4f}  (Δblend-vs-market={bb-bk:+.4f})')

        # Threshold edge — overall
        n_b, mean_b, lo_b, hi_b = threshold_edge(valid, 'p_blend_cp', BET_THR)
        n_r, mean_r, lo_r, hi_r = threshold_edge(valid, 'p_model_team_a', BET_THR)
        print(f'\nBet-when-|edge|>=0.10 (all):')
        print(f'  raw model:    n={n_r:4d}  realized={mean_r:+.4f} '
              f'[{lo_r:+.4f}, {hi_r:+.4f}]')
        print(f'  blend (per-cp): n={n_b:4d}  realized={mean_b:+.4f} '
              f'[{lo_b:+.4f}, {hi_b:+.4f}]')

        # LCKC-only
        lckc = valid[valid['league'] == 'LCKC']
        if len(lckc) > 30:
            n_b2, mean_b2, lo_b2, hi_b2 = threshold_edge(lckc, 'p_blend_cp',
                                                           BET_THR)
            n_r2, mean_r2, lo_r2, hi_r2 = threshold_edge(lckc,
                                                           'p_model_team_a',
                                                           BET_THR)
            print(f'\nBet-when-|edge|>=0.10 (LCKC, n_rows={len(lckc)}):')
            print(f'  raw model:    n={n_r2:4d}  realized={mean_r2:+.4f} '
                  f'[{lo_r2:+.4f}, {hi_r2:+.4f}]')
            print(f'  blend (per-cp): n={n_b2:4d}  realized={mean_b2:+.4f} '
                  f'[{lo_b2:+.4f}, {hi_b2:+.4f}]')

        rows.append({
            'filter': label,
            'rows': len(sub),
            'markets': n_markets,
            'coefs': coefs,
            'brier_blend': bb,
            'brier_market': bk,
        })

    print(f'\n=================================================')
    print(f'== summary: c/(b+c) by checkpoint and filter ==')
    print(f'=================================================')
    print(f'  {"cp":>4s} | ' + ' | '.join(f'{r["filter"]:>10s}'
                                            for r in rows))
    for cp in [10, 15, 20, 25]:
        row = f'  {cp:>4d} | '
        cells = []
        for r in rows:
            if cp in r['coefs']:
                a, b, c = r['coefs'][cp]
                share = c / (b + c) if (b + c) > 0 else float('nan')
                cells.append(f'{share:>10.3f}')
            else:
                cells.append(f'{"-":>10s}')
        print(row + ' | '.join(cells))


if __name__ == '__main__':
    main()

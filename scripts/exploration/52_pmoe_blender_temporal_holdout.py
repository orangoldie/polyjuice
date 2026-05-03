"""Honest temporal-holdout + bootstrap-CI evaluation of the production blender.

Selection: same as the persisted blender — S-tier ∩ |offset_s|<=60s.

Test 1 (temporal holdout): split games by checkpoint_t_unix; fit per-cp blender
on the earliest 80% of GAMES, evaluate on the latest 20%. Reports Brier(blend)
vs Brier(model) vs Brier(market) on the held-out window.

Test 2 (bootstrap CI on Brier deltas): on the existing OOF (group-k-fold)
predictions over the full clean S-tier population, paired-bootstrap the
quantities (Brier_blend - Brier_market) and (Brier_blend - Brier_model) so we
can tell whether the gaps survive sampling noise.
"""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from _tee import tee_stdout
tee_stdout('52_pmoe_blender_temporal_holdout.txt')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

EVAL = 'data/polymarket/eval_table.csv'
OFFSETS = 'data/exploration/49_pmoe_date_anchor_full.csv'
TIER_FILE = 'data/exploration/07_league_tier_matching.json'
EPS = 1e-4
N_FOLDS = 5
N_BOOT = 2000
TIMING_THRESHOLD_S = 60
TRAIN_FRAC = 0.8


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def fit(X, y):
    lr = LogisticRegression(C=1e6, fit_intercept=True, solver='lbfgs',
                             max_iter=1000)
    lr.fit(X, y)
    return lr


def boot_paired_brier_diff(p_a, p_b, y, n_boot=N_BOOT, seed=42):
    """CI on Brier(p_a) - Brier(p_b). <0 means p_a is better."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = brier(p_a[idx], y[idx]) - brier(p_b[idx], y[idx])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def oof_predict(X, y, groups):
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return np.full(len(y), np.nan)
    k = min(N_FOLDS, n_groups)
    gkf = GroupKFold(n_splits=k)
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        oof[te] = fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return oof


def main():
    with open(TIER_FILE) as f:
        s_tier = sorted({m['league'] for m in json.load(f)['matched']
                          if m.get('tier') == 'S'})

    eval_df = pd.read_csv(EVAL).dropna(
        subset=['p_model_team_a', 'market_p_team_a', 'realized_team_a_won',
                'oe_gameid', 'checkpoint', 'league', 'condition_id',
                'checkpoint_t_unix'])
    off = pd.read_csv(OFFSETS)[['condition_id', 'offset_s']]
    df = eval_df.merge(off, on='condition_id', how='left')
    sub = df[df['league'].isin(s_tier) & df['offset_s'].notna() &
             (df['offset_s'].abs() <= TIMING_THRESHOLD_S)].copy()
    print(f'clean S-tier population: {len(sub)} rows / '
          f'{sub["oe_gameid"].nunique()} games')

    # === Test 1: temporal holdout ===
    g_first = sub.groupby('oe_gameid')['checkpoint_t_unix'].min().reset_index()
    g_first = g_first.sort_values('checkpoint_t_unix')
    n_games = len(g_first)
    n_train_games = int(np.floor(n_games * TRAIN_FRAC))
    train_games = set(g_first.iloc[:n_train_games]['oe_gameid'])
    test_games = set(g_first.iloc[n_train_games:]['oe_gameid'])
    train = sub[sub['oe_gameid'].isin(train_games)]
    test = sub[sub['oe_gameid'].isin(test_games)]

    train_cut_dt = pd.to_datetime(g_first.iloc[n_train_games - 1]
                                    ['checkpoint_t_unix'], unit='s')
    test_start_dt = pd.to_datetime(g_first.iloc[n_train_games]
                                     ['checkpoint_t_unix'], unit='s')
    print(f'\n=== Test 1: temporal holdout ===')
    print(f'train: {len(train)} rows / {len(train_games)} games '
          f'(through {train_cut_dt:%Y-%m-%d})')
    print(f'test:  {len(test):4d} rows / {len(test_games):3d} games '
          f'(from {test_start_dt:%Y-%m-%d} onward)')

    print(f'\n  per-cp Brier on temporal holdout:')
    print(f'  {"cp":>4s} | {"n_test":>6s} | {"model":>7s} | {"market":>7s} | '
          f'{"blend":>7s} | {"Δ blend-mkt":>11s} {"[2.5%, 97.5%]":>20s}')
    for cp in [10, 15, 20, 25]:
        tr = train[train['checkpoint'] == cp]
        te = test[test['checkpoint'] == cp]
        if len(tr) < 30 or len(te) < 5:
            print(f'  {cp:>4d} | tr={len(tr)} te={len(te)} too small')
            continue
        Xtr = np.column_stack([logit(tr['p_model_team_a']),
                                logit(tr['market_p_team_a'])])
        ytr = tr['realized_team_a_won'].to_numpy()
        Xte = np.column_stack([logit(te['p_model_team_a']),
                                logit(te['market_p_team_a'])])
        yte = te['realized_team_a_won'].to_numpy()
        lr = fit(Xtr, ytr)
        p_blend = lr.predict_proba(Xte)[:, 1]
        p_model = te['p_model_team_a'].to_numpy()
        p_mkt = te['market_p_team_a'].to_numpy()
        bm = brier(p_model, yte)
        bk = brier(p_mkt, yte)
        bb = brier(p_blend, yte)
        lo, hi = boot_paired_brier_diff(p_blend, p_mkt, yte)
        print(f'  {cp:>4d} | {len(te):6d} | {bm:7.4f} | {bk:7.4f} | {bb:7.4f} | '
              f'{bb-bk:+11.4f} [{lo:+.4f}, {hi:+.4f}]')

    # All cp combined
    Xtr = np.column_stack([logit(train['p_model_team_a']),
                            logit(train['market_p_team_a'])])
    ytr = train['realized_team_a_won'].to_numpy()
    # NB: shared fit (single model across cp) for the combined row only
    lr_all = fit(Xtr, ytr)
    Xte = np.column_stack([logit(test['p_model_team_a']),
                            logit(test['market_p_team_a'])])
    yte = test['realized_team_a_won'].to_numpy()
    p_blend = lr_all.predict_proba(Xte)[:, 1]
    bm = brier(test['p_model_team_a'], yte)
    bk = brier(test['market_p_team_a'], yte)
    bb = brier(p_blend, yte)
    lo, hi = boot_paired_brier_diff(p_blend,
                                      test['market_p_team_a'].to_numpy(), yte)
    print(f'  {"shared":>4s} | {len(test):6d} | {bm:7.4f} | {bk:7.4f} | '
          f'{bb:7.4f} | {bb-bk:+11.4f} [{lo:+.4f}, {hi:+.4f}]')

    # === Test 2: bootstrap CI on OOF Brier deltas ===
    print(f'\n=== Test 2: bootstrap CI on OOF Brier deltas (full clean S-tier) ===')
    print(f'  {"cp":>4s} | {"n":>4s} | {"Δ blend-mkt":>11s} {"[2.5%, 97.5%]":>20s}'
          f' | {"Δ blend-mdl":>11s} {"[2.5%, 97.5%]":>20s}')
    for cp in [10, 15, 20, 25]:
        sc = sub[sub['checkpoint'] == cp]
        if len(sc) < 30:
            continue
        X = np.column_stack([logit(sc['p_model_team_a']),
                              logit(sc['market_p_team_a'])])
        y = sc['realized_team_a_won'].to_numpy()
        groups = sc['oe_gameid'].to_numpy()
        oof = oof_predict(X, y, groups)
        valid = ~np.isnan(oof)
        oof = oof[valid]; y = y[valid]
        p_mkt = sc['market_p_team_a'].to_numpy()[valid]
        p_mdl = sc['p_model_team_a'].to_numpy()[valid]
        d_mk = brier(oof, y) - brier(p_mkt, y)
        d_md = brier(oof, y) - brier(p_mdl, y)
        lo_k, hi_k = boot_paired_brier_diff(oof, p_mkt, y)
        lo_m, hi_m = boot_paired_brier_diff(oof, p_mdl, y)
        print(f'  {cp:>4d} | {len(y):4d} | {d_mk:+11.4f} '
              f'[{lo_k:+.4f}, {hi_k:+.4f}] | {d_md:+11.4f} '
              f'[{lo_m:+.4f}, {hi_m:+.4f}]')


if __name__ == '__main__':
    main()

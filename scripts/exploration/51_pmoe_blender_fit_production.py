"""Fit the production model-market blender on the clean-timing S-tier subset
and persist the coefficients to data/models/model_market_blender/.

Selection rule:
  - league in S_TIER_ALLOWLIST (matches GPR-eligible leagues)
  - |offset_s| <= 60   (clean game-start timing)

For each checkpoint T in {10, 15, 20, 25} fit:
    logit(p_blend) = a + b · logit(p_model) + c · logit(p_market)

Persisted artifact: data/models/model_market_blender/coefficients.json
  {
    "fit_date": ...,
    "selection": {...},
    "n_rows_total": ...,
    "checkpoints": {
       "10": {"a": ..., "b": ..., "c": ..., "n_fit": ...,
              "oof_brier_blend": ..., "oof_brier_model": ...,
              "oof_brier_market": ...},
       ...
    }
  }

Also writes a runs.csv-style log row so the fit is greppable later.
"""
import sys, io, json, datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from _tee import tee_stdout
tee_stdout('51_pmoe_blender_fit_production.txt')

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

EVAL = 'data/polymarket/eval_table.csv'
OFFSETS = 'data/exploration/49_pmoe_date_anchor_full.csv'
TIER_FILE = 'data/exploration/07_league_tier_matching.json'
OUT_DIR = 'data/models/model_market_blender'
EPS = 1e-4
N_FOLDS = 5
TIMING_THRESHOLD_S = 60


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def load_s_tier_leagues():
    with open(TIER_FILE) as f:
        d = json.load(f)
    return sorted({m['league'] for m in d['matched'] if m.get('tier') == 'S'})


def fit_blender(X, y):
    lr = LogisticRegression(C=1e6, fit_intercept=True, solver='lbfgs',
                             max_iter=1000)
    lr.fit(X, y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0]), float(lr.coef_[0, 1])


def oof_predict(X, y, groups, n_folds=N_FOLDS):
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return np.full(len(y), np.nan)
    k = min(n_folds, n_groups)
    gkf = GroupKFold(n_splits=k)
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        lr = LogisticRegression(C=1e6, fit_intercept=True, solver='lbfgs',
                                 max_iter=1000)
        lr.fit(X[tr], y[tr])
        oof[te] = lr.predict_proba(X[te])[:, 1]
    return oof


def main():
    s_tier = load_s_tier_leagues()
    print(f's-tier allowlist ({len(s_tier)}): {s_tier}')

    eval_df = pd.read_csv(EVAL)
    eval_df = eval_df.dropna(subset=['p_model_team_a', 'market_p_team_a',
                                       'realized_team_a_won', 'oe_gameid',
                                       'checkpoint', 'league',
                                       'condition_id']).copy()
    off = pd.read_csv(OFFSETS)[['condition_id', 'offset_s']]
    df = eval_df.merge(off, on='condition_id', how='left')

    in_allow = df['league'].isin(s_tier)
    has_off = df['offset_s'].notna()
    clean = df['offset_s'].abs() <= TIMING_THRESHOLD_S

    print(f'\neval rows: {len(df)}')
    print(f'  in S-tier allowlist:           {in_allow.sum()}')
    print(f'  with offset_s available:       {has_off.sum()}')
    print(f'  S-tier AND |offset| <= {TIMING_THRESHOLD_S}s:    '
          f'{(in_allow & has_off & clean).sum()}')

    sub = df[in_allow & has_off & clean].copy()
    print(f'\nfit population: {len(sub)} rows from {sub["oe_gameid"].nunique()} '
          f'games across {sub["league"].nunique()} leagues')
    print('\n  per-league row counts in fit population:')
    for lg, n in sub['league'].value_counts().items():
        print(f'    {lg:>8s}  {n:5d}')

    # Per-checkpoint fit + OOF eval
    artifact = {
        'fit_date_utc': datetime.datetime.utcnow().isoformat() + 'Z',
        'selection': {
            'league_allowlist_tier': 'S',
            'league_allowlist': s_tier,
            'timing_threshold_s': TIMING_THRESHOLD_S,
            'eval_table': EVAL,
            'offsets_source': OFFSETS,
        },
        'n_rows_total': int(len(sub)),
        'n_games_total': int(sub['oe_gameid'].nunique()),
        'checkpoints': {},
    }

    print(f'\n=== per-checkpoint fits ===')
    print(f'  {"cp":>4s} | {"n_fit":>6s} | {"a":>8s} | {"b (model)":>10s} | '
          f'{"c (market)":>11s} | {"c/(b+c)":>8s} | '
          f'{"oof Brier blend":>16s} | {"vs market":>10s}')
    for cp in [10, 15, 20, 25]:
        sc = sub[sub['checkpoint'] == cp]
        if len(sc) < 30:
            print(f'  {cp:>4d} | n={len(sc)} too small, skipping')
            continue
        X = np.column_stack([logit(sc['p_model_team_a']),
                              logit(sc['market_p_team_a'])])
        y = sc['realized_team_a_won'].to_numpy()
        groups = sc['oe_gameid'].to_numpy()

        a, b, c = fit_blender(X, y)
        oof = oof_predict(X, y, groups)
        valid = ~np.isnan(oof)
        bb = brier(oof[valid], y[valid])
        bm = brier(sc['p_model_team_a'].to_numpy()[valid], y[valid])
        bk = brier(sc['market_p_team_a'].to_numpy()[valid], y[valid])
        share = c / (b + c) if (b + c) > 0 else float('nan')
        print(f'  {cp:>4d} | {len(sc):6d} | {a:+8.4f} | {b:+10.4f} | '
              f'{c:+11.4f} | {share:8.3f} | {bb:16.4f} | {bb-bk:+10.4f}')

        artifact['checkpoints'][str(cp)] = {
            'a': a, 'b': b, 'c': c,
            'n_fit': int(len(sc)),
            'oof_brier_blend': bb,
            'oof_brier_model': bm,
            'oof_brier_market': bk,
        }

    # Bonus: shared (all-checkpoints) fit, for reference
    X = np.column_stack([logit(sub['p_model_team_a']),
                          logit(sub['market_p_team_a'])])
    y = sub['realized_team_a_won'].to_numpy()
    groups = sub['oe_gameid'].to_numpy()
    a, b, c = fit_blender(X, y)
    oof = oof_predict(X, y, groups)
    valid = ~np.isnan(oof)
    bb = brier(oof[valid], y[valid])
    bk = brier(sub['market_p_team_a'].to_numpy()[valid], y[valid])
    share = c / (b + c) if (b + c) > 0 else float('nan')
    print(f'\n  shared (all cp) | n={len(sub):5d} | a={a:+.4f}  b={b:+.4f}  '
          f'c={c:+.4f}  c/(b+c)={share:.3f}  oof Brier={bb:.4f} '
          f'(market {bk:.4f})')
    artifact['shared_reference'] = {'a': a, 'b': b, 'c': c, 'n_fit': int(len(sub)),
                                      'oof_brier_blend': bb,
                                      'oof_brier_market': bk}

    # Persist
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'coefficients.json')
    with open(out_path, 'w') as f:
        json.dump(artifact, f, indent=2)
    print(f'\nwrote {out_path}')

    # Brief inference snippet next to it
    snippet = '''"""Apply the persisted blender at inference time.

    import json, math
    with open('data/models/model_market_blender/coefficients.json') as f:
        coefs = json.load(f)['checkpoints']

    def logit(p, eps=1e-4):
        p = min(max(p, eps), 1 - eps)
        return math.log(p / (1 - p))

    def sigmoid(z):
        return 1.0 / (1.0 + math.exp(-z))

    def blend(p_model, p_market, checkpoint):
        c = coefs[str(int(checkpoint))]
        z = c['a'] + c['b'] * logit(p_model) + c['c'] * logit(p_market)
        return sigmoid(z)

Bet rule (v1):
    edge = blend(p_model, p_market, T) - p_market
    league must be in coefficients.json's selection.league_allowlist.
    Bet when |edge| >= 0.10. Quarter-Kelly with 2% cap.
"""
'''
    with open(os.path.join(OUT_DIR, 'README.txt'), 'w') as f:
        f.write(snippet)
    print(f'wrote {os.path.join(OUT_DIR, "README.txt")}')


if __name__ == '__main__':
    main()

"""Build the calibration / edge eval table.

One row per (condition_id, checkpoint) where:
  - the market joined to an OE gameid (status=ok in markets_oe_joined.csv)
  - the checkpoint is reachable: T*60 < gamelength
  - the unified Polymarket price has at least one fill at-or-before checkpoint_t
    within MAX_STALENESS_S

Columns:
  oe_gameid, condition_id, event_slug, league, checkpoint,
  checkpoint_t_unix,
  oe_team_a, oe_team_b,                    # team_a := the team mapped from outcome_0
  market_p_team_a,                         # unified P(outcome_0 wins) at checkpoint
  staleness_s, n_fills_pm60s,              # diagnostics
  p_model_team_a, p_model_calibrated_kind, # baseline_logistic_gpr production
  realized_team_a_won,                     # OE result for team_a
  gamelength_s

Output: data/polymarket/eval_table.csv
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import json
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

# Make scripts.* importable
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from scripts.models.baseline_logistic_gpr import prepare_Xy, predict_proba, numeric_cols_in

CHECKPOINTS = [10, 15, 20, 25]
MAX_STALENESS_S = 5 * 60        # 5 minutes — beyond this we drop the row
DENSITY_WINDOW_S = 60            # for n_fills_pm60s diagnostic

OUT = 'data/polymarket/eval_table.csv'
PROD = 'data/models/baseline_logistic_gpr/production'
PROCESSED = 'data/processed'


def main():
    joined = pd.read_csv('data/polymarket/markets_oe_joined.csv')
    joined = joined[joined['status'] == 'ok'].reset_index(drop=True)
    print(f'joined ok markets: {len(joined)}')
    cids = set(joined['condition_id'])
    gameids = set(joined['oe_gameid'])
    print(f'  unique cids: {len(cids)}, unique gameids: {len(gameids)}')

    # OE meta + per-team results
    oe_frames = []
    for yr in [2022, 2023, 2024, 2025, 2026]:
        df = pd.read_csv(
            f'data/lol-historical-data/{yr}_LoL_esports_match_data_from_OraclesElixir.csv',
            usecols=['gameid', 'date', 'gamelength', 'league', 'position',
                     'teamname', 'result'],
            low_memory=False,
        )
        df = df[df['position'] == 'team']
        oe_frames.append(df)
    oe = pd.concat(oe_frames, ignore_index=True)
    oe = oe[oe['gameid'].isin(gameids)].copy()
    oe['date'] = pd.to_datetime(oe['date'], utc=True, errors='coerce')
    oe_meta = oe.groupby('gameid').agg(
        date=('date', 'first'),
        gamelength=('gamelength', 'first'),
        league=('league', 'first'),
    ).to_dict('index')
    # per-team result: {(gameid, teamname) -> 0/1}
    oe_result = {(r['gameid'], r['teamname']): int(r['result'])
                 for _, r in oe.iterrows()}
    print(f'  OE meta loaded for {len(oe_meta)} gameids')

    # ---- Run baseline_logistic_gpr inference on the joined gameids per checkpoint ----
    # p_model[(T, gameid, teamname)] = P(teamname wins) under production model.
    p_model: dict[tuple[int, str, str], float] = {}
    cal_kinds: dict[int, str | None] = {}
    for T in CHECKPOINTS:
        ckpt_dir = os.path.join(PROD, f'checkpoint_{T}')
        with open(os.path.join(ckpt_dir, 'model.pkl'), 'rb') as f:
            payload = pickle.load(f)
        with open(os.path.join(ckpt_dir, 'calibrator.pkl'), 'rb') as f:
            calibrator = pickle.load(f)
        with open(os.path.join(ckpt_dir, 'metadata.json')) as f:
            meta = json.load(f)
        cal_kinds[T] = meta.get('calibrator_kind')

        df = pd.read_csv(os.path.join(PROCESSED, f'checkpoint_{T}_gpr.csv'),
                         low_memory=False)
        df = df[df['gameid'].isin(gameids)].reset_index(drop=True)
        if len(df) == 0:
            print(f'  T={T}: no processed rows for joined gameids')
            continue
        X, _, _ = prepare_Xy(df, T)
        numeric_cols = numeric_cols_in(X)
        scaler = payload['scaler']
        model = payload['model']
        mu = payload['mu_gpr']
        probs = predict_proba(scaler, model, mu, X, numeric_cols)
        if calibrator is not None:
            probs = np.asarray(calibrator.predict(probs))
        n_filled = 0
        for i, row in df.iterrows():
            p_model[(T, row['gameid'], row['teamname'])] = float(probs[i])
            n_filled += 1
        print(f'  T={T}: model probs filled for {n_filled} (gameid,teamname) rows  '
              f'(calibrator={cal_kinds[T] or "none"})')

    # ---- Stream trades.csv for our cids; build unified p0 series per cid ----
    print('streaming trades.csv...')
    parts = []
    for chunk in pd.read_csv('data/polymarket/trades.csv', chunksize=500_000,
                             dtype={'condition_id': str, 'asset': str,
                                    'transaction_hash': str, 'proxy_wallet': str}):
        sub = chunk[chunk['condition_id'].isin(cids)]
        if len(sub):
            parts.append(sub[['condition_id', 'timestamp_unix', 'price',
                              'outcome_index']].copy())
    trades = pd.concat(parts, ignore_index=True)
    print(f'  loaded {len(trades)} fills for {trades["condition_id"].nunique()} cids')
    # Unified P(outcome_0): outcome_idx=0 -> price; idx=1 -> 1 - price
    trades['p0'] = np.where(trades['outcome_index'].astype(int) == 0,
                            trades['price'],
                            1.0 - trades['price'])
    trades = trades.sort_values(['condition_id', 'timestamp_unix']).reset_index(drop=True)
    # Index by cid for fast slicing
    cid_groups = {cid: g for cid, g in trades.groupby('condition_id')}

    # ---- Build eval rows ----
    rows = []
    skipped = defaultdict(int)
    for _, r in joined.iterrows():
        cid = r['condition_id']
        gid = r['oe_gameid']
        meta = oe_meta.get(gid)
        if meta is None or pd.isna(meta['date']) or pd.isna(meta['gamelength']):
            skipped['no_oe_meta'] += 1
            continue
        oe_date = meta['date']  # tz-aware UTC
        glen = int(meta['gamelength'])
        league = meta['league']

        team_a, team_b = r['oe_team_a'], r['oe_team_b']
        team_a_won = oe_result.get((gid, team_a))
        if team_a_won is None:
            skipped['no_team_result'] += 1
            continue

        tr = cid_groups.get(cid)
        if tr is None or len(tr) == 0:
            skipped['no_trades'] += 1
            continue
        ts_arr = tr['timestamp_unix'].to_numpy()
        p0_arr = tr['p0'].to_numpy()

        for T in CHECKPOINTS:
            if T * 60 >= glen:
                skipped[f'past_game_end_T{T}'] += 1
                continue
            cp_ts = int(oe_date.timestamp()) + T * 60
            # last fill at-or-before
            idx = np.searchsorted(ts_arr, cp_ts, side='right') - 1
            if idx < 0:
                skipped[f'no_fill_before_T{T}'] += 1
                continue
            stale = cp_ts - int(ts_arr[idx])
            if stale > MAX_STALENESS_S:
                skipped[f'too_stale_T{T}'] += 1
                continue
            mkt_p_a = float(p0_arr[idx])
            # density: fills in [cp_ts - DENSITY, cp_ts + DENSITY]
            lo = np.searchsorted(ts_arr, cp_ts - DENSITY_WINDOW_S, side='left')
            hi = np.searchsorted(ts_arr, cp_ts + DENSITY_WINDOW_S, side='right')
            n_density = int(hi - lo)

            pm = p_model.get((T, gid, team_a))
            if pm is None:
                skipped[f'no_model_prob_T{T}'] += 1
                continue

            rows.append({
                'oe_gameid': gid,
                'condition_id': cid,
                'event_slug': r['event_slug'],
                'league': league,
                'checkpoint': T,
                'checkpoint_t_unix': cp_ts,
                'oe_team_a': team_a,
                'oe_team_b': team_b,
                'market_p_team_a': mkt_p_a,
                'staleness_s': stale,
                'n_fills_pm60s': n_density,
                'p_model_team_a': pm,
                'p_model_calibrator': cal_kinds[T] or '',
                'realized_team_a_won': team_a_won,
                'gamelength_s': glen,
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f'\nwrote {OUT}')
    print(f'  rows: {len(out)}')
    print(f'  unique markets: {out["condition_id"].nunique()}')
    print(f'  unique games:   {out["oe_gameid"].nunique()}')
    print('  rows per checkpoint:')
    print(out['checkpoint'].value_counts().sort_index().to_string())
    print('  skipped:')
    for k, v in sorted(skipped.items(), key=lambda x: -x[1]):
        print(f'    {k:30s} {v}')

    # Quick sanity: distribution of edges
    out['edge'] = out['p_model_team_a'] - out['market_p_team_a']
    print('\n  edge (p_model - p_market) percentiles:')
    print(out['edge'].quantile([.01, .05, .25, .5, .75, .95, .99]).to_string())
    print(f'  mean abs edge: {out["edge"].abs().mean():.4f}')
    print(f'  fraction |edge| > 0.05: {(out["edge"].abs() > 0.05).mean():.3f}')
    print(f'  fraction |edge| > 0.10: {(out["edge"].abs() > 0.10).mean():.3f}')


if __name__ == '__main__':
    main()

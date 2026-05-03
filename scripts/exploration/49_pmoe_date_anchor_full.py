"""Validate OE 'date' as game start across ALL joined markets, grouped by league.

For each ok-joined market: find the earliest timestamp at which the polymarket
price pins to its resolution (>=0.99 or <=0.01) and stays pinned through the
final fill. Compare to (oe_date + gamelength) -> offset_s. Distribution of
offset_s per league tells us whether timing is trustworthy and where it leaks.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from _tee import tee_stdout
tee_stdout('49_pmoe_date_anchor_full.txt')

import numpy as np
import pandas as pd

PIN_HI = 0.99
PIN_LO = 0.01

joined = pd.read_csv('data/polymarket/markets_oe_joined.csv')
joined = joined[joined['status'] == 'ok'].copy()
print(f'joined ok markets: {len(joined)}')

# OE: date + gamelength
oe_frames = []
for yr in [2022, 2023, 2024, 2025, 2026]:
    df = pd.read_csv(
        f'data/lol-historical-data/{yr}_LoL_esports_match_data_from_OraclesElixir.csv',
        usecols=['gameid', 'date', 'gamelength', 'position'],
        low_memory=False,
    )
    df = df[df['position'] == 'team']
    oe_frames.append(df)
oe = pd.concat(oe_frames, ignore_index=True)
oe['date'] = pd.to_datetime(oe['date'], utc=True, errors='coerce')
oe_meta = oe.groupby('gameid').agg(
    date=('date', 'first'),
    gamelength=('gamelength', 'first'),
).reset_index()

joined = joined.merge(oe_meta, left_on='oe_gameid', right_on='gameid',
                       how='left')
joined = joined.dropna(subset=['date', 'gamelength']).copy()
joined['oe_end'] = joined['date'] + pd.to_timedelta(
    joined['gamelength'].astype(int), unit='s')
print(f'with OE meta: {len(joined)}')

cid_to_idx = {cid: i for i, cid in enumerate(joined['condition_id'])}
target_cids = set(cid_to_idx)

# Stream trades and bucket per cid -> (min_ts_pinned_run_start, last_p0)
print('streaming trades.csv...')
parts = []
for chunk in pd.read_csv('data/polymarket/trades.csv', chunksize=500_000,
                         dtype={'condition_id': str, 'asset': str},
                         usecols=['condition_id', 'timestamp_unix', 'price',
                                  'outcome_index']):
    sub = chunk[chunk['condition_id'].isin(target_cids)]
    if len(sub):
        parts.append(sub)
trades = pd.concat(parts, ignore_index=True)
print(f'fills loaded: {len(trades)}')
trades['ts'] = pd.to_datetime(trades['timestamp_unix'], unit='s')
trades['p0'] = np.where(trades['outcome_index'].astype(int) == 0,
                         trades['price'], 1.0 - trades['price'])
trades = trades.sort_values(['condition_id', 'ts']).reset_index(drop=True)


def find_first_pin(sub_p0, sub_ts, oe_date):
    """Earliest ts at/after oe_date such that all subsequent rows are pinned
    on the side the final fill resolved to. Returns (first_pin_ts, final_p0)."""
    if len(sub_p0) == 0:
        return pd.NaT, float('nan')
    final_p = sub_p0[-1]
    if 0.5 < final_p:
        cond = sub_p0 >= PIN_HI
    else:
        cond = sub_p0 <= PIN_LO
    after = sub_ts >= oe_date
    valid_idx = np.where(after)[0]
    if len(valid_idx) == 0:
        return pd.NaT, final_p
    n = len(sub_p0)
    # cond_all_after[i] = True iff cond[i:] all True. Compute via backward AND.
    cond_all_after = np.empty(n, dtype=bool)
    cond_all_after[-1] = cond[-1]
    for i in range(n - 2, -1, -1):
        cond_all_after[i] = cond[i] and cond_all_after[i + 1]
    cand = valid_idx[cond_all_after[valid_idx]]
    if len(cand) == 0:
        return pd.NaT, final_p
    return sub_ts[cand[0]], final_p


print('computing first_pin per market...')
results = []
for cid, grp in trades.groupby('condition_id', sort=False):
    j = joined[joined['condition_id'] == cid].iloc[0]
    p0_arr = grp['p0'].to_numpy()
    ts_arr = grp['ts'].to_numpy()
    first_pin, final_p = find_first_pin(p0_arr, ts_arr,
                                         np.datetime64(j['date'].tz_convert(None)))
    if pd.notna(first_pin):
        offset_s = (pd.Timestamp(first_pin) -
                     j['oe_end'].tz_convert(None)).total_seconds()
    else:
        offset_s = float('nan')
    results.append({
        'condition_id': cid,
        'league': j['oe_league'],
        'event_slug': j['event_slug'],
        'n_fills': len(grp),
        'oe_date': j['date'],
        'gamelength_s': int(j['gamelength']),
        'first_pin': first_pin,
        'final_p0': final_p,
        'offset_s': offset_s,
    })
res = pd.DataFrame(results)
print(f'markets analyzed: {len(res)}  with first_pin: {res["offset_s"].notna().sum()}')

# Filter: only well-resolved markets (final pinned; n_fills sane)
ok = res[res['final_p0'].notna() &
         ((res['final_p0'] >= PIN_HI) | (res['final_p0'] <= PIN_LO)) &
         res['offset_s'].notna() &
         (res['n_fills'] >= 20)].copy()
print(f'cleanly resolved (>=20 fills, pinned): {len(ok)}')


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float('nan')


print('\n=== overall offset_s = first_pin - (oe_date + gamelength) ===')
o = ok['offset_s'].to_numpy()
print(f'  n={len(o)}  mean={o.mean():+.1f}s  median={pct(o, 50):+.1f}s')
print(f'  p05={pct(o,5):+.1f}  p25={pct(o,25):+.1f}  '
      f'p75={pct(o,75):+.1f}  p95={pct(o,95):+.1f}')
for thr in [30, 60, 120, 300]:
    print(f'  |offset|<= {thr:3d}s: {(np.abs(o) <= thr).mean():6.1%}')

print('\n=== per-league offset_s distribution ===')
print(f'  {"league":>10s} | {"n":>4s} | {"mean":>7s} | {"med":>7s} | '
      f'{"p25":>7s} | {"p75":>7s} | {"|<=60s|":>8s} | {"|<=300s|":>9s}')
for lg, grp in ok.groupby('league'):
    if len(grp) < 5:
        continue
    o = grp['offset_s'].to_numpy()
    print(f'  {lg:>10s} | {len(o):4d} | {o.mean():+7.1f} | {pct(o,50):+7.1f} '
          f'| {pct(o,25):+7.1f} | {pct(o,75):+7.1f} | '
          f'{(np.abs(o) <= 60).mean():8.1%} | {(np.abs(o) <= 300).mean():9.1%}')

# Outlier inspection: top 10 |offset|
print('\n=== top 15 outliers by |offset_s| ===')
worst = ok.assign(abs_off=ok['offset_s'].abs()).nlargest(15, 'abs_off')
for _, r in worst.iterrows():
    print(f'  {r["event_slug"]:40s} lg={r["league"]:>6s}  '
          f'n_fills={r["n_fills"]:5d}  glen={r["gamelength_s"]:5d}s  '
          f'offset={r["offset_s"]:+8.1f}s  final={r["final_p0"]:.3f}')

# Save full table for downstream
ok[['condition_id', 'league', 'event_slug', 'n_fills', 'oe_date',
    'gamelength_s', 'first_pin', 'final_p0', 'offset_s']].to_csv(
    'data/exploration/49_pmoe_date_anchor_full.csv', index=False)
print('\nwrote data/exploration/49_pmoe_date_anchor_full.csv')
